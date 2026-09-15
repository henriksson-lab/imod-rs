//! Translation of `IMOD/include/iimage.h` and `IMOD/libiimod/iimage.c`.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, ADOC_ZVALUE_NAME, adoc_get_collection_name, adoc_get_image_meta_info,
    adoc_get_num_collections, adoc_get_number_of_sections, adoc_get_section_name, adoc_new,
    adoc_read, adoc_set_current, adoc_transfer_section,
};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{b3d_error, b3d_output_file_type};
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
    MAX_TIFF_THREADS, Tiff, ii_tiff_check, tiff_filter_warnings, tiff_get_max_eer_super_res,
    tiff_num_read_threads, tiff_open_new, tiff_parallel_read, tiff_set_eer_read_properties,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_HALF_FLOAT,
    MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_complex_smin_smax, mrc_getdcsize, mrc_head_new,
};
use core::ffi::c_char;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicI32, Ordering};
use std::cell::RefCell;
use std::sync::{LazyLock, Mutex};

/// C `IISectionFunc` (`iimage.h`): `int (*)(ImodImageFile *, char *buf, int)`.
///
/// `buf` is a pixel buffer, not a string, and it stays a raw pointer: the
/// Fortran bridge in `unit_fileio.rs` receives the array from Fortran with no
/// length at all (`iiuReadSecPart` takes `array` and a line width), so there is
/// no slice to build there.  It is `*mut u8` rather than `*mut c_char` because
/// nothing about it is a C string.
pub type IiSectionFunc = Option<unsafe fn(*mut ImodImageFile, *mut u8, i32) -> i32>;
pub type IiFileCheckFunction = Option<unsafe fn(*mut ImodImageFile) -> i32>;
/// C `IIRawCheckFunction` (`iimage.h`).  The registry is crate-private Rust
/// state, so probes receive their file and result through ordinary borrows.
/// No foreign caller can install or invoke one of these entries.
pub type IiRawCheckFunction = Option<fn(&mut ImodFile, &[u8], &mut RawImageInfo) -> i32>;

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
/// Registered image format probes, in the same ordered process-wide sequence
/// used by `iimage.c`.  Functions are copied out before invoking them so a
/// callback cannot retain the registry lock.
static S_CHECK_LIST: LazyLock<Mutex<Vec<IiFileCheckFunction>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
/// Borrowed `ImodImageFile` identities for currently opened files.  The list
/// never owns the pointed-to image files; their addresses are only retained to
/// support the legacy `FILE *` to image-file lookup API.
static S_OPENED_FILES: LazyLock<Mutex<Vec<usize>>> = LazyLock::new(|| Mutex::new(Vec::new()));
thread_local! {
    /// TIFF thread copies are owned by the existing image-file lifecycle; these
    /// are borrowed cursors kept per calling thread for parallel section reads.
    static S_TIFF_THREADS: RefCell<([Option<NonNull<ImodImageFile>>; MAX_TIFF_THREADS], i32)> =
        const { RefCell::new(([None; MAX_TIFF_THREADS], 0)) };
}
/// Process-wide C callback registration.  Calls snapshot this value before
/// entering foreign code so a callback may safely register a replacement.
static S_QUIT_CHECK_FUNC: Mutex<Option<unsafe fn(i32) -> i32>> = Mutex::new(None);

/// One dataset in an HDF image stack.  It is entirely crate-owned state: HDF5
/// receives the identifier and a temporary C-compatible path separately.
#[derive(Clone)]
pub struct StackSetData {
    /// Dataset path owned as ordinary Rust text.  A temporary `CString` is
    /// constructed only for an HDF5 call.
    pub name: Option<String>,
    pub dset_id: i64,
    pub is_open: bool,
}

/// Crate-owned image-file state, derived from C `ImodImageFile` (`iimage.h`).
///
/// This is deliberately not a C-layout type: it contains Rust-owned strings,
/// collections, and file handles.  The callback pointers are an internal
/// dispatch table; no foreign library receives this structure by value or
/// relies on its field offsets.
///
/// `Clone` stands in for `iiCopyOpen`'s `memcpy` (`iimage.c:541`): three of the
/// fields the source copies bitwise now own heap storage, so a bitwise copy
/// would give two structs the same buffer and a double free (NATIVE.md 4d,
/// disguise 3).  Cloning duplicates them instead, and the four fields the
/// source clears right afterwards are cleared just the same.
#[derive(Clone)]
pub struct ImodImageFile {
    /// C `char *filename`, represented as text within Rust.  `None` is the
    /// source's NULL; a NUL-terminated temporary is made only at an external
    /// C API boundary.
    pub filename: Option<String>,
    /// C `char fmode[4]`, represented as an ordinary Rust file-mode string.
    pub fmode: String,
    /// C `FILE *fp`.  See [`ImodFile`]; `None` is NULL, and the four places
    /// the source stores a non-file identity here use [`ImodFile::Token`].
    pub fp: Option<ImodFile>,
    /// C `char *description`: the TIFF `ImageDescription`, built out of the MRC
    /// labels by `tiffSyncFromMrcHeader` (`iitif.c:775-809`). This is owned
    /// metadata text without a C terminator; the TIFF boundary adds one only
    /// while calling libtiff.
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
    /// Opaque handle owned by an external image backend.  It is used only for
    /// libtiff's `TIFF *`; crate-owned backend data has a typed field instead.
    pub backend_handle: *mut Tiff,
    /// Decoded pixels owned by the native raster-image backend.  Pixels are
    /// top-to-bottom, with either one luminance byte or three RGB bytes each.
    pub native_image_pixels: Option<Vec<u8>>,
    /// Whether [`Self::native_image_pixels`] contains RGB triples.
    pub native_image_rgb: bool,
    // `HANDLE` on Windows and an `int` file descriptor on POSIX (`iimage.h`).
    // Keeping a pointer-sized slot is required for the Windows mapping handle.
    pub shr_mem_file: isize,
    /// C `char *userData`: `iishrmem.c` stores the `mmap` base address here and
    /// does pointer arithmetic on it.  Not a string.
    pub user_data: *mut u8,
    pub user_flags: u32,
    pub user_count: i32,
    /// TIFF palette data, owned by this image file.
    pub colormap: Option<Vec<u8>>,
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
    /// TIFF directory numbers collected while inspecting a multi-directory
    /// file.  This is internal Rust ownership.
    pub directory_nums: Option<Vec<i32>>,
    /// HDF stack dataset entries owned by this image file.
    pub stack_set_list: Option<Vec<StackSetData>>,
    /// Maps Z sections to HDF datasets.  This was a manually managed C array;
    /// it is crate-owned state, so keep it as an owned Rust collection.
    pub z_to_data_set_map: Vec<i32>,
    pub z_map_size: i32,
    /// C `char *datasetName`, represented as text until it crosses the HDF5
    /// boundary.
    pub dataset_name: Option<String>,
    // `iimage.h` declared this `int`, but it is passed to HDF5 as a `hid_t`.
    // Keep the native HDF identifier width so IDs from current HDF5 are not truncated.
    pub dataset_id: i64,
    pub dataset_is_open: i32,
    pub num_volumes: i32,
    /// The related HDF volume image records.  The records themselves remain
    /// address-stable allocations because the image API hands out their
    /// addresses.  A vacant slot records a volume that has been deleted while
    /// other legacy image cursors remain open.
    pub ii_volumes: Vec<Option<NonNull<ImodImageFile>>>,
    /// Address-stable storage for secondary HDF volume records.  `ii_volumes`
    /// holds typed, non-null legacy cursors exposed to image APIs.
    pub owned_hdf_volumes: Vec<Box<ImodImageFile>>,
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
    pub clean_up: Option<unsafe fn(*mut ImodImageFile)>,
    pub close: Option<unsafe fn(*mut ImodImageFile)>,
    pub reopen: Option<unsafe fn(*mut ImodImageFile) -> i32>,
    pub fill_mrc_header: Option<unsafe fn(*mut ImodImageFile, *mut MrcHeader) -> i32>,
    pub sync_from_mrc_header: Option<unsafe fn(*mut ImodImageFile, *mut MrcHeader) -> i32>,
    pub write_header: Option<unsafe fn(*mut ImodImageFile) -> i32>,
    /// Rust-owned MRC header storage used by the native MRC and like-MRC
    /// backends.
    pub mrc_header: Option<Box<MrcHeader>>,
}

impl Default for ImodImageFile {
    /// `iiNew` (`iimage.c:96`) `malloc`s and then `memset`s the whole struct to
    /// zero before setting its non-zero fields; this is that `memset`, written
    /// out because a struct carrying an `Option<ImodFile>` cannot be produced
    /// by `mem::zeroed` (NATIVE.md 4b).
    fn default() -> ImodImageFile {
        ImodImageFile {
            filename: None,
            fmode: String::new(),
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
            backend_handle: core::ptr::null_mut(),
            native_image_pixels: None,
            native_image_rgb: false,
            shr_mem_file: 0,
            user_data: core::ptr::null_mut(),
            user_flags: 0,
            user_count: 0,
            colormap: None,
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
            directory_nums: None,
            stack_set_list: None,
            z_to_data_set_map: Vec::new(),
            z_map_size: 0,
            dataset_name: None,
            dataset_id: 0,
            dataset_is_open: 0,
            num_volumes: 0,
            ii_volumes: Vec::new(),
            owned_hdf_volumes: Vec::new(),
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
            mrc_header: None,
        }
    }
}

/// Crate-owned raw-image probe result, derived from C `RawImageInfo`.
///
/// It is passed only between Rust format probes and setup code, so it has no
/// foreign-layout contract.
#[derive(Clone, Copy, Default)]
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
#[derive(Default)]
pub struct LineProcData {
    pub x_start: i32,
    pub x_end: i32,
    pub convert: i32,
    /// Borrowed input cursor into an external image backend's mapped/read
    /// buffer.  It does not own allocation because MRC, HDF5, and POSIX
    /// shared-memory backends retarget it to their current input chunk.
    pub bdata: *mut u8,
    /// Borrowed base of the caller's output buffer.  The mutable output cursor
    /// is stored as `bufp_offset`, so section processing never keeps two raw
    /// pointers for the same owned output allocation.
    pub buf: *mut u8,
    pub bufp_offset: isize,
    /// Pixel-conversion lookup data, owned for the duration of a section read.
    pub map: Vec<u8>,
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

pub fn init_check_list() -> i32 {
    let mut checks = S_CHECK_LIST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !checks.is_empty() {
        return 0;
    }
    let initial_checks: [IiFileCheckFunction; 6] = [
        Some(ii_tiff_check),
        Some(ii_mrc_check),
        Some(ii_like_mrc_check),
        Some(hdf_check_callback),
        Some(ii_jpeg_check),
        Some(ii_adoc_check),
    ];
    checks.extend(initial_checks);
    drop(checks);
    tiff_filter_warnings();
    0
}
pub fn ii_add_check_function(func: IiFileCheckFunction) {
    if init_check_list() == 0 {
        S_CHECK_LIST
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(func);
    }
}
pub fn ii_insert_check_function(func: IiFileCheckFunction, index: i32) {
    if init_check_list() != 0 {
        return;
    }
    let mut checks = S_CHECK_LIST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if index < 0 {
        return;
    }
    if (index as usize) < checks.len() {
        checks.insert(index as usize, func);
    } else {
        checks.push(func);
    }
}
pub fn ii_delete_check_list() {
    S_CHECK_LIST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}
/// Matches C `iiRegisterQuitCheck(int (*)(int))` (`iimage.c:121`).
pub fn ii_register_quit_check(func: Option<unsafe fn(i32) -> i32>) {
    *S_QUIT_CHECK_FUNC.lock().unwrap() = func;
}
/// Matches C `iiCheckForQuit(int)` (`iimage.c:130`).
pub fn ii_check_for_quit(param: i32) -> i32 {
    let func = *S_QUIT_CHECK_FUNC.lock().unwrap();
    if let Some(func) = func {
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
pub fn ii_new_box() -> Box<ImodImageFile> {
    let mut ofile = Box::new(ImodImageFile::default());
    ofile.xscale = 1.0;
    ofile.yscale = 1.0;
    ofile.zscale = 1.0;
    ofile.slope = 1.0;
    ofile.smax = 255.0;
    ofile.axis = 3;
    ofile.mirror_fft = 0;
    ofile.any_tiff_pix_size = 0;
    ofile.raw_palette_bytes = 0;
    ofile.tiff_compression = 1;
    ofile.format = IIFILE_UNKNOWN;
    ofile.fp = None;
    ofile.read_section = None;
    ofile.read_section_byte = None;
    ofile.read_section_ushort = None;
    ofile.read_section_float = None;
    ofile.write_section = None;
    ofile.write_section_float = None;
    ofile.fill_mrc_header = None;
    ofile.sync_from_mrc_header = None;
    ofile.write_header = None;
    ofile.clean_up = None;
    ofile.reopen = None;
    ofile.close = None;
    ofile.write_section = None;
    ofile.colormap = None;
    ofile.user_data = core::ptr::null_mut();
    ofile.shr_mem_file = 0;
    ofile.llx = 0;
    ofile.lly = 0;
    ofile.llz = 0;
    ofile.urx = -1;
    ofile.ury = -1;
    ofile.urz = -1;
    ofile.pad_left = 0;
    ofile.pad_right = 0;
    ofile.nx = 0;
    ofile.ny = 0;
    ofile.nz = 0;
    ofile.rms = -1.0;
    ofile.last_written_z = -1;
    ofile.packed4bits = 0;
    ofile.half_floats = 0;
    ofile.read_eer_as_super_res = 0;
    ofile.num_frames_in_eerfile = 0;
    ofile.antialias_eerfilter = 0;
    ofile.directory_nums = None;
    ofile.adoc_index = -1;
    ofile.global_adoc_index = -1;
    ofile.stack_set_list = None;
    ofile.z_to_data_set_map.clear();
    ofile.dataset_name = None;
    ofile.ii_volumes.clear();
    ofile.owned_hdf_volumes.clear();
    ofile.num_volumes = 0;
    ofile.hdf_compression = -1;
    ofile
}

/// Allocate an image record for the legacy raw-pointer API.
///
/// This is the explicit ownership boundary: the returned pointer owns the box
/// made by [`ii_new_box`] and must be returned exactly once to [`ii_delete`].
/// Internal callers that can retain ownership use `ii_new_box` directly.
pub fn ii_new() -> *mut ImodImageFile {
    Box::into_raw(ii_new_box())
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
    let shared_memory_name = String::from_utf8_lossy(filename);
    if shared_memory_name.starts_with(SHR_MEM_NAME_TAG) {
        if ii_shr_mem_check_size(&shared_memory_name) != 0 {
            let file = ii_shr_mem_open(&shared_memory_name, mode);
            if file.is_null() {
                return core::ptr::null_mut();
            }
            (*file).state = IISTATE_READY;
            if add_to_opened_list(&mut *file) == 0 {
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
        (*file).fp = if filename.is_empty() {
            Some(ImodFile::Stdin)
        } else {
            ImodFile::open(&*String::from_utf8_lossy(filename), mode)
        };
        if (*file).fp.is_none() || init_check_list() != 0 {
            if (*file).fp.is_none() {
                let system_error = std::io::Error::last_os_error().to_string();
                let system_error = system_error
                    .split(" (os error ")
                    .next()
                    .unwrap_or(&system_error);
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiOpen - Opening file {} ({})\n",
                        String::from_utf8_lossy(filename),
                        system_error
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
        (*file).filename = Some(String::from_utf8_lossy(filename).into_owned());
        // `iimage.c:284` copies at most three bytes of the file mode.
        (*file).fmode = mode.chars().take(3).collect();
        // `iimage.c:239` declares `err = 0`, so an empty check list leaves it
        // zero and the "unknown format" report below is not made.
        let mut err = 0;
        let checks = S_CHECK_LIST
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for check in checks {
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
                    if add_to_opened_list(&mut *file) == 0 {
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
    let shared_memory_name = String::from_utf8_lossy(filename);
    let file = if shared_memory_name.starts_with(SHR_MEM_NAME_TAG) {
        if ii_shr_mem_check_size(&shared_memory_name) == 0 {
            return core::ptr::null_mut();
        }
        file_kind = IIFILE_SHR_MEM;
        ii_shr_mem_open(&shared_memory_name, mode)
    } else {
        let new_file = ii_new();
        if new_file.is_null() {
            return core::ptr::null_mut();
        }
        (*new_file).filename = Some(String::from_utf8_lossy(filename).into_owned());
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
            IIFILE_HDF => ii_hdf_open_new(&mut *file, mode),
            IIFILE_TIFF => tiff_open_new(file),
            IIFILE_JPEG => jpeg_open_new(file),
            IIFILE_SHR_MEM => 0,
            _ => 1,
        };
    }
    if err == 0 {
        (*file).file = file_kind;
        (*file).new_file = 1;
        (*file).fmode = "rb+".into();
        (*file).state = IISTATE_READY;
        if add_to_opened_list(&mut *file) == 0 {
            return file;
        }
    }
    ii_delete(file);
    core::ptr::null_mut()
}
pub fn ii_reopen(in_file: &mut ImodImageFile) -> i32 {
    if in_file.fp.is_some() {
        return 1;
    }
    {
        if in_file.fmode.is_empty() {
            // `iimage.c:411-412`.
            in_file.fmode = "rb+".into();
        }
        if let Some(reopen) = in_file.reopen {
            if unsafe { reopen(in_file) } != 0 {
                return 2;
            }
            in_file.state = IISTATE_READY;
            add_to_opened_list(in_file);
            return 0;
        }
        let Some(name) = in_file.filename.clone() else {
            return 2;
        };
        in_file.fp = ImodFile::open(&name, &in_file.fmode);
        if in_file.fp.is_none() {
            return 2;
        }
        add_to_opened_list(in_file);
        if in_file.state != IISTATE_NOTINIT {
            in_file.state = IISTATE_READY;
            return 0;
        }
        (*in_file).format = IIFILE_UNKNOWN;
        let checks = S_CHECK_LIST
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        for check in checks {
            if unsafe { check.unwrap()(in_file) } == 0 {
                in_file.state = IISTATE_READY;
                return 0;
            }
        }
    }
    -1
}
/// Matches C `iiSetMM` (`iimage.c:454`).
pub fn ii_set_mm(
    in_file: &mut ImodImageFile,
    mut in_min: f32,
    mut in_max: f32,
    scale_max: f32,
) -> i32 {
    if in_min != in_max {
        in_file.smin = in_min;
        in_file.smax = in_max;
    }
    if in_file.smin == in_file.smax {
        in_file.smin = 0.;
        in_file.smax = 255.;
    }
    in_min = in_file.smin;
    in_max = in_file.smax;
    if in_file.format == IIFORMAT_COMPLEX {
        (in_min, in_max) = mrc_complex_smin_smax(in_min, in_max);
    }
    in_file.slope = scale_max / (in_max - in_min);
    in_file.offset = -in_min * in_file.slope;
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
        remove_from_opened_list(&mut *in_file);
        if (*in_file).state != IISTATE_NOTINIT {
            (*in_file).state = IISTATE_PARK;
        }
    }
}
/// Destroy an image record obtained from [`ii_new`] or a legacy open API.
///
/// # Safety
///
/// `in_file` must be null or the unique, still-owned raw cursor produced by
/// [`ii_new`] (or explicitly handed off to this legacy boundary by an image
/// backend).  It must not point into a locally owned `Box<ImodImageFile>`.
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
        (*in_file).colormap = None;
        // `iiNew` hands out a `Box`; reclaim it the same way, which is also
        // what runs the `Option<ImodFile>` destructor and closes the file.
        drop(Box::from_raw(in_file));
    }
}
pub fn ii_copy_open(in_file: &mut ImodImageFile) -> Option<Box<ImodImageFile>> {
    if in_file.colormap.is_some() {
        // `iimage.c:535`.
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiCopyOpen - Not allowed for a file with a colormap\n"),
        );
        return None;
    }
    // `iimage.c:541` is `memcpy(copy, inFile, sizeof(ImodImageFile))`; see the
    // type's `Clone` note for why this is a clone rather than a byte copy.
    let mut copy = Box::new(in_file.clone());
    copy.fp = None;
    copy.backend_handle = core::ptr::null_mut();
    copy.native_image_pixels = None;
    copy.native_image_rgb = false;
    copy.mrc_header = None;
    copy.owned_hdf_volumes.clear();
    copy.filename = None;
    copy.description = None;
    if in_file.state != IISTATE_NOTINIT {
        in_file.state = IISTATE_PARK;
    }
    copy.filename = in_file.filename.clone();
    if copy.filename.is_none() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiCopyOpen - Memory error copying filename\n"),
        );
        return None;
    }
    if let Some(directory_nums) = in_file.directory_nums.as_ref() {
        copy.directory_nums = Some(directory_nums.clone());
    }
    let err = ii_reopen(&mut copy);
    if err != 0 {
        // `iimage.c:569` writes `%d` with no argument at all, so the reference
        // prints whatever is in the next varargs slot.  Pass the error code the
        // source plainly meant; the garbage it actually reads is not matchable.
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiCopyOpen - Error {err} calling iiReopen on copy of file\n"),
        );
        return None;
    }
    Some(copy)
}
pub fn ii_use_tiff_threads(in_file: &mut ImodImageFile, mut max_threads: i32) -> i32 {
    if in_file.file != IIFILE_TIFF || S_TIFF_THREADS.with(|state| state.borrow().1 > 0) {
        return 0;
    }
    if max_threads <= 0 {
        max_threads = MAX_TIFF_THREADS as i32;
    }
    max_threads = tiff_num_read_threads(
        in_file.nx,
        in_file.ny,
        in_file.tiff_compression,
        max_threads,
    );
    if max_threads > 1 {
        max_threads = S_TIFF_THREADS.with(|state| {
            let mut state = state.borrow_mut();
            state.0[0] = Some(NonNull::from(in_file));
            for index in 1..max_threads {
                let Some(file) = (unsafe { state.0[0].and_then(|file| file.as_ptr().as_mut()) })
                    .and_then(ii_copy_open)
                else {
                    return index;
                };
                state.0[index as usize] = NonNull::new(Box::into_raw(file));
            }
            max_threads
        });
    }
    if max_threads > 1 {
        S_TIFF_THREADS.with(|state| state.borrow_mut().1 = max_threads);
    }
    S_TIFF_THREADS.with(|state| state.borrow().1)
}
pub fn ii_use_tiff_threads_for_fp(fp: &ImodFile, max_threads: i32) -> i32 {
    match ii_lookup_file_from_fp(fp) {
        None => 0,
        Some(file) => unsafe { ii_use_tiff_threads(&mut *file, max_threads) },
    }
}
pub fn ii_close_tiff_copies(in_file: &mut ImodImageFile) {
    S_TIFF_THREADS.with(|state| {
        let mut state = state.borrow_mut();
        if state.1 <= 0
            || state.0[0].is_none_or(|file| !core::ptr::eq(in_file, unsafe { file.as_ref() }))
        {
            return;
        }
        for index in 1..state.1 {
            unsafe { ii_delete(state.0[index as usize].expect("open TIFF copy").as_ptr()) };
        }
        state.0.fill(None);
        state.1 = 0;
    });
}
pub fn ii_close_tiff_copies_for_fp(fp: &ImodFile) {
    if let Some(file) = ii_lookup_file_from_fp(fp) {
        unsafe { ii_close_tiff_copies(&mut *file) };
    }
}
pub fn ii_open_copies_for_threads(
    file_copies: &mut [*mut ImodImageFile; MAX_TIFF_THREADS],
    max_threads: i32,
) -> i32 {
    for index in 1..max_threads {
        let Some(file) = (unsafe { file_copies[0].as_mut() }).and_then(ii_copy_open) else {
            return index;
        };
        file_copies[index as usize] = Box::into_raw(file);
    }
    max_threads
}

/// Matches C `iiFillMrcHeader(ImodImageFile *, MrcHeader *)` (`iimage.c:661`).
pub unsafe fn ii_fill_mrc_header(in_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    if in_file.is_null() {
        return 1;
    }
    let Some(fill_mrc_header) = (unsafe { (*in_file).fill_mrc_header }) else {
        return 1;
    };
    unsafe { fill_mrc_header(in_file, hdata) }
}

/// Matches C `iiSimpleFillMrcHeader(ImodImageFile *, MrcHeader *)` (`iimage.c:674`).
pub fn ii_simple_fill_mrc_header(in_file: &ImodImageFile, hdata: &mut MrcHeader) -> i32 {
    mrc_head_new(hdata, in_file.nx, in_file.ny, in_file.nz, in_file.mode);
    hdata.bytes_signed = 0;
    hdata.fp = in_file.fp.clone();
    hdata.amin = in_file.amin;
    hdata.amax = in_file.amax;
    hdata.amean = in_file.amean;
    hdata.xlen = in_file.nx as f32 * in_file.xscale;
    hdata.ylen = in_file.ny as f32 * in_file.yscale;
    hdata.zlen = in_file.nz as f32 * in_file.zscale;
    0
}

/// Stored callback adapter for image backends that expose this default fill
/// operation through the legacy C dispatch table.
pub(crate) unsafe fn ii_simple_fill_mrc_header_callback(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    let (Some(in_file), Some(hdata)) = (unsafe { in_file.as_ref() }, unsafe { hdata.as_mut() })
    else {
        return 1;
    };
    ii_simple_fill_mrc_header(in_file, hdata)
}
/// Matches C `iiSyncFromMrcHeader(ImodImageFile *, MrcHeader *)` (`iimage.c:693`).
pub fn ii_sync_from_mrc_header(in_file: &mut ImodImageFile, hdata: &mut MrcHeader) {
    let bytes_signed =
        if hdata.bytes_signed != 0 && !(in_file.file == IIFILE_TIFF && in_file.new_file != 0) {
            1
        } else {
            0
        };
    if hdata.mode != in_file.mode || hdata.mode == 0 {
        ii_mrc_mode_to_format_type(in_file, hdata.mode, bytes_signed);
    }
    in_file.nx = hdata.nx;
    in_file.ny = hdata.ny;
    in_file.nz = hdata.nz;
    in_file.amin = hdata.amin;
    in_file.amax = hdata.amax;
    in_file.amean = hdata.amean;
    in_file.rms = hdata.rms;
    in_file.xscale = 1.0;
    in_file.yscale = 1.0;
    in_file.zscale = 1.0;
    if hdata.xlen != 0.0 && hdata.mx != 0 {
        in_file.xscale = hdata.xlen / hdata.mx as f32;
    }
    if hdata.ylen != 0.0 && hdata.my != 0 {
        in_file.yscale = hdata.ylen / hdata.my as f32;
    }
    if hdata.xlen != 0.0 && hdata.mz != 0 {
        in_file.zscale = hdata.zlen / hdata.mz as f32;
    }
    in_file.xtrans = hdata.xorg;
    in_file.ytrans = hdata.yorg;
    in_file.ztrans = hdata.zorg;
    in_file.xrot = hdata.tiltangles[3];
    in_file.yrot = hdata.tiltangles[4];
    in_file.zrot = hdata.tiltangles[5];
    in_file.header_size = hdata.header_size;
    in_file.section_skip = hdata.section_skip;
    if let Some(sync_from_mrc_header) = in_file.sync_from_mrc_header {
        unsafe { sync_from_mrc_header(in_file, hdata) };
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
pub fn ii_write_header(in_file: &mut ImodImageFile) -> i32 {
    if let Some(write_header) = in_file.write_header {
        return unsafe { write_header(in_file) };
    }
    0
}
/// Matches C `iiAddToOpenedList(ImodImageFile *)` (`iimage.c:783`).
pub fn ii_add_to_opened_list(ii_file: &mut ImodImageFile) -> i32 {
    add_to_opened_list(ii_file)
}

/// Matches C static `addToOpenedList(ImodImageFile *)` (`iimage.c:791`).
pub fn add_to_opened_list(ii_file: &mut ImodImageFile) -> i32 {
    S_OPENED_FILES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(ii_file as *mut ImodImageFile as usize);
    0
}

/// Matches C static `removeFromOpenedList(ImodImageFile *)` (`iimage.c:801`).
pub fn remove_from_opened_list(ii_file: &mut ImodImageFile) {
    let index = unsafe { find_file_in_list(ii_file, None) };
    if index >= 0 {
        S_OPENED_FILES
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(index as usize);
    }
}

/// Matches C static `findFileInList(ImodImageFile *, FILE *)` (`iimage.c:811`).
///
/// The source's `(*listPtr)->fp == fp` is an identity test on the C library's
/// own handle; [`ImodFile::ptr_eq`] is that test.
pub unsafe fn find_file_in_list(ii_file: *mut ImodImageFile, fp: Option<&ImodFile>) -> i32 {
    let opened_files = S_OPENED_FILES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (index, address) in opened_files.iter().copied().enumerate() {
        let list_pointer = address as *mut ImodImageFile;
        if (!ii_file.is_null() && list_pointer == ii_file)
            || fp.is_some_and(|fp| {
                unsafe { (*list_pointer).fp.as_ref() }.is_some_and(|other| other.ptr_eq(fp))
            })
        {
            return index as i32;
        }
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
        unsafe { remove_from_opened_list(&mut *old_file) };
    }
    // `iimage.c:834`: an HDF file carries its own address in `fp` as an
    // identity token, so relocating the struct means restamping the token.
    if unsafe { (*new_file).fp.as_ref() }
        .is_some_and(|fp| fp.ptr_eq(&ImodFile::Token(old_file as usize)))
    {
        unsafe {
            (*new_file).fp = Some(ImodFile::Token(new_file as usize));
            if (*new_file).file == IIFILE_HDF {
                if let Some(header) = (*new_file).mrc_header.as_deref_mut() {
                    header.fp = (*new_file).fp.clone();
                }
            }
        }
    }
    if !unsafe { (&(*new_file).ii_volumes).is_empty() } && unsafe { (*new_file).num_volumes } != 0 {
        for index in 0..unsafe { (*new_file).num_volumes } {
            if unsafe { (&(*new_file).ii_volumes)[index as usize] }
                .is_some_and(|volume| volume.as_ptr() == old_file)
            {
                unsafe {
                    (&mut (*new_file).ii_volumes)[index as usize] =
                        Some(NonNull::new_unchecked(new_file))
                };
            }
        }
    }
    if change_list != 0 {
        unsafe { add_to_opened_list(&mut *new_file) };
    }
}
pub fn ii_fopen(filename: &[u8], mode: &str) -> Option<ImodFile> {
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
    let file = S_OPENED_FILES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())[index as usize]
        as *mut ImodImageFile;
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
pub fn ii_fopen_volume(in_file: &mut ImodImageFile, vol_index: i32) -> Option<ImodFile> {
    if in_file.file != IIFILE_HDF || in_file.num_volumes < 2 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenVolume - Attempting to open a secondary volume for a non-HDF file or an HDF file with only a stack or one volume"
            ),
        );
        return None;
    }
    if vol_index < 1 || vol_index >= in_file.num_volumes {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenVolume - Requested volume index {vol_index} out of range\n"
            ),
        );
        return None;
    }
    let volume = in_file.ii_volumes[vol_index as usize]
        .expect("an open HDF volume has a cursor")
        .as_ptr();
    if unsafe { ii_reopen(&mut *volume) } != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiFOpenVolume - Error calling iiReopen on volume at index {vol_index}\n"
            ),
        );
        return None;
    }
    unsafe { (*volume).fp.clone() }
}
/// Matches C `iiFOpenNewVolume` (`iimage.c:925`).
pub fn ii_fopen_new_volume(in_file: &mut ImodImageFile) -> Option<ImodFile> {
    if in_file.file != IIFILE_HDF || (in_file.stack_set_list.is_some() && in_file.nz > 1) {
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
    let file = in_file.ii_volumes[(in_file.num_volumes - 1) as usize]
        .expect("the newly created HDF volume has a cursor")
        .as_ptr();
    if unsafe { add_to_opened_list(&mut *file) } != 0 {
        return None;
    }
    unsafe { (*file).fp.clone() }
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
pub fn ii_set_chunk_sizes(
    in_file: &mut ImodImageFile,
    x_size: i32,
    y_size: i32,
    z_size: i32,
) -> i32 {
    if in_file.file != IIFILE_HDF || in_file.stack_set_list.is_some() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiSetChunkSizes - Attempting to set chunk sizes for a non-HDF file or an HDF file with a stack in it\n"
            ),
        );
        return 1;
    }
    if in_file.dataset_name.is_some() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiSetChunkSizes - The volume dataset properties have already been set and cannot be changed\n"
            ),
        );
        return 1;
    }
    if x_size < 0 || y_size < 0 || z_size <= 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiSetChunkSizes - X and Y chunk sizes must be non-negative and Z size must be positive\n"
            ),
        );
        return 1;
    }
    in_file.tile_size_x = x_size;
    in_file.tile_size_y = y_size;
    in_file.z_chunk_size = z_size;
    0
}
pub fn ii_get_adoc_index(in_file: &mut ImodImageFile, global: i32, open_mdoc_or_new: i32) -> i32 {
    if in_file.file == IIFILE_HDF {
        return if in_file.stack_set_list.is_some() || in_file.global_adoc_index < 0 || global == 0 {
            in_file.adoc_index
        } else {
            in_file.global_adoc_index
        };
    }
    if in_file.adoc_index >= 0 || open_mdoc_or_new == 0 {
        return in_file.adoc_index;
    }
    if open_mdoc_or_new < 0 {
        in_file.adoc_index = adoc_new();
    } else {
        // `iimage.c:1020-1026`: the image name with ".mdoc" appended.
        let name = format!("{}.mdoc", in_file.filename.as_deref().unwrap_or_default());
        in_file.adoc_index = adoc_read(name.as_bytes());
    }
    if in_file.adoc_index < 0 {
        -2
    } else {
        in_file.adoc_index
    }
}
pub fn ii_transfer_adoc_sections(from_file: &ImodImageFile, to_file: &ImodImageFile) -> i32 {
    if from_file.adoc_index < 0 || to_file.adoc_index < 0 {
        return 1;
    }
    if adoc_set_current(from_file.adoc_index) != 0
        || adoc_transfer_section(
            ADOC_GLOBAL_NAME,
            0,
            to_file.adoc_index,
            Some(ADOC_GLOBAL_NAME),
            0,
        ) != 0
    {
        return 1;
    }
    let from_doc = if from_file.global_adoc_index >= 0 {
        from_file.global_adoc_index
    } else {
        from_file.adoc_index
    };
    let to_doc = if to_file.global_adoc_index >= 0 {
        to_file.global_adoc_index
    } else {
        to_file.adoc_index
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
pub fn ii_read_section(image: &mut ImodImageFile, buf: &mut [u8], in_section: i32) -> i32 {
    let mut bytes = 0;
    let mut channels = 0;
    if mrc_getdcsize(image.mode, &mut bytes, &mut channels) != 0 {
        // TIFF has no MRC integer mode for 32-bit signed or unsigned pixels.
        // `iiTIFFCheck` deliberately records those as mode -1, while its
        // section callback still transfers their native four-byte samples.
        // Keep that representation usable through the bounded Rust API.
        if matches!(image.type_, IITYPE_INT | IITYPE_UINT) {
            bytes = 4;
            channels = 1;
        } else {
            return IIERR_BAD_CALL;
        }
    }
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let pixel_bytes = if image.mode == MRC_MODE_HALF_FLOAT || image.half_floats != 0 {
        2
    } else {
        (bytes * channels) as usize
    };
    let Some(length) = width
        .checked_mul(rows)
        .and_then(|pixels| pixels.checked_mul(pixel_bytes))
    else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < length {
        return IIERR_BAD_CALL;
    }
    read_write_section(
        image,
        &mut buf[..length],
        in_section,
        image.read_section,
        "reading from",
    )
}

/// Raw callback retained solely for the legacy MRC callback table.
pub unsafe fn ii_read_section_callback(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let (Some(image), false) = (unsafe { in_file.as_mut() }, buf.is_null()) else {
        return IIERR_BAD_CALL;
    };
    let mut bytes = 0;
    let mut channels = 0;
    if mrc_getdcsize(image.mode, &mut bytes, &mut channels) != 0 {
        return IIERR_BAD_CALL;
    }
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let pixel_bytes = if image.mode == MRC_MODE_HALF_FLOAT || image.half_floats != 0 {
        2
    } else {
        (bytes * channels) as usize
    };
    let Some(length) = width
        .checked_mul(rows)
        .and_then(|pixels| pixels.checked_mul(pixel_bytes))
    else {
        return IIERR_BAD_CALL;
    };
    unsafe {
        ii_read_section(
            image,
            core::slice::from_raw_parts_mut(buf, length),
            in_section,
        )
    }
}
pub fn ii_read_section_byte(image: &mut ImodImageFile, buf: &mut [u8], in_section: i32) -> i32 {
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let Some(length) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < length {
        return IIERR_BAD_CALL;
    }
    read_write_section(
        image,
        &mut buf[..length],
        in_section,
        image.read_section_byte,
        "reading and converting to bytes for",
    )
}

/// Raw callback retained solely for the legacy MRC callback table.
pub unsafe fn ii_read_section_byte_callback(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let (Some(image), false) = (unsafe { in_file.as_mut() }, buf.is_null()) else {
        return IIERR_BAD_CALL;
    };
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let Some(length) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    unsafe {
        ii_read_section_byte(
            image,
            core::slice::from_raw_parts_mut(buf, length),
            in_section,
        )
    }
}

pub fn ii_read_section_ushort(image: &mut ImodImageFile, buf: &mut [u16], in_section: i32) -> i32 {
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let Some(length) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < length {
        return IIERR_BAD_CALL;
    }
    let bytes = unsafe {
        core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), length * size_of::<u16>())
    };
    read_write_section(
        image,
        bytes,
        in_section,
        image.read_section_ushort,
        "reading and converting to shorts for",
    )
}

/// Raw callback retained solely for the legacy MRC callback table.
pub unsafe fn ii_read_section_ushort_callback(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let (Some(image), false) = (unsafe { in_file.as_mut() }, buf.is_null()) else {
        return IIERR_BAD_CALL;
    };
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let Some(length) = width
        .checked_mul(rows)
        .and_then(|pixels| pixels.checked_mul(2))
    else {
        return IIERR_BAD_CALL;
    };
    unsafe {
        read_write_section(
            image,
            core::slice::from_raw_parts_mut(buf, length),
            in_section,
            image.read_section_ushort,
            "reading and converting to shorts for",
        )
    }
}
pub fn ii_read_section_float(image: &mut ImodImageFile, buf: &mut [f32], in_section: i32) -> i32 {
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let Some(pixels) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < pixels {
        return IIERR_BAD_CALL;
    }
    let bytes = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), pixels * 4) };
    read_write_section(
        image,
        bytes,
        in_section,
        image.read_section_float,
        "reading and converting to floats for",
    )
}

/// Raw callback retained solely for the legacy MRC callback table.
pub unsafe fn ii_read_section_float_callback(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let (Some(image), false) = (unsafe { in_file.as_mut() }, buf.is_null()) else {
        return IIERR_BAD_CALL;
    };
    let width = (image.urx - image.llx + 1 + image.pad_left.max(0) + image.pad_right.max(0)).max(0)
        as usize;
    let rows = (if image.axis == 2 {
        image.urz - image.llz + 1
    } else {
        image.ury - image.lly + 1
    })
    .max(0) as usize;
    let Some(pixels) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    unsafe {
        ii_read_section_float(
            image,
            core::slice::from_raw_parts_mut(buf.cast(), pixels),
            in_section,
        )
    }
}
pub fn ii_read_section_any(
    in_file: &mut ImodImageFile,
    buf: &mut [u8],
    in_section: i32,
    convert_to: i32,
) -> i32 {
    match convert_to {
        MRSA_NOPROC => read_write_section(
            in_file,
            buf,
            in_section,
            in_file.read_section,
            "reading from",
        ),
        MRSA_BYTE => read_write_section(
            in_file,
            buf,
            in_section,
            in_file.read_section_byte,
            "reading and converting to bytes for",
        ),
        MRSA_USHORT => read_write_section(
            in_file,
            buf,
            in_section,
            in_file.read_section_ushort,
            "reading and converting to shorts for",
        ),
        MRSA_FLOAT => read_write_section(
            in_file,
            buf,
            in_section,
            in_file.read_section_float,
            "reading and converting to floats for",
        ),
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
pub fn ii_write_section(in_file: &mut ImodImageFile, buf: &mut [u8], in_section: i32) -> i32 {
    let mut bytes = 0;
    let mut channels = 0;
    if mrc_getdcsize(in_file.mode, &mut bytes, &mut channels) != 0 {
        return IIERR_BAD_CALL;
    }
    let width = (in_file.urx - in_file.llx + 1 + in_file.pad_left.max(0) + in_file.pad_right.max(0))
        .max(0) as usize;
    let rows = (if in_file.axis == 2 {
        in_file.urz - in_file.llz + 1
    } else {
        in_file.ury - in_file.lly + 1
    })
    .max(0) as usize;
    let pixel_bytes = if in_file.mode == MRC_MODE_HALF_FLOAT || in_file.half_floats != 0 {
        2
    } else {
        (bytes * channels) as usize
    };
    let Some(length) = width
        .checked_mul(rows)
        .and_then(|pixels| pixels.checked_mul(pixel_bytes))
    else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < length {
        return IIERR_BAD_CALL;
    }
    let func = in_file.write_section;
    read_write_section(in_file, &mut buf[..length], in_section, func, "writing to")
}
pub fn ii_write_section_float(
    in_file: &mut ImodImageFile,
    buf: &mut [f32],
    in_section: i32,
) -> i32 {
    let width = (in_file.urx - in_file.llx + 1 + in_file.pad_left.max(0) + in_file.pad_right.max(0))
        .max(0) as usize;
    let rows = (if in_file.axis == 2 {
        in_file.urz - in_file.llz + 1
    } else {
        in_file.ury - in_file.lly + 1
    })
    .max(0) as usize;
    let Some(pixels) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    if buf.len() < pixels {
        return IIERR_BAD_CALL;
    }
    let func = in_file.write_section_float;
    // `f32` is four contiguous bytes, so this is a bounded view of the
    // caller-owned native float storage until the legacy section callback is
    // itself converted to a typed API.
    let bytes = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), pixels * 4) };
    read_write_section(
        in_file,
        bytes,
        in_section,
        func,
        "converting floats to write to",
    )
}
pub fn read_write_section(
    in_file: &mut ImodImageFile,
    buf: &mut [u8],
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
    {
        if in_file.fp.is_none() && ii_reopen(in_file) != 0 {
            return -1;
        }
        let width =
            (in_file.urx - in_file.llx + 1 + in_file.pad_left.max(0) + in_file.pad_right.max(0))
                .max(0) as usize;
        let rows = (if in_file.axis == 2 {
            in_file.urz - in_file.llz + 1
        } else {
            in_file.ury - in_file.lly + 1
        })
        .max(0) as usize;
        let element_bytes =
            if Some(func) == in_file.read_section || Some(func) == in_file.write_section {
                let mut bytes = 0;
                let mut channels = 0;
                if mrc_getdcsize(in_file.mode, &mut bytes, &mut channels) != 0 {
                    // See `ii_read_section`: the TIFF reader represents
                    // 32-bit integer pixels with mode -1 because MRC has no
                    // corresponding storage mode.
                    if matches!(in_file.type_, IITYPE_INT | IITYPE_UINT) {
                        bytes = 4;
                        channels = 1;
                    } else {
                        return IIERR_BAD_CALL;
                    }
                }
                if in_file.mode == MRC_MODE_HALF_FLOAT || in_file.half_floats != 0 {
                    2
                } else {
                    (bytes * channels) as usize
                }
            } else if Some(func) == in_file.read_section_byte {
                1
            } else if Some(func) == in_file.read_section_ushort {
                2
            } else if Some(func) == in_file.read_section_float
                || Some(func) == in_file.write_section_float
            {
                4
            } else {
                return IIERR_BAD_CALL;
            };
        let Some(required) = width
            .checked_mul(rows)
            .and_then(|pixels| pixels.checked_mul(element_bytes))
        else {
            return IIERR_BAD_CALL;
        };
        if buf.len() < required {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiRead/WriteSection - Buffer is too small for {} this type of file\n",
                    mess
                ),
            );
            return IIERR_BAD_CALL;
        }
        let mut data_size = 0;
        let mut convert = 0;
        if S_TIFF_THREADS.with(|state| {
            let state = state.borrow();
            state.1 > 1 && state.0[0].is_some_and(|file| core::ptr::eq(in_file, file.as_ptr()))
        }) && in_file.axis == 3
        {
            let mut dsize = 0;
            let mut csize = 0;
            if mrc_getdcsize(in_file.mode, &mut dsize, &mut csize) == 0 {
                if Some(func) == in_file.read_section {
                    data_size = dsize * csize;
                    convert = MRSA_NOPROC;
                } else if Some(func) == in_file.read_section_byte {
                    data_size = 1;
                    convert = MRSA_BYTE;
                } else if Some(func) == in_file.read_section_ushort {
                    data_size = 2;
                    convert = MRSA_USHORT;
                } else if Some(func) == in_file.read_section_float {
                    data_size = 4;
                    convert = MRSA_FLOAT;
                }
            }
        }
        ii_change_call_count(1);
        let err = if data_size != 0 {
            S_TIFF_THREADS.with(|state| {
                let mut state = state.borrow_mut();
                let mut file_copies = state
                    .0
                    .map(|file| file.map_or(core::ptr::null_mut(), NonNull::as_ptr));
                unsafe {
                    tiff_parallel_read(
                        file_copies.as_mut_ptr(),
                        state.1,
                        in_file.llx,
                        in_file.urx,
                        in_file.lly,
                        in_file.ury,
                        data_size,
                        buf.as_mut_ptr(),
                        in_section,
                        convert,
                    )
                }
            })
        } else {
            unsafe { func(in_file, buf.as_mut_ptr(), in_section) }
        };
        ii_change_call_count(-1);
        err
    }
}
/// Matches C `iiReadPoint` (`iimage.c:1214`).
pub fn ii_read_point(in_file: &mut ImodImageFile, x: i32, y: i32, z: i32) -> f32 {
    let mut value = in_file.amin;
    if x < 0 || y < 0 || z < 0 || x >= in_file.nx || y >= in_file.ny || z >= in_file.nz {
        return value;
    }
    let mut save = ImodImageFile::default();
    ii_save_load_params(in_file, &mut save);
    in_file.llx = x;
    in_file.urx = x;
    in_file.lly = y;
    in_file.ury = y;
    in_file.axis = 3;
    if in_file.mode == MRC_MODE_COMPLEX_SHORT {
        let mut data = [0i16; 2];
        let bytes = unsafe { core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), 4) };
        if ii_read_section(in_file, bytes, z) == 0 {
            value =
                ((data[0] as f64 * data[0] as f64 + data[1] as f64 * data[1] as f64).sqrt()) as f32;
        }
    } else if in_file.mode == MRC_MODE_COMPLEX_FLOAT {
        let mut data = [0f32; 2];
        let bytes = unsafe { core::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), 8) };
        if ii_read_section(in_file, bytes, z) == 0 {
            value =
                ((data[0] as f64 * data[0] as f64 + data[1] as f64 * data[1] as f64).sqrt()) as f32;
        }
    } else {
        ii_read_section_float(in_file, core::slice::from_mut(&mut value), z);
    }
    ii_restore_load_params(0, in_file, &save);
    value
}
pub fn ii_load_pcoord(
    in_file: &mut ImodImageFile,
    use_mdoc: i32,
    li: &mut crate::imod::libiimod::mrcfiles::LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    if in_file.file == IIFILE_HDF || in_file.file == IIFILE_ADOC {
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
    if unsafe { ii_mrc_check(in_file) } != 0 {
        return 0;
    }
    if use_mdoc < 2 {
        ii_mrc_load_pcoord(in_file, li, nx, ny, nz);
    }
    if li.plist == 0 && use_mdoc != 0 {
        crate::imod::libiimod::plist::ii_plist_from_metadata(
            in_file.filename.as_deref().unwrap_or_default(),
            1,
            li,
            nx,
            ny,
            nz,
        );
    }
    0
}
pub fn ii_make_buffer_convert_if_float(
    in_file: &ImodImageFile,
    float_buffer: Option<&[f32]>,
    inverted: &mut bool,
    routine: &str,
) -> Result<Option<Vec<u8>>, ()> {
    if float_buffer.is_none() || in_file.type_ == IITYPE_FLOAT {
        return Ok(None);
    }

    let nx = match usize::try_from(in_file.nx) {
        Ok(value) if value > 0 => value,
        _ => return Ok(None),
    };
    let ny = match usize::try_from(in_file.ny) {
        Ok(value) if value > 0 => value,
        _ => return Ok(None),
    };
    let pixsize = if in_file.mode == MRC_MODE_BYTE { 1 } else { 2 };
    let Some(data_size) = nx
        .checked_mul(ny)
        .and_then(|pixels| pixels.checked_mul(pixsize))
    else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: {} - Array size for converting floats overflows\n",
                routine
            ),
        );
        return Err(());
    };
    let float_buffer = float_buffer.expect("checked above");
    let Some(float_count) = nx.checked_mul(ny) else {
        return Err(());
    };
    if float_buffer.len() < float_count {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: {} - Float buffer is too short for conversion\n",
                routine
            ),
        );
        return Err(());
    }
    let mut use_buf = Vec::new();
    if use_buf.try_reserve_exact(data_size).is_err() {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: {} - Allocating array for converting floats\n",
                routine
            ),
        );
        return Err(());
    }
    use_buf.resize(data_size, 0);
    *inverted = true;
    for (source, destination) in float_buffer[..float_count]
        .chunks_exact(nx)
        .zip(use_buf.chunks_exact_mut(nx * pixsize).rev())
    {
        ii_convert_line_of_floats(source, destination, in_file.mode, false, false);
    }
    Ok(Some(use_buf))
}
pub fn ii_convert_line_of_floats(
    floats: &[f32],
    output: &mut [u8],
    mrc_mode: i32,
    bytes_signed: bool,
    pack_4bits: bool,
) {
    match mrc_mode {
        MRC_MODE_BYTE => {
            if pack_4bits {
                for (packed, values) in output.iter_mut().zip(floats.chunks(2)) {
                    let mut ival = (values[0] + 0.5) as i32;
                    ival = ival.clamp(0, 15);
                    let hval = values
                        .get(1)
                        .map_or(0, |value| ((value + 0.5) as i32).clamp(0, 15));
                    *packed = (ival + (hval << 4)) as u8;
                }
            } else if bytes_signed {
                for (value, byte) in floats.iter().zip(output.iter_mut()) {
                    // `iimage.c:1339`: `127.5` is a double literal and `floor`
                    // is the double version, so the float promotes and the
                    // whole expression evaluates in double.  Doing it in f32
                    // lands one off on values near a .5 boundary.
                    let mut ival = (*value as f64 - 127.5).floor() as i32;
                    ival = ival.clamp(-128, 127);
                    *byte = ival as i8 as u8;
                }
            } else {
                for (value, byte) in floats.iter().zip(output.iter_mut()) {
                    let mut ival = (*value + 0.5) as i32;
                    ival = ival.clamp(0, 255);
                    *byte = ival as u8;
                }
            }
        }
        MRC_MODE_SHORT => {
            for (value, bytes) in floats.iter().zip(output.chunks_exact_mut(2)) {
                // `iimage.c:1357`: `0.5` is a double literal and `floor` is the
                // double version -- unlike the unsigned arms, which use `0.5f`.
                let mut ival = (*value as f64 + 0.5).floor() as i32;
                ival = ival.clamp(-32768, 32767);
                bytes.copy_from_slice(&(ival as i16).to_ne_bytes());
            }
        }
        MRC_MODE_USHORT => {
            for (value, bytes) in floats.iter().zip(output.chunks_exact_mut(2)) {
                let mut ival = (*value + 0.5) as i32;
                ival = ival.clamp(0, 65535);
                bytes.copy_from_slice(&(ival as u16).to_ne_bytes());
            }
        }
        MRC_MODE_HALF_FLOAT => {
            let mut halves = vec![0_u16; floats.len()];
            imnp_floatbuf_to_halfs(floats, &mut halves, floats.len() as i32);
            for (half, bytes) in halves.iter().zip(output.chunks_exact_mut(2)) {
                bytes.copy_from_slice(&half.to_ne_bytes());
            }
        }
        _ => {}
    }
}
pub fn ii_save_load_params(ii_file: &ImodImageFile, ii_save: &mut ImodImageFile) {
    ii_save.llx = ii_file.llx;
    ii_save.urx = ii_file.urx;
    ii_save.lly = ii_file.lly;
    ii_save.ury = ii_file.ury;
    ii_save.llz = ii_file.llz;
    ii_save.urz = ii_file.urz;
    ii_save.axis = ii_file.axis;
    ii_save.pad_left = ii_file.pad_left;
    ii_save.pad_right = ii_file.pad_right;
    ii_save.slope = ii_file.slope;
    ii_save.offset = ii_file.offset;
}
pub fn ii_restore_load_params(
    ret_val: i32,
    ii_file: &mut ImodImageFile,
    ii_save: &ImodImageFile,
) -> i32 {
    ii_file.llx = ii_save.llx;
    ii_file.urx = ii_save.urx;
    ii_file.lly = ii_save.lly;
    ii_file.ury = ii_save.ury;
    ii_file.llz = ii_save.llz;
    ii_file.urz = ii_save.urz;
    ii_file.axis = ii_save.axis;
    ii_file.pad_left = ii_save.pad_left;
    ii_file.pad_right = ii_save.pad_right;
    ii_file.slope = ii_save.slope;
    ii_file.offset = ii_save.offset;
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
    let cstr = crate::imod::libcfshr::b3dutil::fortran_string(filename, name_len);
    native_ii_test_if_hdf(cstr.as_bytes())
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
unsafe fn hdf_check_callback(in_file: *mut ImodImageFile) -> i32 {
    in_file.as_mut().map_or(IIERR_IO_ERROR, native_ii_hdf_check)
}

/// Legacy C callback entry point.  Native Rust callers use the borrowed-image
/// APIs from `iihdf` directly.
pub unsafe fn ii_hdf_check(in_file: *mut ImodImageFile) -> i32 {
    in_file.as_mut().map_or(IIERR_IO_ERROR, native_ii_hdf_check)
}
pub unsafe fn ii_hdfopen_new(in_file: *mut ImodImageFile, mode: &str) -> i32 {
    in_file
        .as_mut()
        .map_or(1, |file| ii_hdf_open_new(file, mode))
}
pub unsafe fn hdf_write_global_adoc(in_file: *mut ImodImageFile) -> i32 {
    in_file.as_mut().map_or(1, native_hdf_write_global_adoc)
}
pub unsafe fn hdf_write_dummy_section(in_file: *mut ImodImageFile, buf: *mut u8, cz: i32) -> i32 {
    native_hdf_write_dummy_section(in_file, buf, cz)
}
pub unsafe fn ii_test_if_hdf(filename: &[u8]) -> i32 {
    native_ii_test_if_hdf(filename)
}
pub unsafe fn ii_reorder_hdfstack(in_file: *mut ImodImageFile, sect_order: *mut i32) -> i32 {
    let Some(file) = in_file.as_mut() else {
        return 1;
    };
    if sect_order.is_null() || file.nz < 0 {
        return 1;
    }
    ii_reorder_hdf_stack(
        file,
        core::slice::from_raw_parts(sect_order, file.nz as usize),
    )
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
    native_init_new_hdf_file(&mut *in_file)
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
    fn set_mm_uses_owned_image_state_and_complex_scaling() {
        let mut image = ii_new_box();
        image.format = IIFORMAT_COMPLEX;
        assert_eq!(ii_set_mm(&mut image, 3., 3., 255.), 0);
        let (complex_min, complex_max) = mrc_complex_smin_smax(0., 255.);
        assert_eq!(image.slope, 255. / (complex_max - complex_min));
        assert_eq!(image.offset, -complex_min * image.slope);
    }

    #[test]
    fn convert_line_of_floats_preserves_source_rounding_clamping_and_packing() {
        let floats = [-2., 0.49, 255.6];
        let mut bytes = [0u8; 3];
        ii_convert_line_of_floats(&floats, &mut bytes, MRC_MODE_BYTE, false, false);
        assert_eq!(bytes, [0, 0, 255]);

        let signed = [0., 127.5, 255.];
        ii_convert_line_of_floats(&signed, &mut bytes, MRC_MODE_BYTE, true, false);
        assert_eq!(bytes.map(|value| value as i8), [-128, 0, 127]);

        let packed = [1., 2., 15.];
        ii_convert_line_of_floats(&packed, &mut bytes, MRC_MODE_BYTE, false, true);
        assert_eq!(&bytes[..2], &[0x21, 15]);

        let values = [-1.1, 32767.9];
        let mut short_bytes = [0u8; 4];
        ii_convert_line_of_floats(&values, &mut short_bytes, MRC_MODE_SHORT, false, false);
        assert_eq!(
            short_bytes
                .chunks_exact(2)
                .map(|value| i16::from_ne_bytes([value[0], value[1]]))
                .collect::<Vec<_>>(),
            [-1, 32767]
        );

        let mut ushort_bytes = [0u8; 4];
        ii_convert_line_of_floats(&values, &mut ushort_bytes, MRC_MODE_USHORT, false, false);
        assert_eq!(
            ushort_bytes
                .chunks_exact(2)
                .map(|value| u16::from_ne_bytes([value[0], value[1]]))
                .collect::<Vec<_>>(),
            [0, 32768]
        );

        let half_values = [0., 1.];
        ii_convert_line_of_floats(
            &half_values,
            &mut ushort_bytes,
            MRC_MODE_HALF_FLOAT,
            false,
            false,
        );
        assert_eq!(
            ushort_bytes
                .chunks_exact(2)
                .map(|value| u16::from_ne_bytes([value[0], value[1]]))
                .collect::<Vec<_>>(),
            [0, 0x3c00]
        );
    }

    #[test]
    fn make_buffer_convert_if_float_preserves_source_y_inversion_and_identity_return() {
        let mut in_file = ii_new_box();
        in_file.nx = 2;
        in_file.ny = 2;
        in_file.mode = MRC_MODE_BYTE;
        in_file.type_ = IITYPE_UBYTE;
        let floats = [1f32, 2., 3., 4.];
        let mut inverted = false;
        let converted =
            ii_make_buffer_convert_if_float(&in_file, Some(&floats), &mut inverted, "test")
                .unwrap()
                .unwrap();
        assert!(inverted);
        assert_eq!(converted, [3, 4, 1, 2]);

        in_file.type_ = IITYPE_FLOAT;
        inverted = false;
        assert!(
            ii_make_buffer_convert_if_float(&in_file, Some(&floats), &mut inverted, "test",)
                .unwrap()
                .is_none()
        );
        assert!(!inverted);
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
            let mut image_file_owner = ii_new_box();
            let image_file = image_file_owner.as_mut() as *mut ImodImageFile;
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
            ii_sync_from_mrc_header(&mut *image_file, &mut header);
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
            ii_sync_from_mrc_header(&mut *image_file, &mut header);
            assert_eq!((*image_file).zscale, 1.0);
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

    #[test]
    fn section_dispatch_requires_a_bounded_native_buffer() {
        unsafe fn read_two_bytes(
            _image: *mut ImodImageFile,
            buffer: *mut u8,
            _section: i32,
        ) -> i32 {
            unsafe {
                let buffer = core::slice::from_raw_parts_mut(buffer, 2);
                buffer.copy_from_slice(&[17, 29]);
            }
            0
        }

        let mut image = ImodImageFile {
            fp: Some(ImodFile::Token(1)),
            nx: 2,
            ny: 1,
            nz: 1,
            llx: 0,
            urx: 1,
            lly: 0,
            ury: 0,
            llz: 0,
            urz: 0,
            axis: 3,
            mode: MRC_MODE_BYTE,
            read_section: Some(read_two_bytes),
            ..ImodImageFile::default()
        };
        let func = image.read_section;
        let mut short = [0_u8; 1];
        assert_eq!(
            read_write_section(&mut image, &mut short, 0, func, "reading from"),
            IIERR_BAD_CALL
        );
        let mut pixels = [0_u8; 2];
        assert_eq!(
            read_write_section(&mut image, &mut pixels, 0, func, "reading from"),
            0
        );
        assert_eq!(pixels, [17, 29]);

        let mut dispatched = [0_u8; 2];
        assert_eq!(
            ii_read_section_any(&mut image, &mut dispatched, 0, MRSA_NOPROC),
            0
        );
        assert_eq!(dispatched, [17, 29]);
        assert_eq!(
            ii_read_section_any(&mut image, &mut dispatched[..1], 0, MRSA_NOPROC),
            IIERR_BAD_CALL
        );
    }

    #[test]
    fn image_file_constructor_and_initializer_match_source_defaults() {
        unsafe {
            let mut image_file_owner = ii_new_box();
            let image_file = image_file_owner.as_mut() as *mut ImodImageFile;
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
        }
    }

    #[test]
    fn opened_file_registry_uses_the_original_ilist_pointer_identity_rules() {
        unsafe {
            let mut first_owner = ii_new_box();
            let first = first_owner.as_mut() as *mut ImodImageFile;
            let mut second_owner = ii_new_box();
            let second = second_owner.as_mut() as *mut ImodImageFile;
            (*first).fp = Some(ImodFile::Token(1));
            (*second).fp = Some(ImodFile::Token(2));
            assert_eq!(add_to_opened_list(&mut *first), 0);
            assert_eq!(ii_add_to_opened_list(&mut *second), 0);
            assert_eq!(find_file_in_list(first, None), 0);
            assert_eq!(
                find_file_in_list(core::ptr::null_mut(), (*second).fp.as_ref()),
                1
            );
            assert_eq!(
                ii_lookup_file_from_fp((*first).fp.as_ref().unwrap()),
                Some(first)
            );
            remove_from_opened_list(&mut *first);
            assert!(ii_lookup_file_from_fp((*first).fp.as_ref().unwrap()).is_none());
            assert_eq!(
                ii_lookup_file_from_fp((*second).fp.as_ref().unwrap()),
                Some(second)
            );
            remove_from_opened_list(&mut *second);
        }
    }

    #[test]
    fn file_address_change_replaces_the_opened_file_registry_entry() {
        unsafe {
            let mut old_file_owner = ii_new_box();
            let old_file = old_file_owner.as_mut() as *mut ImodImageFile;
            let mut new_file_owner = ii_new_box();
            let new_file = new_file_owner.as_mut() as *mut ImodImageFile;
            (*old_file).fp = Some(ImodFile::Token(1));
            (*new_file).fp = Some(ImodFile::Token(old_file as usize));
            assert_eq!(add_to_opened_list(&mut *old_file), 0);
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
            remove_from_opened_list(&mut *new_file);
        }
    }

    #[test]
    fn save_and_restore_load_params_copy_only_the_source_load_fields() {
        let mut ii_file = ii_new_box();
        let mut ii_save = ii_new_box();
        ii_file.llx = 1;
        ii_file.urx = 2;
        ii_file.lly = 3;
        ii_file.ury = 4;
        ii_file.llz = 5;
        ii_file.urz = 6;
        ii_file.axis = 2;
        ii_file.pad_left = 7;
        ii_file.pad_right = 8;
        ii_file.slope = 1.5;
        ii_file.offset = -2.5;
        ii_save_load_params(&ii_file, &mut ii_save);
        ii_file.llx = 0;
        ii_file.urx = 0;
        ii_file.lly = 0;
        ii_file.ury = 0;
        ii_file.llz = 0;
        ii_file.urz = 0;
        ii_file.axis = 0;
        ii_file.pad_left = 0;
        ii_file.pad_right = 0;
        ii_file.slope = 0.;
        ii_file.offset = 0.;
        assert_eq!(ii_restore_load_params(-7, &mut ii_file, &ii_save), -7);
        assert_eq!(
            (
                ii_file.llx,
                ii_file.urx,
                ii_file.lly,
                ii_file.ury,
                ii_file.llz,
                ii_file.urz,
                ii_file.axis,
                ii_file.pad_left,
                ii_file.pad_right,
                ii_file.slope,
                ii_file.offset,
            ),
            (1, 2, 3, 4, 5, 6, 2, 7, 8, 1.5, -2.5)
        );
    }

    #[test]
    fn lookup_ii_file_saves_and_sets_axis_specific_load_parameters() {
        unsafe {
            let mut ii_file_owner = ii_new_box();
            let ii_file = ii_file_owner.as_mut() as *mut ImodImageFile;
            (*ii_file).fp = Some(ImodFile::Token(1));
            (*ii_file).file = IIFILE_TIFF;
            (*ii_file).llx = 99;
            (*ii_file).lly = 98;
            (*ii_file).axis = 1;
            assert_eq!(add_to_opened_list(&mut *ii_file), 0);
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
            remove_from_opened_list(&mut *ii_file);
        }
    }

    #[test]
    fn quit_callback_returns_the_source_quitting_status() {
        unsafe fn quit_on_seven(value: i32) -> i32 {
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

            let path = std::env::temp_dir()
                .join(format!("imod-rs-iimage-open-{}.mrc", std::process::id()));
            let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::open(&path, "wb").unwrap();
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
            let image = ii_open(path.as_os_str().as_encoded_bytes(), "rb");
            assert!(!image.is_null());
            assert_eq!(
                ((*image).state, (*image).file, (*image).nx, (*image).ny),
                (IISTATE_READY, IIFILE_MRC, 2, 2)
            );
            let mut pixels = [0_u8; 4];
            assert_eq!(ii_read_section(&mut *image, &mut pixels, 0), 0);
            assert_eq!(pixels, [129, 130, 131, 132]);
            assert_eq!(ii_read_point(&mut *image, 1, 1, 0), 132.);
            assert_eq!(ii_read_point(&mut *image, -1, 1, 0), (*image).amin);
            ii_delete(image);
            ii_delete_check_list();
            std::fs::remove_file(path).unwrap();
        }
    }
}
