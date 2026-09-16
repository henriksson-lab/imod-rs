//! Translation of `IMOD/3dmod/imodview.cpp`, `imodview.h` and `imodviewP.h`,
//! together with the `ViewInfo`/`ivwSlice`/`imod_showslice_struct` types that
//! `imodP.h` declares for it.
//!
//! This unit owns the `ImodView` that every other `3dmod` window reads.  The
//! source's file-scope statics are kept as thread-locals with the same
//! lifetime and sharing, exactly as `mv_input.rs` does; the source's two
//! function-pointer globals (`best_ivwGetValue`, `ivwFastGetValue`) are kept
//! as function pointers so the selection logic is the source's.
//!
//! Every call this unit makes into a widget, a dialog, a preferences object,
//! an OpenGL context, or a `3dmod` unit whose state has no host yet crosses
//! [`ImodviewNativeBoundary`], whose default bodies name the missing unit
//! rather than silently dropping the action.
#![allow(dead_code, unused_variables)]

use core::ffi::c_void;
use core::ptr::{self, NonNull};
use std::cell::{Cell, RefCell};
use std::io::{Read, Write};

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, adoc_clear, adoc_get_float, adoc_get_image_meta_info, adoc_get_integer,
    adoc_get_number_of_sections, adoc_lookup_by_name_value, adoc_open_image_metadata,
    adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_addressable_memory, set_or_clear_flags};
use crate::imod::libcfshr::islice::{Islice, slice_create};
use crate::imod::libcfshr::reduce_by_binning::reduce_by_binning;
use crate::imod::libiimod::iimage::{
    IIFILE_ADOC, IIFILE_HDF, IIFILE_JPEG, IIFILE_MRC, IIFILE_QIMAGE, IIFILE_RAW, IIFILE_TIFF,
    IIFORMAT_COLORMAP, IIFORMAT_COMPLEX, IIFORMAT_LUMINANCE, IIFORMAT_RGB, IITYPE_BYTE,
    IITYPE_FLOAT, IITYPE_UBYTE, ImodImageFile, MRSA_BYTE, MRSA_NOPROC, MRSA_USHORT,
    ii_add_to_opened_list, ii_allow_multi_volume, ii_close, ii_default_min_max_mean, ii_delete,
    ii_file_change_address, ii_get_adoc_index, ii_new, ii_open, ii_open_copies_for_threads,
    ii_read_point, ii_read_section_any, ii_reopen, ii_set_mm,
};
use crate::imod::libiimod::iishrmem::IIFILE_SHR_MEM;
use crate::imod::libiimod::iitif::{
    IICOMPRESSION_EER_7BIT, IICOMPRESSION_EER_8BIT, IICOMPRESSION_NONE, tiff_num_read_threads,
    tiff_parallel_read,
};
use crate::imod::libiimod::mrcfiles::{
    IMOD_MRC_STAMP, LoadInfo, MRC_FLAGS_BAD_RMS_NEG, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT,
    MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT,
    MRC_RAMP_LIN, MrcHeader, get_short_map, mrc_fix_li, mrc_get_standard_version,
    mrc_mirror_source,
};
use crate::imod::libiimod::plist::{ii_plist_load, ii_plist_load_f};
use crate::imod::libimod::icont::imod_contours_delete;
use crate::imod::libimod::imesh::imod_meshes_delete;
use crate::imod::libimod::imodel::{
    ICONT_WILD, IMOD_UNIT_PIXEL, IMODF_FLIPYZ, IMODF_OTRANS_ORIGIN, IMODF_ROT90X, IMODF_TILTOK,
    Icont, Iindex, Imod, Iobj, Ipoint, Iref_image, imod_contour_get, imod_flip_yz,
    imod_insert_point, imod_new_contour, imod_object_get, imod_rot90x, imod_trans_from_ref_image,
};
use crate::imod::libimod::iobj::{
    IOBJ_EX_PNT_LIMIT, imod_object_default, imod_object_new, iobj_flag_time,
};
use crate::imod::three_dmod::b3dgfx::{SNAPSHOT_JPG, SNAPSHOT_PNG, SNAPSHOT_RGB, SNAPSHOT_TIF};
use crate::imod::three_dmod::control::{SLICER_WINDOW_TYPE, ZAP_WINDOW_TYPE};
use crate::imod::three_dmod::iirawimage::ii_raw_scan;
use crate::imod::three_dmod::imod::{
    APP, IMOD_CWD_PATH, IMOD_IFD_PATH, IMOD_TRANS, imod_depth, imod_error, imod_trace, wprint,
};
use crate::imod::three_dmod::imod_io::IMOD_IO_SUCCESS;
use crate::imod::three_dmod::utilities::FLIP_TO_ROTATION;
use crate::imod::three_dmod::xcramp::Cramp;

/// `IMOD_MM_TOGGLE` (`imod.h`).
pub const IMOD_MM_TOGGLE: i32 = 0;
/// `IMOD_MMOVIE` (`imod.h`).
pub const IMOD_MMOVIE: i32 = 0;
/// `IMOD_MMODEL` (`imod.h`).
pub const IMOD_MMODEL: i32 = 1;
/// `IMOD_DRAW_IMAGE` (`imod.h`).
pub const IMOD_DRAW_IMAGE: i32 = 1;
/// `IMOD_DRAW_XYZ` (`imod.h`).
pub const IMOD_DRAW_XYZ: i32 = 1 << 1;
/// `IMOD_DRAW_MOD` (`imod.h`).
pub const IMOD_DRAW_MOD: i32 = 1 << 2;
/// `IMOD_DRAW_NOSYNC` (`imod.h`).
pub const IMOD_DRAW_NOSYNC: i32 = 1 << 12;
/// `IMOD_DRAW_ALL` (`imod.h`).
pub const IMOD_DRAW_ALL: i32 = IMOD_DRAW_IMAGE | IMOD_DRAW_XYZ | IMOD_DRAW_MOD;
/// `MAX_READ_THREADS` (`imodP.h:36`).
pub const MAX_READ_THREADS: usize = 16;
/// `IFDLINE_SIZE` (`imodview.cpp:2368`).
pub const IFDLINE_SIZE: usize = 255;
/// `DEFAULT_TILE_CACHE_LIMIT` (`pyramidcache.h:16`).
pub const DEFAULT_TILE_CACHE_LIMIT: f64 = 20000.;
/// `IMOD_GHOST_SECTION` (`imodP.h`).
pub const IMOD_GHOST_SECTION: i32 = 3;
/// `IMOD_GHOST_2SHADES` (`imodP.h`).
pub const IMOD_GHOST_2SHADES: i32 = 1 << 5;

/// C `ivwSlice` (`imodP.h:100`).
pub struct IvwSlice {
    pub cz: i32,
    pub ct: i32,
    pub used: i32,
    pub sec: Islice,
}

/// C `struct imod_showslice_struct` (`imodP.h:106`).
///
/// This is viewer-owned coordinate state. It is not passed to a foreign
/// function or read as bytes, so its field order is an implementation detail
/// of the Rust view rather than a C ABI contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImodShowsliceStruct {
    pub zx1: i32,
    pub zx2: i32,
    pub zy1: i32,
    pub zy2: i32,
    pub xy1: i32,
    pub xy2: i32,
    pub xz1: i32,
    pub xz2: i32,
    pub yx1: i32,
    pub yx2: i32,
    pub yz1: i32,
    pub yz2: i32,
}

/// C `ViewInfo` / `ImodView` (`imodP.h:119`), in declaration order.
///
/// Three members deviate, each because the translated type they hold is a
/// Rust-owning type rather than the C pointer the source reallocates:
/// `extraObj`/`extraObjInUse` are `Vec` because `Iobj` in `libimod` owns its
/// contours in a `Vec` and cannot be moved by `realloc`; `tiltAngles` is a
/// `Vec<f32>` for the same reason `ivwReadAngleFile` steals an `Ilist`'s
/// buffer; and `ctrlist` is the translated `ImodControlList` value that
/// `control.rs` already owns.  `numExtraObj` is retained as a field so the
/// source's own bookkeeping is reproduced rather than inferred.
///
/// `ImodView` is never transferred across an FFI boundary: the raw view
/// pointers in the surrounding 3dmod code are Rust-internal cursors. Its
/// owned `Vec`, `Box`, and `Option` members also make a C layout actively
/// misleading, so this is a native Rust aggregate.
pub struct ImodView {
    /// `ViewInfo::idata`.
    pub idata: *mut *mut u8,
    /// Rust ownership for the source `idata` section allocation.  C keeps a
    /// separately allocated `unsigned char **` plus section buffers; these
    /// two vectors preserve those address-stable cursors while the normal
    /// Rust-native image host owns the view.
    pub idata_storage: Vec<Vec<u8>>,
    pub idata_ptrs: Vec<*mut u8>,
    pub xsize: i32,
    pub ysize: i32,
    pub zsize: i32,
    pub xysize: usize,
    pub full_xsize: i32,
    pub full_ysize: i32,
    pub full_zsize: i32,
    pub xmouse: f32,
    pub ymouse: f32,
    pub zmouse: f32,
    pub x_unbin_size: i32,
    pub y_unbin_size: i32,
    pub z_unbin_size: i32,
    pub num_times: i32,
    pub cur_time: i32,
    pub li: *mut LoadInfo,
    pub image: *mut ImodImageFile,
    pub image_list: *mut ImodImageFile,
    /// Owned backing for the multi-file image-list pointer.
    ///
    /// The pointer remains because image I/O APIs expose stable C-layout
    /// records, but it always borrows this vector after image loading.
    pub image_list_storage: Vec<ImodImageFile>,
    pub hdr: *mut ImodImageFile,
    pub model_view_vi: i32,
    pub vm_size: i32,
    pub vm_entered_as_gb: bool,
    pub vm_cache: Vec<IvwSlice>,
    pub vm_count: i32,
    pub cache_index: Vec<i32>,
    pub vm_tdim: i32,
    pub vm_tbase: i32,
    pub full_cache_flipped: i32,
    pub keep_cache_full: i32,
    pub strip_or_tile_cache: i32,
    pub loading_image: i32,
    pub doing_initial_load: i32,
    pub did_model_init_in_load: i32,
    pub xybin: i32,
    pub zbin: i32,
    pub eer_super_res: i32,
    pub eer_zbinning: i32,
    pub image_pyramid: i32,
    /// `ViewInfo::pyrCache`; the `PyramidCache` object of `pyramidcache.cpp`.
    pub pyr_cache: *mut crate::imod::three_dmod::pyramidcache::PyramidCache,
    pub max_read_threads: i32,
    pub num_read_threads: i32,
    pub file_copies: [*mut ImodImageFile; MAX_READ_THREADS],
    pub pix_size_index: Vec<i32>,
    pub adoc_pix_sizes: Vec<Vec<f32>>,
    pub bapc_zval: Vec<i32>,
    pub bapc_xpiece: Vec<i32>,
    pub bapc_ypiece: Vec<i32>,
    pub bapc_xcoord: Vec<i32>,
    pub bapc_ycoord: Vec<i32>,
    pub bapc_xsize: i32,
    pub bapc_ysize: i32,
    pub bapc_xoverlap: i32,
    pub bapc_yoverlap: i32,
    pub rampbase: i32,
    pub rampsize: i32,
    pub black: i32,
    pub white: i32,
    pub range_low: i32,
    pub range_high: i32,
    pub white_in_range: i32,
    pub black_in_range: i32,
    pub movierate: i32,
    pub xmovie: i32,
    pub ymovie: i32,
    pub zmovie: i32,
    pub tmovie: i32,
    pub movie_interval: u32,
    pub movie_running: i32,
    /// `ViewInfo::timers`, owned by the view after `ivw_init`.
    pub timers: Option<Box<crate::imod::three_dmod::workprocs::ImodWorkproc>>,
    pub doing_snap_draw: i32,
    pub slice: ImodShowsliceStruct,
    pub lslice: ImodShowsliceStruct,
    pub cramp: *mut Cramp,
    pub imod: *mut Imod,
    /// `ViewInfo::extraObj`; see the type note above.
    pub extra_obj: Vec<Iobj>,
    pub num_extra_obj: i32,
    /// `ViewInfo::extraObjInUse`; see the type note above.
    pub extra_obj_in_use: Vec<i32>,
    /// `ViewInfo::selectionList`, with each selection stored as its native index.
    pub selection_list: Vec<Iindex>,
    pub num_tilt_angles: i32,
    /// `ViewInfo::tiltAngles`; see the type note above.
    pub tilt_angles: Vec<f32>,
    /// Scratch row handles assembled from image/cache storage.  This vector
    /// owns the table, never the pixels; it is converted to the legacy raw
    /// pointer table only at a rendering boundary.
    pub line_ptrs: Vec<Option<NonNull<u8>>>,
    pub line_ptr_max: i32,
    pub blank_line: Vec<u8>,
    /// `ViewInfo::ax`, the `Autox` of `autox.cpp`.
    pub ax: *mut crate::imod::three_dmod::autox::Autox,
    /// `ViewInfo::ctrlist`; see the type note above.
    pub ctrlist: Option<crate::imod::three_dmod::control::ImodControlList>,
    /// `ViewInfo::undo`, owned by the view after `ivw_init`.
    pub undo: Option<Box<crate::imod::three_dmod::undoredo::UndoRedo>>,
    pub dim: i32,
    pub obj_moveto: i32,
    pub ghostmode: i32,
    pub ghostlast: i32,
    pub ghostdist: i32,
    pub insertmode: i32,
    pub fastdraw: i32,
    pub drawcursor: i32,
    pub ifd: i32,
    pub overlay_sec: i32,
    pub overlay_ramp: i32,
    pub which_green: i32,
    pub reverse_overlay: i32,
    pub draw_stipple: i32,
    pub track_mouse_for_plugs: i32,
    pub flippable: i32,
    pub fake_image: i16,
    pub raw_image_store: i16,
    pub int_opt_entered: i32,
    pub ushort_store: i32,
    pub rgb_store: i32,
    pub colormap_image: i32,
    pub gray_rgbs: i32,
    pub multi_file_z: i32,
    pub volume_stack: i32,
    pub scale_scan_type: i32,
    pub store_scan_in_mrc: i32,
    pub switch_to_ushort: i32,
    pub reloadable: i32,
    pub no_readable_image: i32,
    pub equal_scaling: i32,
    pub pixel_size_varies: i32,
    /// C `FILE *fp` (`imodview.h`).  See [`ImodFile`]; this one is only ever
    /// tested for non-NULL and handed to `iiPlistLoadF`.
    pub fp: Option<crate::imod::libcfshr::b3dutil::ImodFile>,
}

impl Default for ImodView {
    /// The C struct is allocated without an initializer and given its values
    /// by `ivwInit`; this is the zero state plus the members `ivwInit` sets
    /// unconditionally, so a `Default` view is already usable by the units
    /// that construct one directly.
    fn default() -> Self {
        let mut view = Self {
            idata: ptr::null_mut(),
            idata_storage: Vec::new(),
            idata_ptrs: Vec::new(),
            xsize: 0,
            ysize: 0,
            zsize: 0,
            xysize: 0,
            full_xsize: 0,
            full_ysize: 0,
            full_zsize: 0,
            xmouse: 0.,
            ymouse: 0.,
            zmouse: 0.,
            x_unbin_size: 0,
            y_unbin_size: 0,
            z_unbin_size: 0,
            num_times: 0,
            cur_time: 0,
            li: ptr::null_mut(),
            image: ptr::null_mut(),
            image_list: ptr::null_mut(),
            image_list_storage: Vec::new(),
            hdr: ptr::null_mut(),
            model_view_vi: 0,
            vm_size: 0,
            vm_entered_as_gb: false,
            vm_cache: Vec::new(),
            vm_count: 0,
            cache_index: Vec::new(),
            vm_tdim: 0,
            vm_tbase: 0,
            full_cache_flipped: 0,
            keep_cache_full: 1,
            strip_or_tile_cache: 0,
            loading_image: 0,
            doing_initial_load: 0,
            did_model_init_in_load: 0,
            xybin: 1,
            zbin: 1,
            eer_super_res: 1,
            eer_zbinning: 10,
            image_pyramid: 0,
            pyr_cache: ptr::null_mut(),
            max_read_threads: 1,
            num_read_threads: 1,
            file_copies: [ptr::null_mut(); MAX_READ_THREADS],
            pix_size_index: Vec::new(),
            adoc_pix_sizes: Vec::new(),
            bapc_zval: Vec::new(),
            bapc_xpiece: Vec::new(),
            bapc_ypiece: Vec::new(),
            bapc_xcoord: Vec::new(),
            bapc_ycoord: Vec::new(),
            bapc_xsize: 0,
            bapc_ysize: 0,
            bapc_xoverlap: 0,
            bapc_yoverlap: 0,
            rampbase: 0,
            rampsize: 256,
            black: 0,
            white: 255,
            range_low: 0,
            range_high: 65535,
            white_in_range: 255,
            black_in_range: 0,
            movierate: 0,
            xmovie: 0,
            ymovie: 0,
            zmovie: 0,
            tmovie: 0,
            movie_interval: 17,
            movie_running: 0,
            timers: None,
            doing_snap_draw: 0,
            slice: ImodShowsliceStruct::default(),
            lslice: ImodShowsliceStruct::default(),
            cramp: ptr::null_mut(),
            imod: ptr::null_mut(),
            extra_obj: Vec::new(),
            num_extra_obj: 0,
            extra_obj_in_use: Vec::new(),
            selection_list: Vec::new(),
            num_tilt_angles: 0,
            tilt_angles: Vec::new(),
            line_ptrs: Vec::new(),
            line_ptr_max: 0,
            blank_line: Vec::new(),
            ax: ptr::null_mut(),
            ctrlist: None,
            undo: None,
            dim: 1 + 2 + 4,
            obj_moveto: 1,
            ghostmode: IMOD_GHOST_2SHADES,
            ghostlast: IMOD_GHOST_SECTION | IMOD_GHOST_2SHADES,
            ghostdist: 0,
            insertmode: 0,
            fastdraw: 0,
            drawcursor: 1,
            ifd: 0,
            overlay_sec: 0,
            overlay_ramp: -1,
            which_green: 0,
            reverse_overlay: 0,
            draw_stipple: 0,
            track_mouse_for_plugs: 0,
            flippable: 1,
            fake_image: 0,
            raw_image_store: 0,
            int_opt_entered: 0,
            ushort_store: 0,
            rgb_store: 0,
            colormap_image: 0,
            gray_rgbs: 0,
            multi_file_z: 0,
            volume_stack: 0,
            scale_scan_type: 0,
            store_scan_in_mrc: 0,
            switch_to_ushort: 0,
            reloadable: 0,
            no_readable_image: 0,
            equal_scaling: 0,
            pixel_size_varies: 0,
            fp: None,
        };
        start_extra_object_if_none(&mut view);
        view
    }
}

/// The calls `imodview.cpp` makes into Qt objects, `App` members, and the
/// `3dmod` units whose state has no native host yet.
///
/// Each method is named after the source call it stands for.  A method whose
/// default body can do the whole job on this platform does it; every other
/// default body reports once, naming the missing unit, and returns the value
/// the source's own "nothing there" path returns.
pub trait ImodviewNativeBoundary {
    /// `imcSetMovierate(vi, rate)` (`moviecon.cpp:222`).  The translated
    /// routine needs `moviecon.cpp`'s file-static `MovieConState`, which no
    /// unit owns yet.
    fn imc_set_movierate(&mut self, vi: &mut ImodView, rate: i32) {
        report_once(
            "imcSetMovierate",
            "the movie controller state of moviecon.cpp",
        );
    }
    /// `imcResetAll(vi)` (`moviecon.cpp:101`).
    fn imc_reset_all(&mut self, vi: &mut ImodView) {
        report_once("imcResetAll", "the movie controller state of moviecon.cpp");
    }
    /// `QDir::setCurrent(path)` (`imodview.cpp:193`).
    fn qdir_set_current(&mut self, path: &str) -> bool {
        std::env::set_current_dir(path).is_ok()
    }
    /// `QDir::cleanPath(path)`.  Qt removes redundant separators, `.`, and
    /// resolvable `..` components without touching the file system.
    fn qdir_clean_path(&mut self, path: &str) -> String {
        let absolute = path.starts_with('/');
        let mut parts: Vec<&str> = Vec::new();
        for part in path.split('/') {
            match part {
                "" | "." => continue,
                ".." => {
                    if matches!(parts.last(), Some(&last) if last != "..") {
                        parts.pop();
                    } else if !absolute {
                        parts.push("..");
                    }
                }
                other => parts.push(other),
            }
        }
        let joined = parts.join("/");
        if absolute {
            format!("/{joined}")
        } else if joined.is_empty() {
            ".".to_owned()
        } else {
            joined
        }
    }
    /// `QDir::toNativeSeparators(path)`; the identity on this platform.
    fn qdir_to_native_separators(&mut self, path: &str) -> String {
        path.to_owned()
    }
    /// `QDir(dir).absoluteFilePath(file)` (`imodview.cpp:3996`).
    fn qdir_absolute_file_path(&mut self, dir: &str, file: &str) -> String {
        if file.starts_with('/') {
            return file.to_owned();
        }
        let base = if dir.is_empty() {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            dir.to_owned()
        };
        format!("{base}/{file}")
    }
    /// `App->cvi` (`imodP.h:42`), the current view.
    fn app_cvi(&mut self) -> *mut ImodView {
        report_once("App->cvi", "the ImodApp of imod.cpp");
        ptr::null_mut()
    }
    /// `App->objbase` (`imodP.h:58`).
    fn app_objbase(&mut self) -> i32 {
        report_once("App->objbase", "the colour setup of display.cpp");
        0
    }
    /// `App->DevicePixelRatio` (`imodP.h:75`).
    fn app_device_pixel_ratio(&mut self) -> f32 {
        1.
    }
    /// `icfGetAutofill()` (`cachefill.cpp:516`), which reads the fill
    /// dialog's own data.
    fn icf_get_autofill(&mut self) -> i32 {
        0
    }
    /// `icfDoAutofill(vi, section)` (`cachefill.cpp:535`).
    fn icf_do_autofill(&mut self, vi: *mut ImodView, section: i32) -> *mut u8 {
        report_once("icfDoAutofill", "the cache fill dialog of cachefill.cpp");
        ptr::null_mut()
    }
    /// `imodCacheFill(vi)` (`cachefill.cpp:521`).
    fn imod_cache_fill(&mut self, vi: *mut ImodView) -> i32 {
        report_once("imodCacheFill", "the cache filler of cachefill.cpp");
        0
    }
    /// `imod_info_input()` (`info_cb.cpp`), which pumps the Qt event loop.
    fn imod_info_input(&mut self) {}
    /// `vi->pyrCache->getFullSection(cz)` (`pyramidcache.cpp:702`).  The
    /// translated method returns a borrowed slice and needs a
    /// `PyramidCacheBoundary`, so a host must supply the line pointers.
    fn pyr_cache_get_full_section(&mut self, vi: *mut ImodView, cz: i32) -> *mut *mut u8 {
        report_once(
            "PyramidCache::getFullSection",
            "the tile-cache loader host of pyramidcache.cpp",
        );
        ptr::null_mut()
    }
    /// `vi->pyrCache->freeFullSection()` (`pyramidcache.cpp:742`).
    fn pyr_cache_free_full_section(&mut self, vi: *mut ImodView) {
        report_once(
            "PyramidCache::freeFullSection",
            "the tile-cache loader host of pyramidcache.cpp",
        );
    }
    /// `vi->pyrCache->setupFastAccess(...)` (`pyramidcache.cpp:750`), which
    /// fills the `imdata`/`vmdataxsize` arrays this unit owns.
    fn pyr_cache_setup_fast_access(
        &mut self,
        vi: *mut ImodView,
        cache_ind: i32,
        imdata: *mut *mut u8,
        vmdataxsize: *mut i32,
        cache_sum: &mut i32,
        tile_x_delta: &mut i32,
        tile_y_delta: &mut i32,
        tile_x_offset: &mut i32,
        tile_y_offset: &mut i32,
    ) {
        report_once(
            "PyramidCache::setupFastAccess",
            "the tile-cache loader host of pyramidcache.cpp",
        );
    }
    /// `iprocRethink(vi)` (`iproc.cpp:738`), which needs the image-processing
    /// dialog's own `ImodIproc`.
    fn iproc_rethink(&mut self, vi: *mut ImodView) {
        report_once("iprocRethink", "the image processing dialog of iproc.cpp");
    }
    /// `autox_newsize(vi)` (`autox.cpp:435`), which needs the `Autox` the
    /// auto-contour dialog owns.
    fn autox_newsize(&mut self, vi: *mut ImodView) {
        report_once("autox_newsize", "the auto-contour dialog of autox.cpp");
    }
    /// `imod_info_float_clear(section, time)` (`info_cb.cpp:532`).
    fn imod_info_float_clear(&mut self, section: i32, time: i32) {
        report_once(
            "imod_info_float_clear",
            "the info window state of info_cb.cpp",
        );
    }
    /// `ImodInfoWidget->setLHSliders(...)` (`form_info.cpp`).
    fn info_widget_set_lh_sliders(
        &mut self,
        low: i32,
        high: i32,
        smin: f32,
        smax: f32,
        is_float: bool,
    ) {
        report_once(
            "ImodInfoWidget->setLHSliders",
            "the info window form of form_info.cpp",
        );
    }
    /// `ImodInfoWidget->hideLowHighGrid()` (`form_info.cpp`).
    fn info_widget_hide_low_high_grid(&mut self) {
        report_once(
            "ImodInfoWidget->hideLowHighGrid",
            "the info window form of form_info.cpp",
        );
    }
    /// `ImodInfoWidget->showOrHideRamps()` (`form_info.cpp`).
    fn info_widget_show_or_hide_ramps(&mut self) {
        report_once(
            "ImodInfoWidget->showOrHideRamps",
            "the info window form of form_info.cpp",
        );
    }
    /// `ImodInfoWin->setWindowTitle(imodwfname("3dmod:"))` (`info_setup.cpp`).
    fn info_win_set_window_title(&mut self, prefix: &str) {
        report_once(
            "ImodInfoWin->setWindowTitle",
            "the info window of info_setup.cpp",
        );
    }
    /// `ImodInfoWin->openSelectedWindows(keys, 0)` (`info_setup.cpp`).
    fn info_win_open_selected_windows(&mut self, keys: &str, flag: i32) {
        report_once(
            "ImodInfoWin->openSelectedWindows",
            "the info window of info_setup.cpp",
        );
    }
    /// `imodImageScaleUpdate(vi)` (`rescale.cpp:147`).
    fn imod_image_scale_update(&mut self, vi: *mut ImodView) {
        report_once(
            "imodImageScaleUpdate",
            "the image scale dialog of rescale.cpp",
        );
    }
    /// `imodDraw(vi, flags)` (`display.cpp:459`).
    fn imod_draw(&mut self, vi: *mut ImodView, flags: i32) {
        report_once("imodDraw", "the image display host of display.cpp");
    }
    /// `imod_set_mmode(mode)` (`info_cb.cpp:782`).
    fn imod_set_mmode(&mut self, mode: i32) {
        report_once("imod_set_mmode", "the info window state of info_cb.cpp");
    }
    /// `imod_color_init(App)` (`display.cpp:306`).
    fn imod_color_init(&mut self) {
        report_once("imod_color_init", "the colour setup of display.cpp");
    }
    /// `imod_info_setbw(black, white)` (`info_cb.cpp:395`).
    fn imod_info_setbw(&mut self, black: i32, white: i32) {
        report_once("imod_info_setbw", "the info window state of info_cb.cpp");
    }
    /// `imod_info_setobjcolor()` (`info_cb.cpp:315`).
    fn imod_info_setobjcolor(&mut self) {
        report_once(
            "imod_info_setobjcolor",
            "the info window state of info_cb.cpp",
        );
    }
    /// `imod_object_edit_draw()` (`object_edit.cpp:234`).
    fn imod_object_edit_draw(&mut self) {
        report_once(
            "imod_object_edit_draw",
            "the object edit dialog of object_edit.cpp",
        );
    }
    /// `imodvObjedNewView()` (`mv_objed.cpp:336`).
    fn imodv_objed_new_view(&mut self) {
        report_once("imodvObjedNewView", "the object editor of mv_objed.cpp");
    }
    /// `ImodPrefs->setInfoGeometry()` (`preferences.cpp`).
    fn prefs_set_info_geometry(&mut self) {
        report_once(
            "ImodPrefs->setInfoGeometry",
            "the settings object of preferences.cpp",
        );
    }
    /// `ImodPrefs->loadIntIfMeanSD()` (`preferences.cpp`).
    fn prefs_load_int_if_mean_sd(&mut self) -> bool {
        false
    }
    /// `ImodPrefs->preferMeanSD()` (`preferences.cpp`).
    fn prefs_prefer_mean_sd(&mut self) -> bool {
        false
    }
    /// `ImodPrefs->numSDsForScaling()` (`preferences.cpp`).
    fn prefs_num_sds_for_scaling(&mut self) -> f32 {
        3.
    }
    /// `ImodPrefs->restoreSnapFormat()` (`preferences.cpp`).
    fn prefs_restore_snap_format(&mut self) {
        report_once(
            "ImodPrefs->restoreSnapFormat",
            "the settings object of preferences.cpp",
        );
    }
    /// `ImodPrefs->saveGenericSettings(key, numVals, values)`.
    fn prefs_save_generic_settings(&mut self, key: &str, values: &[f64]) -> i32 {
        report_once(
            "ImodPrefs->saveGenericSettings",
            "the settings object of preferences.cpp",
        );
        0
    }
    /// `ImodPrefs->getGenericSettings(key, values, maxVals)`.
    fn prefs_get_generic_settings(&mut self, key: &str, values: &mut [f64]) -> i32 {
        report_once(
            "ImodPrefs->getGenericSettings",
            "the settings object of preferences.cpp",
        );
        0
    }
    /// `b3dSetNonTiffSnapFormat(format)` (`b3dgfx.cpp:949`).
    fn b3d_set_non_tiff_snap_format(&mut self, format: i32) -> i32 {
        report_once(
            "b3dSetNonTiffSnapFormat",
            "the snapshot writer of b3dgfx.cpp",
        );
        0
    }
    /// `zapSetMouseTracking()` (`xzap.cpp:1111`).
    fn zap_set_mouse_tracking(&mut self) {
        report_once("zapSetMouseTracking", "the Zap window host of xzap.cpp");
    }
    /// `slicerSetMouseTracking()` (`slicer.cpp`).
    fn slicer_set_mouse_tracking(&mut self) {
        report_once(
            "slicerSetMouseTracking",
            "the Slicer window host of slicer.cpp",
        );
    }
    /// `getTopZapWindow(withBand)` (`xzap.cpp:1078`).
    fn get_top_zap_window(
        &mut self,
        with_band: bool,
    ) -> *mut crate::imod::three_dmod::xzap::ZapFuncs {
        report_once("getTopZapWindow", "the Zap window host of xzap.cpp");
        ptr::null_mut()
    }
    /// `getTopSlicer()` (`slicer.cpp`).
    fn get_top_slicer(&mut self) -> *mut crate::imod::three_dmod::slicer::SlicerFuncs {
        report_once("getTopSlicer", "the Slicer window host of slicer.cpp");
        ptr::null_mut()
    }
    /// `getTopZapMouse(imagePt)` (`xzap.cpp:1116`).
    fn get_top_zap_mouse(&mut self, image_pt: &mut Ipoint) -> i32 {
        report_once("getTopZapMouse", "the Zap window host of xzap.cpp");
        1
    }
    /// `zap->draw()` (`xzap.cpp:398`).
    fn zap_draw(&mut self, zap: *mut crate::imod::three_dmod::xzap::ZapFuncs) {
        report_once("ZapFuncs::draw", "the Zap window host of xzap.cpp");
    }
    /// `zap->namedSnapshot(name, format, checkGrayConvert, fullArea)`
    /// (`xzap.cpp:792`).
    fn zap_named_snapshot(
        &mut self,
        zap: *mut crate::imod::three_dmod::xzap::ZapFuncs,
        name: &mut String,
        format: i32,
        check_gray_convert: bool,
        full_area: bool,
    ) -> i32 {
        report_once("ZapFuncs::namedSnapshot", "the Zap window host of xzap.cpp");
        1
    }
    /// `slicer->namedSnapshot(...)` (`slicer.cpp`).
    fn slicer_named_snapshot(
        &mut self,
        slicer: *mut crate::imod::three_dmod::slicer::SlicerFuncs,
        name: &mut String,
        format: i32,
        check_gray_convert: bool,
        full_area: bool,
    ) -> i32 {
        report_once(
            "SlicerFuncs::namedSnapshot",
            "the Slicer window host of slicer.cpp",
        );
        1
    }
    /// `zap->startAddedArrow()` (`xzap.cpp:880`).
    fn zap_start_added_arrow(&mut self, zap: *mut crate::imod::three_dmod::xzap::ZapFuncs) {
        report_once(
            "ZapFuncs::startAddedArrow",
            "the Zap window host of xzap.cpp",
        );
    }
    /// `slicer->startAddedArrow()` (`slicer.cpp:399`).
    fn slicer_start_added_arrow(
        &mut self,
        slicer: *mut crate::imod::three_dmod::slicer::SlicerFuncs,
    ) {
        report_once(
            "SlicerFuncs::startAddedArrow",
            "the Slicer window host of slicer.cpp",
        );
    }
    /// `zap->clearArrows()` (`xzap.cpp:873`).
    fn zap_clear_arrows(&mut self, zap: *mut crate::imod::three_dmod::xzap::ZapFuncs) {
        report_once("ZapFuncs::clearArrows", "the Zap window host of xzap.cpp");
    }
    /// `slicer->clearArrows()` (`slicer.cpp:388`).
    fn slicer_clear_arrows(&mut self, slicer: *mut crate::imod::three_dmod::slicer::SlicerFuncs) {
        report_once(
            "SlicerFuncs::clearArrows",
            "the Slicer window host of slicer.cpp",
        );
    }
    /// `imodDialogManager.windowList(&objList, -1, windowType)`
    /// (`control.cpp`).  The returned handles are passed straight back to
    /// [`Self::window_list_zap`] / [`Self::window_list_slicer`].
    fn imod_dialog_manager_window_list(&mut self, window_type: i32) -> Vec<*mut c_void> {
        report_once(
            "imodDialogManager.windowList",
            "the dialog manager of control.cpp",
        );
        Vec::new()
    }
    /// `((ZapWindow *)objList.at(i))->mZap` (`imodview.cpp:3960`).
    fn window_list_zap(
        &mut self,
        window: *mut c_void,
    ) -> *mut crate::imod::three_dmod::xzap::ZapFuncs {
        ptr::null_mut()
    }
    /// `((SlicerWindow *)objList.at(i))->mFuncs` (`imodview.cpp:3963`).
    fn window_list_slicer(
        &mut self,
        window: *mut c_void,
    ) -> *mut crate::imod::three_dmod::slicer::SlicerFuncs {
        ptr::null_mut()
    }
    /// `vbCleanupVBD(obj)` (`vertexbuffer.cpp`).
    fn vb_cleanup_vbd(&mut self, obj: *mut Iobj) {
        report_once("vbCleanupVBD", "the vertex buffer host of vertexbuffer.cpp");
    }
    /// `imod_io_image_load(vi)` (`imod_io.cpp:535`).
    fn imod_io_image_load(&mut self, vi: *mut ImodView) -> *mut *mut u8 {
        // The paired `imod_io.cpp` source unit owns the normal non-cached
        // read path.  Its Rust storage lives in `ImodView`, so no Qt/window
        // host is needed merely to load pixels.
        unsafe { crate::imod::three_dmod::imod_io::imod_io_image_load(vi) }
    }
    /// The `Model` global (`imodP.h:151`).
    fn model_global(&mut self) -> *mut Imod {
        ptr::null_mut()
    }
    /// The `Imod_filename` global (`imodP.h:153`).
    fn imod_filename(&mut self) -> String {
        String::new()
    }
    /// `initReadInModelData(Model, false)` (`imod_io.cpp:427`).
    fn init_read_in_model_data(&mut self, model: *mut Imod, keep_bw: bool) {
        report_once(
            "initReadInModelData",
            "the model input/output host of imod_io.cpp",
        );
    }
    /// `createNewModel(Imod_filename)` (`imod_io.cpp:454`).
    fn create_new_model(&mut self, filename: &str) -> i32 {
        report_once(
            "createNewModel",
            "the model input/output host of imod_io.cpp",
        );
        IMOD_IO_SUCCESS
    }
    /// `ClipHandler->doneWithLoad()` (`client_message.cpp`); a null
    /// `ClipHandler` is the source's own no-message-passing case.
    fn clip_handler_done_with_load(&mut self) -> bool {
        false
    }
    /// `utilExchangeFlipRotation(imod, direction)` (`utilities.cpp:394`),
    /// which needs the `UtilitiesBoundary` no unit owns yet.
    fn util_exchange_flip_rotation(&mut self, imod: *mut Imod, direction: i32) {
        report_once(
            "utilExchangeFlipRotation",
            "the drawing utility host of utilities.cpp",
        );
    }
    /// `imodError(out, format, ...)` (`utilities.cpp:1377`).  A `None` `out`
    /// is the source's NULL and means `dia_err`, the message box of
    /// `libdiaqt`; any other stream is written directly.
    fn imod_error(&mut self, out: Option<&mut ImodFile>, message: &str) {
        match out {
            None => {
                report_once("dia_err", "the message box of libdiaqt");
                eprint!("{message}");
            }
            // `fprintf(out, "%s", errorMess)`.  The C stream, not
            // `std::io` — `3dmod` has native event-loop siblings and a
            // Rust-buffered write would reorder a redirected capture.
            Some(out) => {
                let _ = out.write_all(
                    crate::imod::libcfshr::b3dutil::c_format_bytes(
                        "%s",
                        &[crate::imod::libcfshr::b3dutil::CArg::Str(message)],
                    )
                    .as_slice(),
                );
            }
        }
    }
}

/// The reporting shape of a default boundary body: the missing unit is named
/// once per process, never silently skipped.
fn report_once(call: &'static str, needs: &'static str) {
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;
    static REPORTED: StdMutex<Option<HashSet<&'static str>>> = StdMutex::new(None);
    let mut guard = REPORTED.lock().unwrap();
    let set = guard.get_or_insert_with(HashSet::new);
    if set.insert(call) {
        eprintln!("3dmod: {call} needs {needs}, which has no native host yet");
    }
}

thread_local! {
    /// The source resolves these calls through process-global Qt objects and
    /// `App`; the Rust host supplies the same UI-thread ownership boundary.
    pub static IMODVIEW_NATIVE_BOUNDARY: RefCell<Option<Box<dyn ImodviewNativeBoundary>>> =
        const { RefCell::new(None) };
}

/// The reporting boundary used whenever no host has been installed.
pub struct ImodviewReportingBoundary;
impl ImodviewNativeBoundary for ImodviewReportingBoundary {}

/// Runs `action` against the installed boundary, or against the reporting one.
fn with_boundary<T>(action: impl FnOnce(&mut dyn ImodviewNativeBoundary) -> T) -> T {
    IMODVIEW_NATIVE_BOUNDARY.with(|slot| match slot.borrow_mut().as_deref_mut() {
        Some(boundary) => action(boundary),
        None => action(&mut ImodviewReportingBoundary),
    })
}

/// `plistBuf` (`imodview.cpp:365`).
thread_local! {
    /// Rust-owned temporary storage for piece-list section reads.
    static S_PLIST_STORAGE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// `startPos` (`imodview.cpp:900`).  glibc's `fpos_t.__pos` is not exposed
    /// by the `libc` crate; `ftello` returns the same byte offset, which is
    /// the only member the source reads.
    static S_START_POS: Cell<i64> = const { Cell::new(0) };
    /// `skipDumping` (`imodview.cpp:901`).
    static S_SKIP_DUMPING: Cell<bool> = const { Cell::new(false) };
    /// `best_ivwGetValue` (`imodview.cpp:940`).
    static S_BEST_IVW_GET_VALUE: Cell<fn(&ImodView, i32, i32, i32) -> i32> =
        const { Cell::new(fake_ivw_get_value) };
    /// `imdataxsize` (`imodview.cpp:1046`).
    static S_IMDATAXSIZE: Cell<i32> = const { Cell::new(0) };
    /// Rust-owned backing storage for the fast-access lookup table.
    static S_VMDATAXSIZE_STORAGE: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    /// Rust-owned fast-access handles into image/cache-owned pixels.  A
    /// missing cache plane is represented explicitly instead of a null raw
    /// pointer.
    static S_IMDATA_STORAGE: RefCell<Vec<Option<NonNull<u8>>>> = const { RefCell::new(Vec::new()) };
    /// `imdataMax` (`imodview.cpp:1050`).
    static S_IMDATA_MAX: Cell<i32> = const { Cell::new(0) };
    /// `vmnullvalue` (`imodview.cpp:1051`).
    static S_VMNULLVALUE: Cell<i32> = const { Cell::new(0) };
    /// `rgbChan` (`imodview.cpp:1052`).
    static S_RGB_CHAN: Cell<i32> = const { Cell::new(0) };
    /// `zsizeFac` (`imodview.cpp:1053`).
    static S_ZSIZE_FAC: Cell<i32> = const { Cell::new(0) };
    /// `xzsizeFac` (`imodview.cpp:1054`).
    static S_XZSIZE_FAC: Cell<i32> = const { Cell::new(0) };
    /// `xzsizeBigFac` (`imodview.cpp:1055`).
    static S_XZSIZE_BIG_FAC: Cell<usize> = const { Cell::new(0) };
    /// `numXfastTiles` (`imodview.cpp:1056`).
    static S_NUM_XFAST_TILES: Cell<i32> = const { Cell::new(0) };
    /// `numYfastTiles` (`imodview.cpp:1056`).
    static S_NUM_YFAST_TILES: Cell<i32> = const { Cell::new(0) };
    /// `fastTileXdelta` (`imodview.cpp:1056`).
    static S_FAST_TILE_XDELTA: Cell<i32> = const { Cell::new(0) };
    /// `fastTileYdelta` (`imodview.cpp:1056`).
    static S_FAST_TILE_YDELTA: Cell<i32> = const { Cell::new(0) };
    /// `fastTileXoffset` (`imodview.cpp:1057`).
    static S_FAST_TILE_XOFFSET: Cell<i32> = const { Cell::new(0) };
    /// `fastTileYoffset` (`imodview.cpp:1057`).
    static S_FAST_TILE_YOFFSET: Cell<i32> = const { Cell::new(0) };
    /// `ivwFastGetValue` (`imodview.cpp:1061`), the global the slicer, tumbler
    /// and xyz windows call through.
    static S_IVW_FAST_GET_VALUE: Cell<fn(i32, i32, i32) -> i32> =
        const { Cell::new(fake_get_value) };
}

/// The `ivwFastGetValue` global function pointer (`imodP.h:163`).
pub fn ivw_fast_get_value(x: i32, y: i32, z: i32) -> i32 {
    S_IVW_FAST_GET_VALUE.with(|f| f.get())(x, y, z)
}

/// `ivwInit` (`imodview.cpp:63`); default settings for the view info structure.
pub fn ivw_init(vi: &mut ImodView, modview: bool) {
    vi.xmouse = 0.0;
    vi.ymouse = 0.0;
    vi.zmouse = 0.0;

    vi.xmovie = 0;
    vi.ymovie = 0;
    vi.zmovie = 0;
    vi.tmovie = 0;
    vi.xsize = 0;
    vi.ysize = 0;
    vi.zsize = 0;
    vi.full_xsize = 0;
    vi.full_ysize = 0;
    vi.full_zsize = 0;
    vi.xysize = 0;

    vi.num_times = 0;
    vi.cur_time = 0;

    // Initialize things needed for model view and then stop if model view only
    // Standalone model view puts vi under Imodv not App
    vi.imod = ptr::null_mut();
    vi.selection_list.clear();
    vi.extra_obj.clear();
    vi.num_extra_obj = 0;
    vi.extra_obj_in_use.clear();
    start_extra_object_if_none(vi);
    vi.undo = Some(Box::new(crate::imod::three_dmod::undoredo::UndoRedo::new()));
    vi.model_view_vi = i32::from(modview);
    vi.xybin = 1;
    vi.zbin = 1;
    vi.eer_super_res = 1;
    vi.eer_zbinning = 10;
    vi.timers = Some(Box::new(
        crate::imod::three_dmod::workprocs::ImodWorkproc::new(vi),
    ));
    if modview {
        return;
    }

    with_boundary(|n| n.imc_set_movierate(vi, 0));

    vi.vm_size = 0;
    vi.vm_entered_as_gb = false;
    vi.keep_cache_full = 1;
    vi.full_cache_flipped = 0;
    vi.image_pyramid = 0;
    vi.strip_or_tile_cache = 0;
    vi.pyr_cache = ptr::null_mut();
    vi.max_read_threads = 1;
    vi.num_read_threads = 1;
    vi.loading_image = 0;
    vi.doing_initial_load = 0;
    vi.did_model_init_in_load = 0;
    vi.black = 0;
    vi.white = 255;
    vi.black_in_range = 0;
    vi.white_in_range = 255;
    vi.range_low = 0;
    vi.range_high = 65535;
    vi.fastdraw = 0;
    vi.dim = 1 + 2 + 4;
    vi.ax = ptr::null_mut();
    vi.ctrlist = None;

    vi.idata = ptr::null_mut();
    vi.fp = None;

    vi.image_list = ptr::null_mut();
    vi.image = ptr::null_mut();
    vi.num_tilt_angles = 0;
    vi.tilt_angles.clear();
    vi.bapc_xsize = 0;

    vi.movie_interval = 17;
    vi.movie_running = 0;
    vi.doing_snap_draw = 0;
    vi.ghostmode = IMOD_GHOST_2SHADES;
    vi.ghostlast = IMOD_GHOST_SECTION | IMOD_GHOST_2SHADES;
    vi.ghostdist = 0;
    vi.obj_moveto = 1;
    vi.drawcursor = 1;
    vi.insertmode = 0;
    vi.overlay_sec = 0;
    vi.overlay_ramp = -1;
    vi.draw_stipple = 0;
    vi.track_mouse_for_plugs = 0;

    vi.fake_image = 0;
    vi.raw_image_store = 0;
    vi.int_opt_entered = 0;
    vi.ushort_store = 0;
    vi.rgb_store = 0;
    vi.switch_to_ushort = 0;
    vi.multi_file_z = 0;
    vi.volume_stack = 0;
    vi.no_readable_image = 0;
    vi.line_ptrs.clear();
    vi.line_ptr_max = 0;
    vi.blank_line.clear();
    vi.flippable = 1;
    vi.gray_rgbs = 0;
    vi.reloadable = 0;
    vi.colormap_image = 0;
    vi.equal_scaling = 0;
    vi.pixel_size_varies = 0;
}

/*
 *
 *  Image data service functions.
 *
 */

/// `ivwGetCurrentSection` (`imodview.cpp:171`).
pub fn ivw_get_current_section(vi: &mut ImodView) -> *mut *mut u8 {
    let cz = (vi.zmouse + 0.5f32) as i32;
    ivw_get_z_section(vi, cz)
}

/// `ivwGetCurrentZSection` (`imodview.cpp:179`).
pub fn ivw_get_current_z_section(vi: &mut ImodView) -> *mut *mut u8 {
    let cz = (vi.zmouse + 0.5f32) as i32;
    if !vi.pyr_cache.is_null() {
        return with_boundary(|n| n.pyr_cache_get_full_section(vi, cz));
    }
    ivw_get_z_section(vi, cz)
}

/// `ivwReopen` (`imodview.cpp:189`).
pub fn ivw_reopen(in_file: &mut ImodImageFile) -> i32 {
    let ifd_path = IMOD_IFD_PATH.lock().unwrap().clone();
    if !ifd_path.is_empty() {
        with_boundary(|n| n.qdir_set_current(&ifd_path));
    }
    let retval = ii_reopen(in_file);

    // If multiple threads are allowed and not in use, open the file copies
    let cvi = with_boundary(|n| n.app_cvi());
    if !cvi.is_null()
        && unsafe { (*cvi).max_read_threads > 1 }
        && unsafe { (*cvi).num_read_threads < 2 }
        && in_file.file == IIFILE_TIFF
    {
        unsafe {
            (*cvi).file_copies[0] = in_file;
            (*cvi).num_read_threads =
                ii_open_copies_for_threads(&mut (*cvi).file_copies, (*cvi).max_read_threads);
        }
    }
    if !ifd_path.is_empty() {
        let cwd = IMOD_CWD_PATH.lock().unwrap().clone();
        with_boundary(|n| n.qdir_set_current(&cwd));
    }
    retval
}

/// `ivwClose` (`imodview.cpp:209`).
pub unsafe fn ivw_close(vi: *mut ImodView, in_file: *mut ImodImageFile) {
    unsafe {
        if (*vi).num_read_threads > 1 && in_file == (*vi).file_copies[0] {
            for ind in 1..(*vi).num_read_threads {
                ii_delete((*vi).file_copies[ind as usize]);
            }
            (*vi).num_read_threads = 1;
        }
        ii_close(in_file);
    }
}

/// `ivwGetZSectionTime` (`imodview.cpp:220`).
pub fn ivw_get_z_section_time(vi: &mut ImodView, section: i32, time: i32) -> *mut *mut u8 {
    let mut old_time = 0;

    if vi.num_times == 0 {
        return ivw_get_z_section(vi, section);
    }

    /* DNM: make test > instead of >= */
    if time < 1 || time > vi.num_times {
        return ptr::null_mut();
    }

    ivw_get_time(vi, Some(&mut old_time));
    if time == old_time {
        return ivw_get_z_section(vi, section);
    }

    let Some(time_index) = usize::try_from(time - 1).ok() else {
        return ptr::null_mut();
    };
    let Some(old_index) = usize::try_from(old_time - 1).ok() else {
        return ptr::null_mut();
    };
    if time_index >= vi.image_list_storage.len() || old_index >= vi.image_list_storage.len() {
        return ptr::null_mut();
    }

    vi.cur_time = time;
    // `image` is a temporary legacy cursor into the owned image-list vector.
    let image = unsafe { vi.image_list_storage.as_mut_ptr().add(time_index) };
    vi.image = image;
    vi.hdr = image;
    unsafe { ivw_reopen(&mut *image) };
    let image_data = ivw_get_z_section(vi, section);
    unsafe { ivw_close(vi, image) };
    vi.cur_time = old_time;
    let image = unsafe { vi.image_list_storage.as_mut_ptr().add(old_index) };
    vi.image = image;
    vi.hdr = image;
    image_data
}

/// `ivwScaleDepth8` (`imodview.cpp:250`).
pub fn ivw_scale_depth8(vi: &ImodView, temp_slice: &mut IvwSlice) {
    if imod_depth() != 8 || vi.raw_image_store != 0 {
        return;
    }
    let rbase = vi.rampbase;
    let scale = vi.rampsize as f32 / 256.0f32;
    let Some(pixel_count) = usize::try_from(temp_slice.sec.xsize)
        .ok()
        .and_then(|width| {
            usize::try_from(temp_slice.sec.ysize)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
    else {
        return;
    };
    for pixel in temp_slice.sec.data.iter_mut().take(pixel_count) {
        *pixel = (*pixel as f32 * scale + rbase as f32) as u8;
    }
}

/// `ivwGetZSection` (`imodview.cpp:269`); returns line pointers for a Z section.
pub fn ivw_get_z_section(vi: &mut ImodView, mut section: i32) -> *mut *mut u8 {
    let mut slmin: usize = 0;

    if section < 0 || section >= vi.zsize {
        return ptr::null_mut();
    }
    if vi.fp.is_none() || vi.fake_image != 0 || vi.loading_image != 0 || !vi.pyr_cache.is_null() {
        return ptr::null_mut();
    }
    let load_info = unsafe { vi.li.as_ref() };
    if vi.full_cache_flipped == 0 && ivw_plist_blank(load_info, section) != 0 {
        return ptr::null_mut();
    }

    // Flip -> rotation: invert Z if flipped
    if load_info.is_some_and(|load_info| load_info.axis == 2) {
        section = vi.zsize - 1 - section;
    }

    /* Plain, uncached data: make line pointers if not flipped */
    if vi.vm_size == 0 {
        if load_info.is_some_and(|load_info| load_info.axis == 3) {
            let bytes = usize::try_from(vi.xsize)
                .ok()
                .and_then(|width| {
                    usize::try_from(vi.ysize)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| {
                    pixels.checked_mul(ivw_get_pixel_bytes(vi.raw_image_store as i32) as usize)
                });
            let Some(bytes) = bytes else {
                return ptr::null_mut();
            };
            let Some(data) = (unsafe { vi.idata.add(section as usize).as_ref() }) else {
                return ptr::null_mut();
            };
            let data = unsafe { core::slice::from_raw_parts_mut(*data, bytes) };
            return ivw_make_line_pointers(
                &mut vi.line_ptrs,
                &mut vi.line_ptr_max,
                data,
                vi.xsize,
                vi.ysize,
                vi.raw_image_store as i32,
            )
            .map_or(ptr::null_mut(), |lines| lines.as_mut_ptr().cast());
        } else {
            /* If flipped, check the pointer allocation, get the */
            if ivw_check_line_ptr_allocation(vi, vi.ysize) != 0 {
                return ptr::null_mut();
            }
            let pix_size = ivw_get_pixel_bytes(vi.raw_image_store as i32);
            if vi.idata.is_null() {
                return ptr::null_mut();
            }
            for (sl, line) in vi.line_ptrs.iter_mut().enumerate().take(vi.ysize as usize) {
                let data = unsafe { *vi.idata.add(sl) };
                if data.is_null() {
                    return ptr::null_mut();
                }
                let Some(offset) = vi
                    .xsize
                    .checked_mul(pix_size)
                    .and_then(|stride| stride.checked_mul(section))
                    .and_then(|offset| usize::try_from(offset).ok())
                else {
                    return ptr::null_mut();
                };
                *line = NonNull::new(unsafe { data.add(offset) });
            }
            return vi.line_ptrs.as_mut_ptr().cast();
        }
    }

    /* Cached data with a full cache upon flipping - make line pointers */
    if vi.vm_size != 0 && vi.full_cache_flipped != 0 {
        if ivw_check_line_ptr_allocation(vi, vi.ysize) != 0 {
            return ptr::null_mut();
        }
        let pix_size = ivw_get_pixel_bytes(vi.raw_image_store as i32);
        let Some(section_offset) = vi
            .xsize
            .checked_mul(pix_size)
            .and_then(|stride| stride.checked_mul(section))
            .and_then(|offset| usize::try_from(offset).ok())
        else {
            return ptr::null_mut();
        };
        for (sl, line) in vi.line_ptrs.iter_mut().enumerate().take(vi.ysize as usize) {
            let Some(cache_key) = i32::try_from(sl)
                .ok()
                .and_then(|sl| sl.checked_mul(vi.vm_tdim))
                .and_then(|key| key.checked_add(vi.cur_time - vi.vm_tbase))
                .and_then(|key| usize::try_from(key).ok())
            else {
                return ptr::null_mut();
            };
            let Some(&slice) = vi.cache_index.get(cache_key) else {
                return ptr::null_mut();
            };
            if slice < 0 {
                *line = NonNull::new(vi.blank_line.as_mut_ptr());
            } else {
                let Some(cached) = vi.vm_cache.get_mut(slice as usize) else {
                    return ptr::null_mut();
                };
                let Some(data) = cached.sec.data.get_mut(section_offset..) else {
                    return ptr::null_mut();
                };
                *line = NonNull::new(data.as_mut_ptr());
            }
        }
        return vi.line_ptrs.as_mut_ptr().cast();
    }

    /* Cached data otherwise */
    let Some(cache_key) = section
        .checked_mul(vi.vm_tdim)
        .and_then(|key| key.checked_add(vi.cur_time - vi.vm_tbase))
        .and_then(|key| usize::try_from(key).ok())
    else {
        return ptr::null_mut();
    };
    let Some(&mut_sl) = vi.cache_index.get(cache_key) else {
        return ptr::null_mut();
    };
    let mut sl = mut_sl;
    if sl < 0 {
        /* Didn't find slice in cache, need to load it in. */

        /* DNM 12/12/01: add call to cache filler */
        if with_boundary(|n| n.icf_get_autofill()) != 0 {
            let filled = with_boundary(|n| n.icf_do_autofill(vi, section));
            let bytes = usize::try_from(vi.xsize)
                .ok()
                .and_then(|width| {
                    usize::try_from(vi.ysize)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| {
                    pixels.checked_mul(ivw_get_pixel_bytes(vi.raw_image_store as i32) as usize)
                });
            let Some(bytes) = bytes else {
                return ptr::null_mut();
            };
            let data = unsafe { core::slice::from_raw_parts_mut(filled, bytes) };
            return ivw_make_line_pointers(
                &mut vi.line_ptrs,
                &mut vi.line_ptr_max,
                data,
                vi.xsize,
                vi.ysize,
                vi.raw_image_store as i32,
            )
            .map_or(ptr::null_mut(), |lines| lines.as_mut_ptr().cast());
        }

        /* Find oldest slice to replace */
        let mut minused = vi.vm_count + 1;
        for (index, cached) in vi.vm_cache.iter().enumerate() {
            if cached.used < minused {
                minused = cached.used;
                slmin = index;
            }
        }

        sl = slmin as i32;
        let old_cz = vi.vm_cache[slmin].cz;
        let old_ct = vi.vm_cache[slmin].ct;
        if let Some(old_key) = old_cz
            .checked_mul(vi.vm_tdim)
            .and_then(|key| key.checked_add(old_ct - vi.vm_tbase))
            .and_then(|key| usize::try_from(key).ok())
            && let Some(old_index) = vi.cache_index.get_mut(old_key)
        {
            *old_index = -1;
        }

        /* Load in image */
        let data = vi.vm_cache[slmin].sec.data.as_mut_ptr();
        unsafe { ivw_read_z(vi, data, section) };
        let mut temp_slice = vi.vm_cache.remove(slmin);
        ivw_scale_depth8(vi, &mut temp_slice);
        temp_slice.cz = section;
        temp_slice.ct = vi.cur_time;
        vi.vm_cache.insert(slmin, temp_slice);
        vi.cache_index[cache_key] = sl;
    }

    /* Adjust use count, assign to slice */
    vi.vm_count += 1;
    let Some(temp_slice) = vi.vm_cache.get_mut(sl as usize) else {
        return ptr::null_mut();
    };
    temp_slice.used = vi.vm_count;

    ivw_make_line_pointers(
        &mut vi.line_ptrs,
        &mut vi.line_ptr_max,
        &mut temp_slice.sec.data,
        temp_slice.sec.xsize,
        temp_slice.sec.ysize,
        vi.raw_image_store as i32,
    )
    .map_or(ptr::null_mut(), |lines| lines.as_mut_ptr().cast())
}

/// `ivwPlistBlank` (`imodview.cpp:368`).
pub fn ivw_plist_blank(load_info: Option<&LoadInfo>, mut cz: i32) -> i32 {
    let Some(load_info) = load_info else {
        return 0;
    };
    if load_info.plist == 0 {
        return 0;
    }
    let Some(coords) = load_info.pcoords.as_deref() else {
        return 0;
    };
    cz += load_info.zmin;
    for i in 0..load_info.plist as usize {
        let Some(&piece_z) = coords.get(i * 3 + 2) else {
            return 0;
        };
        if piece_z == cz {
            return 0;
        }
    }
    1
}

/// `ivwReadZ` (`imodview.cpp:381`); read a section of data into the cache.
pub unsafe fn ivw_read_z(vi: *mut ImodView, buf: *mut u8, mut cz: i32) {
    unsafe {
        /* Image in not a stack but loaded into pieces. */
        if (*(*vi).li).plist != 0 {
            let mut pix_loaded: f32 = 0.;
            let check_crit: f32 = 1.0e7;

            /* DNM 1/3/04: use function instead of explicit tests */
            let pix_size = ivw_get_pixel_bytes((*vi).raw_image_store as i32);

            let mx = (*vi).xsize;
            let my = (*vi).ysize;
            let ox = (*(*vi).li).xmin;
            let oy = (*(*vi).li).ymin;
            let mxy = (mx * my) as u32;

            let nxbin = (*(*vi).hdr).nx / (*vi).xybin;
            let nybin = (*(*vi).hdr).ny / (*vi).xybin;

            /* DNM: make the buffer the size of input pieces */
            let bxy = (nxbin * nybin) as u32;
            let image_bytes = (mxy as usize) * (pix_size as usize);

            // Clear image buffer we will write to with scaled value of image mean for real
            // data.  Otherwise just set to midrange
            if (*(*vi).image).format == IIFORMAT_LUMINANCE {
                let mut fill =
                    ((*(*vi).image).amean * (*(*vi).image).slope + (*(*vi).image).offset) as i32;
                if (*vi).ushort_store != 0 {
                    fill = fill.clamp(0, 65535);
                    core::slice::from_raw_parts_mut(buf.cast::<u16>(), mxy as usize)
                        .fill(fill as u16);
                } else {
                    fill = fill.clamp(0, 255);
                    core::slice::from_raw_parts_mut(buf, image_bytes).fill(fill as u8);
                }
            } else {
                core::slice::from_raw_parts_mut(buf, image_bytes).fill(127);
            }

            /* Setup load buffer. */
            S_PLIST_STORAGE.with(|storage| {
                let mut storage = storage.borrow_mut();
                if storage.len() < (bxy as usize) * (pix_size as usize) {
                    storage.resize((bxy as usize) * (pix_size as usize), 0);
                }
            });
            cz += (*(*vi).li).zmin;

            /* Check each piece and copy its parts into the section. */
            for i in 0..(*(*vi).li).plist {
                if (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3 + 2) as usize] == cz {
                    let mut iox = (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3) as usize];
                    let mut ioy = (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3 + 1) as usize];

                    /* DNM: compute the bounding coordinates to read in, and
                    skip if there is nothing that overlaps the image */
                    let mut llx = ox - iox;
                    if llx < 0 {
                        llx = 0;
                    }
                    let mut urx = (*(*vi).li).xmax - iox;
                    if urx >= nxbin {
                        urx = nxbin - 1;
                    }

                    let mut lly = oy - ioy;
                    if lly < 0 {
                        lly = 0;
                    }
                    let mut ury = (*(*vi).li).ymax - ioy;
                    if ury >= nybin {
                        ury = nybin - 1;
                    }

                    if llx > urx || lly > ury {
                        continue;
                    }

                    (*(*vi).image).llx = llx;
                    (*(*vi).image).urx = urx;
                    (*(*vi).image).lly = lly;
                    (*(*vi).image).ury = ury;

                    S_PLIST_STORAGE.with(|storage| {
                        ivw_read_binned_section(vi, storage.borrow_mut().as_mut_ptr(), i);
                    });

                    /* set up size of copy, offsets and skip on each line
                    for copying into image buffer */
                    let xsize = urx + 1 - llx;
                    let ysize = ury + 1 - lly;
                    let iskip = mx - xsize;
                    iox += llx - ox;
                    ioy += lly - oy;
                    let fox = 0;
                    let foy = 0;
                    let fskip = 0;

                    /* Draw our piece into the image buffer. */
                    S_PLIST_STORAGE.with(|storage| {
                        memreccpy(
                            core::slice::from_raw_parts_mut(buf, image_bytes),
                            storage.borrow().as_slice(),
                            xsize,
                            ysize,
                            pix_size,
                            iskip,
                            iox,
                            ioy,
                            fskip,
                            fox,
                            foy,
                        );
                    });

                    // Periodically check for a user exit
                    pix_loaded +=
                        xsize as f32 * ysize as f32 * (*vi).xybin as f32 * (*vi).xybin as f32;
                    if pix_loaded > check_crit {
                        with_boundary(|n| n.imod_info_input());
                        if APP.lock().unwrap().as_ref().is_some_and(|a| a.exiting != 0) {
                            std::process::exit(0);
                        }
                        pix_loaded = 0.;
                    }
                }
            }
            return;
        }

        /* normal data - set up z to read based on axis, flipped or not */
        let mut zread = cz + (*(*vi).li).zmin;
        if (*(*vi).li).axis == 2 {
            zread = cz + (*(*vi).li).ymin;
        }

        /* For multi-file Z, close current file, open proper one,  read z = 0 */
        if (*vi).multi_file_z > 0 {
            ivw_close(vi, (*vi).image);
            (*vi).image = (*vi).image_list.add(zread as usize);
            (*vi).hdr = (*vi).image;
            ivw_reopen(&mut *(*vi).image);
            zread = 0;
        }

        /* DNM 1/2/04: simplify to call one place for raw, regular, or binned read */
        ivw_read_binned_section(vi, buf, ivw_adjusted_z_if_vol_stack(&*vi, zread));
    }
}

/// `ivwGetPixelBytes` (`imodview.cpp:521`); determine size of data unit.
pub fn ivw_get_pixel_bytes(mode: i32) -> i32 {
    match mode {
        MRC_MODE_BYTE => 1,
        MRC_MODE_SHORT | MRC_MODE_USHORT => 2,
        MRC_MODE_FLOAT | MRC_MODE_COMPLEX_SHORT => 4,
        MRC_MODE_COMPLEX_FLOAT => 8,
        MRC_MODE_RGB => 3,
        _ => 1,
    }
}

/// `ivwCheckLinePtrAllocation` (`imodview.cpp:541`).
pub fn ivw_check_line_ptr_allocation(vi: &mut ImodView, ysize: i32) -> i32 {
    if ysize < 0 {
        return 1;
    }
    if ysize > vi.line_ptr_max {
        vi.line_ptrs.resize(ysize as usize, None);
        vi.line_ptr_max = ysize;
    }
    0
}

/// `ivwAdjustedZIfVolStack` (`imodview.cpp:557`).
pub fn ivw_adjusted_z_if_vol_stack(vi: &ImodView, z_in: i32) -> i32 {
    if vi.volume_stack == 0 {
        return z_in;
    }
    z_in + (vi.cur_time - 1) * vi.zsize
}

/// `ivwMakeLinePointers` (`imodview.cpp:566`).
pub fn ivw_make_line_pointers<'a>(
    line_ptrs: &'a mut Vec<Option<NonNull<u8>>>,
    line_ptr_max: &mut i32,
    data: &mut [u8],
    xsize: i32,
    ysize: i32,
    mode: i32,
) -> Option<&'a mut [Option<NonNull<u8>>]> {
    let row_bytes = usize::try_from(xsize)
        .ok()?
        .checked_mul(usize::try_from(ivw_get_pixel_bytes(mode)).ok()?)?;
    let rows = usize::try_from(ysize).ok()?;
    if row_bytes == 0 || data.len() < row_bytes.checked_mul(rows)? || ysize < 0 {
        return None;
    }
    if ysize > *line_ptr_max || line_ptrs.len() < rows {
        line_ptrs.resize(rows, None);
        *line_ptr_max = (*line_ptr_max).max(ysize);
    }

    for (line, row) in line_ptrs
        .iter_mut()
        .take(rows)
        .zip(data.chunks_exact_mut(row_bytes))
    {
        *line = NonNull::new(row.as_mut_ptr());
    }
    Some(&mut line_ptrs[..rows])
}

/// `ivwReadBinnedSection(ImodView *, char *, int)` (`imodview.cpp:589`).
pub unsafe fn ivw_read_binned_section(vi: *mut ImodView, buf: *mut u8, section: i32) -> i32 {
    unsafe { ivw_read_binned_section_image(vi, (*vi).image, buf, section) }
}

/// `ivwReadBinnedSection(ImodView *, ImodImageFile *, char *, int)`
/// (`imodview.cpp:594`).  C++ overloading has no Rust equivalent, so the
/// four-argument form carries the `_image` suffix.
pub unsafe fn ivw_read_binned_section_image(
    vi: *mut ImodView,
    image: *mut ImodImageFile,
    buf: *mut u8,
    mut section: i32,
) -> i32 {
    unsafe {
        let (mut x_offset, mut left_xpad, mut right_xpad) = (0i32, 0i32, 0i32);
        let (mut y_offset, mut left_ypad, mut right_ypad) = (0i32, 0i32, 0i32);
        let (mut z_offset, mut left_zpad, mut right_zpad) = (0i32, 0i32, 0i32);
        let (mut blank_x, mut blank_y, mut blank_z) = (false, false, false);
        let mut buf = buf;
        let ucbuf = buf;
        let mut usbuf = buf.cast::<u16>();

        let pixsize = ivw_get_pixel_bytes((*vi).raw_image_store as i32);

        // Copy the record for its adjusted load-in coordinates.  `Clone`
        // retains the file handle and other owned fields correctly, unlike
        // the source's bytewise structure copy.
        let mut im = (*image).clone();
        if (*(*vi).li).plist == 0 && (*vi).pyr_cache.is_null() && (*vi).volume_stack == 0 {
            blank_x = ivw_fix_under_size_coords(
                (*vi).full_xsize,
                im.nx / (*vi).xybin,
                &mut im.llx,
                &mut im.urx,
                &mut x_offset,
                &mut left_xpad,
                &mut right_xpad,
            );
            blank_y = ivw_fix_under_size_coords(
                (*vi).full_ysize,
                im.ny / (*vi).xybin,
                &mut im.lly,
                &mut im.ury,
                &mut y_offset,
                &mut left_ypad,
                &mut right_ypad,
            );
            if (*vi).multi_file_z <= 0 {
                blank_z = ivw_fix_under_size_coords(
                    (*vi).full_zsize,
                    im.nz / (*vi).zbin,
                    &mut im.llz,
                    &mut im.urz,
                    &mut z_offset,
                    &mut left_zpad,
                    &mut right_zpad,
                );
            }

            // Adjust the section number appropriately for the axis and see if section
            // exists
            if (*vi).multi_file_z <= 0 && !(blank_x || blank_y || blank_z) {
                if im.axis == 3 {
                    section -= z_offset;
                    blank_z = section < 0 || section >= im.nz / (*vi).zbin;
                } else {
                    section -= y_offset;
                    blank_y = section < 0 || section >= im.ny / (*vi).xybin;
                    left_ypad = left_zpad;
                    right_ypad = right_zpad;
                }
            }
        }

        // Fill a blank image if any axis is out of range
        let num_pix = (*vi).xsize as usize * (*vi).ysize as usize;
        let num_bytes = num_pix * pixsize as usize;
        if blank_x || blank_y || blank_z {
            if (*vi).ushort_store != 0 {
                for _ in 0..num_pix {
                    *usbuf = 32767;
                    usbuf = usbuf.add(1);
                }
            } else {
                for _ in 0..num_pix {
                    *buf = 127;
                    buf = buf.add(1);
                }
            }
            return 0;
        }

        // Now load the image normally with these adjusted coordinates
        ivw_get_file_start_pos(&*image);
        let xbinned = (im.urx + 1 - im.llx) as usize;
        let ybinned = if im.axis == 3 {
            (im.ury + 1 - im.lly) as usize
        } else {
            (im.urz + 1 - im.llz) as usize
        };
        let xybinned = xbinned * ybinned;

        // Set conversion type for the read calls
        let convert = if (*vi).rgb_store != 0 || (*vi).colormap_image != 0 {
            MRSA_NOPROC
        } else if (*vi).ushort_store != 0 {
            MRSA_USHORT
        } else {
            MRSA_BYTE
        };

        // If there is no binning, just call the raw or byte routines
        im.raw_palette_bytes = (*vi).colormap_image;
        if (*vi).xybin * (*vi).zbin == 1 {
            if (*vi).num_read_threads > 1 && image == (*vi).file_copies[0] {
                tiff_parallel_read(
                    (*vi).file_copies.as_mut_ptr(),
                    (*vi).num_read_threads,
                    im.llx,
                    im.urx,
                    im.lly,
                    im.ury,
                    pixsize,
                    buf.cast(),
                    section,
                    convert,
                );
            } else {
                ii_read_section_any(
                    &mut im,
                    core::slice::from_raw_parts_mut(buf, num_bytes),
                    section,
                    convert,
                );
            }
        } else {
            im.llx = (*vi).xybin * im.llx;
            im.urx = (*vi).xybin * im.urx + (*vi).xybin - 1;
            im.lly = (*vi).xybin * im.lly;
            im.ury = (*vi).xybin * im.ury + (*vi).xybin - 1;
            im.llz = (*vi).xybin * im.llz;
            im.urz = (*vi).xybin * im.urz + (*vi).xybin - 1;

            // Get unbinned size, and get buffers for unbinned data and for adding
            // up binned data if there is Z binning
            let xsize = im.urx + 1 - im.llx;
            let ysize = if im.axis == 3 {
                im.ury + 1 - im.lly
            } else {
                im.urz + 1 - im.llz
            };
            let mut unbinbuf = Vec::new();
            let unbinbytes = xsize as usize * ysize as usize * pixsize as usize;
            if unbinbuf.try_reserve_exact(unbinbytes).is_err() {
                return 1;
            }
            unbinbuf.resize(unbinbytes, 0);
            let mut binbuf = Vec::new();
            if (*vi).zbin > 1 {
                if binbuf.try_reserve_exact(xybinned).is_err() {
                    return 1;
                }
                binbuf.resize(xybinned, 0);
            }

            // Loop through the unbinned sections to read and bin them into buf
            let mut ix = 0;
            let mut iy = 0;
            for iz in 0..(*vi).zbin {
                if (*vi).num_read_threads > 1 && image == (*vi).file_copies[0] {
                    tiff_parallel_read(
                        (*vi).file_copies.as_mut_ptr(),
                        (*vi).num_read_threads,
                        im.llx,
                        im.urx,
                        im.lly,
                        im.ury,
                        pixsize,
                        unbinbuf.as_mut_ptr().cast(),
                        (*vi).zbin * section + iz,
                        convert,
                    );
                } else {
                    ii_read_section_any(&mut im, &mut unbinbuf, (*vi).zbin * section + iz, convert);
                }
                reduce_by_binning(
                    core::slice::from_raw_parts(
                        unbinbuf.as_ptr(),
                        xsize as usize * ysize as usize * pixsize as usize,
                    ),
                    (*vi).raw_image_store as i32,
                    xsize,
                    ysize,
                    (*vi).xybin,
                    core::slice::from_raw_parts_mut(buf, xybinned * pixsize as usize),
                    1,
                    &mut ix,
                    &mut iy,
                );

                // For multiple sections, move or add to the binned buffer
                if (*vi).zbin > 1 {
                    if (*vi).ushort_store != 0 {
                        for i in 0..xybinned {
                            binbuf[i] += *usbuf.add(i) as i32;
                        }
                    } else {
                        for i in 0..xybinned {
                            binbuf[i] += *ucbuf.add(i) as i32;
                        }
                    }
                }
            }

            // And divide binned value into final buffer
            if (*vi).zbin > 1 {
                if (*vi).ushort_store != 0 {
                    for i in 0..xybinned {
                        *usbuf.add(i) = (binbuf[i] / (*vi).zbin) as u16;
                    }
                } else {
                    for i in 0..xybinned {
                        *buf.add(i) = (binbuf[i] / (*vi).zbin) as u8;
                    }
                }
            }
        }
        ivw_dump_file_sys_cache(&*image);

        // If the image is at all undersized, now it needs to be copied up in array and
        // padding applied
        if left_xpad != 0 || left_ypad != 0 || right_xpad != 0 || right_ypad != 0 {
            // Get an edge mean for byte data
            let mut fill: u16 = 127;
            if (*vi).rgb_store == 0 {
                let mut sum = 0.;
                if (*vi).ushort_store != 0 {
                    for i in 0..xbinned {
                        sum +=
                            *usbuf.add(i) as f64 + *usbuf.add(i + (ybinned - 1) * xbinned) as f64;
                    }
                    for i in 1..ybinned - 1 {
                        sum += *usbuf.add(i * xbinned) as f64
                            + *usbuf.add(xbinned - 1 + i * xbinned) as f64;
                    }
                } else {
                    for i in 0..xbinned {
                        sum +=
                            *ucbuf.add(i) as f64 + *ucbuf.add(i + (ybinned - 1) * xbinned) as f64;
                    }
                    for i in 1..ybinned - 1 {
                        sum += *ucbuf.add(i * xbinned) as f64
                            + *ucbuf.add(xbinned - 1 + i * xbinned) as f64;
                    }
                }
                fill = ((sum / (2. * (xbinned + ybinned - 2) as f64)) + 0.5).floor() as u16;
            }

            if (*vi).ushort_store != 0 {
                let mut usout = usbuf.add(num_pix - 1);
                let mut usin = usbuf.add(xybinned - 1);

                // Do fill at end
                for _ in 0..((*vi).xsize * right_ypad + right_xpad) {
                    *usout = fill;
                    usout = usout.sub(1);
                }

                // For each line, copy data and fill left side and right side of previous
                // line
                for iy in (0..ybinned as i32).rev() {
                    for _ in 0..xbinned {
                        *usout = *usin;
                        usout = usout.sub(1);
                        usin = usin.sub(1);
                    }
                    for _ in 0..(left_xpad + if iy != 0 { right_xpad } else { 0 }) {
                        *usout = fill;
                        usout = usout.sub(1);
                    }
                }

                // Do fill at start
                for _ in 0..((*vi).xsize * left_ypad) {
                    *usout = fill;
                    usout = usout.sub(1);
                }
            } else {
                let mut outbuf = ucbuf.add(num_bytes - 1);
                let mut inbuf = ucbuf.add(xybinned * pixsize as usize - 1);

                // Do fill at end
                for _ in 0..(((*vi).xsize * right_ypad + right_xpad) * pixsize) {
                    *outbuf = fill as u8;
                    outbuf = outbuf.sub(1);
                }

                // For each line, copy data and fill left side and right side of previous
                // line
                for iy in (0..ybinned as i32).rev() {
                    for _ in 0..(xbinned * pixsize as usize) {
                        *outbuf = *inbuf;
                        outbuf = outbuf.sub(1);
                        inbuf = inbuf.sub(1);
                    }
                    for _ in 0..((left_xpad + if iy != 0 { right_xpad } else { 0 }) * pixsize) {
                        *outbuf = fill as u8;
                        outbuf = outbuf.sub(1);
                    }
                }

                // Do fill at start
                for _ in 0..(((*vi).xsize * left_ypad) * pixsize) {
                    *outbuf = fill as u8;
                    outbuf = outbuf.sub(1);
                }
            }
        }
        0
    }
}

/// `ivwFixUnderSizeCoords` (`imodview.cpp:812`).
pub fn ivw_fix_under_size_coords(
    size: i32,
    nx: i32,
    llx: &mut i32,
    urx: &mut i32,
    offset: &mut i32,
    left_pad: &mut i32,
    right_pad: &mut i32,
) -> bool {
    *offset = (size - nx) / 2;
    *left_pad = 0;
    *right_pad = 0;
    *llx -= *offset;
    *urx -= *offset;
    if *llx < 0 {
        *left_pad = -*llx;
        *llx = 0;
    }
    if *urx >= nx {
        *right_pad = *urx + 1 - nx;
        *urx = nx - 1;
    }
    *urx < 0 || *llx >= nx
}

/// `ivwGetImagePadding` (`imodview.cpp:837`).
#[allow(clippy::too_many_arguments)]
pub fn ivw_get_image_padding(
    vi: &ImodView,
    image: Option<&ImodImageFile>,
    load_info: Option<&LoadInfo>,
    cy: i32,
    section: i32,
    time: i32,
    ll_x: &mut i32,
    left_xpad: &mut i32,
    right_xpad: &mut i32,
    ll_y: &mut i32,
    left_ypad: &mut i32,
    right_ypad: &mut i32,
    ll_z: &mut i32,
    left_zpad: &mut i32,
    right_zpad: &mut i32,
) -> i32 {
    let mut fz: i32;
    let mut blank_z = false;

    if vi.fake_image != 0 || vi.volume_stack != 0 {
        *ll_x = 0;
        *left_xpad = 0;
        *right_xpad = 0;
        *ll_y = 0;
        *left_ypad = 0;
        *right_ypad = 0;
        *ll_z = 0;
        *left_zpad = 0;
        *right_zpad = 0;
        return if vi.fake_image != 0 { 1 } else { 0 };
    }
    let Some(load_info) = load_info else {
        return -1;
    };

    // Copy the right image file structure for the situation.
    *left_zpad = 0;
    *right_zpad = 0;
    *ll_z = 0;
    let mut im = if vi.multi_file_z > 0 {
        fz = if load_info.axis == 3 { section } else { cy };
        if fz < 0 || fz >= vi.multi_file_z {
            return -1;
        }
        let Some(image) = vi.image_list_storage.get(fz as usize) else {
            return -1;
        };
        image.clone()
    } else if time > 0 {
        let Some(image) = vi.image_list_storage.get((time - 1) as usize) else {
            return -1;
        };
        image.clone()
    } else {
        let Some(image) = image else {
            return -1;
        };
        image.clone()
    };

    if load_info.plist != 0 {
        *left_xpad = 0;
        *right_xpad = 0;
        *left_ypad = 0;
        *right_ypad = 0;
        *left_zpad = 0;
        *right_zpad = 0;
        *ll_x = im.llx;
        *ll_y = im.lly;
        *ll_z = im.llz;
        return 0;
    }

    // Get the padding on each axis.
    fz = 0;
    let blank_x = ivw_fix_under_size_coords(
        vi.full_xsize,
        im.nx / vi.xybin,
        &mut im.llx,
        &mut im.urx,
        &mut fz,
        left_xpad,
        right_xpad,
    );
    let blank_y = ivw_fix_under_size_coords(
        vi.full_ysize,
        im.ny / vi.xybin,
        &mut im.lly,
        &mut im.ury,
        &mut fz,
        left_ypad,
        right_ypad,
    );
    *ll_x = im.llx;
    *ll_y = im.lly;
    if vi.multi_file_z <= 0 {
        blank_z = ivw_fix_under_size_coords(
            vi.full_zsize,
            im.nz / vi.zbin,
            &mut im.llz,
            &mut im.urz,
            &mut fz,
            left_zpad,
            right_zpad,
        );
        *ll_z = im.llz;
    }

    // Swap Y and Z padding if flipped.
    if load_info.axis == 2 {
        fz = *left_ypad;
        *left_ypad = *left_zpad;
        *left_zpad = fz;
        fz = *right_ypad;
        *right_ypad = *right_zpad;
        *right_zpad = fz;
    }
    if blank_x || blank_y || blank_z { 1 } else { 0 }
}

/// `ivwGetFileStartPos` (`imodview.cpp:903`).
pub fn ivw_get_file_start_pos(image: &ImodImageFile) {
    S_SKIP_DUMPING.with(|s| s.set(std::env::var_os("IMOD_DUMP_FSCACHE").is_none()));
    if S_SKIP_DUMPING.with(|s| s.get())
        || image.fp.is_none()
        || (image.file != IIFILE_MRC && image.file != IIFILE_RAW)
    {
        return;
    }
    S_START_POS.with(|s| s.set(image.fp.clone().unwrap().tell()));
}

/// `ivwDumpFileSysCache` (`imodview.cpp:914`).
pub fn ivw_dump_file_sys_cache(image: &ImodImageFile) {
    if S_SKIP_DUMPING.with(|s| s.get())
        || image.fp.is_none()
        || (image.file != IIFILE_MRC && image.file != IIFILE_RAW)
    {
        return;
    }
    let filedes = image.fp.as_ref().unwrap().fileno();
    let end = image.fp.clone().unwrap().tell();
    let start = S_START_POS.with(|s| s.get());
    if end <= start {
        return;
    }
    let diff = end - start;
    // POSIX owns the file-descriptor cache policy; all Rust image state above
    // is borrowed and validated before crossing this OS boundary.
    unsafe {
        libc::posix_fadvise(
            filedes,
            start as libc::off_t,
            diff as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

/*
 * Routines for getting a value based on type of data
 */

/// `ivwGetValue` (`imodview.cpp:942`).
pub fn ivw_get_value(vi: &ImodView, x: i32, y: i32, z: i32) -> i32 {
    S_BEST_IVW_GET_VALUE.with(|f| f.get())(vi, x, y, z)
}

/// `idata_ivwGetValue` (`imodview.cpp:947`).
fn idata_ivw_get_value(vi: &ImodView, x: i32, y: i32, z: i32) -> i32 {
    if vi.li.is_null() || vi.idata.is_null() {
        return 0;
    }
    unsafe {
        /* DNM: calling routine is responsible for limit checks */
        if (*vi.li).axis == 3 {
            let image = *vi.idata.add(z as usize);
            if image.is_null() {
                return 0;
            }
            *image.add((x + y * vi.xsize) as usize) as i32
        } else {
            let image = *vi.idata.add(y as usize);
            if image.is_null() {
                return 0;
            }
            *image.add((x + (vi.zsize - 1 - z) * vi.xsize) as usize) as i32
        }
    }
}

/* 1/3/04: eliminated fileScale_ivwGetValue which was unussed, incorrect,
and the only user of li->slope and offset */

/// `cache_ivwGetValue` (`imodview.cpp:959`).
fn cache_ivw_get_value(vi: &ImodView, x: i32, mut y: i32, mut z: i32) -> i32 {
    /* find pixel in cache */
    // Flip -> rotation: invert Z
    if vi.li.is_null() {
        return 0;
    }
    if unsafe { (*vi.li).axis } == 2 {
        z = vi.zsize - 1 - z;
    }

    /* If full cache and flipped, swap y and z */
    if vi.full_cache_flipped != 0 {
        (y, z) = (z, y);
    }

    let Some(cache_key) = usize::try_from(z * vi.vm_tdim + vi.cur_time - vi.vm_tbase).ok() else {
        return 0;
    };
    let Some(&sl) = vi.cache_index.get(cache_key) else {
        return 0;
    };
    let Some(temp_slice) = usize::try_from(sl)
        .ok()
        .and_then(|index| vi.vm_cache.get(index))
    else {
        return 0;
    };
    let Some(index) = usize::try_from(y)
        .ok()
        .and_then(|row| row.checked_mul(temp_slice.sec.xsize as usize))
        .and_then(|row| {
            usize::try_from(x)
                .ok()
                .and_then(|column| row.checked_add(column))
        })
    else {
        return 0;
    };

    /* DNM: calling routine is responsible for limit checks */
    if vi.ushort_store != 0 {
        let Some(offset) = index.checked_mul(std::mem::size_of::<u16>()) else {
            return 0;
        };
        let Some(bytes) = temp_slice
            .sec
            .data
            .get(offset..offset + std::mem::size_of::<u16>())
        else {
            return 0;
        };
        return u16::from_ne_bytes([bytes[0], bytes[1]]) as i32;
    }
    temp_slice.sec.data.get(index).copied().unwrap_or(0) as i32
}

/// `fake_ivwGetValue` (`imodview.cpp:1000`).
fn fake_ivw_get_value(vi: &ImodView, x: i32, y: i32, z: i32) -> i32 {
    0
}

/// `tiles_ivwGetValue` (`imodview.cpp:1005`).
fn tiles_ivw_get_value(vi: &ImodView, x: i32, y: i32, z: i32) -> i32 {
    if vi.pyr_cache.is_null() {
        return 0;
    }
    unsafe { (*vi.pyr_cache).get_value_from_base_cache(x, y, z) }
}

/// `ivwUShortInRangeToByteMap` (`imodview.cpp:1012`).
pub fn ivw_ushort_in_range_to_byte_map(vi: &ImodView) -> Vec<u8> {
    let slope = (255. / (vi.range_high - vi.range_low) as f64) as f32;
    let offset = -slope * vi.range_low as f32;
    get_short_map(slope, offset, 0, 255, MRC_RAMP_LIN, 0, 0)
}

/*
 * Routines for fast access from slicer, tumbler, and xyz
 */

/// `idata_GetValue` (`imodview.cpp:1043`).
fn idata_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[z as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add((x + y * S_IMDATAXSIZE.with(|s| s.get())) as usize) as i32
    }
}

/// `idata_BigGetValue` (`imodview.cpp:1048`).
fn idata_big_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[z as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(x as usize + y as usize * S_IMDATAXSIZE.with(|s| s.get()) as usize) as i32
    }
}

/// `flipped_GetValue` (`imodview.cpp:1053`).
fn flipped_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[y as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(
            (x + S_XZSIZE_FAC.with(|s| s.get()) - (z * S_IMDATAXSIZE.with(|s| s.get()))) as usize,
        ) as i32
    }
}

/// `flipped_BigGetValue` (`imodview.cpp:1058`).
fn flipped_big_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[y as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(
            x as usize + S_XZSIZE_BIG_FAC.with(|s| s.get())
                - (z as usize * S_IMDATAXSIZE.with(|s| s.get()) as usize),
        ) as i32
    }
}

/// `cache_GetValue` (`imodview.cpp:1063`).
fn cache_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[z as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[z as usize]);
        *imdata.add((x + y * vmdataxsize) as usize) as i32
    }
}

/// `cache_BigGetValue` (`imodview.cpp:1070`).
fn cache_big_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[z as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[z as usize]);
        *imdata.add(x as usize + y as usize * vmdataxsize as usize) as i32
    }
}

/// `cache_GetFlipped` (`imodview.cpp:1077`).
fn cache_get_flipped(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[y as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[y as usize]);
        *imdata.add((x + (S_ZSIZE_FAC.with(|s| s.get()) - z) * vmdataxsize) as usize) as i32
    }
}

/// `cache_BigGetFlipped` (`imodview.cpp:1084`).
fn cache_big_get_flipped(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[y as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[y as usize]);
        *imdata
            .add(x as usize + (S_ZSIZE_FAC.with(|s| s.get()) - z) as usize * vmdataxsize as usize)
            as i32
    }
}

/// `idata_ChanValue` (`imodview.cpp:1091`).
fn idata_chan_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[z as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(
            (3 * (x + y * S_IMDATAXSIZE.with(|s| s.get())) + S_RGB_CHAN.with(|s| s.get())) as usize,
        ) as i32
    }
}

/// `idata_BigChanValue` (`imodview.cpp:1096`).
fn idata_big_chan_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[z as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(
            3 * (x as usize + y as usize * S_IMDATAXSIZE.with(|s| s.get()) as usize)
                + S_RGB_CHAN.with(|s| s.get()) as usize,
        ) as i32
    }
}

/// `flipped_ChanValue` (`imodview.cpp:1101`).
fn flipped_chan_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[y as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(
            (3 * (x + S_XZSIZE_FAC.with(|s| s.get()) - (z * S_IMDATAXSIZE.with(|s| s.get())))
                + S_RGB_CHAN.with(|s| s.get())) as usize,
        ) as i32
    }
}

/// `flipped_BigChanValue` (`imodview.cpp:1106`).
fn flipped_big_chan_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let imdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[y as usize])
            .expect("fast access has a loaded plane")
            .as_ptr();
        *imdata.add(
            3 * (x as usize + S_XZSIZE_BIG_FAC.with(|s| s.get())
                - (z as usize * S_IMDATAXSIZE.with(|s| s.get()) as usize))
                + S_RGB_CHAN.with(|s| s.get()) as usize,
        ) as i32
    }
}

/// `cache_ChanValue` (`imodview.cpp:1111`).
fn cache_chan_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[z as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[z as usize]);
        *imdata.add((3 * (x + y * vmdataxsize) + S_RGB_CHAN.with(|s| s.get())) as usize) as i32
    }
}

/// `cache_BigChanValue` (`imodview.cpp:1118`).
fn cache_big_chan_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[z as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[z as usize]);
        *imdata.add(
            3 * (x as usize + y as usize * vmdataxsize as usize)
                + S_RGB_CHAN.with(|s| s.get()) as usize,
        ) as i32
    }
}

/// `cache_ChanFlipped` (`imodview.cpp:1125`).
fn cache_chan_flipped(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[y as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[y as usize]);
        *imdata.add(
            (3 * (x + (S_ZSIZE_FAC.with(|s| s.get()) - z) * vmdataxsize)
                + S_RGB_CHAN.with(|s| s.get())) as usize,
        ) as i32
    }
}

/// `cache_BigChanFlipped` (`imodview.cpp:1132`).
fn cache_big_chan_flipped(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(imdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[y as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let imdata = imdata.as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[y as usize]);
        *imdata.add(
            3 * x as usize
                + (S_ZSIZE_FAC.with(|s| s.get()) - z) as usize * vmdataxsize as usize
                + S_RGB_CHAN.with(|s| s.get()) as usize,
        ) as i32
    }
}

/// `idata_GetUSValue` (`imodview.cpp:1139`).
fn idata_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let usimdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[z as usize])
            .expect("fast access has a loaded plane")
            .cast::<u16>()
            .as_ptr();
        *usimdata.add((x + y * S_IMDATAXSIZE.with(|s| s.get())) as usize) as i32
    }
}

/// `idata_BigGetUSValue` (`imodview.cpp:1144`).
fn idata_big_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let usimdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[z as usize])
            .expect("fast access has a loaded plane")
            .cast::<u16>()
            .as_ptr();
        *usimdata.add(x as usize + y as usize * S_IMDATAXSIZE.with(|s| s.get()) as usize) as i32
    }
}

/// `flipped_GetUSValue` (`imodview.cpp:1149`).
fn flipped_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let usimdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[y as usize])
            .expect("fast access has a loaded plane")
            .cast::<u16>()
            .as_ptr();
        *usimdata.add(
            (x + S_XZSIZE_FAC.with(|s| s.get()) - (z * S_IMDATAXSIZE.with(|s| s.get()))) as usize,
        ) as i32
    }
}

/// `flipped_BigGetUSValue` (`imodview.cpp:1154`).
fn flipped_big_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let usimdata = S_IMDATA_STORAGE
            .with(|storage| storage.borrow()[y as usize])
            .expect("fast access has a loaded plane")
            .cast::<u16>()
            .as_ptr();
        *usimdata.add(
            x as usize + S_XZSIZE_BIG_FAC.with(|s| s.get())
                - (z as usize * S_IMDATAXSIZE.with(|s| s.get()) as usize),
        ) as i32
    }
}

/// `cache_GetUSValue` (`imodview.cpp:1159`).
fn cache_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(usimdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[z as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let usimdata = usimdata.cast::<u16>().as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[z as usize]);
        *usimdata.add((x + y * vmdataxsize) as usize) as i32
    }
}

/// `cache_BigGetUSValue` (`imodview.cpp:1166`).
fn cache_big_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(usimdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[z as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let usimdata = usimdata.cast::<u16>().as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[z as usize]);
        *usimdata.add(x as usize + y as usize * vmdataxsize as usize) as i32
    }
}

/// `cache_GetUSFlipped` (`imodview.cpp:1173`).
fn cache_get_us_flipped(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(usimdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[y as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let usimdata = usimdata.cast::<u16>().as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[y as usize]);
        *usimdata.add((x + (S_ZSIZE_FAC.with(|s| s.get()) - z) * vmdataxsize) as usize) as i32
    }
}

/// `cache_BigGetUSFlipped` (`imodview.cpp:1180`).
fn cache_big_get_us_flipped(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let Some(usimdata) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[y as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let usimdata = usimdata.cast::<u16>().as_ptr();
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[y as usize]);
        *usimdata
            .add(x as usize + (S_ZSIZE_FAC.with(|s| s.get()) - z) as usize * vmdataxsize as usize)
            as i32
    }
}

/// `fake_GetValue` (`imodview.cpp:1187`).
fn fake_get_value(x: i32, y: i32, z: i32) -> i32 {
    0
}

/// `tilecache_GetValue` (`imodview.cpp:1206`), from the `TILECACHE_VALUE`
/// macro (`imodview.cpp:1192`).
fn tilecache_get_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let num_x = S_NUM_XFAST_TILES.with(|s| s.get());
        let num_y = S_NUM_YFAST_TILES.with(|s| s.get());
        let x_off = S_FAST_TILE_XOFFSET.with(|s| s.get());
        let y_off = S_FAST_TILE_YOFFSET.with(|s| s.get());
        let mut xtile = (x + x_off) / S_FAST_TILE_XDELTA.with(|s| s.get());
        xtile = xtile.clamp(0, num_x - 1);
        let mut ytile = (y + y_off) / S_FAST_TILE_YDELTA.with(|s| s.get());
        ytile = ytile.clamp(0, num_y - 1);
        let index = xtile + (ytile + z * num_y) * num_x;
        let Some(data) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[index as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let data = data.as_ptr();
        let x_in_tile = if xtile != 0 {
            (x + x_off) - xtile * S_FAST_TILE_XDELTA.with(|s| s.get())
        } else {
            x
        };
        let y_in_tile = if ytile != 0 {
            (y + y_off) - ytile * S_FAST_TILE_YDELTA.with(|s| s.get())
        } else {
            y
        };
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[index as usize]);
        *data.add((x_in_tile + y_in_tile * vmdataxsize) as usize) as i32
    }
}

/// `tilecache_GetUSValue` (`imodview.cpp:1211`).
fn tilecache_get_us_value(x: i32, y: i32, z: i32) -> i32 {
    unsafe {
        let num_x = S_NUM_XFAST_TILES.with(|s| s.get());
        let num_y = S_NUM_YFAST_TILES.with(|s| s.get());
        let x_off = S_FAST_TILE_XOFFSET.with(|s| s.get());
        let y_off = S_FAST_TILE_YOFFSET.with(|s| s.get());
        let mut xtile = (x + x_off) / S_FAST_TILE_XDELTA.with(|s| s.get());
        xtile = xtile.clamp(0, num_x - 1);
        let mut ytile = (y + y_off) / S_FAST_TILE_YDELTA.with(|s| s.get());
        ytile = ytile.clamp(0, num_y - 1);
        let index = xtile + (ytile + z * num_y) * num_x;
        let Some(data) = S_IMDATA_STORAGE.with(|storage| storage.borrow()[index as usize]) else {
            return S_VMNULLVALUE.with(|s| s.get());
        };
        let data = data.cast::<u16>().as_ptr();
        let x_in_tile = if xtile != 0 {
            (x + x_off) - xtile * S_FAST_TILE_XDELTA.with(|s| s.get())
        } else {
            x
        };
        let y_in_tile = if ytile != 0 {
            (y + y_off) - ytile * S_FAST_TILE_YDELTA.with(|s| s.get())
        } else {
            y
        };
        let vmdataxsize = S_VMDATAXSIZE_STORAGE.with(|storage| storage.borrow()[index as usize]);
        *data.add((x_in_tile + y_in_tile * vmdataxsize) as usize) as i32
    }
}

/// `setupFastArrays` (`imodview.cpp:1217`).
fn setup_fast_arrays(size: i32, do_xsize: i32) -> i32 {
    /* If array(s) are not big enough, grow their owned backing storage. */
    if S_IMDATA_MAX.with(|s| s.get()) < size {
        S_IMDATA_STORAGE.with(|storage| {
            let mut storage = storage.borrow_mut();
            storage.resize(size as usize, None);
        });
        if do_xsize != 0 {
            S_VMDATAXSIZE_STORAGE.with(|storage| {
                let mut storage = storage.borrow_mut();
                storage.resize(size as usize, 0);
            });
        }
        S_IMDATA_MAX.with(|s| s.set(size));
    }
    0
}

/// `ivwSetupFastAccess` (`imodview.cpp:1246`); `time` defaults to -1.
pub unsafe fn ivw_setup_fast_access(
    vi: *mut ImodView,
    out_imdata: *mut *mut *mut u8,
    in_nullvalue: i32,
    cache_sum: &mut i32,
    mut time: i32,
) -> i32 {
    unsafe {
        let mut size = (*vi).zsize;

        // Time is an optional argument with default of -1 for current time
        if time < 0 {
            time = (*vi).cur_time;
        }

        *cache_sum = 0;
        let mut big_gets = if ((*vi).xsize as f64) * (*vi).ysize as f64 > 2.0e9 {
            1
        } else {
            0
        };
        S_RGB_CHAN.with(|s| s.set(0));

        if ((*vi).vm_size == 0 || (*vi).full_cache_flipped != 0) && (*(*vi).li).axis == 2 {
            size = (*vi).ysize;
            big_gets = if ((*vi).xsize as f64) * (*vi).zsize as f64 > 2.0e9 {
                1
            } else {
                0
            };
        }

        if setup_fast_arrays(size, (*vi).vm_size) != 0 {
            return 1;
        }
        if (*vi).fake_image != 0 {
            S_IVW_FAST_GET_VALUE.with(|s| s.set(fake_get_value));
        } else if (*vi).vm_size != 0 {
            /* Cached data: fill up pointers that exist
            The cache section is the inverse of an internal section number, so take the
            inverse when looking up the index for a flipped cache */
            S_IMDATA_STORAGE.with(|image_storage| {
                S_VMDATAXSIZE_STORAGE.with(|xsize_storage| {
                    let mut imdata = image_storage.borrow_mut();
                    let mut vmdataxsize = xsize_storage.borrow_mut();
                    for iz in 0..size {
                        let i = if (*(*vi).li).axis == 2 && (*vi).full_cache_flipped == 0 {
                            (&(*vi).cache_index)[(((*vi).zsize - 1 - iz) * (*vi).vm_tdim + time
                                - (*vi).vm_tbase)
                                as usize]
                        } else {
                            (&(*vi).cache_index)
                                [(iz * (*vi).vm_tdim + time - (*vi).vm_tbase) as usize]
                        };
                        if i < 0 {
                            imdata[iz as usize] = None;
                        } else {
                            imdata[iz as usize] = NonNull::new(
                                (&mut (*vi).vm_cache)[i as usize].sec.data.as_mut_ptr(),
                            );
                            vmdataxsize[iz as usize] = (&(*vi).vm_cache)[i as usize].sec.xsize;
                            *cache_sum += iz;
                        }
                    }
                })
            });

            S_VMNULLVALUE.with(|s| s.set(in_nullvalue));
            S_ZSIZE_FAC.with(|s| s.set((*vi).zsize - 1));
            if (*vi).ushort_store != 0 {
                if (*vi).full_cache_flipped != 0 {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            cache_big_get_us_flipped
                        } else {
                            cache_get_us_flipped
                        })
                    });
                } else {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            cache_big_get_us_value
                        } else {
                            cache_get_us_value
                        })
                    });
                }
            } else if (*vi).rgb_store != 0 {
                if (*vi).full_cache_flipped != 0 {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            cache_big_chan_flipped
                        } else {
                            cache_chan_flipped
                        })
                    });
                } else {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            cache_big_chan_value
                        } else {
                            cache_chan_value
                        })
                    });
                }
            } else if (*vi).full_cache_flipped != 0 {
                S_IVW_FAST_GET_VALUE.with(|s| {
                    s.set(if big_gets != 0 {
                        cache_big_get_flipped
                    } else {
                        cache_get_flipped
                    })
                });
            } else {
                S_IVW_FAST_GET_VALUE.with(|s| {
                    s.set(if big_gets != 0 {
                        cache_big_get_value
                    } else {
                        cache_get_value
                    })
                });
            }
        } else {
            /* for loaded data, get pointers from idata */

            S_IMDATA_STORAGE.with(|storage| {
                let mut imdata = storage.borrow_mut();
                for i in 0..size {
                    imdata[i as usize] = NonNull::new(*(*vi).idata.add(i as usize));
                }
            });
            S_IMDATAXSIZE.with(|s| s.set((*vi).xsize));
            let imdataxsize = S_IMDATAXSIZE.with(|s| s.get());
            if big_gets != 0 {
                S_XZSIZE_BIG_FAC.with(|s| s.set(((*vi).zsize - 1) as usize * imdataxsize as usize));
            } else {
                S_XZSIZE_FAC.with(|s| s.set(((*vi).zsize - 1) * imdataxsize));
            }
            if (*vi).ushort_store != 0 {
                if (*(*vi).li).axis == 3 {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            idata_big_get_us_value
                        } else {
                            idata_get_us_value
                        })
                    });
                } else {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            flipped_big_get_us_value
                        } else {
                            flipped_get_us_value
                        })
                    });
                }
            } else if (*vi).rgb_store != 0 {
                if (*(*vi).li).axis == 3 {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            idata_big_chan_value
                        } else {
                            idata_chan_value
                        })
                    });
                } else {
                    S_IVW_FAST_GET_VALUE.with(|s| {
                        s.set(if big_gets != 0 {
                            flipped_big_chan_value
                        } else {
                            flipped_chan_value
                        })
                    });
                }
            } else if (*(*vi).li).axis == 3 {
                S_IVW_FAST_GET_VALUE.with(|s| {
                    s.set(if big_gets != 0 {
                        idata_big_get_value
                    } else {
                        idata_get_value
                    })
                });
            } else {
                S_IVW_FAST_GET_VALUE.with(|s| {
                    s.set(if big_gets != 0 {
                        flipped_big_get_value
                    } else {
                        flipped_get_value
                    })
                });
            }
        }
        if !out_imdata.is_null() {
            *out_imdata = S_IMDATA_STORAGE.with(|storage| storage.borrow_mut().as_mut_ptr().cast());
        }
        0
    }
}

/// `ivwSetRGBChannel` (`imodview.cpp:1342`).
pub fn ivw_set_rgb_channel(value: i32) {
    S_RGB_CHAN.with(|s| s.set(0.max(2.min(value))));
}

/// `ivwSetupFastTileAccess` (`imodview.cpp:1350`).
pub unsafe fn ivw_setup_fast_tile_access(
    vi: *mut ImodView,
    cache_ind: i32,
    in_nullvalue: i32,
    cache_sum: &mut i32,
) -> i32 {
    unsafe {
        let Ok((num_x, num_y, nz)) = (*(*vi).pyr_cache).get_cache_tile_numbers(cache_ind) else {
            return 1;
        };
        S_NUM_XFAST_TILES.with(|s| s.set(num_x));
        S_NUM_YFAST_TILES.with(|s| s.set(num_y));
        if setup_fast_arrays(num_x * num_y * nz, 1) != 0 {
            return 1;
        }
        let mut x_delta = 0;
        let mut y_delta = 0;
        let mut x_offset = 0;
        let mut y_offset = 0;
        S_IMDATA_STORAGE.with(|image_storage| {
            S_VMDATAXSIZE_STORAGE.with(|xsize_storage| {
                with_boundary(|n| {
                    n.pyr_cache_setup_fast_access(
                        vi,
                        cache_ind,
                        image_storage.borrow_mut().as_mut_ptr().cast(),
                        xsize_storage.borrow_mut().as_mut_ptr(),
                        cache_sum,
                        &mut x_delta,
                        &mut y_delta,
                        &mut x_offset,
                        &mut y_offset,
                    )
                });
            });
        });
        S_FAST_TILE_XDELTA.with(|s| s.set(x_delta));
        S_FAST_TILE_YDELTA.with(|s| s.set(y_delta));
        S_FAST_TILE_XOFFSET.with(|s| s.set(x_offset));
        S_FAST_TILE_YOFFSET.with(|s| s.set(y_offset));
        if (*vi).ushort_store != 0 {
            S_IVW_FAST_GET_VALUE.with(|s| s.set(tilecache_get_us_value));
        } else {
            S_IVW_FAST_GET_VALUE.with(|s| s.set(tilecache_get_value));
        }
        S_VMNULLVALUE.with(|s| s.set(in_nullvalue));
        0
    }
}

/* DNM 1/19/03: eliminated ivwShowstatus in favor of imod_imgcnt */

/****************************************************************************/
/* CACHE INITIALIZATION ROUTINES */

/// `ivwFreeCache` (`imodview.cpp:1375`).
pub fn ivw_free_cache(vi: &mut ImodView) {
    vi.vm_cache.clear();
    vi.cache_index.clear();
    vi.blank_line.clear();
}

/// `ivwFlushCache` (`imodview.cpp:1391`).
pub fn ivw_flush_cache(vi: &mut ImodView, load_info: Option<&LoadInfo>, time: i32) {
    let Some(load_info) = load_info else {
        return;
    };
    let mut zsize = load_info.zmax - load_info.zmin + 1;
    if load_info.axis == 2 {
        zsize = load_info.ymax - load_info.ymin + 1;
    }

    for slice in &mut vi.vm_cache {
        if time < 0 || slice.ct == time {
            slice.cz = -1;
            slice.ct = 0;
            slice.used = -1;
        }
    }
    let tst;
    let tnd;
    if time < 0 {
        vi.vm_count = 0;
        tst = if vi.num_times != 0 { 1 } else { 0 };
        tnd = if vi.num_times != 0 { vi.num_times } else { 0 };
    } else {
        tst = time;
        tnd = time;
    }

    for t in tst..=tnd {
        for i in 0..zsize {
            let index = i * vi.vm_tdim + t - vi.vm_tbase;
            if let Some(value) = usize::try_from(index)
                .ok()
                .and_then(|index| vi.cache_index.get_mut(index))
            {
                *value = -1;
            }
        }
    }
}

/// `ivwInitCache` (`imodview.cpp:1420`).
pub fn ivw_init_cache(vi: &mut ImodView, load_info: &LoadInfo) -> i32 {
    let xsize = load_info.xmax - load_info.xmin + 1;
    let mut ysize = load_info.ymax - load_info.ymin + 1;
    let mut zsize = load_info.zmax - load_info.zmin + 1;

    vi.vm_tdim = if vi.num_times != 0 { vi.num_times } else { 1 };
    vi.vm_tbase = if vi.num_times != 0 { 1 } else { 0 };

    if load_info.axis == 2 {
        ysize = load_info.zmax - load_info.zmin + 1;
        zsize = load_info.ymax - load_info.ymin + 1;
    }

    let pixels = xsize * ivw_get_pixel_bytes(vi.raw_image_store as i32);
    let Some(index_len) = (vi.vm_tdim as usize).checked_mul(zsize as usize) else {
        return 9;
    };
    let mut cache = Vec::new();
    if cache.try_reserve_exact(vi.vm_size as usize).is_err() {
        return 9;
    }
    for _ in 0..vi.vm_size {
        let Some(sec) = slice_create(xsize, ysize, vi.raw_image_store as i32) else {
            return 10;
        };
        cache.push(IvwSlice {
            cz: -1,
            ct: 0,
            used: -1,
            sec,
        });
    }
    vi.vm_cache = cache;
    vi.cache_index = vec![-1; index_len];
    vi.blank_line = vec![0; pixels as usize];
    ivw_flush_cache(vi, Some(load_info), -1);
    0
}

/// `ivwSetCacheSize` (`imodview.cpp:1475`).
unsafe fn ivw_set_cache_size(vi: *mut ImodView, phys_limit: f64) -> i32 {
    unsafe {
        let xsize = (*(*vi).li).xmax - (*(*vi).li).xmin + 1;
        let ysize = (*(*vi).li).ymax - (*(*vi).li).ymin + 1;
        let zsize = (*(*vi).li).zmax - (*(*vi).li).zmin + 1;
        let mut dzsize = zsize;

        // 3/8/11: This was the size of the data in the file!  It needs to be loaded data
        let pix_size = ivw_get_pixel_bytes((*vi).raw_image_store as i32);
        let mut mem_limit = 0.;
        let mut pyr_limit;

        if xsize == 0 || ysize == 0 || zsize == 0 {
            return -1;
        }

        /* If negative size, it is megabytes, convert to sections */
        if (*vi).vm_size < 0 {
            if !(*vi).pyr_cache.is_null() {
                pyr_limit = -1000000. * (*vi).vm_size as f64;
                if phys_limit > 0. {
                    pyr_limit = pyr_limit.min(phys_limit);
                }
                (*(*vi).pyr_cache).vm_pixels = pyr_limit / pix_size as f64;
            }
            (*vi).vm_size = ((-1000000. * (*vi).vm_size as f64)
                / (((xsize as f32 * ysize as f32) * pix_size as f32) as f64))
                as i32;
            if (*vi).vm_size == 0 {
                (*vi).vm_size = 1;
            }
        } else if !(*vi).pyr_cache.is_null() {
            if (*vi).vm_size != 0 {
                pyr_limit = ((*vi).vm_size as f64 * xsize as f64) * ysize as f64 * pix_size as f64;
                if phys_limit > 0. {
                    pyr_limit = pyr_limit.min(phys_limit);
                }
                (*(*vi).pyr_cache).vm_pixels = pyr_limit / pix_size as f64;
            } else {
                if let Some(env_limit) = std::env::var_os("TILECACHE_LIMIT_MB") {
                    mem_limit = env_limit
                        .to_string_lossy()
                        .trim()
                        .parse::<f64>()
                        .unwrap_or(0.);
                }
                if mem_limit == 0. {
                    mem_limit = DEFAULT_TILE_CACHE_LIMIT;
                }
                (*(*vi).pyr_cache).vm_pixels = 1000000. * mem_limit / pix_size as f64;
            }
        }

        if (*(*vi).li).plist != 0 {
            /* For montage, make the maximum cache size be the minimum of the
            number of sections with data and the number actually being
            loaded */
            dzsize = (*(*vi).li).pdz;
            if dzsize < 1 {
                dzsize = 1;
            }
            if zsize < dzsize {
                dzsize = zsize;
            }

            /* find first actually existing data and set mouse there */
            (*vi).zmouse = zsize as f32;
            for i in 0..(*(*vi).li).plist {
                if (*vi).zmouse
                    > ((&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i + 2) as usize]
                        - (*(*vi).li).zmin) as f32
                {
                    (*vi).zmouse = ((&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i + 2) as usize]
                        - (*(*vi).li).zmin) as f32;
                }
            }
        } else {
            /* For non-montage, maximum cache is size * number of files */
            dzsize *= if (*vi).num_times > 0 {
                (*vi).num_times
            } else {
                1
            };
        }
        let mut phys_sects = (phys_limit * 1024. * 1024.
            / (((xsize as f32 * ysize as f32) * pix_size as f32) as f64))
            as i32;
        if phys_limit > 0. && phys_sects < dzsize {
            phys_sects = phys_sects.max(1);
            imod_trace(
                '1',
                &format!(
                    "Cache limited to {phys_sects} sections to use less than {phys_limit:.0} MB"
                ),
            );
            dzsize = phys_sects;
        }

        /* If no entry, just take the maximum size */
        /* Otherwise, limit the entry to the maximum size needed */
        if (*vi).vm_size == 0 {
            (*vi).vm_size = dzsize;
        } else if (*vi).vm_size > dzsize {
            (*vi).vm_size = dzsize;
        }

        0
    }
}

/* DNM 1/3/04; eliminated ivwSetScale as unneeded and confusing */

/*
 * FLIP A TOMOGRAM
 */
/// `ivwFlip` (`imodview.cpp:1561`).
pub unsafe fn ivw_flip(vi: *mut ImodView) -> i32 {
    unsafe {
        /* DNM 12/10/02: if loading image, flip the axis but defer until done */
        if (*vi).doing_initial_load > 0 {
            (*(*vi).li).axis = if (*(*vi).li).axis == 2 { 3 } else { 2 };
            return 1;
        }

        /* but if it is not the inital load, it is just to be ignored */
        if (*vi).loading_image != 0 {
            return 2;
        }

        /* find out if cache is full */
        let mut cache_full = 1;
        if (*vi).vm_size != 0 && (*vi).full_cache_flipped == 0 {
            for i in 0..(*vi).vm_tdim * (*vi).zsize {
                if (&(*vi).cache_index)[i as usize] < 0
                    && ivw_plist_blank((*vi).li.as_ref(), i) == 0
                {
                    cache_full = 0;
                    break;
                }
            }
        }

        /* Flipping is always allowed unless the cache is not full */
        if ((*vi).flippable == 0 || (*(*vi).li).plist != 0) && cache_full == 0 {
            wprint(
                "\x07Sorry, these image data can't be flipped unless they are completely \
                 loaded into memory.\n",
            );
            return -1;
        }

        let old_ymouse = ((*vi).ymouse + 0.5f32) as i32;
        let old_zmouse = ((*vi).zmouse + 0.5f32) as i32;

        wprint("Flipping image data.\n");

        /* DNM: restore data before flipping, as well as resetting when done */
        with_boundary(|n| n.iproc_rethink(vi));
        let mut nx = (*vi).xsize;
        let newy = (*vi).zsize;
        let newz = (*vi).ysize;

        (*(*vi).li).axis = if (*(*vi).li).axis == 2 { 3 } else { 2 };

        if (*vi).vm_size != 0 && cache_full != 0 {
            (*vi).full_cache_flipped = 1 - (*vi).full_cache_flipped;
        } else if (*vi).vm_size != 0 {
            /* If Image data is cached from disk */
            /* tell images to flipaxis */

            if (*vi).num_times != 0 && !(*vi).image_list.is_null() {
                for t in 0..(*vi).num_times {
                    (*(*vi).image_list.add(t as usize)).axis = (*(*vi).li).axis;
                }
            }
            if !(*vi).image.is_null() {
                (*(*vi).image).axis = (*(*vi).li).axis;
            }

            ivw_free_cache(&mut *vi);
            /* DNM: if the cache size equalled old # of Z planes, set it to new
            number of planes, including ones for each file
            Otherwise, set it to occupy same amount of memory, rounding up
            to avoid erosion on repeated flips */
            let t = if (*vi).num_times > 0 {
                (*vi).num_times
            } else {
                1
            };
            if (*vi).vm_size == t * (*vi).zsize {
                (*vi).vm_size = t * newz;
            } else {
                (*vi).vm_size = ((*vi).vm_size * (*vi).ysize + newy / 2) / newy;
                if (*vi).vm_size == 0 {
                    (*vi).vm_size = 1;
                }
            }
            ivw_init_cache(&mut *vi, &*(*vi).li);
        }

        //vi->xsize = nx;
        (*vi).ysize = newy;
        (*vi).zsize = newz;
        (*vi).xysize = (*vi).xsize as usize * (*vi).ysize as usize;
        //vi->xmouse = 0;
        (*vi).ymouse = 0.;
        (*vi).zmouse = 0.;

        nx = (*vi).y_unbin_size;
        (*vi).y_unbin_size = (*vi).z_unbin_size;
        (*vi).z_unbin_size = nx;

        // Rotate here instead of flipping
        if (*vi).doing_initial_load == 0 {
            if let (Some(model), Some(load_info)) = ((*vi).imod.as_mut(), (*vi).li.as_ref()) {
                ivw_flip_model(model, Some(load_info), true);
            }
        }
        with_boundary(|n| n.iproc_rethink(vi));
        with_boundary(|n| n.autox_newsize(vi));
        with_boundary(|n| n.imod_info_float_clear(-1, -1));

        if (*(*vi).li).axis == 2 {
            (*vi).ymouse = old_zmouse as f32;
            (*vi).zmouse = (newz - 1 - old_ymouse) as f32;
        } else {
            (*vi).ymouse = (newy - 1 - old_zmouse) as f32;
            (*vi).zmouse = old_ymouse as f32;
        }

        /* Keep it in bounds */
        if (*vi).zmouse > (newz - 1) as f32 {
            (*vi).zmouse = (newz - 1) as f32;
        }

        /* DNM: need to reset the movie controller because ny and nz changed */
        with_boundary(|n| n.imc_reset_all(&mut *vi));

        0
    }
}

/// `ivwScale` (`imodview.cpp:1679`); scale image data to fit in 8-bit colorramp.
pub unsafe fn ivw_scale(vi: *mut ImodView) -> i32 {
    unsafe {
        let ysize = (*vi).ysize;
        let xsize = (*vi).xsize;

        if (*vi).vm_size != 0 {
            return -1;
        }

        let rbase = (*vi).rampbase;
        let scale = (*vi).rampsize as f32 / 256.0f32;

        for k in 0..(*vi).zsize {
            for j in 0..ysize {
                for i in 0..xsize {
                    let at = (i + j * (*vi).xsize) as usize;
                    let mut pix = (*(*(*vi).idata.add(k as usize)).add(at) as f32 * scale) as i32;
                    pix += rbase;
                    *(*(*vi).idata.add(k as usize)).add(at) = pix as u8;
                }
            }
        }

        0
    }
}

/// `ivwBindMouse` (`imodview.cpp:1705`).
pub fn ivw_bind_mouse(vi: &mut ImodView) {
    if vi.xmouse < 0. {
        vi.xmouse = 0.;
    }
    if vi.ymouse < 0. {
        vi.ymouse = 0.;
    }
    if vi.zmouse < 0. {
        vi.zmouse = 0.;
    }
    if vi.xmouse >= vi.xsize as f32 {
        vi.xmouse = (vi.xsize - 1) as f32;
    }
    if vi.ymouse >= vi.ysize as f32 {
        vi.ymouse = (vi.ysize - 1) as f32;
    }
    if vi.zmouse > (vi.zsize - 1) as f32 {
        vi.zmouse = (vi.zsize - 1) as f32;
    }
}

/// `ivwGetLocation` (`imodview.cpp:1722`).
pub fn ivw_get_location(vi: &ImodView, x: &mut i32, y: &mut i32, z: &mut i32) {
    *x = vi.xmouse as i32;
    *y = vi.ymouse as i32;
    *z = (vi.zmouse as f64 + 0.5).floor() as i32;
}

/// `ivwGetLocationPoint` (`imodview.cpp:1730`).
pub fn ivw_get_location_point(in_imod_view: &ImodView, out_point: &mut Ipoint) {
    out_point.x = in_imod_view.xmouse;
    out_point.y = in_imod_view.ymouse;
    out_point.z = in_imod_view.zmouse;
}

/// `ivwGetTime` (`imodview.cpp:1742`).
pub fn ivw_get_time(vi: &ImodView, time: Option<&mut i32>) -> i32 {
    if let Some(time) = time {
        *time = vi.cur_time;
    }
    vi.num_times
}

/// `ivwSetTime` (`imodview.cpp:1753`); set the current time index.
pub fn ivw_set_time(vi: &mut ImodView, time: i32) {
    unsafe {
        if vi.num_times == 0 {
            vi.cur_time = 0;
            if !vi.imod.is_null() {
                (*vi.imod).ctime = 0;
            }
            return;
        }

        /* DNM 6/17/01: Don't do this */
        /* inputSetModelTime(vi, time); */
        /* set model point to a good value. */

        // keep file open for volume stack
        if vi.cur_time > 0 && vi.fake_image == 0 && vi.volume_stack == 0 {
            let current_image = vi
                .image_list_storage
                .get_mut((vi.cur_time - 1) as usize)
                .map(|image| image as *mut ImodImageFile);
            if let Some(current_image) = current_image {
                ivw_close(vi, current_image);
            }
        }

        vi.cur_time = time;
        if vi.cur_time > vi.num_times {
            vi.cur_time = vi.num_times;
        }
        if vi.cur_time <= 0 {
            vi.cur_time = 1;
        }

        if vi.fake_image == 0 {
            let image_index = (vi.cur_time - 1) as usize;
            if let Some(image) = vi.image_list_storage.get_mut(image_index) {
                let image_ptr = image as *mut ImodImageFile;
                let slider_values = (vi.ushort_store != 0).then_some((
                    vi.range_low,
                    vi.range_high,
                    image.smin,
                    image.smax,
                    image.type_ == IITYPE_FLOAT,
                ));

                // ivwSetScale(vi);
                if vi.volume_stack == 0 {
                    ivw_reopen(image);
                }
                vi.image = image_ptr;
                vi.hdr = image_ptr;

                if let Some((low, high, smin, smax, is_float)) = slider_values {
                    with_boundary(|n| {
                        n.info_widget_set_lh_sliders(low, high, smin, smax, is_float)
                    });
                }
            }
        }
        /* DNM: update scale window */
        let view = vi as *mut ImodView;
        with_boundary(|n| n.imod_image_scale_update(view));
        if !vi.imod.is_null() {
            (*vi.imod).ctime = vi.cur_time;
        }
        with_boundary(|n| n.info_win_set_window_title("3dmod:"));
    }
}

/// `ivwGetTimeIndexLabel` (`imodview.cpp:1794`).
pub fn ivw_get_time_index_label(in_imod_view: &ImodView, in_index: i32) -> &[u8] {
    if in_index < 1 || in_index > in_imod_view.num_times || in_imod_view.fake_image != 0 {
        return b"";
    }
    // The image-list backing is owned by the view; an absent source
    // `description` remains an empty borrowed label.
    in_imod_view
        .image_list_storage
        .get((in_index - 1) as usize)
        .and_then(|image| image.description.as_deref())
        .unwrap_or(b"")
}

/// `ivwGetTimeLabel` (`imodview.cpp:1803`).
///
/// Time labels are owned by `image_list_storage`; the legacy `image` cursor
/// merely mirrors the selected element for image-I/O and UI callbacks.
pub fn ivw_get_time_label(in_imod_view: &ImodView) -> &[u8] {
    ivw_get_time_index_label(in_imod_view, in_imod_view.cur_time)
}

/// `ivwGetMaxTime` (`imodview.cpp:1808`).
pub fn ivw_get_max_time(in_imod_view: &ImodView) -> i32 {
    in_imod_view.num_times
}

/// `ivwSetNewContourTime` (`imodview.cpp:1815`).
pub fn ivw_set_new_contour_time(vw: &ImodView, obj: Option<&Iobj>, cont: Option<&mut Icont>) {
    if vw.num_times != 0
        && let Some(obj) = obj
        && let Some(cont) = cont
        && iobj_flag_time(obj) != 0
    {
        cont.time = vw.cur_time;
    }
}

/// `ivwSetLocation` (`imodview.cpp:1824`).
pub fn ivw_set_location(vi: &mut ImodView, x: i32, y: i32, z: i32) {
    vi.xmouse = x as f32;
    vi.ymouse = y as f32;
    vi.zmouse = z as f32;
    ivw_bind_mouse(vi);
    let view = vi as *mut ImodView;
    with_boundary(|n| n.imod_draw(view, IMOD_DRAW_ALL));
}

/// `ivwSetLocationPoint` (`imodview.cpp:1834`).
pub fn ivw_set_location_point(vi: &mut ImodView, pnt: &Ipoint) {
    vi.xmouse = pnt.x;
    vi.ymouse = pnt.y;
    vi.zmouse = ((pnt.z as f64 + 0.5).floor() as i32) as f32;
    ivw_bind_mouse(vi);
    let view = vi as *mut ImodView;
    with_boundary(|n| n.imod_draw(view, IMOD_DRAW_ALL));
}

/// `imod_setxyzmouse` (`imodview.cpp:1849`).
pub fn imod_setxyzmouse() -> i32 {
    let cvi = with_boundary(|n| n.app_cvi());
    unsafe { imod_redraw(cvi) }
}

/// `imod_redraw` (`imodview.cpp:1854`).
pub unsafe fn imod_redraw(vw: *mut ImodView) -> i32 {
    unsafe {
        let Some(imod) = ivw_get_model(vw.as_ref()) else {
            with_boundary(|n| n.imod_draw(vw, IMOD_DRAW_MOD));
            return 1;
        };

        let index = imod.cindex.point;
        if index < 0 {
            with_boundary(|n| n.imod_draw(vw, IMOD_DRAW_MOD));
            return 1;
        }

        let Some(cont) = imod_contour_get(Some(imod)) else {
            with_boundary(|n| n.imod_draw(vw, IMOD_DRAW_MOD));
            return 1;
        };
        if cont.pts.is_empty() || cont.pts.len() as i32 <= index {
            with_boundary(|n| n.imod_draw(vw, IMOD_DRAW_MOD));
            return 1;
        }

        let time = cont.time;
        let point = cont.pts[index as usize];
        let obj = imod_object_get(Some(imod));
        if obj.is_some_and(|obj| iobj_flag_time(obj) != 0) {
            ivw_set_time(&mut *vw, time);
        }

        ivw_set_location_point(&mut *vw, &point);

        0
    }
}

/// `ivwSetMovieModelMode` (`imodview.cpp:1886`).
pub fn ivw_set_movie_model_mode(vi: &mut ImodView, mode: i32) {
    with_boundary(|n| n.imod_set_mmode(mode));
}

/// `ivwPointVisible` (`imodview.cpp:1893`).
pub fn ivw_point_visible(vi: &ImodView, pnt: &Ipoint) -> i32 {
    if (vi.zmouse as f64 + 0.5).floor() as i32 == (pnt.z as f64 + 0.5).floor() as i32 {
        1
    } else {
        0
    }
}

/// `ivwReadBinnedPoint` (`imodview.cpp:1903`).
unsafe fn ivw_read_binned_point(
    vi: *mut ImodView,
    image: *mut ImodImageFile,
    cx: i32,
    cy: i32,
    cz: i32,
) -> f32 {
    unsafe {
        let mut sum = 0.;
        let mut nsum = 0;
        if (*vi).xybin * (*vi).zbin == 1 && (*(*vi).image).mirror_fft == 0 {
            return ii_read_point(&mut *image, cx, cy, cz);
        }
        for iz in 0..(*vi).zbin {
            let ubz = cz * (*vi).zbin + iz;
            if ubz < (*image).nz {
                for iy in 0..(*vi).xybin {
                    let uby = cy * (*vi).xybin + iy;
                    if uby < (*image).ny {
                        for ix in 0..(*vi).xybin {
                            let ubx = cx * (*vi).xybin + ix;
                            if (*(*vi).image).mirror_fft != 0 && ubx < (*(*vi).image).nx {
                                let (mirx, miry) = mrc_mirror_source(
                                    (*(*vi).image).nx,
                                    (*(*vi).image).ny,
                                    ubx,
                                    uby,
                                );
                                sum += ii_read_point(&mut *image, mirx, miry, ubz) as f64;
                                nsum += 1;
                            } else if ubx < (*image).nx {
                                sum += ii_read_point(&mut *image, ubx, uby, ubz) as f64;
                                nsum += 1;
                            }
                        }
                    }
                }
            }
        }
        if nsum == 0 {
            return 0.;
        }
        (sum / nsum as f64) as f32
    }
}

/// `ivwGetFileValue` (`imodview.cpp:1939`); takes unadjusted Z values.
pub unsafe fn ivw_get_file_value(vi: *mut ImodView, cx: i32, cy: i32, mut cz: i32) -> f32 {
    unsafe {
        /* cx, cy, cz are in model file coords. */
        /* fx, fy, fz are in image file coords. */
        /* px, py, pz are in piece list coords. */
        let (mut fx, mut fy, mut fz) = (0i32, 0i32, 0i32);
        let (mut llx, mut lly, mut llz) = (0, 0, 0);
        let (mut xpad, mut ypad, mut zpad) = (0, 0, 0);

        if (*vi).image.is_null() || cz < 0 || cz >= (*vi).zsize {
            return 0.0f32;
        }
        if (*vi).no_readable_image != 0
            || (*(*vi).image).tiff_compression == IICOMPRESSION_EER_7BIT
            || (*(*vi).image).tiff_compression == IICOMPRESSION_EER_8BIT
        {
            if (*vi).raw_image_store == 0 {
                return ivw_get_value(&*vi, cx, cy, cz) as f32;
            }
            return 0.0f32;
        }
        if (*(*vi).li).axis == 2 {
            cz = (*vi).zsize - 1 - cz;
        }
        cz = ivw_adjusted_z_if_vol_stack(&*vi, cz);

        // For a tile cache, translate the coordinates and read it
        if !(*vi).pyr_cache.is_null() {
            if let Ok((bfx, bfy, bfz)) = (*(*vi).pyr_cache).get_base_file_coords(cx, cy, cz) {
                return ivw_read_binned_point(vi, (*vi).image, bfx, bfy, bfz);
            }
            return 0.;
        }

        if !(*vi).li.is_null() {
            /* get to index values in file from screen index values */
            if ivw_get_image_padding(
                &*vi,
                (*vi).image.as_ref(),
                (*vi).li.as_ref(),
                cy,
                cz,
                (*vi).cur_time,
                &mut llx,
                &mut xpad,
                &mut fx,
                &mut lly,
                &mut ypad,
                &mut fy,
                &mut llz,
                &mut zpad,
                &mut fz,
            ) < 0
            {
                return 0.;
            }

            /* DNM 7/13/04: changed to apply ymin, zmin after switching y and z */
            fx = cx + llx - xpad;
            if (*(*vi).li).axis == 3 {
                fy = cy + lly - ypad;
                fz = cz + llz - zpad;
            } else {
                fy = cz + lly - zpad;
                fz = cy + llz - ypad;
            }

            /* For multi-file sections in Z, make sure z is legal, reopen the right
            section if necessary, and set z to 0 */
            if (*vi).multi_file_z > 0 {
                if fz >= 0
                    && fz < (*vi).multi_file_z
                    && (*vi).image != (*vi).image_list.add(fz as usize)
                {
                    /* Don't mess with image files while loading is going on */
                    if (*vi).loading_image != 0 {
                        return 0.;
                    }
                    ivw_close(vi, (*vi).image);
                    (*vi).image = (*vi).image_list.add(fz as usize);
                    (*vi).hdr = (*vi).image;
                    ivw_reopen(&mut *(*vi).image);
                }
                fz = 0;
            }

            if (*(*vi).li).plist != 0 {
                /* montaged: find piece with coordinates in it and get data there */
                let mi = (*(*vi).li).plist;
                let px = fx;
                let py = fy;
                let pz = fz;
                for i in 0..mi {
                    if pz == (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3 + 2) as usize]
                        && px >= (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3) as usize]
                        && px
                            < (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3) as usize]
                                + (*(*vi).hdr).nx / (*vi).xybin
                        && py >= (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3 + 1) as usize]
                        && py
                            < (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3 + 1) as usize]
                                + (*(*vi).hdr).ny / (*vi).xybin
                    {
                        fz = i;
                        fx = px - (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3) as usize];
                        fy = py - (&(*(*vi).li).pcoords).as_ref().unwrap()[(i * 3 + 1) as usize];
                        return ivw_read_binned_point(vi, (*vi).image, fx, fy, fz);
                    }
                }
                return (*(*vi).hdr).amean;
            }
            return ivw_read_binned_point(vi, (*vi).image, fx, fy, fz);
        }
        ivw_read_binned_point(vi, (*vi).image, cx, cy, cz)
    }
}

/// `memreccpy` (`imodview.cpp:2029`); copy a portion of one buffer into another.
#[allow(clippy::too_many_arguments)]
pub fn memreccpy(
    tb: &mut [u8],
    fb: &[u8],
    xcpy: i32,
    ycpy: i32,
    psize: i32,
    tskip: i32,
    tox: i32,
    toy: i32,
    fskip: i32,
    fox: i32,
    foy: i32,
) {
    if xcpy < 0
        || ycpy < 0
        || psize < 0
        || tskip < 0
        || fskip < 0
        || tox < 0
        || toy < 0
        || fox < 0
        || foy < 0
    {
        return;
    }
    let Some(width) = (xcpy as usize).checked_mul(psize as usize) else {
        return;
    };
    let Some(target_stride) = (xcpy as usize)
        .checked_add(tskip as usize)
        .and_then(|pixels| pixels.checked_mul(psize as usize))
    else {
        return;
    };
    let Some(source_stride) = (xcpy as usize)
        .checked_add(fskip as usize)
        .and_then(|pixels| pixels.checked_mul(psize as usize))
    else {
        return;
    };
    let Some(target_start) = (tox as usize)
        .checked_mul(psize as usize)
        .and_then(|offset| {
            (toy as usize)
                .checked_mul(target_stride)
                .and_then(|row| offset.checked_add(row))
        })
    else {
        return;
    };
    let Some(source_start) = (fox as usize)
        .checked_mul(psize as usize)
        .and_then(|offset| {
            (foy as usize)
                .checked_mul(source_stride)
                .and_then(|row| offset.checked_add(row))
        })
    else {
        return;
    };

    for row in 0..ycpy as usize {
        let Some(target_offset) = row
            .checked_mul(target_stride)
            .and_then(|offset| target_start.checked_add(offset))
        else {
            return;
        };
        let Some(source_offset) = row
            .checked_mul(source_stride)
            .and_then(|offset| source_start.checked_add(offset))
        else {
            return;
        };
        let Some(target_end) = target_offset.checked_add(width) else {
            return;
        };
        let Some(source_end) = source_offset.checked_add(width) else {
            return;
        };
        let (Some(target), Some(source)) = (
            tb.get_mut(target_offset..target_end),
            fb.get(source_offset..source_end),
        ) else {
            return;
        };
        target.copy_from_slice(source);
    }
}

/// `memLineCpy` (`imodview.cpp:2064`).
#[allow(clippy::too_many_arguments)]
pub fn mem_line_cpy(
    tlines: &mut [&mut [u8]],
    fb: &[u8],
    xcpy: i32,
    ycpy: i32,
    psize: i32,
    tox: i32,
    toy: i32,
    fxsize: i32,
    fox: i32,
    foy: i32,
) {
    if xcpy < 0 || ycpy < 0 || psize < 0 || tox < 0 || toy < 0 || fxsize < 0 || fox < 0 || foy < 0 {
        return;
    }
    let Some(width) = (xcpy as usize).checked_mul(psize as usize) else {
        return;
    };
    let Some(source_stride) = (fxsize as usize).checked_mul(psize as usize) else {
        return;
    };
    let Some(source_start) = (fox as usize)
        .checked_mul(psize as usize)
        .and_then(|offset| {
            (foy as usize)
                .checked_mul(source_stride)
                .and_then(|row| offset.checked_add(row))
        })
    else {
        return;
    };
    let Some(target_start) = (tox as usize).checked_mul(psize as usize) else {
        return;
    };

    for row in 0..ycpy as usize {
        let Some(source_offset) = row
            .checked_mul(source_stride)
            .and_then(|offset| source_start.checked_add(offset))
        else {
            return;
        };
        let Some(source_end) = source_offset.checked_add(width) else {
            return;
        };
        let Some(target_line) = (toy as usize)
            .checked_add(row)
            .and_then(|line| tlines.get_mut(line))
        else {
            return;
        };
        let Some(target_end) = target_start.checked_add(width) else {
            return;
        };
        let (Some(target), Some(source)) = (
            target_line.get_mut(target_start..target_end),
            fb.get(source_offset..source_end),
        ) else {
            return;
        };
        target.copy_from_slice(source);
    }
}

/// `ivwCopyImageToByteBuffer` (`imodview.cpp:2078`).
pub fn ivw_copy_image_to_byte_buffer(vi: &ImodView, image: &[&[u8]], buf: &mut [u8]) -> i32 {
    let (Ok(width), Ok(height)) = (usize::try_from(vi.xsize), usize::try_from(vi.ysize)) else {
        return 1;
    };
    let Some(byte_count) = width.checked_mul(height) else {
        return 1;
    };
    if image.len() < height || buf.len() < byte_count {
        return 1;
    }
    if vi.ushort_store != 0 {
        let byte_width = match width.checked_mul(std::mem::size_of::<u16>()) {
            Some(value) => value,
            None => return 1,
        };
        let map = ivw_ushort_in_range_to_byte_map(vi);
        for (source, destination) in image.iter().take(height).zip(buf.chunks_exact_mut(width)) {
            let Some(source) = source.get(..byte_width) else {
                return 1;
            };
            for (pixel, destination) in source.chunks_exact(2).zip(destination) {
                *destination = map[u16::from_ne_bytes([pixel[0], pixel[1]]) as usize];
            }
        }
    } else {
        for (source, destination) in image.iter().take(height).zip(buf.chunks_exact_mut(width)) {
            let Some(source) = source.get(..width) else {
                return 1;
            };
            destination.copy_from_slice(source);
        }
    }
    0
}

/// `ivwGrayScaleImageLoaded` (`imodview.cpp:2101`).
pub fn ivw_gray_scale_image_loaded(in_imod_view: &ImodView) -> bool {
    in_imod_view.fake_image == 0 && in_imod_view.rgb_store == 0
}

/// `ivwCheckWildFlag` (`imodview.cpp:2109`).
pub fn ivw_check_wild_flag(imod: &mut Imod) {
    for ob in 0..imod.obj.len() {
        let obj = &mut imod.obj[ob];
        for co in 0..obj.cont.len() {
            let cont = &mut obj.cont[co];
            cont.flags &= !ICONT_WILD;
            if !cont.pts.is_empty() {
                let iz = (cont.pts[0].z as f64 + 0.5).floor() as i32;
                for pt in 1..cont.pts.len() {
                    if iz != (cont.pts[pt].z as f64 + 0.5).floor() as i32 {
                        cont.flags |= ICONT_WILD;
                        break;
                    }
                }
            }
        }
    }
}

/// `ivwGetImageRef` (`imodview.cpp:2145`).
///
/// The source `malloc`s the returned `IrefImage` and the callers `free` it;
/// the translated `Iref_image` is an owned value, so ownership rides in the
/// `Option` instead.
pub fn ivw_get_image_ref(
    image: Option<&ImodImageFile>,
    load_info: Option<&LoadInfo>,
    xybin: i32,
    zbin: i32,
) -> Option<Iref_image> {
    let (Some(image), Some(load_info)) = (image, load_info) else {
        return None;
    };
    let mut ref_ = Iref_image {
        oscale: Ipoint::default(),
        otrans: Ipoint::default(),
        orot: Ipoint::default(),
        cscale: Ipoint::default(),
        ctrans: Ipoint::default(),
        crot: Ipoint::default(),
    };

    let xscale = image.xscale;
    let yscale = image.yscale;
    let zscale = image.zscale;

    ref_.cscale.x = xscale;
    ref_.cscale.y = yscale;
    ref_.cscale.z = zscale;

    /* DNM 11/5/98: need to scale the load-in offsets before adding them */
    ref_.ctrans.x = image.xtrans - xscale * load_info.xmin as f32 * xybin as f32;
    ref_.ctrans.y = image.ytrans - yscale * load_info.ymin as f32 * xybin as f32;
    ref_.ctrans.z = image.ztrans - zscale * load_info.zmin as f32 * zbin as f32;
    if image.mirror_fft != 0 {
        ref_.ctrans.x = (ref_.ctrans.x as f64 + (xscale * image.nx as f32) as f64 / 2.) as f32;
    }

    /* DNM 12/19/98: if using piece lists, need to subtract the minimum
    values as well */
    if load_info.plist != 0 {
        ref_.ctrans.x -= xscale * load_info.opx * xybin as f32;
        ref_.ctrans.y -= yscale * load_info.opy * xybin as f32;
        ref_.ctrans.z -= zscale * load_info.opz;
    }

    /* DNM 11/5/98: tilt angles were not being passed back.  Start passing
    them through so that they will start being saved in model IrefImage */
    ref_.crot.x = image.xrot;
    ref_.crot.y = image.yrot;
    ref_.crot.z = image.zrot;
    Some(ref_)
}

/// `ivwSetModelTrans` (`imodview.cpp:2198`).
pub fn ivw_set_model_trans(vi: &mut ImodView) {
    let Some(imod) = (unsafe { vi.imod.as_mut() }) else {
        return;
    };

    // Unconditionally set the maxes in the model; this works for fakeimage because
    // size was set from model right away
    imod.xmax = vi.xsize;
    imod.ymax = vi.ysize;
    imod.zmax = vi.zsize;

    if vi.fake_image != 0 {
        return;
    }

    let Some(image) = (unsafe { vi.image.as_ref() }) else {
        return;
    };
    // Get the current image transformation data and copy to model structure.
    let Some(iref) = ivw_get_image_ref(Some(image), unsafe { vi.li.as_ref() }, vi.xybin, vi.zbin)
    else {
        return;
    };
    let (xt, yt, zt) = (image.xtrans, image.ytrans, image.ztrans);

    // If there is not an existing refImage, get a new one
    let ref_ = imod.ref_image.get_or_insert_with(|| Iref_image {
        oscale: Ipoint::default(),
        otrans: Ipoint::default(),
        orot: Ipoint::default(),
        cscale: Ipoint::default(),
        ctrans: Ipoint::default(),
        crot: Ipoint::default(),
    });
    ref_.cscale = iref.cscale;
    ref_.ctrans = iref.ctrans;
    ref_.crot = iref.crot;

    /* DNM 7/20/02: the old values in the model seem never to be used, so
    use otrans to store image origin information so programs can get
    back to full volume index coordinates from info in model header.
    Also set a new flag to indicate this info exists */
    ref_.otrans.x = xt;
    ref_.otrans.y = yt;
    ref_.otrans.z = zt;
    /* DNM 11/5/98: set this flag that tilt angles were properly saved */
    imod.flags |= IMODF_TILTOK;
    imod.flags |= IMODF_OTRANS_ORIGIN;
}

/// `ivwFlipModel` (`imodview.cpp:2246`); `rotate` defaults to false.
///
/// The view itself is Rust-owned.  Its model and load-info fields are legacy
/// viewer cursors, so their conversion is limited to this boundary.
pub fn ivw_flip_model(imod: &mut Imod, load_info: Option<&LoadInfo>, rotate: bool) {
    let Some(load_info) = load_info else {
        return;
    };
    /* flip model y and z and manage the flag state */
    let flag = if rotate { IMODF_ROT90X } else { IMODF_FLIPYZ };
    let cur_state = i32::from(imod.flags & flag != 0);
    if load_info.axis == 2 && cur_state != 0 {
        return;
    }

    if (load_info.axis == 3 || load_info.axis == 0) && cur_state == 0 {
        return;
    }

    if cur_state != 0 {
        imod.flags &= !flag;
    } else {
        imod.flags |= flag;
    }

    if rotate {
        imod_rot90x(imod, cur_state);
    } else {
        imod_flip_yz(imod);
    }
}

/// `ivwTransModel` (`imodview.cpp:2275`).
pub fn ivw_trans_model(vi: &mut ImodView) {
    let imod_trans = IMOD_TRANS.load(core::sync::atomic::Ordering::Relaxed);

    /* If model doesn't have a reference coordinate system from an image, then use
     * this image's coordinate system and return, unless there is binning; */
    let has_ref_image = unsafe { vi.imod.as_ref() }.is_some_and(|imod| imod.ref_image.is_some());
    if !imod_trans || !has_ref_image {
        // When loading with no trans, if the model is flipped, need to invert it to
        // restore handedness and mark it as rotated if image is, to avoid further
        // operations
        if !imod_trans {
            let Some(imod) = (unsafe { vi.imod.as_mut() }) else {
                return;
            };
            with_boundary(|n| n.util_exchange_flip_rotation(imod, FLIP_TO_ROTATION));
            let state = i32::from(unsafe { vi.li.as_ref() }.is_some_and(|li| li.axis == 2));
            set_or_clear_flags(&mut imod.flags, IMODF_ROT90X, state);
        } else {
            // Otherwise is needs to be in the right flip state before setting the
            // trans data
            let load_info = unsafe { vi.li.as_ref() };
            if let Some(imod) = unsafe { vi.imod.as_mut() } {
                ivw_flip_model(imod, load_info, false);
            }
        }
        ivw_set_model_trans(vi);
        let has_ref_image =
            unsafe { vi.imod.as_ref() }.is_some_and(|imod| imod.ref_image.is_some());
        if !imod_trans || !has_ref_image || vi.xybin * vi.zbin == 1 {
            return;
        }
    }

    /* Try and get the coordinate system that we will transform the model to match.
     * Set the old members if iref to the model's current members */
    let iref = ivw_get_image_ref(
        unsafe { vi.image.as_ref() },
        unsafe { vi.li.as_ref() },
        vi.xybin,
        vi.zbin,
    );
    if let (Some(mut iref), Some(imod)) = (iref, unsafe { vi.imod.as_mut() }) {
        if let Some(current) = imod.ref_image.as_ref() {
            iref.orot = current.crot;
            iref.otrans = current.ctrans;
            iref.oscale = current.cscale;
            let bin_scale = Ipoint {
                x: vi.xybin as f32,
                y: vi.xybin as f32,
                z: vi.zbin as f32,
            };

            /* transform model to new coords (it will be unflipped if necessary) */
            imod_trans_from_ref_image(imod, &iref, bin_scale);
        }
    }

    let load_info = unsafe { vi.li.as_ref() };
    if let Some(imod) = unsafe { vi.imod.as_mut() } {
        ivw_flip_model(imod, load_info, false);
    }
    ivw_set_model_trans(vi);
}

/*****************************************************************************/
/**** IMOD IFD Files. ****/
/*****************************************************************************/

/// `imodImageFileDesc` (`imodview.cpp:2338`); returns the type of image file.
pub unsafe fn imod_image_file_desc(fin: &mut ImodFile) -> i32 {
    unsafe {
        let mut isifd = 0;
        let mut buf = [0u8; 128];

        crate::imod::libcfshr::b3dutil::b3d_rewind(fin);
        crate::imod::libimod::imodel_files::imod_fgetline(fin, &mut buf, 127);

        if buf.starts_with(b"IMOD image list") {
            isifd = 1;
        }

        if isifd != 0 {
            isifd = 0;

            while crate::imod::libimod::imodel_files::imod_fgetline(fin, &mut buf, 127) > 0 {
                if buf.starts_with(b"VERSION") {
                    // `atoi(&buf[8])`.  The C library's conversion is
                    // translated in `clip/clip.rs`, which is where this tree
                    // keeps that boundary.
                    isifd = crate::imod::clip::clip::atoi_bytes(&buf[8..]);
                    if isifd == 0 {
                        isifd = 1;
                    }
                    crate::imod::libcfshr::b3dutil::b3d_rewind(fin);
                    return isifd;
                }
            }
        }
        crate::imod::libcfshr::b3dutil::b3d_rewind(fin);
        isifd
    }
}

/// `ivwLoadIMODifd` (`imodview.cpp:2370`); load the IMOD image list file
/// description.
pub unsafe fn ivw_load_imod_ifd(
    vi: *mut ImodView,
    pl_file_names: &mut Vec<String>,
    any_have_piece_list: &mut bool,
    any_image_fail: &mut bool,
) -> i32 {
    unsafe {
        (*vi).image_list_storage.clear();
        (*vi).image_list_storage.reserve(32);
        let mut image: *mut ImodImageFile;
        let mut line = [0u8; IFDLINE_SIZE + 1];
        let mut xsize = 0;
        let mut ysize = 0;
        let mut zsize = 0;
        let mut smin = 0.;
        let mut smax = 0.;
        let mut version = 0;
        let mut need_version = 1;
        let mut imgdir: Option<String> = None;

        *any_have_piece_list = false;
        *any_image_fail = false;
        ii_allow_multi_volume(0);
        crate::imod::libcfshr::b3dutil::b3d_rewind((&mut (*vi).fp).as_mut().unwrap());
        crate::imod::libimod::imodel_files::imod_fgetline(
            (&mut (*vi).fp).as_mut().unwrap(),
            &mut line,
            IFDLINE_SIZE as i32,
        );

        while crate::imod::libimod::imodel_files::imod_fgetline(
            (&mut (*vi).fp).as_mut().unwrap(),
            &mut line,
            IFDLINE_SIZE as i32,
        ) > 0
        {
            /* clear the return from the line. */
            for i in 0..line.len() {
                if line[i] == 0 {
                    break;
                }
                if line[i] == b'\n' || line[i] == b'\r' {
                    line[i] = 0x00;
                    break;
                }
            }

            if line[0] == b'#' {
                need_version = 2;
                continue;
            }

            if line.starts_with(b"VERSION") {
                version = crate::imod::clip::clip::atoi_bytes(&line[8..]);
                continue;
            }

            /* supply size in case first file is not found. */
            if line.starts_with(b"SIZE") {
                // `sscanf` over the line up to its terminator.  The scan
                // semantics -- a partial parse leaves the later targets
                // untouched -- are what the source relies on, so this goes
                // through the translated `fscanf`, not `str::parse`.
                let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
                let mut pos = 0usize;
                crate::imod::clip::clip::fscanf(
                    &line[..end],
                    &mut pos,
                    "SIZE %d%*c%d%*c%d\n",
                    &mut [
                        crate::imod::clip::clip::ScanArg::Int(&mut xsize),
                        crate::imod::clip::clip::ScanArg::Int(&mut ysize),
                        crate::imod::clip::clip::ScanArg::Int(&mut zsize),
                    ],
                );
                continue;
            }

            /* define a root pathname for all image files. */
            if line.starts_with(b"IMGDIR") {
                let rest = line_tail(&line, 7);
                imgdir = Some(with_boundary(|n| n.qdir_clean_path(rest.trim())));
                continue;
            }

            // Define a scale for images that follow; can be defined again
            if line.starts_with(b"SCALE") {
                let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
                let mut pos = 0usize;
                crate::imod::clip::clip::fscanf(
                    &line[..end],
                    &mut pos,
                    "SCALE %f%*c%f\n",
                    &mut [
                        crate::imod::clip::clip::ScanArg::Flt(&mut smin),
                        crate::imod::clip::clip::ScanArg::Flt(&mut smax),
                    ],
                );
                need_version = 2;
                continue;
            }

            if line.starts_with(b"PYRAMID") {
                (*vi).image_pyramid = 1;
                need_version = 2;
                continue;
            }

            /* DNM: XYZ label now supported; require one image file */
            if line.starts_with(b"XYZ") {
                let li = (*vi).li;
                if (*vi).image_list_storage.len() == 1 {
                    image = (*vi).image_list_storage.last_mut().unwrap();
                } else {
                    with_boundary(|n| {
                        n.imod_error(
                            None,
                            "3DMOD Error: Image list file must specify one image file before \
                             the XYZ option.\n",
                        )
                    });
                    std::process::exit(3);
                }
                ii_plist_load_f(
                    (&mut (*vi).fp).as_mut().unwrap(),
                    &mut *li,
                    (*image).nx,
                    (*image).ny,
                    (*image).nz,
                );

                /* DNM 1/2/04: move adjusting of loading coordinates to fix_li call,
                move that call into list processing, eliminate setting cache size,
                since it will happen later, and break instead of continuing */
                (*vi).flippable = 0;
                break;
            }

            // TIME label replaces the filename in the description string
            if line.starts_with(b"TIME") {
                if let Some(last) = (*vi).image_list_storage.last_mut() {
                    image = last;
                    let tail = line_tail(&line, 5);
                    (*image).description = Some(tail.as_bytes().to_vec());
                }
                continue;
            }

            // Origin is applied to the current image file
            if line.starts_with(b"ORIGIN") {
                if let Some(last) = (*vi).image_list_storage.last_mut() {
                    image = last;
                    let end = line.iter().position(|&b| b == 0).unwrap_or(line.len());
                    let mut pos = 0usize;
                    crate::imod::clip::clip::fscanf(
                        &line[..end],
                        &mut pos,
                        "ORIGIN %f%*c%f%*c%f\n",
                        &mut [
                            crate::imod::clip::clip::ScanArg::Flt(&mut (*image).xtrans),
                            crate::imod::clip::clip::ScanArg::Flt(&mut (*image).ytrans),
                            crate::imod::clip::clip::ScanArg::Flt(&mut (*image).ztrans),
                        ],
                    );
                }
                need_version = 2;
                continue;
            }

            if line.starts_with(b"IMAGE") {
                /* Load image file */
                let tail = line_tail(&line, 6);
                let cleaned = with_boundary(|n| n.qdir_clean_path(tail.trim()));
                let filename = match &imgdir {
                    Some(dir) => format!("{dir}/{cleaned}"),
                    None => cleaned,
                };

                let native = with_boundary(|n| n.qdir_to_native_separators(&filename));
                image = ii_open(native.as_bytes(), "rb");
                if image.is_null() {
                    if xsize == 0 || ysize == 0 {
                        with_boundary(|n| {
                            n.imod_error(
                                None,
                                &format!(
                                    "3DMOD Error: couldn't open {filename}, first file in image \
                                     list,\n and no SIZE specified before this.\n"
                                ),
                            )
                        });
                        std::process::exit(3);
                    }
                    wprint(&format!("Warning: couldn't open {filename}\n\r"));
                    let error = std::io::Error::last_os_error();
                    let sys = if error.raw_os_error().is_some_and(|code| code != 0) {
                        format!("System error: {error}")
                    } else {
                        String::new()
                    };
                    with_boundary(|n| {
                        n.imod_error(
                            Some(&mut ImodFile::Stdout),
                            &format!("Warning: couldn't open {filename}\n{sys}"),
                        )
                    });
                    image = ii_new();
                    (*image).nx = xsize;
                    (*image).ny = ysize;
                    (*image).nz = zsize;
                    (*image).filename = Some(native.clone());
                    *any_image_fail = true;
                }

                /* DNM: set up scaling for this image, leave last file in hdr/image */
                if (((*image).file == IIFILE_RAW && (*image).amin == 0. && (*image).amax == 0.)
                    || get_valid_scale(image, None, None) == 0)
                    && smin >= smax
                {
                    ii_raw_scan(&mut *image);
                }
                if smin < smax {
                    (*image).smin = smin;
                    (*image).smax = smax;
                }

                ii_close(image);
                if (*image).has_piece_coords != 0 {
                    *any_have_piece_list = true;
                }

                /* DNM: Make filename with directory stripped be the default descriptor */
                let bytes = filename.as_bytes();
                let mut pathlen = bytes.len();
                while pathlen > 0 && bytes[pathlen - 1] != b'/' {
                    pathlen -= 1;
                }
                (*image).description = Some(bytes[pathlen..].to_vec());

                if (*vi).image_list_storage.len() == (*vi).image_list_storage.capacity() {
                    let old_len = (*vi).image_list_storage.len();
                    let mut grown_images = Vec::with_capacity(old_len + 32);
                    grown_images.extend((*vi).image_list_storage.iter().cloned());
                    let old_images = (*vi).image_list_storage.as_mut_ptr();
                    let new_images = grown_images.as_mut_ptr();
                    for index in 0..old_len {
                        ii_file_change_address(old_images.add(index), new_images.add(index));
                    }
                    (*vi).image_list_storage.clear();
                    (*vi).image_list_storage = grown_images;
                }
                let old_image = image;
                (*vi).image_list_storage.push((*image).clone());
                image = (*vi).image_list_storage.last_mut().unwrap();
                ii_file_change_address(old_image, image);
                ii_delete(old_image);
                (*vi).image = image;
                (*vi).hdr = image;
                /* set xsize etc from size of first file if not set */
                if xsize == 0 && ysize == 0 {
                    xsize = (*image).nx;
                    ysize = (*image).ny;
                    zsize = (*image).nz;
                }

                /* DNM: set time and increment time counter here, not with the TIME
                label */
                (*image).time = (*vi).num_times;
                (*vi).num_times += 1;

                continue;
            }

            if line.starts_with(b"PIECEFILE") {
                for _ in pl_file_names.len() as i32..(*vi).image_list_storage.len() as i32 - 1 {
                    pl_file_names.push("NONE".to_owned());
                }
                let mut qname = line_tail(&line, 10).trim().to_owned();
                if let Some(dir) = &imgdir {
                    qname = format!("{dir}/{qname}");
                }
                let cleaned = with_boundary(|n| n.qdir_clean_path(&qname));
                pl_file_names.push(with_boundary(|n| n.qdir_to_native_separators(&cleaned)));
                need_version = 2;
                continue;
            }

            let text = line_tail(&line, 0);
            with_boundary(|n| {
                n.imod_error(
                    None,
                    &format!("3dmod warning: Unknown image list option ({text})\n"),
                )
            });
        }
        crate::imod::libcfshr::b3dutil::b3d_rewind((&mut (*vi).fp).as_mut().unwrap());
        /* end of while (getline) */

        (*vi).image_list = (*vi).image_list_storage.as_mut_ptr();
        if version < need_version {
            with_boundary(|n| {
                n.imod_error(
                    None,
                    &format!(
                        "3DMOD Error: The image list file must specify version {need_version} \
                         or higher\n"
                    ),
                )
            });
            std::process::exit(3);
        }

        0
    }
}

/// The tail of a C string in a line buffer starting at `offset`; the source
/// writes `&line[offset]`, which Rust cannot index directly on a NUL-padded
/// byte array.
fn line_tail(line: &[u8], offset: usize) -> &str {
    let bytes = &line[offset.min(line.len())..];
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    core::str::from_utf8(&bytes[..end]).unwrap_or("")
}

/// `ivwLoadIFDpieceList` (`imodview.cpp:2592`).
pub unsafe fn ivw_load_ifd_piece_list(
    pl_name: &str,
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    unsafe {
        let ifd_path = IMOD_IFD_PATH.lock().unwrap().clone();
        if !ifd_path.is_empty() {
            with_boundary(|n| n.qdir_set_current(&ifd_path));
        }
        let retval = ii_plist_load(pl_name, &mut *li, nx, ny, nz);
        if !ifd_path.is_empty() {
            let cwd = IMOD_CWD_PATH.lock().unwrap().clone();
            with_boundary(|n| n.qdir_set_current(&cwd));
        }
        retval
    }
}

/// `ivwMultipleFiles` (`imodview.cpp:2605`); take a list of multiple files
/// from the argument list and compose an image list.
pub unsafe fn ivw_multiple_files(
    vi: *mut ImodView,
    argv: &[Vec<u8>],
    firstfile: i32,
    lastimage: i32,
    any_have_piece_list: &mut bool,
) {
    unsafe {
        (*vi).image_list_storage.clear();
        (*vi)
            .image_list_storage
            .reserve((lastimage - firstfile + 1).max(0) as usize);
        let mut image: *mut ImodImageFile;
        let mut base_image: *mut ImodImageFile = ptr::null_mut();
        let mut convarg: Vec<u8> = Vec::new();
        *any_have_piece_list = false;
        ii_allow_multi_volume(1);

        for i in firstfile..=lastimage {
            let arg = argv[i as usize].as_slice();
            let mut num_vols = 1;
            if !arg.is_empty() && arg[0] != 0 {
                let entered = String::from_utf8_lossy(arg).into_owned();
                let cleaned = with_boundary(|n| n.qdir_clean_path(&entered));
                convarg = cleaned.as_bytes().to_vec();
                let native = with_boundary(|n| n.qdir_to_native_separators(&cleaned));
                image = ii_open(native.as_bytes(), "rb");
            } else {
                image = ii_open(arg, "rb");
            }
            let mut ind_vol;
            if !image.is_null() && (*image).num_volumes > 1 {
                base_image = image;
                num_vols = (*image).num_volumes;
                let needed = (*vi).image_list_storage.len() + num_vols as usize;
                if needed > (*vi).image_list_storage.capacity() {
                    let old_len = (*vi).image_list_storage.len();
                    let mut grown_images = Vec::with_capacity(needed);
                    grown_images.extend((*vi).image_list_storage.iter().cloned());
                    let old_images = (*vi).image_list_storage.as_mut_ptr();
                    let new_images = grown_images.as_mut_ptr();
                    for index in 0..old_len {
                        ii_file_change_address(old_images.add(index), new_images.add(index));
                    }
                    (*vi).image_list_storage.clear();
                    (*vi).image_list_storage = grown_images;
                }
                let adoc_ind = ii_get_adoc_index(&mut *image, 1, 0);
                let mut vol_flag = 0;
                if adoc_ind >= 0
                    && adoc_set_current(adoc_ind) == 0
                    && adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"image_pyramid", &mut vol_flag) == 0
                    && vol_flag > 0
                {
                    (*vi).image_pyramid = 1;
                }
            }
            ind_vol = 0;
            while ind_vol < num_vols {
                if ind_vol != 0 {
                    let volume = (&(*base_image).ii_volumes)[ind_vol as usize]
                        .expect("listed HDF volume has a cursor")
                        .as_ptr();
                    if ii_reopen(&mut *volume) != 0 {
                        crate::imod::three_dmod::imod::imod_print_stderr(&format!(
                            "Failed to reopen volume #{} in image file.\n",
                            ind_vol + 1
                        ));
                        image = ptr::null_mut();
                    } else {
                        image = (&(*base_image).ii_volumes)[ind_vol as usize]
                            .expect("reopened HDF volume has a cursor")
                            .as_ptr();
                        convarg = (*image)
                            .filename
                            .as_deref()
                            .unwrap_or_default()
                            .as_bytes()
                            .to_vec();
                    }
                }
                if image.is_null() {
                    let entered = String::from_utf8_lossy(arg);
                    // `imodview.cpp:2643` prepends `b3dGetError()` to the
                    // `SPRINTF`ed message before handing it to `imodError`.
                    let error = crate::imod::libcfshr::b3dutil::b3d_get_error();
                    with_boundary(|n| {
                        n.imod_error(
                            None,
                            &format!("{error}3DMOD Error: couldn't open image file {entered}.\n"),
                        )
                    });
                    std::process::exit(3);
                }

                /* set up scaling for this image, scanning if needed */
                if (((*image).file == IIFILE_RAW && (*image).amin == 0. && (*image).amax == 0.)
                    || get_valid_scale(image, None, None) == 0)
                    && ii_raw_scan(&mut *image) != 0
                {
                    let entered = String::from_utf8_lossy(arg);
                    let error = crate::imod::libcfshr::b3dutil::b3d_get_error();
                    with_boundary(|n| {
                        n.imod_error(
                            None,
                            &format!(
                                "{error}3DMOD Error: Scanning for scaling limits in \
                                 {entered}.\n"
                            ),
                        )
                    });
                    std::process::exit(3);
                }

                (*image).time = (*vi).num_times;
                (*vi).num_times += 1;

                // This just needs to be non-NULL when there is a file, doesn't need to be
                // accurate
                (*vi).fp = (*image).fp.clone();

                /* Copy filename with directory stripped to the descriptor */
                if !arg.is_empty() && arg[0] != 0 {
                    ii_close(image);
                    let mut pathlen = convarg.len();
                    while pathlen > 0 && convarg[pathlen - 1] != b'/' {
                        pathlen -= 1;
                    }
                    (*image).description = Some(convarg[pathlen..].to_vec());
                    convarg = Vec::new();
                } else {
                    (*image).description = Some(arg.to_vec());
                }
                if (*image).has_piece_coords != 0 {
                    *any_have_piece_list = true;
                }

                /* Move the opened record into typed view-owned staging, then
                update the image-I/O address registry for its new location. */
                (*vi).image_list_storage.push((*image).clone());
                let new_image = (*vi).image_list_storage.last_mut().unwrap();
                ii_file_change_address(image, new_image);
                ii_delete(image);
                image = new_image;
                if ind_vol == 0 {
                    base_image = image;
                }

                /* Anyway, leave last file in vi->hdr/image */
                (*vi).image = image;
                (*vi).hdr = image;
                ind_vol += 1;
            }
        }

        if *any_have_piece_list && (*vi).num_times > 1 && (*vi).image_pyramid == 0 {
            (*vi).image_pyramid = -1;
        }

        (*vi).image_list = (*vi).image_list_storage.as_mut_ptr();
    }
}

/// `ivwLoadImage` (`imodview.cpp:2705`); load images initially, use for all
/// kinds of data.
pub unsafe fn ivw_load_image(vi: *mut ImodView) -> i32 {
    unsafe {
        let usable_mem = b3d_addressable_memory() / (1024. * 1024.);
        let mut mem_limit = 0.75 * usable_mem;
        if let Some(env) = std::env::var_os("3DMOD_MEMORY_LIMIT") {
            let env_val = env.to_string_lossy().trim().parse::<f32>().unwrap_or(0.);
            if env_val > 0.1 && env_val <= 1.0 {
                mem_limit *= usable_mem;
            } else if env_val > 10. && (env_val as f64) <= usable_mem {
                mem_limit = env_val as f64;
            }
        }

        if (*vi).fake_image != 0 {
            // Initialize various things for no image; access Model not vi->imod
            let model = with_boundary(|n| n.model_global());
            (*vi).xsize = (*model).xmax;
            (*vi).ysize = (*model).ymax;
            (*vi).zsize = (*model).zmax;
            (*vi).xybin = 1;
            (*vi).zbin = 1;
            (*vi).x_unbin_size = (*vi).xsize;
            (*vi).y_unbin_size = (*vi).ysize;
            (*vi).z_unbin_size = (*vi).zsize;
            if (*vi).num_times > 1 {
                ivw_set_time(&mut *vi, 1);
            }

            (*vi).raw_image_store = 0;
            with_boundary(|n| n.info_widget_hide_low_high_grid());
            with_boundary(|n| n.imod_color_init());

            wprint(&format!(
                "Image size {} x {}, {} sections.\n",
                (*vi).xsize,
                (*vi).ysize,
                (*vi).zsize
            ));
            S_BEST_IVW_GET_VALUE.with(|s| s.set(fake_ivw_get_value));

            /* DNM: set the axis flag based on the model flip flag */
            if (*(*vi).li).axis == 2 {
                with_boundary(|n| {
                    n.imod_error(
                        None,
                        "The -Y flag is ignored when loading a model without an image.\nUse \
                         Edit-Image-Flip to flip the model if desired",
                    )
                });
            }
            (*(*vi).li).axis = 3;
            if (*model).flags & IMODF_FLIPYZ != 0 {
                (*(*vi).li).axis = 2;
            }
            with_boundary(|n| n.prefs_set_info_geometry());
            return initialize_flip_and_model(vi);
        }

        ivw_process_image_list(vi);

        // Set info window up now that size is known
        with_boundary(|n| n.prefs_set_info_geometry());
        (*vi).doing_initial_load = 1;

        /* Set up the cache and load it for a variety of conditions */
        let pix_size = ivw_get_pixel_bytes((*vi).raw_image_store as i32);
        if (*(*vi).li).plist != 0
            || (*vi).num_times != 0
            || (*vi).multi_file_z > 0
            || (*vi).vm_size != 0
            || (*vi).ushort_store != 0
            || !(*vi).pyr_cache.is_null()
            || (mem_limit > 0.
                && ((((*vi).xsize as f32 * (*vi).ysize as f32)
                    * (*vi).zsize as f32
                    * pix_size as f32) as f64
                    / 1024.
                    * 1024.)
                    > mem_limit)
        {
            /* DNM: only one mode won't work now; just exit in either case */
            if (*(*vi).hdr).mode == MRC_MODE_COMPLEX_SHORT {
                with_boundary(|n| {
                    n.imod_error(
                        None,
                        "3DMOD Error: Image cache and piece lists do not work with complex \
                         short data.\n",
                    )
                });
                std::process::exit(3);
            }

            (*vi).idata = ptr::null_mut();

            /* print load status */
            wprint(&format!(
                "Image size {} x {}, {} sections.\n",
                (*vi).xsize,
                (*vi).ysize,
                (*vi).zsize
            ));

            ivw_set_cache_size(vi, mem_limit);

            if !(*vi).pyr_cache.is_null() {
                (*(*vi).pyr_cache).initialize_caches();
                S_BEST_IVW_GET_VALUE.with(|s| s.set(tiles_ivw_get_value));
            } else {
                /* initialize ordinary cache, make sure axis is set for all data
                structures.  Set axis to 3 for first initialization because vmSize has
                been computed based on unflipped Z dimensions */
                let axis_save = (*(*vi).li).axis;
                (*(*vi).li).axis = 3;
                let eret = ivw_init_cache(&mut *vi, &*(*vi).li);
                if eret != 0 {
                    return eret;
                }

                S_BEST_IVW_GET_VALUE.with(|s| s.set(cache_ivw_get_value));

                /* If we are to keep cache full, fill it now and restore axis for possible
                flip later, unless user flipped it via menu */
                if (*vi).keep_cache_full != 0 {
                    let eret = with_boundary(|n| n.imod_cache_fill(vi));
                    if eret != 0 {
                        return eret;
                    }
                }
                if (*(*vi).li).axis == 3 {
                    (*(*vi).li).axis = axis_save;
                } else {
                    (*(*vi).li).axis = if axis_save == 3 { 2 } else { 3 };
                }
            }
        } else {
            /* Finally, here is what happens for non-cached data of all kinds */

            /* DNM 9/25/03: it probably no longer matters if data are ever contiguous
            so switch to noncontiguous for anything above 1 GB */
            (*(*vi).li).contig = 1;
            if (1000000000 / (*vi).xsize) / (*vi).ysize < (*vi).zsize {
                (*(*vi).li).contig = 0;
            }

            S_BEST_IVW_GET_VALUE.with(|s| s.set(idata_ivw_get_value));
            (*vi).idata = with_boundary(|n| n.imod_io_image_load(vi));
            if (*vi).idata.is_null() {
                /* Let caller do error message */
                return -1;
            }
        }

        if imod_depth() == 8 {
            ivw_scale(vi);
        }

        initialize_flip_and_model(vi)
    }
}

/// `ivwProcessImageList` (`imodview.cpp:2837`); process an image list and
/// determine sizes and types of files.
unsafe fn ivw_process_image_list(vi: *mut ImodView) -> i32 {
    unsafe {
        let image_list = &mut (*vi).image_list_storage;
        let mut image: *mut ImodImageFile = ptr::null_mut();
        let mut pix_size_vec: Vec<f32> = Vec::new();
        let (mut xsize, mut ysize, mut zsize) = (0i32, 0i32, 0i32);
        let mut tiff_compression = IICOMPRESSION_NONE;
        let (mut smin, mut smax) = (0.0f32, 0.0f32);
        let mut pixel_size = 0.0f32;
        let pix_tol = 1.0e-4f32;
        let (mut rgbs, mut cmaps, mut all_byte, mut all_can_read_int) = (0, 0, 1, 1);
        let mut any_tiled = false;
        let mut any_tiffs = false;
        let sect_names: [&[u8]; 4] = [
            crate::imod::libcfshr::autodoc::ADOC_ZVALUE_NAME,
            b"Image",
            crate::imod::libcfshr::autodoc::ADOC_ZVALUE_NAME,
            b"MontSection",
        ];

        if image_list.is_empty() {
            return -1;
        }

        (*vi).pix_size_index = vec![-1; image_list.len()];

        /* First get minimum x, y, z sizes of all the files and count up rgbs */
        for i in 0..image_list.len() {
            image = image_list.as_mut_ptr().add(i);
            if i != 0 {
                smin = smin.min((*image).smin);
                smax = smax.max((*image).smax);
            } else {
                smin = (*image).smin;
                smax = (*image).smax;
            }

            if get_valid_scale(image, None, None) > 1
                && with_boundary(|n| n.prefs_load_int_if_mean_sd())
            {
                (*vi).switch_to_ushort = 1;
            }

            if ((*image).type_ != IITYPE_UBYTE && (*image).type_ != IITYPE_BYTE)
                || (*image).format != IIFORMAT_LUMINANCE
            {
                all_byte = 0;
                if (*image).read_section_ushort.is_none() {
                    all_can_read_int = 0;
                }
            }

            // See if mirroring of an FFT is needed: Not forbidden by option
            // (MRC complex float odd size not reliable, eliminated 7/16/13)
            // Set flags and increase the nx
            if ((*image).file == IIFILE_MRC
                || (*image).file == IIFILE_RAW
                || (*image).file == IIFILE_SHR_MEM
                || (*image).file == IIFILE_HDF)
                && (*image).format == IIFORMAT_COMPLEX
                && (*image).type_ == IITYPE_FLOAT
                && (*(*vi).li).mirror_fft >= 0
            {
                (*image).mirror_fft = 1;
                if (*image).file == IIFILE_MRC {
                    // Analyze real MRC file for whether the hot pixel in the middle of
                    // of the file is much higher than at the bottom, if so don't mirror
                    ivw_reopen(&mut *image);
                    let midy = (*image).ny / 2;
                    let midz = (*image).nz / 2;
                    let mut naysum = ii_read_point(&mut *image, 1, midy, 0)
                        + ii_read_point(&mut *image, 0, midy - 1, 0)
                        + ii_read_point(&mut *image, 1, midy + 1, 0);
                    let mut zratio = 0.;
                    if naysum > 0. {
                        zratio = ii_read_point(&mut *image, 0, midy, 0) / naysum;
                    }
                    naysum = ii_read_point(&mut *image, 1, midy, midz)
                        + ii_read_point(&mut *image, 0, midy - 1, midz)
                        + ii_read_point(&mut *image, 1, midy + 1, midz);
                    let mut mratio = 0.;
                    if naysum > 0. {
                        mratio = ii_read_point(&mut *image, 0, midy, midz) / naysum;
                    }
                    if zratio != 0. && mratio != 0. && mratio > 10. * zratio {
                        (*image).mirror_fft = 0;
                    }
                    ii_close(image);
                }
                if (*image).mirror_fft != 0 {
                    (*(*vi).li).mirror_fft = 1;
                    (*image).nx = ((*image).nx - 1) * 2;
                }
            }

            // Keep track of largest image size
            if i == 0 || (*image).nx > xsize {
                xsize = (*image).nx;
            }
            if i == 0 || (*image).ny > ysize {
                ysize = (*image).ny;
            }
            if i == 0 || (*image).nz > zsize {
                zsize = (*image).nz;
            }

            /* Add to count if RGB or not, to see if all the same type.  Similarly
            for colormap images */
            if (*image).format == IIFORMAT_RGB
                && !(((*image).file == IIFILE_MRC
                    || (*image).file == IIFILE_RAW
                    || (*image).file == IIFILE_HDF
                    || (*image).file == IIFILE_TIFF
                    || (*image).file == IIFILE_JPEG
                    || (*image).file == IIFILE_QIMAGE
                    || (*image).file == IIFILE_SHR_MEM)
                    && (*vi).gray_rgbs != 0)
            {
                rgbs += 1;
            }
            if (*image).format == IIFORMAT_COLORMAP && (*vi).gray_rgbs == 0 {
                cmaps += 1;
            }
            if (*vi).strip_or_tile_cache != 0
                && ((*image).format == IIFORMAT_COMPLEX || rgbs != 0 || cmaps != 0)
            {
                with_boundary(|n| {
                    n.imod_error(
                        None,
                        "3DMOD Error: You cannot cache tiles or strips with complex,\ncolormap, \
                         or RGB data, except when loading RGB as grayscale",
                    )
                });
                std::process::exit(3);
            }

            // lLook for TIFFs for evaluating parallel reading
            if (*image).file == IIFILE_TIFF {
                any_tiffs = true;
                if (*image).tile_size_x != 0 {
                    any_tiled = true;
                }
                if (*image).tiff_compression != IICOMPRESSION_NONE {
                    tiff_compression = (*image).tiff_compression;
                }
            }

            // Look for pixel size variations
            if i == 0 {
                pixel_size = (*image).xscale;
            }
            if ((pixel_size / (*image).xscale) as f64 - 1.).abs() > pix_tol as f64 {
                (*vi).pixel_size_varies = 1;
            }

            // Skip single-image files and look for an mdoc if there is not an adoc index
            if (*image).nz > 1 {
                let mut vary_in_adoc = false;
                pix_size_vec.clear();
                let mut adoc_ind = (*image).adoc_index;
                let mut if_montage = 0;
                let mut num_adoc_sect = 0;
                let mut sect_type = 0;
                if adoc_ind < 0 {
                    adoc_ind = adoc_open_image_metadata(
                        (*image).filename.as_deref().unwrap_or("").as_bytes(),
                        if (*image).file == IIFILE_ADOC { 0 } else { 1 },
                        &mut if_montage,
                        &mut num_adoc_sect,
                        &mut sect_type,
                    );
                }

                // In either case, get the properties, and set up to get MontSection for
                // montage
                if adoc_ind >= 0
                    && adoc_set_current(adoc_ind) == 0
                    && adoc_get_image_meta_info(&mut if_montage, &mut num_adoc_sect, &mut sect_type)
                        == 0
                {
                    if if_montage > 0 {
                        sect_type = 4;
                        num_adoc_sect =
                            adoc_get_number_of_sections(sect_names[(sect_type - 1) as usize]);
                    }
                    for iz in 0..(*image).nz.min(num_adoc_sect) {
                        let mut sect_ind = iz;
                        if sect_type != 2 {
                            sect_ind =
                                adoc_lookup_by_name_value(sect_names[(sect_type - 1) as usize], iz);
                        }
                        let mut sec_pixel = 0.0f32;
                        if sect_ind >= 0
                            && adoc_get_float(
                                sect_names[(sect_type - 1) as usize],
                                sect_ind,
                                b"PixelSpacing",
                                &mut sec_pixel,
                            ) == 0
                        {
                            if ((sec_pixel / (*image).xscale) as f64 - 1.).abs() < pix_tol as f64 {
                                vary_in_adoc = true;
                            }
                            pix_size_vec.push(sec_pixel);
                        } else {
                            pix_size_vec.push(0.);
                        }
                    }
                }

                // Save the pixel sizes and set index if they varied
                if vary_in_adoc {
                    (*vi).pixel_size_varies = 1;
                    (&mut (*vi).pix_size_index)[i as usize] = (*vi).adoc_pix_sizes.len() as i32;
                    (*vi).adoc_pix_sizes.push(pix_size_vec.clone());
                }
                if (*image).adoc_index < 0 {
                    adoc_clear(adoc_ind);
                }
            }
        }

        // Switch to loading shorts if anybody was scaled by mean/SD or by estimation and
        // preference was to do so
        if (*vi).switch_to_ushort != 0 && (*vi).raw_image_store == 0 {
            (*vi).raw_image_store = MRC_MODE_USHORT as i16;
            (*vi).switch_to_ushort = 2;
        }

        // Cancel integer loading if all files are bytes or there is color loading
        if all_byte != 0 || rgbs != 0 || cmaps != 0 {
            (*vi).raw_image_store = 0;
        }
        if (*vi).raw_image_store as i32 == MRC_MODE_USHORT && app_rgba() == 0 {
            with_boundary(|n| {
                n.imod_error(
                    None,
                    "3DMOD Error: You can not store data as integers with the -ci option.\n",
                )
            });
            std::process::exit(3);
        }
        if (*vi).raw_image_store as i32 == MRC_MODE_USHORT && all_can_read_int == 0 {
            if (*vi).switch_to_ushort > 1 || (*vi).int_opt_entered == 0 {
                (*vi).raw_image_store = 0;
            } else {
                with_boundary(|n| {
                    n.imod_error(
                        None,
                        "3DMOD Error: -I option cannot be used because some files cannot be read \
                         from as integers.\n",
                    )
                });
                std::process::exit(3);
            }
        }

        // And now the color ramps can be initialized and other flags set
        if (*vi).raw_image_store as i32 == MRC_MODE_USHORT {
            set_app_rgba(2);
            (*vi).ushort_store = 1;
            (*vi).white = 65535;
            (*(*vi).li).outmax = 65535;
        } else {
            with_boundary(|n| n.info_widget_hide_low_high_grid());
        }
        with_boundary(|n| n.imod_color_init());

        /* Deal with color files */
        if rgbs != 0 || cmaps != 0 {
            if (rgbs != 0 && rgbs < image_list.len() as i32)
                || (cmaps != 0 && cmaps < image_list.len() as i32)
            {
                let count = if rgbs != 0 { rgbs } else { cmaps };
                let kind = if rgbs != 0 { "RGB" } else { "colormap" };
                let total = image_list.len();
                with_boundary(|n| {
                    n.imod_error(
                        None,
                        &format!(
                            "3DMOD Error: Only {count} files out of {total} are {kind} type and \
                             all files must be.\n"
                        ),
                    )
                });
                std::process::exit(3);
            }

            if app_rgba() == 0 {
                let kind = if rgbs != 0 { "RGB" } else { "colormap" };
                with_boundary(|n| {
                    n.imod_error(
                        None,
                        &format!(
                            "3DMOD Error: You must not start 3dmod with the -ci option to \
                             display {kind} files.\n"
                        ),
                    )
                });
                std::process::exit(3);
            }

            /* For RGB, set the flag for storing raw images with the mode, and set
            rgba to indicate the number of bytes being stored */
            if rgbs != 0 {
                set_app_rgba(3);
                (*vi).raw_image_store = MRC_MODE_RGB as i16;
                (*vi).rgb_store = 1;
            } else {
                (*vi).colormap_image = 1;
                (*(*vi).cramp).falsecolor = 2;
                (*(*vi).li).axis = 3;
            }
        }

        /* Set the scaling including equal scaling of intensities */
        for i in 0..image_list.len() {
            image = image_list.as_mut_ptr().add(i);
            if (*vi).equal_scaling != 0
                || ((*vi).image_pyramid != 0 && (*(*vi).li).smin == (*(*vi).li).smax)
            {
                (*(*vi).li).smin = smin;
                (*(*vi).li).smax = smax;
            }
            if (*(*vi).li).smin == (*(*vi).li).smax {
                get_valid_scale(image, Some(&mut smin), Some(&mut smax));
                ii_set_mm(
                    &mut *image,
                    smin,
                    smax,
                    if (*vi).ushort_store != 0 {
                        65535.
                    } else {
                        255.
                    },
                );
            } else {
                ii_set_mm(
                    &mut *image,
                    (*(*vi).li).smin,
                    (*(*vi).li).smax,
                    if (*vi).ushort_store != 0 {
                        65535.
                    } else {
                        255.
                    },
                );
            }
        }

        if (*(*vi).li).plist == 0 {
            /* Deal with non-montage case */
            if (*vi).image_pyramid != 0 {
                xsize = (*vi).x_unbin_size;
                ysize = (*vi).y_unbin_size;
                zsize = (*vi).z_unbin_size;
            } else if image_list.len() > 1 && zsize == 1 && (*vi).multi_file_z >= 0 {
                /* If maximum Z is 1 and multifile treatment in Z is allowed, set zsize
                to number of files, and cancel treatment as times */
                zsize = image_list.len() as i32;
                (*vi).multi_file_z = image_list.len() as i32;
                with_boundary(|n| n.info_widget_show_or_hide_ramps());
                (*vi).cur_time = 0;
                (*vi).num_times = 0;
            } else if image_list.len() == 1
                && zsize > 1
                && (*vi).multi_file_z == 0
                && (*vi).image_pyramid == 0
                && (*image).file == IIFILE_MRC
            {
                // If only one file and -T option is not given, check for a volume stack
                // with at least 2 volumes
                let header = (*image)
                    .mrc_header
                    .as_ref()
                    .expect("MRC image has an owned header");
                if header.ispg == 401 && header.nz / header.mz > 1 {
                    zsize = header.mz;
                    (*vi).num_times = header.nz / header.mz;
                    (*vi).cur_time = 1;
                    (*vi).volume_stack = 1;

                    // Have to reopen to get the fp
                    if ii_reopen(&mut *image) != 0 {
                        with_boundary(|n| {
                            n.imod_error(None, "3DMOD Error: Reopening volume stack file")
                        });
                        std::process::exit(3);
                    }

                    let mut volume_images = Vec::with_capacity((*vi).num_times as usize);
                    volume_images.push(image_list[0].clone());
                    ii_file_change_address(image_list.as_mut_ptr(), volume_images.as_mut_ptr());
                    image_list.clear();
                    *image_list = volume_images;
                    image = image_list.as_mut_ptr();
                    let stable_image = (*image).clone();
                    for i in 1..(*vi).num_times {
                        image_list.push(stable_image.clone());
                        let new_image = image_list.last_mut().unwrap();
                        if ii_add_to_opened_list(&mut *new_image) != 0 {
                            with_boundary(|n| {
                                n.imod_error(
                                    None,
                                    "3DMOD Error: Memory error setting up duplicate files for  \
                                     volume stack",
                                )
                            });
                            std::process::exit(3);
                        }
                        (*new_image).description = Some(format!("Volume {}", i + 1).into_bytes());
                    }
                    (*image_list.first_mut().unwrap()).description = Some(b"Volume 1".to_vec());
                }
            }

            /* Use this to fix the load-in coordinates, then use those to set the
            lower left and upper right coords in each file - except for Z in the
            multifile Z case, which is set to 0 - 0 */
            mrc_fix_li(&mut *(*vi).li, xsize, ysize, zsize);
            if any_tiffs
                && (*vi).image_pyramid == 0
                && !((*vi).strip_or_tile_cache != 0 && any_tiled)
            {
                (*vi).max_read_threads = tiff_num_read_threads(
                    (*(*vi).li).xmax + 1 - (*(*vi).li).xmin,
                    (*(*vi).li).ymax + 1 - (*(*vi).li).ymin,
                    tiff_compression,
                    MAX_READ_THREADS as i32,
                );
            }
            ivw_check_binning(vi, xsize, ysize, zsize);
            if (*vi).image_pyramid == 0 {
                for i in 0..image_list.len() {
                    image = image_list.as_mut_ptr().add(i);
                    (*image).llx = (*(*vi).li).xmin;
                    (*image).lly = (*(*vi).li).ymin;
                    (*image).llz = if (*vi).multi_file_z > 0 {
                        0
                    } else {
                        (*(*vi).li).zmin
                    };
                    (*image).urx = (*(*vi).li).xmax;
                    (*image).ury = (*(*vi).li).ymax;
                    (*image).urz = if (*vi).multi_file_z > 0 {
                        0
                    } else {
                        (*(*vi).li).zmax
                    };

                    // If not an MRC file, or if multifile in Z, set to no flipping unless
                    // cache full
                    if ((*image).file != IIFILE_MRC
                        && (*image).file != IIFILE_RAW
                        && (*image).file != IIFILE_HDF
                        && (*image).file != IIFILE_SHR_MEM)
                        || (*vi).multi_file_z > 0
                    {
                        (*vi).flippable = 0;
                    }
                }
            }
        } else {
            /* For montage, do the fix_li and see if it is rgb */
            mrc_fix_li(&mut *(*vi).li, 0, 0, 0);
            image = image_list.as_mut_ptr();
            ivw_check_binning(vi, (*image).nx, (*image).ny, (*image).nz);
        }

        // 3/17/11: This fixes bugs from various places not testing <= 0 vs > 0
        (*vi).multi_file_z = 0.max((*vi).multi_file_z);

        if image_list.len() == 1 {
            /* for single file, cancel times and copy "list" to vi->image */
            (*vi).image = ii_new();
            (*vi).hdr = (*vi).image;
            if (*vi).image.is_null() {
                with_boundary(|n| n.imod_error(None, "Not enough memory.\n"));
                std::process::exit(3);
            }
            let staged_image = image_list.as_mut_ptr();
            *(*vi).image = (*staged_image).clone();
            ii_file_change_address(staged_image, (*vi).image);
            image_list.clear();
            ivw_reopen(&mut *(*vi).image);
            (*vi).cur_time = 0;
            (*vi).num_times = 0;
            (*vi).image_list = ptr::null_mut();
            if cmaps != 0 {
                let map = (*(*vi).image)
                    .colormap
                    .as_deref()
                    .unwrap()
                    .as_ptr()
                    .add(768 * (*(*vi).li).zmin as usize)
                    .cast::<[[u8; 256]; 3]>();
                crate::imod::three_dmod::xcramp::xcramp_copyfalsemap(&*map);
                crate::imod::three_dmod::xcramp::xcramp_ramp(&mut *(*vi).cramp);
            }
        } else {
            (*vi).image_list = image_list.as_mut_ptr();

            let base = if (*vi).image_pyramid != 0 {
                (*(*vi).pyr_cache).base_index
            } else {
                0
            };
            (*vi).image = (*vi).image_list.add(base as usize);
            (*vi).hdr = (*vi).image;

            /* for times, set up initial time; for multifile Z, reopen first image */
            if (*vi).image_pyramid != 0 {
                (*vi).cur_time = 0;
                (*vi).num_times = 0;
                ivw_reopen(&mut *(*vi).image);
            } else if (*vi).multi_file_z == 0 {
                ivw_set_time(&mut *vi, 1);
                (*vi).dim |= 8;
            } else {
                ivw_reopen(&mut *(*vi).image);
            }
        }
        if (*vi).ushort_store != 0 {
            let (low, high) = ((*vi).range_low, (*vi).range_high);
            let (smn, smx) = ((*(*vi).image).smin, (*(*vi).image).smax);
            let is_float = (*(*vi).image).type_ == IITYPE_FLOAT;
            with_boundary(|n| n.info_widget_set_lh_sliders(low, high, smn, smx, is_float));
        }

        0
    }
}

/// `App->rgba` (`imodP.h:46`).
fn app_rgba() -> i32 {
    APP.lock().unwrap().as_ref().map_or(0, |app| app.rgba)
}

/// Assignment to `App->rgba`.
fn set_app_rgba(value: i32) {
    if let Some(app) = APP.lock().unwrap().as_mut() {
        app.rgba = value;
    }
}

/// `getValidScale` (`imodview.cpp:3249`); returns 1 if the image has a valid
/// scaling from min/max, 2 if from mean/SD, or 0 if none.
unsafe fn get_valid_scale(
    image: *mut ImodImageFile,
    min: Option<&mut f32>,
    max: Option<&mut f32>,
) -> i32 {
    unsafe {
        // The min and max are valid if they are not crossed and not both zero
        let min_max_valid =
            (*image).amax >= (*image).amin && ((*image).amax != 0. || (*image).amin != 0.);
        let mut rms_valid = (*image).rms > 0.;

        // The RMS is valid if it is positive or if 0 is supposed to be valid due to flags
        if (*image).file == IIFILE_MRC && (*image).rms == 0. {
            if let Some(hdata) = (*image).mrc_header.as_ref() {
                rms_valid = mrc_get_standard_version(Some(hdata)) > 0
                    || (hdata.imod_stamp == IMOD_MRC_STAMP
                        && (hdata.imod_flags | MRC_FLAGS_BAD_RMS_NEG) != 0);
            }
        }

        // The mean is valid if it is bigger than the min of min and max, and not bigger
        // than a valid max, and something is non-zero; mean & SD valid if RMS is valid too
        let mean_sd_valid = (*image).amean >= (*image).amin.min((*image).amax)
            && !((*image).amean as f64
                > (*image).amax as f64 + 1.001 * ((*image).amax - (*image).amin) as f64
                && min_max_valid)
            && ((*image).amin != 0. || (*image).amax != 0. || (*image).amean != 0.)
            && rms_valid;

        imod_trace(
            'r',
            &format!(
                "mmvalid {}  meanvalid {}  rmsvalid {}",
                i32::from(min_max_valid),
                i32::from(mean_sd_valid),
                i32::from(rms_valid)
            ),
        );
        // Return mean/SD based value if no valid min/max or pref is to use mean/SD
        if mean_sd_valid && (!min_max_valid || with_boundary(|n| n.prefs_prefer_mean_sd())) {
            let (Some(min), Some(max)) = (min, max) else {
                return 2;
            };
            let num_sds = with_boundary(|n| n.prefs_num_sds_for_scaling());
            *min = (*image).amean - num_sds * (*image).rms;
            *max = (*image).amean + num_sds * (*image).rms;

            // Limit the values for integer modes
            let mut min_lim = 0.0f32;
            let mut max_lim = 0.0f32;
            let mut dummy = 0.0f32;
            if (*image).type_ != IITYPE_FLOAT
                && ii_default_min_max_mean(
                    if (*image).type_ == IITYPE_BYTE {
                        IITYPE_UBYTE
                    } else {
                        (*image).type_
                    },
                    &mut min_lim,
                    &mut max_lim,
                    &mut dummy,
                ) == 0
            {
                *min = min.max(min_lim);
                *max = max.min(max_lim);
            }
            imod_trace(
                'r',
                &format!(
                    "mean {:.6}  rms {:.6}  return mni/max {:.6} {:.6}",
                    (*image).amean,
                    (*image).rms,
                    *min,
                    *max
                ),
            );
            return 2;
        }

        // return valid min/max
        if min_max_valid {
            let (Some(min), Some(max)) = (min, max) else {
                return 1;
            };
            *min = (*image).amin;
            *max = (*image).amax;
            imod_trace(
                'r',
                &format!("return min/max {:.6}  {:.6}", (*image).amin, (*image).amax),
            );
            return 1;
        }
        0
    }
}

/// `initializeFlipAndModel` (`imodview.cpp:3305`).
unsafe fn initialize_flip_and_model(vi: *mut ImodView) -> i32 {
    unsafe {
        // Set this to -1 so image flipping works, but model initializing can skip things
        (*vi).doing_initial_load = -1;

        /* Flip data if called for, but do not generate error if it is not flippable */
        if (*vi).fake_image == 0 {
            let flipit = if (*(*vi).li).axis == 2 { 1 } else { 0 };
            (*(*vi).li).axis = 3;
            if flipit != 0 {
                let retcode = ivw_flip(vi);

                // This leads to an exit so just returning is OK
                if retcode != 0 && retcode != -1 {
                    return retcode;
                }
            }
        }

        // Now that flipping is done, set up read-in or new model by standard routes
        let model = with_boundary(|n| n.model_global());
        if !model.is_null() {
            if (*vi).did_model_init_in_load == 0 {
                with_boundary(|n| n.init_read_in_model_data(model, false));
            }
        } else {
            let filename = with_boundary(|n| n.imod_filename());
            let retcode = with_boundary(|n| n.create_new_model(&filename));
            if retcode != IMOD_IO_SUCCESS {
                return retcode;
            }
        }
        (*vi).doing_initial_load = 0;
        with_boundary(|n| n.clip_handler_done_with_load());
        0
    }
}

/// `ivwCheckBinning` (`imodview.cpp:3342`); check for binning and modify all
/// parameters as necessary.
unsafe fn ivw_check_binning(vi: *mut ImodView, nx: i32, ny: i32, nz: i32) -> i32 {
    unsafe {
        // Save original sizes of image
        (*vi).x_unbin_size = (*(*vi).li).xmax - (*(*vi).li).xmin + 1;
        (*vi).y_unbin_size = (*(*vi).li).ymax - (*(*vi).li).ymin + 1;
        (*vi).z_unbin_size = (*(*vi).li).zmax - (*(*vi).li).zmin + 1;
        if ((*vi).x_unbin_size as usize * (*vi).y_unbin_size as usize) / (*vi).x_unbin_size as usize
            != (*vi).y_unbin_size as usize
        {
            with_boundary(|n| {
                n.imod_error(
                    None,
                    "This image is too large in X and Y to load on a 32-bit computer.\n",
                )
            });
            std::process::exit(3);
        }

        // Adjust binning to be positive and not larger than dimensions
        if (*vi).xybin < 1 {
            (*vi).xybin = 1;
        }
        if (*vi).zbin < 1 {
            (*vi).zbin = 1;
        }
        if (*vi).xybin > nx {
            (*vi).xybin = nx;
        }
        if (*vi).xybin > ny {
            (*vi).xybin = ny;
        }
        if (*vi).zbin > nz {
            (*vi).zbin = nz;
        }

        // Forbid z binning for RGB images
        if (*vi).rgb_store != 0 && (*vi).zbin > 1 {
            (*vi).zbin = 1;
            wprint("\x07\nBinning in Z cannot be used with RGB data.\n");
        }

        // forbid Z binning for multifile Z or montage (need to test multiFileZ this way)
        if ((*vi).multi_file_z > 0
            || (*(*vi).li).plist != 0
            || (*vi).image_pyramid != 0
            || (*vi).strip_or_tile_cache != 0)
            && (*vi).zbin > 1
        {
            (*vi).zbin = 1;
            if (*vi).image_pyramid != 0 || (*vi).strip_or_tile_cache != 0 {
                wprint("\x07\nThe Z dimension cannot be binned with an image pyramid.\n");
            } else if (*vi).strip_or_tile_cache != 0 {
                wprint("\x07\nThe Z dimension cannot be binned with strip/tile caching.\n");
            } else if (*(*vi).li).plist != 0 {
                wprint("\x07\nThe Z dimension cannot be binned with montaged data.\n");
            } else {
                wprint(
                    "\x07\nThe Z dimension cannot be binned with multiple single-section \
                     files.\n",
                );
            }
        }

        // Get binned size of image file
        let mut nxbin = nx / (*vi).xybin;
        let mut nybin = ny / (*vi).xybin;
        let nzbin = nz / (*vi).zbin;

        if (*vi).xybin * (*vi).zbin > 1 {
            // Forbid flipped loading for non-isotropic binning
            if (*vi).xybin != (*vi).zbin {
                (*vi).flippable = 0;
            }

            // If montaged, adjust piece coordinates and compute new full size
            if (*(*vi).li).plist != 0 {
                let mut xmax = -1;
                let mut ymax = -1;
                for i in 0..(*(*vi).li).plist {
                    // `pcoords[]` and `xybin` are both `int`, so the source's
                    // `pcoords[3*i] / vi->xybin` truncates before `+ 0.5` widens it.
                    (&mut (*(*vi).li).pcoords).as_mut().unwrap()[(3 * i) as usize] =
                        (((&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i) as usize] / (*vi).xybin)
                            as f64
                            + 0.5) as i32;
                    if xmax < (&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i) as usize] {
                        xmax = (&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i) as usize];
                    }
                    (&mut (*(*vi).li).pcoords).as_mut().unwrap()[(3 * i + 1) as usize] =
                        (((&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i + 1) as usize]
                            / (*vi).xybin) as f64
                            + 0.5) as i32;
                    if ymax < (&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i + 1) as usize] {
                        ymax = (&(*(*vi).li).pcoords).as_ref().unwrap()[(3 * i + 1) as usize];
                    }
                }
                nxbin += xmax;
                nybin += ymax;
                (*(*vi).li).px = nxbin as f32;
                (*(*vi).li).py = nybin as f32;
                (*(*vi).li).opx /= (*vi).xybin as f32;
                (*(*vi).li).opy /= (*vi).xybin as f32;
            }

            // Adjust load-in coordinates
            (*(*vi).li).xmin /= (*vi).xybin;
            (*(*vi).li).ymin /= (*vi).xybin;
            (*(*vi).li).zmin /= (*vi).zbin;
            (*(*vi).li).xmax /= (*vi).xybin;
            (*(*vi).li).ymax /= (*vi).xybin;
            (*(*vi).li).zmax /= (*vi).zbin;
            if (*(*vi).li).xmax >= nxbin {
                (*(*vi).li).xmax = nxbin - 1;
            }
            if (*(*vi).li).ymax >= nybin {
                (*(*vi).li).ymax = nybin - 1;
            }
            if (*(*vi).li).zmax >= nzbin {
                (*(*vi).li).zmax = nzbin - 1;
            }
        }

        // Set loaded image size for current loadin
        (*vi).xsize = (*(*vi).li).xmax - (*(*vi).li).xmin + 1;
        (*vi).ysize = (*(*vi).li).ymax - (*(*vi).li).ymin + 1;
        (*vi).zsize = (*(*vi).li).zmax - (*(*vi).li).zmin + 1;
        (*vi).xysize = (*vi).xsize as usize * (*vi).ysize as usize;
        (*vi).full_xsize = nxbin;
        (*vi).full_ysize = nybin;
        (*vi).full_zsize = nzbin;
        if !(*vi).pyr_cache.is_null() {
            (*(*vi).pyr_cache).adjust_for_binning();
        }

        0
    }
}

/* Tilt angle functions */

/// `ivwReadAngleFile` (`imodview.cpp:3455`).
pub fn ivw_read_angle_file(vi: &mut ImodView, fname: &[u8]) -> i32 {
    let name = String::from_utf8_lossy(fname);
    let Some(mut fin) = ImodFile::open(&*name, "r") else {
        with_boundary(|n| {
            n.imod_error(
                None,
                &format!("3dmod warning: could not open angle file {name}"),
            )
        });
        return 1;
    };
    let mut list: Vec<f32> = Vec::new();
    // `fscanf(fin, "%f", &angle)`: the stream is the file's bytes plus a
    // cursor, which is what the translated `fscanf` takes.
    let mut contents = Vec::new();
    let _ = fin.read_to_end(&mut contents);
    let mut pos = 0usize;
    loop {
        let mut angle: f32 = 0.;
        let scanret = crate::imod::clip::clip::fscanf(
            &contents,
            &mut pos,
            "%f",
            &mut [crate::imod::clip::clip::ScanArg::Flt(&mut angle)],
        );
        if scanret != 1 {
            break;
        }
        list.push(angle);
    }
    vi.num_tilt_angles = list.len() as i32;
    vi.tilt_angles = list;
    0
}

/// `ivwGetTiltAngles` (`imodview.cpp:3488`); return number and pointer to
/// tilt angles, starting at zmin.
pub fn ivw_get_tilt_angles<'a>(vi: &'a mut ImodView, num_angles: &mut i32) -> &'a mut [f32] {
    let zmin = unsafe { vi.li.as_ref() }.map_or(0, |li| 0.max(li.zmin)) as usize;
    let angles = vi.tilt_angles.get_mut(zmin..).unwrap_or_default();
    *num_angles = angles.len() as i32;
    angles
}

/// `ivwReadAlignedPcCoords` (`imodview.cpp:3496`); read a file of aligned
/// piece coordinates for finding edge numbers for midas.
pub unsafe fn ivw_read_aligned_pc_coords(vi: *mut ImodView, fname: &[u8]) -> i32 {
    unsafe {
        let (mut xsum, mut ysum, mut nxsum, mut nysum) = (0, 0, 0, 0);
        let path = String::from_utf8_lossy(fname).into_owned();
        let Ok(contents) = std::fs::read_to_string(&path) else {
            with_boundary(|n| {
                n.imod_error(
                    None,
                    &format!("3dmod warning: could not open aligned piece coordinate file {path}"),
                )
            });
            return 1;
        };

        for qline in contents.lines() {
            if qline.is_empty() {
                break;
            }

            // Get as much as could be present on the first line
            let mut vals = [0i32; 9];
            let (v0, rest) = vals.split_at_mut(1);
            let (v1, rest) = rest.split_at_mut(1);
            let (v2, rest) = rest.split_at_mut(1);
            let (v3, rest) = rest.split_at_mut(1);
            let (v4, rest) = rest.split_at_mut(1);
            let (v5, rest) = rest.split_at_mut(1);
            let (v6, rest) = rest.split_at_mut(1);
            let (v7, v8) = rest.split_at_mut(1);
            let scanret = crate::imod::clip::clip::sscanf(
                qline,
                " %d %d %d %d %d %d %d %d %d",
                &mut [
                    crate::imod::clip::clip::ScanArg::Int(&mut v0[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v1[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v2[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v3[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v4[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v5[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v6[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v7[0]),
                    crate::imod::clip::clip::ScanArg::Int(&mut v8[0]),
                ],
            );
            if scanret == -1 {
                break;
            }

            // Check validity
            if ((*vi).bapc_xsize == 0 && scanret < 9) || scanret < 7 {
                if (*vi).bapc_xsize != 0 {
                    with_boundary(|n| {
                        n.imod_error(
                            None,
                            &format!(
                                "3dmod warning: aligned piece coordinate file has a line with \
                                 only {scanret} values and is unusable"
                            ),
                        )
                    });
                } else {
                    with_boundary(|n| {
                        n.imod_error(
                            None,
                            "3dmod warning: aligned piece coordinate file is from an old version \
                             of Blendmont and is unusable",
                        )
                    });
                }
                std::process::exit(1);
            }

            // Assign size first time
            if (*vi).bapc_xsize == 0 {
                (*vi).bapc_xsize = vals[7];
                (*vi).bapc_ysize = vals[8];
            }

            // Save other values in vectors
            (*vi).bapc_zval.push(vals[2]);
            (*vi).bapc_xpiece.push(vals[3]);
            (*vi).bapc_ypiece.push(vals[4]);
            (*vi).bapc_xcoord.push(vals[5]);
            (*vi).bapc_ycoord.push(vals[6]);
        }

        // Determin mean spacing in X and Y by brute force
        let v = &mut *vi;
        for i1 in 0..v.bapc_zval.len() {
            for i2 in 0..v.bapc_zval.len() {
                if v.bapc_zval[i1] == v.bapc_zval[i2] {
                    if v.bapc_xpiece[i1] + 1 == v.bapc_xpiece[i2]
                        && v.bapc_ypiece[i1] == v.bapc_ypiece[i2]
                    {
                        xsum += v.bapc_xcoord[i2] - v.bapc_xcoord[i1];
                        nxsum += 1;
                    }
                    if v.bapc_ypiece[i1] + 1 == v.bapc_ypiece[i2]
                        && v.bapc_xpiece[i1] == v.bapc_xpiece[i2]
                    {
                        ysum += v.bapc_ycoord[i2] - v.bapc_ycoord[i1];
                        nysum += 1;
                    }
                }
            }
        }
        (*vi).bapc_xoverlap = (*vi).bapc_xsize - xsum / nxsum;
        (*vi).bapc_yoverlap = (*vi).bapc_ysize - ysum / nysum;
        0
    }
}

/* plugin utility functions.*/

/// `ivwGetImageSize` (`imodview.cpp:3578`).
pub fn ivw_get_image_size(
    in_imod_view: &ImodView,
    out_x: &mut i32,
    out_y: &mut i32,
    out_z: &mut i32,
) {
    *out_x = in_imod_view.xsize;
    *out_y = in_imod_view.ysize;
    *out_z = in_imod_view.zsize;
}

/// `ivwGetImageStoreMode` (`imodview.cpp:3585`).
pub fn ivw_get_image_store_mode(vi: &ImodView) -> i32 {
    vi.raw_image_store as i32
}

/// `ivwDataInTileOrStripCache` (`imodview.cpp:3590`).
pub fn ivw_data_in_tile_or_strip_cache(in_imod_view: &ImodView) -> bool {
    !in_imod_view.pyr_cache.is_null()
}

/// `ivwGetTileCachedSection` (`imodview.cpp:3595`).
pub unsafe fn ivw_get_tile_cached_section(
    in_imod_view: *mut ImodView,
    section: i32,
) -> *mut *mut u8 {
    unsafe {
        if !(*in_imod_view).pyr_cache.is_null() {
            return with_boundary(|n| n.pyr_cache_get_full_section(in_imod_view, section));
        }
        ptr::null_mut()
    }
}

/// `ivwFreeTileCachedSection` (`imodview.cpp:3602`).
pub unsafe fn ivw_free_tile_cached_section(in_imod_view: *mut ImodView) {
    unsafe {
        if !(*in_imod_view).pyr_cache.is_null() {
            with_boundary(|n| n.pyr_cache_free_full_section(in_imod_view));
        }
    }
}

/// `ivwGetMovieModelMode` (`imodview.cpp:3608`).
pub fn ivw_get_movie_model_mode(vw: Option<&ImodView>) -> i32 {
    match vw {
        None => 0,
        Some(vw) => {
            if unsafe { (*vw.imod).mousemode } == IMOD_MMOVIE {
                0
            } else {
                1
            }
        }
    }
}

/// `ivwGetModel` (`imodview.cpp:3615`).
pub fn ivw_get_model(in_imod_view: Option<&ImodView>) -> Option<&Imod> {
    in_imod_view.and_then(|view| unsafe { view.imod.as_ref() })
}

/// `startExtraObjectIfNone` (`imodview.cpp:3622`).
pub fn start_extra_object_if_none(vi: &mut ImodView) {
    if vi.extra_obj.is_empty() {
        let object = imod_object_new();
        vi.num_extra_obj = i32::from(object.is_some());
        match object {
            Some(object) => {
                vi.extra_obj.push(object);
                vi.extra_obj_in_use.push(1);
            }
            None => {
                vi.extra_obj.clear();
                vi.extra_obj_in_use.clear();
            }
        }
    }
}

/// `ivwGetExtraObject` (`imodview.cpp:3640`).
pub fn ivw_get_extra_object(in_imod_view: &mut ImodView) -> Option<&mut Iobj> {
    ivw_get_an_extra_object(in_imod_view, 0)
}

/// `ivwGetFreeExtraObjectNumber` (`imodview.cpp:3645`).
pub fn ivw_get_free_extra_object_number(vi: &mut ImodView) -> i32 {
    let mut i = 1;
    while i < vi.num_extra_obj {
        if vi.extra_obj_in_use[i as usize] == 0 {
            vi.extra_obj_in_use[i as usize] = 1;

            // Return object to default when reassigning it
            imod_object_default(&mut vi.extra_obj[i as usize]);
            return i;
        }
        i += 1;
    }
    start_extra_object_if_none(vi);
    if vi.extra_obj.is_empty() {
        return -1;
    }
    vi.num_extra_obj += 1;
    vi.extra_obj.push(Iobj::default());
    vi.extra_obj_in_use.push(0);
    imod_object_default(&mut vi.extra_obj[(vi.num_extra_obj - 1) as usize]);
    vi.extra_obj_in_use[i as usize] = 1;
    vi.num_extra_obj - 1
}

/// `ivwFreeExtraObject` (`imodview.cpp:3677`).
pub fn ivw_free_extra_object(vi: &mut ImodView, obj_num: i32) -> i32 {
    if obj_num < 1 || obj_num >= vi.num_extra_obj || vi.extra_obj_in_use[obj_num as usize] == 0 {
        return 1;
    }
    ivw_clear_an_extra_object(vi, obj_num);
    vi.extra_obj_in_use[obj_num as usize] = 0;
    0
}

/// `ivwGetAnExtraObject` (`imodview.cpp:3687`).
pub fn ivw_get_an_extra_object(in_imod_view: &mut ImodView, obj_num: i32) -> Option<&mut Iobj> {
    if obj_num < 0
        || obj_num >= in_imod_view.num_extra_obj
        || in_imod_view.extra_obj_in_use[obj_num as usize] == 0
    {
        return None;
    }
    Some(&mut in_imod_view.extra_obj[obj_num as usize])
}

/// `ivwClearExtraObject` (`imodview.cpp:3696`); delete all contours in the
/// extra object.
pub fn ivw_clear_extra_object(in_imod_view: &mut ImodView) {
    ivw_clear_an_extra_object(in_imod_view, 0);
}

/// `ivwClearAnExtraObject` (`imodview.cpp:3701`).
pub fn ivw_clear_an_extra_object(in_imod_view: &mut ImodView, obj_num: i32) {
    let Some(obj) = ivw_get_an_extra_object(in_imod_view, obj_num) else {
        return;
    };
    if !obj.cont.is_empty() {
        let size = obj.cont.len() as i32;
        imod_contours_delete(&mut obj.cont, size);
    }
    obj.cont = Vec::new();
    let obj_ptr = obj as *mut Iobj;
    with_boundary(|n| n.vb_cleanup_vbd(obj_ptr));
    let obj = unsafe { &mut *obj_ptr };
    if !obj.mesh.is_empty() {
        let size = obj.mesh.len() as i32;
        imod_meshes_delete(Some(core::mem::take(&mut obj.mesh)), size);
    }
    obj.mesh = Vec::new();
    obj.store = Vec::new();
}

/// `ivwGetCurPixelSize` (`imodview.cpp:3722`); get pixel size that applies to
/// the current image.
pub fn ivw_get_cur_pixel_size(
    vi: &ImodView,
    model: Option<&Imod>,
    current_image: Option<&ImodImageFile>,
) -> Option<f32> {
    let model = model?;
    let time_ind = if vi.num_times != 0 {
        usize::try_from(vi.cur_time - 1).ok()?
    } else {
        0
    };

    // Use model pixel size if no variations or there are no units.
    let units = model.units;
    if vi.pixel_size_varies == 0 || units == IMOD_UNIT_PIXEL {
        return Some(model.pixsize);
    }

    // Otherwise get from the current image, then look for one from adoc lists.
    let mut pix_size = current_image?.xscale;
    let im_z = (vi.zmouse as f64 + 0.5).floor() as i32;
    if vi.multi_file_z != 0 {
        pix_size = vi
            .image_list_storage
            .get(usize::try_from(im_z).ok()?)?
            .xscale;
    } else if vi.num_times != 0 {
        pix_size = vi.image_list_storage.get(time_ind)?.xscale;
    }
    if let Some(&vec_ind) = vi.pix_size_index.get(time_ind)
        && vec_ind >= 0
        && let Some(pix_sizes) = vi.adoc_pix_sizes.get(vec_ind as usize)
        && let Some(&adoc_pix_size) = usize::try_from(im_z)
            .ok()
            .and_then(|index| pix_sizes.get(index))
    {
        pix_size = adoc_pix_size;
    }

    // These are Angstroms; express in same units as model units.
    Some((pix_size as f64 * 10f64.powf(-(10. + units as f64))) as f32)
}

/// `ivwEnableStipple` (`imodview.cpp:3751`).
pub fn ivw_enable_stipple(in_imod_view: &mut ImodView, enable: i32) {
    in_imod_view.draw_stipple = enable;
}

/// `ivwTrackMouseForPlugs` (`imodview.cpp:3756`).
pub fn ivw_track_mouse_for_plugs(in_imod_view: &mut ImodView, enable: i32) {
    in_imod_view.track_mouse_for_plugs =
        0.max(in_imod_view.track_mouse_for_plugs + if enable != 0 { 1 } else { -1 });
    with_boundary(|n| n.zap_set_mouse_tracking());
    with_boundary(|n| n.slicer_set_mouse_tracking());
}

/// `ivwGetTopZapZslice` (`imodview.cpp:3764`).
pub fn ivw_get_top_zap_zslice(in_imod_view: &ImodView, out_z: &mut i32) -> i32 {
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() {
        return 1;
    }
    *out_z = unsafe { (*zap).section };
    0
}

/// `ivwGetTopZapZoom` (`imodview.cpp:3773`).
pub fn ivw_get_top_zap_zoom(in_imod_view: &ImodView, out_zoom: &mut f32) -> i32 {
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() {
        return 1;
    }
    *out_zoom = unsafe { (*zap).zoom };
    0
}

/// `ivwGetTopSlicerZoom` (`imodview.cpp:3782`).
pub fn ivw_get_top_slicer_zoom(in_imod_view: &ImodView, out_zoom: &mut f32) -> i32 {
    let ss = with_boundary(|n| n.get_top_slicer());
    if ss.is_null() {
        return 1;
    }
    *out_zoom = unsafe { (*ss).zoom };
    0
}

/// `ivwSetTopZapZoom` (`imodview.cpp:3791`).
pub fn ivw_set_top_zap_zoom(in_imod_view: &ImodView, in_zoom: f32, draw: bool) -> i32 {
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() || in_zoom < 0.005 || in_zoom > 200. {
        return 1;
    }
    unsafe { (*zap).zoom = in_zoom };
    if draw {
        with_boundary(|n| n.zap_draw(zap));
    }
    0
}

/// `ivwGetTopSlicerThickness` (`imodview.cpp:3802`).
pub fn ivw_get_top_slicer_thickness(in_imod_view: &ImodView, out_thick: &mut i32) -> i32 {
    let ss = with_boundary(|n| n.get_top_slicer());
    if ss.is_null() {
        return 1;
    }
    *out_thick = unsafe { (*ss).nslice };
    0
}

/// `ivwSetTopZapZslice` (`imodview.cpp:3811`).
pub fn ivw_set_top_zap_zslice(in_imod_view: &ImodView, in_z: i32) -> i32 {
    let cvi = with_boundary(|n| n.app_cvi());
    if cvi.is_null() || in_z < 0 || in_z >= unsafe { (*cvi).zsize } {
        return 1;
    }
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() {
        return 1;
    }
    unsafe {
        if (*zap).lock != 0 {
            (*zap).section = in_z;
        } else {
            (*cvi).zmouse = in_z as f32;
        }
    }
    0
}

/// `ivwGetTopZapMouse` (`imodview.cpp:3825`).
pub fn ivw_get_top_zap_mouse(in_imod_view: &ImodView, image_pt: &mut Ipoint) -> i32 {
    with_boundary(|n| n.get_top_zap_mouse(image_pt))
}

/// `ivwGetTopZapCenter` (`imodview.cpp:3830`).
pub fn ivw_get_top_zap_center(
    in_imod_view: &ImodView,
    im_x: &mut f32,
    im_y: &mut f32,
    im_z: &mut i32,
) -> i32 {
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() {
        return 1;
    }
    unsafe {
        if (*zap).lock != 0 {
            *im_z = (*zap).section;
        } else {
            *im_z = (in_imod_view.zmouse as f64 + 0.5).floor() as i32;
        }
        *im_x = (in_imod_view.xsize as f64 / 2. - (*zap).xtrans as f64) as f32;
        *im_y = (in_imod_view.ysize as f64 / 2. - (*zap).ytrans as f64) as f32;
    }
    0
}

/// `ivwSetTopZapCenter` (`imodview.cpp:3844`).
pub fn ivw_set_top_zap_center(
    in_imod_view: &mut ImodView,
    im_x: f32,
    im_y: f32,
    im_z: i32,
    draw: bool,
) -> i32 {
    if im_x < 0.
        || im_x >= in_imod_view.xsize as f32
        || im_y < 0.
        || im_y >= in_imod_view.ysize as f32
        || im_z < 0
        || im_z >= in_imod_view.zsize
    {
        return 1;
    }
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() {
        return 1;
    }

    unsafe {
        (*zap).xtrans = (((*(*zap).vi).xsize as f64 / 2. - im_x as f64) + 0.5).floor() as i32;
        (*zap).ytrans = (((*(*zap).vi).ysize as f64 / 2. - im_y as f64) + 0.5).floor() as i32;
        if (*zap).lock != 0 {
            (*zap).section = im_z;
            if draw {
                with_boundary(|n| n.zap_draw(zap));
            }
        } else {
            in_imod_view.zmouse = im_z as f32;
            if draw {
                let view = in_imod_view as *mut ImodView;
                with_boundary(|n| n.imod_draw(view, IMOD_DRAW_XYZ));
            }
        }
    }
    0
}

/// `ivwGetTopZapDevPixelRatio` (`imodview.cpp:3868`).
pub fn ivw_get_top_zap_dev_pixel_ratio(in_imod_view: &ImodView) -> f32 {
    let zap = with_boundary(|n| n.get_top_zap_window(false));
    if zap.is_null() {
        with_boundary(|n| n.app_device_pixel_ratio())
    } else {
        unsafe { (*zap).device_pixel_ratio }
    }
}

/*
 * Do snapshot in top zap or slicer
 */

/// `ivwSnapshotTopZap` (`imodview.cpp:3877`).
pub fn ivw_snapshot_top_zap(
    name: &mut String,
    format: i32,
    check_gray_convert: bool,
    full_area: bool,
) -> i32 {
    snapshot_top_window(name, format, check_gray_convert, ZAP_WINDOW_TYPE, full_area)
}

/// `ivwSnapshotTopSlicer` (`imodview.cpp:3882`).
pub fn ivw_snapshot_top_slicer(
    name: &mut String,
    format: i32,
    check_gray_convert: bool,
    full_area: bool,
) -> i32 {
    snapshot_top_window(
        name,
        format,
        check_gray_convert,
        SLICER_WINDOW_TYPE,
        full_area,
    )
}

/// `snapshotTopWindow` (`imodview.cpp:3887`).
fn snapshot_top_window(
    name: &mut String,
    mut format: i32,
    check_gray_convert: bool,
    win_type: i32,
    full_area: bool,
) -> i32 {
    let mut zap: *mut crate::imod::three_dmod::xzap::ZapFuncs = ptr::null_mut();
    let mut slicer: *mut crate::imod::three_dmod::slicer::SlicerFuncs = ptr::null_mut();
    let retval = get_top_zap_or_slicer(win_type, &mut zap, &mut slicer);
    if retval != 0 {
        return retval;
    }
    if format != SNAPSHOT_TIF && format != SNAPSHOT_JPG && format != SNAPSHOT_PNG {
        return -2;
    }
    // `restore` is declared and tested but never assigned in the source, so
    // both branches below take its initial value.
    let restore = 0;
    if format != SNAPSHOT_TIF {
        with_boundary(|n| n.b3d_set_non_tiff_snap_format(format));
        if restore < 0 {
            return -3;
        }
        format = SNAPSHOT_RGB;
    }
    let retval = if win_type == ZAP_WINDOW_TYPE {
        with_boundary(|n| n.zap_named_snapshot(zap, name, format, check_gray_convert, full_area))
    } else {
        with_boundary(|n| {
            n.slicer_named_snapshot(slicer, name, format, check_gray_convert, full_area)
        })
    };
    if restore != 0 {
        with_boundary(|n| n.prefs_restore_snap_format());
    }
    retval
}

/// `getTopZapOrSlicer` (`imodview.cpp:3913`).
fn get_top_zap_or_slicer(
    win_type: i32,
    zap: &mut *mut crate::imod::three_dmod::xzap::ZapFuncs,
    slicer: &mut *mut crate::imod::three_dmod::slicer::SlicerFuncs,
) -> i32 {
    if win_type == ZAP_WINDOW_TYPE {
        *zap = with_boundary(|n| n.get_top_zap_window(false));
    } else if win_type == SLICER_WINDOW_TYPE {
        *slicer = with_boundary(|n| n.get_top_slicer());
    } else {
        return -4;
    }
    if (win_type == ZAP_WINDOW_TYPE && zap.is_null())
        || (win_type == SLICER_WINDOW_TYPE && slicer.is_null())
    {
        return -1;
    }
    0
}

/*
 * Functions for starting new arrow or clearing all arrows
 */

/// `startAddedArrow` (`imodview.cpp:3930`).
pub fn start_added_arrow(window_type: i32) -> i32 {
    let mut zap: *mut crate::imod::three_dmod::xzap::ZapFuncs = ptr::null_mut();
    let mut slicer: *mut crate::imod::three_dmod::slicer::SlicerFuncs = ptr::null_mut();
    let retval = get_top_zap_or_slicer(window_type, &mut zap, &mut slicer);
    if retval != 0 {
        return retval;
    }
    if window_type == ZAP_WINDOW_TYPE {
        with_boundary(|n| n.zap_start_added_arrow(zap));
    } else {
        with_boundary(|n| n.slicer_start_added_arrow(slicer));
    }
    0
}

/// `clearAllArrows` (`imodview.cpp:3945`); clear all arrows in top window or
/// all windows of given type.
pub fn clear_all_arrows(window_type: i32, all_windows: bool) -> i32 {
    let mut zap: *mut crate::imod::three_dmod::xzap::ZapFuncs = ptr::null_mut();
    let mut slicer: *mut crate::imod::three_dmod::slicer::SlicerFuncs = ptr::null_mut();

    // Get top window regardless, for easy error check
    let i = get_top_zap_or_slicer(window_type, &mut zap, &mut slicer);
    if i != 0 {
        return i;
    }

    // Top window
    if !all_windows {
        if window_type == ZAP_WINDOW_TYPE {
            with_boundary(|n| n.zap_clear_arrows(zap));
        } else {
            with_boundary(|n| n.slicer_clear_arrows(slicer));
        }
    } else {
        // All windows
        let obj_list = with_boundary(|n| n.imod_dialog_manager_window_list(window_type));
        for window in obj_list {
            if window_type == ZAP_WINDOW_TYPE {
                zap = with_boundary(|n| n.window_list_zap(window));
                with_boundary(|n| n.zap_clear_arrows(zap));
            } else {
                slicer = with_boundary(|n| n.window_list_slicer(window));
                with_boundary(|n| n.slicer_clear_arrows(slicer));
            }
        }
    }
    0
}

/// `ivwCurrentImageFile` (`imodview.cpp:3980`); return the name of the
/// current image file, without any path adjustments.
pub fn ivw_current_image_file(in_imod_view: &ImodView, as_entered: bool) -> String {
    if in_imod_view.fake_image != 0 {
        return String::new();
    }
    let cur_dir = IMOD_IFD_PATH.lock().unwrap().clone();
    let file = if in_imod_view.multi_file_z <= 0 {
        // `image` is the legacy image-I/O cursor for the single-file case.
        unsafe { in_imod_view.image.as_ref() }
            .and_then(|image| image.filename.clone())
            .unwrap_or_default()
    } else {
        let cz = ((in_imod_view.zmouse as f64 + 0.5).floor() as i32)
            .clamp(0, in_imod_view.multi_file_z - 1);
        in_imod_view
            .image_list_storage
            .get(cz as usize)
            .and_then(|image| image.filename.clone())
            .unwrap_or_default()
    };
    if as_entered {
        return file;
    }
    let absolute = with_boundary(|n| n.qdir_absolute_file_path(&cur_dir, &file));
    with_boundary(|n| n.qdir_clean_path(&absolute))
}

/// `ivwOpen3dmodDialogs` (`imodview.cpp:3999`); open 3dmod dialogs based on
/// key letters.
pub fn ivw_open3dmod_dialogs(keys: &str) {
    with_boundary(|n| n.info_win_open_selected_windows(keys, 0));
}

/// `prefSaveGenericSettings` (`imodview.cpp:4004`).
pub fn pref_save_generic_settings(key: &str, num_vals: i32, values: &[f64]) -> i32 {
    with_boundary(|n| n.prefs_save_generic_settings(key, &values[..num_vals as usize]))
}

/// `prefGetGenericSettings` (`imodview.cpp:4009`).
pub fn pref_get_generic_settings(key: &str, values: &mut [f64], max_vals: i32) -> i32 {
    with_boundary(|n| n.prefs_get_generic_settings(key, &mut values[..max_vals as usize]))
}

/// `imodUpdateObjectDialogs` (`imodview.cpp:4014`).
pub fn imod_update_object_dialogs() {
    with_boundary(|n| n.imodv_objed_new_view());
    with_boundary(|n| n.imod_object_edit_draw());
    with_boundary(|n| n.imod_info_setobjcolor());
}

/// `ivwGetContrastReversed` (`imodview.cpp:4021`).
pub fn ivw_get_contrast_reversed(in_imod_view: &ImodView) -> i32 {
    unsafe { in_imod_view.cramp.as_ref() }.map_or(0, |cramp| cramp.reverse)
}

/// `ivwOverlayOK` (`imodview.cpp:4026`).
pub fn ivw_overlay_ok(in_imod_view: &ImodView) -> i32 {
    let cvi = with_boundary(|n| n.app_cvi());
    i32::from(app_rgba() == 1 && !cvi.is_null() && unsafe { (*cvi).raw_image_store } == 0)
}

/// `ivwSetOverlayMode` (`imodview.cpp:4031`).
pub fn ivw_set_overlay_mode(vw: &mut ImodView, sec: i32, reverse: i32, which_green: i32) {
    use crate::imod::three_dmod::xcramp::{
        xcramp_getlevels, xcramp_ramp, xcramp_reverse, xcramp_select_index, xcramp_setlevels,
    };

    let Some(cramp) = (unsafe { vw.cramp.as_mut() }) else {
        return;
    };

    // If changing state, change color ramps
    if (vw.overlay_sec != 0 && sec == 0) || (vw.overlay_sec == 0 && sec != 0) {
        if vw.overlay_ramp < 0 {
            // The first time, save the ramp index, and initialize the next ramp
            // to the same black-white levels
            vw.overlay_ramp = cramp.clevel;
            xcramp_select_index(cramp, (vw.overlay_ramp + 1) % cramp.noflevels);
            xcramp_setlevels(cramp, vw.black, vw.white);
        } else {
            // Otherwise, restore the other color ramp
            let index = if sec != 0 {
                (vw.overlay_ramp + 1) % cramp.noflevels
            } else {
                vw.overlay_ramp
            };
            xcramp_select_index(cramp, index);
            xcramp_ramp(cramp);
            let (black, white) = xcramp_getlevels(cramp);
            vw.black = black;
            vw.white = white;
            with_boundary(|n| n.imod_info_setbw(black, white));
        }

        // Reverse if flag set
        if reverse != 0 {
            let flag = i32::from(cramp.reverse == 0);
            xcramp_reverse(cramp, flag);
        }

        // If state is staying on but reverse is changing, then reverse
    } else if sec != 0 && reverse != vw.reverse_overlay {
        let flag = i32::from(cramp.reverse == 0);
        xcramp_reverse(cramp, flag);
    }

    vw.reverse_overlay = reverse;
    vw.overlay_sec = sec;
    vw.which_green = which_green;
    with_boundary(|n| n.imod_draw(vw, IMOD_DRAW_IMAGE | IMOD_DRAW_NOSYNC));
}

/// `ivwGetOrMakeContour` (`imodview.cpp:4071`); get the current contour, the
/// last contour if it is empty, or a new contour.
///
/// Model selection is represented by `model.cindex`, so this native operation
/// takes the view and model as ordinary Rust borrows.  Callers that originate
/// in a Qt event callback convert their cursors before entering here.
pub fn ivw_get_or_make_contour<'a>(
    view: &mut ImodView,
    model: &'a mut Imod,
    time_lock: i32,
) -> Option<&'a mut Icont> {
    let object_index = usize::try_from(model.cindex.object).ok()?;
    let point_limit_reached = model
        .obj
        .get(object_index)
        .and_then(|object| {
            imod_contour_get(Some(model)).map(|contour| {
                object.extra[IOBJ_EX_PNT_LIMIT] != 0
                    && contour.pts.len() as u32 >= object.extra[IOBJ_EX_PNT_LIMIT]
            })
        })
        .unwrap_or(true);

    if point_limit_reached {
        let contour_count = model.obj.get(object_index)?.cont.len();
        // Use the last contour if it is empty so a newly-created contour
        // inherits its surface and open/closed state.
        model.cindex.contour = contour_count as i32 - 1;
        let last_is_empty =
            imod_contour_get(Some(model)).is_some_and(|contour| contour.pts.is_empty());
        if !last_is_empty {
            crate::imod::three_dmod::undoredo::undo_contour_addition_co(
                view.undo
                    .as_deref_mut()
                    .expect("initialized view undo stack"),
                model,
                contour_count as i32,
            );
            if imod_new_contour(model) != 0 {
                view.undo
                    .as_deref_mut()
                    .expect("initialized view undo stack")
                    .clear_units();
                return None;
            }
            let contour_index = usize::try_from(model.cindex.contour).ok()?;
            let time_enabled = model
                .obj
                .get(object_index)
                .is_some_and(|object| iobj_flag_time(object) != 0);
            if view.num_times != 0 && time_enabled {
                model
                    .obj
                    .get_mut(object_index)?
                    .cont
                    .get_mut(contour_index)?
                    .time = view.cur_time;
            }
        }
    }

    let contour_index = usize::try_from(model.cindex.contour).ok()?;
    let mismatch = {
        let object = model.obj.get(object_index)?;
        let contour = object.cont.get(contour_index)?;
        ivw_time_mismatch(view, time_lock, object, contour)
    };
    let current_time = ivw_window_time(view, time_lock);
    if mismatch && model.obj[object_index].cont[contour_index].pts.is_empty() {
        crate::imod::three_dmod::undoredo::undo_contour_prop_chg_cc(
            view.undo
                .as_deref_mut()
                .expect("initialized view undo stack"),
            model,
        );
        model.obj[object_index].cont[contour_index].time = current_time;
    }

    if model.cindex.point < 0 {
        model.cindex.point = model.obj[object_index].cont[contour_index].pts.len() as i32 - 1;
    }
    model.obj.get_mut(object_index)?.cont.get_mut(contour_index)
}

/// `ivwTimeMismatch` (`imodview.cpp:4114`).
pub fn ivw_time_mismatch(vi: &ImodView, timelock: i32, obj: &Iobj, cont: &Icont) -> bool {
    let time = if timelock != 0 { timelock } else { vi.cur_time };
    vi.num_times > 0 && iobj_flag_time(obj) != 0 && cont.time != 0 && time != cont.time
}

/// `ivwWindowTime` (`imodview.cpp:4121`).
pub fn ivw_window_time(vi: &ImodView, timelock: i32) -> i32 {
    if timelock != 0 { timelock } else { vi.cur_time }
}

/// `ivwRegisterInsertPoint` (`imodview.cpp:4129`).
pub fn ivw_register_insert_point(
    vi: &mut ImodView,
    model: &mut Imod,
    contour_first: Option<(f32, u32)>,
    pt: Ipoint,
    index: i32,
) -> i32 {
    if let Some((first_z, contour_flags)) = contour_first
        && (first_z as f64 + 0.5).floor() as i32 != (pt.z as f64 + 0.5).floor() as i32
        && contour_flags & ICONT_WILD == 0
    {
        crate::imod::three_dmod::undoredo::undo_contour_prop_chg_cc(
            vi.undo.as_deref_mut().expect("initialized view undo stack"),
            model,
        );
    }
    crate::imod::three_dmod::undoredo::undo_point_addition_cc(
        vi.undo.as_deref_mut().expect("initialized view undo stack"),
        model,
        index,
    );
    let result = imod_insert_point(Some(model), Some(pt), index);
    if result <= 0 {
        vi.undo
            .as_deref_mut()
            .expect("initialized view undo stack")
            .flush_unit();
    } else {
        vi.undo
            .as_deref_mut()
            .expect("initialized view undo stack")
            .finish_unit(model);
    }
    result
}

/// `ivwDraw` (`imodview.cpp:4157`).
pub unsafe fn ivw_draw(in_imod_view: *mut ImodView, in_flags: i32) -> i32 {
    with_boundary(|n| n.imod_draw(in_imod_view, in_flags));
    0
}

/// `ivwRedraw` (`imodview.cpp:4163`).
pub unsafe fn ivw_redraw(vi: *mut ImodView) -> i32 {
    unsafe { imod_redraw(vi) }
}

/// `ivwGetRamp` (`imodview.cpp:4168`).
pub fn ivw_get_ramp(in_imod_view: &ImodView, out_ramp_base: &mut i32, out_ramp_size: &mut i32) {
    *out_ramp_base = in_imod_view.rampbase;
    *out_ramp_size = in_imod_view.rampsize;
}

/// `ivwGetObjectColor` (`imodview.cpp:4174`).
pub fn ivw_get_object_color(in_imod_view: &mut ImodView, in_object: i32) -> i32 {
    let obj_index = 0;

    /* check that inObject is within range. */
    if in_object < 0 {
        return obj_index;
    }
    let Some(model) = (unsafe { in_imod_view.imod.as_mut() }) else {
        return obj_index;
    };
    if in_object >= model.obj.len() as i32 {
        return obj_index;
    }

    let objbase = with_boundary(|n| n.app_objbase());
    let obj = &mut model.obj[in_object as usize];

    if imod_depth() <= 8 {
        obj.fgcolor = objbase - in_object;
    } else {
        obj.fgcolor = objbase + in_object;
    }
    obj.fgcolor
}

/// `ivwSetBlackWhiteFromModel` (`imodview.cpp:4192`); set the black and white
/// levels from the values stored in a model, adjusting as needed.
pub fn ivw_set_black_white_from_model(vi: &mut ImodView, model: Option<&Imod>) {
    let Some(model) = model else {
        return;
    };
    vi.black = model.blacklevel;
    vi.white = model.whitelevel;
    if vi.ushort_store != 0 {
        if vi.black < 256 && vi.white < 256 {
            vi.black *= 256;
            vi.white *= 256;
        }
    } else if vi.black > 255 || vi.white > 255 {
        vi.black /= 256;
        vi.white /= 256;
    }
}

/// `ivwBinByN` (`imodview.cpp:4206`); bin an array by the binning factor.
pub fn ivw_bin_by_n(array: &[u8], nxin: i32, nyin: i32, nbin: i32, brray: &mut [u8]) {
    if nxin <= 0 || nyin <= 0 || nbin <= 0 || array.len() != (nxin * nyin) as usize {
        return;
    }
    let nxout = nxin / nbin;
    let nyout = nyin / nbin;
    if brray.len() < (nxout * nyout) as usize {
        return;
    }
    // `imodview.cpp:4222` deliberately omits centered remainders for factor 2.
    let ixofs = if nbin == 2 { 0 } else { (nxin % nbin) / 2 };
    let iyofs = if nbin == 2 { 0 } else { (nyin % nbin) / 2 };
    for iy in 0..nyout {
        for ix in 0..nxout {
            let mut sum = 0i32;
            for by in 0..nbin {
                for bx in 0..nbin {
                    sum += array
                        [(ix * nbin + ixofs + bx + (iy * nbin + iyofs + by) * nxin) as usize]
                        as i32;
                }
            }
            brray[(ix + iy * nxout) as usize] = (sum / (nbin * nbin)) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records which boundary calls a translated routine makes, in order, so
    /// a host-free test can assert the sequence the source emits.
    #[derive(Default)]
    struct RecordingBoundary {
        calls: std::rc::Rc<RefCell<Vec<String>>>,
        zap: *mut crate::imod::three_dmod::xzap::ZapFuncs,
    }

    impl ImodviewNativeBoundary for RecordingBoundary {
        fn imc_set_movierate(&mut self, vi: &mut ImodView, rate: i32) {
            self.calls
                .borrow_mut()
                .push(format!("imcSetMovierate({rate})"));
        }
        fn imod_draw(&mut self, vi: *mut ImodView, flags: i32) {
            self.calls.borrow_mut().push(format!("imodDraw({flags})"));
        }
        fn imod_image_scale_update(&mut self, vi: *mut ImodView) {
            self.calls.borrow_mut().push("imodImageScaleUpdate".into());
        }
        fn info_win_set_window_title(&mut self, prefix: &str) {
            self.calls
                .borrow_mut()
                .push(format!("setWindowTitle({prefix})"));
        }
        fn get_top_zap_window(
            &mut self,
            with_band: bool,
        ) -> *mut crate::imod::three_dmod::xzap::ZapFuncs {
            self.calls.borrow_mut().push("getTopZapWindow".into());
            self.zap
        }
        fn zap_draw(&mut self, zap: *mut crate::imod::three_dmod::xzap::ZapFuncs) {
            self.calls.borrow_mut().push("ZapFuncs::draw".into());
        }
        fn vb_cleanup_vbd(&mut self, obj: *mut Iobj) {
            self.calls.borrow_mut().push("vbCleanupVBD".into());
        }
        fn imod_set_mmode(&mut self, mode: i32) {
            self.calls
                .borrow_mut()
                .push(format!("imod_set_mmode({mode})"));
        }
    }

    fn install(boundary: RecordingBoundary) -> std::rc::Rc<RefCell<Vec<String>>> {
        let calls = boundary.calls.clone();
        IMODVIEW_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = Some(Box::new(boundary)));
        calls
    }

    fn uninstall() {
        IMODVIEW_NATIVE_BOUNDARY.with(|slot| *slot.borrow_mut() = None);
    }

    fn zeroed_load_info() -> LoadInfo {
        unsafe { LoadInfo::default() }
    }

    #[test]
    fn bin_by_three_preserves_upstream_centered_window() {
        let input: Vec<u8> = (0..36).collect();
        let mut output = [0u8; 4];
        ivw_bin_by_n(&input, 6, 6, 3, &mut output);
        assert_eq!(output, [7, 10, 25, 28]);
    }

    #[test]
    fn bin_by_two_omits_the_offsets_the_source_omits() {
        // `imodview.cpp:4222` is the only `ivwBinByN` case that does not add
        // `ixofs`/`iyofs` to the line start, so an odd input starts at 0, 0.
        let input: Vec<u8> = (0..25).collect();
        let mut output = [0u8; 4];
        ivw_bin_by_n(&input, 5, 5, 2, &mut output);
        // Rows 0-1, columns 0-1 => (0+1+5+6)/4 = 3, not the centred (6+7+11+12)/4.
        assert_eq!(output[0], 3);
    }

    #[test]
    fn time_metadata_reads_owned_view_data() {
        let mut view = ImodView {
            num_times: 2,
            cur_time: 2,
            image_list_storage: vec![
                ImodImageFile {
                    description: Some(b"first".to_vec()),
                    ..Default::default()
                },
                ImodImageFile {
                    description: Some(b"second".to_vec()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut time = 0;

        assert_eq!(ivw_get_time(&view, Some(&mut time)), 2);
        assert_eq!(time, 2);
        assert_eq!(ivw_get_time_index_label(&view, 1), b"first");
        assert_eq!(ivw_get_time_index_label(&view, 2), b"second");
        assert_eq!(ivw_get_time_label(&view), b"second");
        assert_eq!(ivw_get_time_index_label(&view, 0), b"");
        view.fake_image = 1;
        assert_eq!(ivw_get_time_index_label(&view, 1), b"");
    }

    #[test]
    fn viewer_state_uses_owned_image_ramp_and_model_data() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView {
            multi_file_z: 2,
            zmouse: 1.,
            image_list_storage: vec![
                ImodImageFile {
                    filename: Some("first.mrc".into()),
                    ..Default::default()
                },
                ImodImageFile {
                    filename: Some("second.mrc".into()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(ivw_current_image_file(&view, true), "second.mrc");
        assert_eq!(ivw_get_contrast_reversed(&view), 0);

        let mut cramp = crate::imod::three_dmod::xcramp::xcramp_allinit(24, None, 0, 255, 0)
            .expect("test ramp initializes");
        view.cramp = &mut cramp;
        ivw_set_overlay_mode(&mut view, 1, 1, 2);
        assert_eq!(
            (view.overlay_sec, view.reverse_overlay, view.which_green),
            (1, 1, 2)
        );
        assert_eq!(ivw_get_contrast_reversed(&view), 1);
        assert!(
            calls
                .borrow()
                .iter()
                .any(|call| call == &format!("imodDraw({})", IMOD_DRAW_IMAGE | IMOD_DRAW_NOSYNC))
        );

        let mut model = Imod {
            obj: vec![Iobj::default()],
            ..Imod::default()
        };
        view.imod = &mut model;
        assert_eq!(ivw_get_object_color(&mut view, -1), 0);
        let color = ivw_get_object_color(&mut view, 0);
        assert_eq!(model.obj[0].fgcolor, color);

        view.fake_image = 1;
        view.xsize = 11;
        view.ysize = 12;
        view.zsize = 13;
        ivw_set_model_trans(&mut view);
        assert_eq!((model.xmax, model.ymax, model.zmax), (11, 12, 13));
        uninstall();
    }

    #[test]
    fn filesystem_cache_accounting_accepts_an_image_reference() {
        let image = ImodImageFile::default();
        ivw_get_file_start_pos(&image);
        ivw_dump_file_sys_cache(&image);
    }

    #[test]
    fn image_reference_uses_typed_image_and_load_info() {
        let image = ImodImageFile {
            xscale: 2.,
            yscale: 3.,
            zscale: 4.,
            xtrans: 50.,
            ytrans: 60.,
            ztrans: 70.,
            xrot: 1.,
            yrot: 2.,
            zrot: 3.,
            nx: 20,
            mirror_fft: 1,
            ..Default::default()
        };
        let load_info = LoadInfo {
            xmin: 2,
            ymin: 3,
            zmin: 4,
            plist: 1,
            opx: 1.5,
            opy: 2.5,
            opz: 3.5,
            ..Default::default()
        };
        let reference = ivw_get_image_ref(Some(&image), Some(&load_info), 2, 3).unwrap();

        assert_eq!(
            reference.cscale,
            Ipoint {
                x: 2.,
                y: 3.,
                z: 4.
            }
        );
        assert_eq!(
            reference.ctrans,
            Ipoint {
                x: 56.,
                y: 27.,
                z: 8.
            }
        );
        assert_eq!(
            reference.crot,
            Ipoint {
                x: 1.,
                y: 2.,
                z: 3.
            }
        );
        assert!(ivw_get_image_ref(None, Some(&load_info), 2, 3).is_none());
    }

    #[test]
    fn tilt_angles_are_an_owned_view_slice_from_load_zmin() {
        let mut load_info = zeroed_load_info();
        load_info.zmin = 1;
        let mut view = ImodView {
            li: &mut load_info,
            num_tilt_angles: 3,
            tilt_angles: vec![-60., 0., 60.],
            ..Default::default()
        };
        let mut number = -1;
        let angles = ivw_get_tilt_angles(&mut view, &mut number);
        assert_eq!(number, 2);
        assert_eq!(angles, [0., 60.]);
        angles[0] = 1.5;
        assert_eq!(view.tilt_angles, [-60., 1.5, 60.]);
    }

    #[test]
    fn location_is_bounded() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView {
            xsize: 4,
            ysize: 3,
            zsize: 2,
            ..Default::default()
        };
        ivw_set_location(&mut view, 99, -3, 8);
        assert_eq!((view.xmouse, view.ymouse, view.zmouse), (3., 0., 1.));
        assert_eq!(*calls.borrow(), vec![format!("imodDraw({IMOD_DRAW_ALL})")]);
        uninstall();
    }

    #[test]
    fn undersize_coordinates_match_source_centering() {
        let (mut llx, mut urx, mut offset, mut left, mut right) = (0, 7, 0, 0, 0);
        assert!(!ivw_fix_under_size_coords(
            10,
            8,
            &mut llx,
            &mut urx,
            &mut offset,
            &mut left,
            &mut right
        ));
        assert_eq!((llx, urx, offset, left, right), (0, 6, 1, 1, 0));
    }

    #[test]
    fn image_padding_uses_typed_view_image_and_load_info() {
        let view = ImodView {
            full_xsize: 10,
            full_ysize: 10,
            full_zsize: 10,
            xybin: 1,
            zbin: 1,
            ..Default::default()
        };
        let image = ImodImageFile {
            nx: 8,
            ny: 8,
            nz: 8,
            urx: 7,
            ury: 7,
            urz: 7,
            ..Default::default()
        };
        let load_info = LoadInfo::default();
        let (mut llx, mut left_x, mut right_x) = (0, 0, 0);
        let (mut lly, mut left_y, mut right_y) = (0, 0, 0);
        let (mut llz, mut left_z, mut right_z) = (0, 0, 0);

        assert_eq!(
            ivw_get_image_padding(
                &view,
                Some(&image),
                Some(&load_info),
                0,
                0,
                0,
                &mut llx,
                &mut left_x,
                &mut right_x,
                &mut lly,
                &mut left_y,
                &mut right_y,
                &mut llz,
                &mut left_z,
                &mut right_z,
            ),
            0
        );
        assert_eq!((llx, left_x, right_x), (0, 1, 0));
        assert_eq!((lly, left_y, right_y), (0, 1, 0));
        assert_eq!((llz, left_z, right_z), (0, 1, 0));
    }

    #[test]
    fn cache_lifecycle_uses_owned_view_storage() {
        let mut view = ImodView {
            vm_tdim: 2,
            vm_count: 9,
            cache_index: vec![4, 5, 6, 7],
            blank_line: vec![0; 4],
            ..Default::default()
        };
        let load_info = LoadInfo {
            zmin: 0,
            zmax: 1,
            ..Default::default()
        };

        ivw_flush_cache(&mut view, Some(&load_info), -1);
        assert_eq!(view.vm_count, 0);
        assert_eq!(view.cache_index, vec![-1, 5, -1, 7]);
        ivw_free_cache(&mut view);
        assert!(view.cache_index.is_empty());
        assert!(view.blank_line.is_empty());
    }

    #[test]
    fn cached_value_reads_owned_slice_bytes() {
        let mut load_info = LoadInfo {
            axis: 3,
            ..Default::default()
        };
        let mut byte_slice = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
        byte_slice.data = vec![1, 2, 3, 4];
        let view = ImodView {
            li: &mut load_info,
            vm_tdim: 1,
            cache_index: vec![0],
            vm_cache: vec![IvwSlice {
                cz: 0,
                ct: 0,
                used: 0,
                sec: byte_slice,
            }],
            ..Default::default()
        };
        assert_eq!(cache_ivw_get_value(&view, 1, 1, 0), 4);
        S_BEST_IVW_GET_VALUE.with(|getter| getter.set(cache_ivw_get_value));
        assert_eq!(ivw_get_value(&view, 1, 1, 0), 4);
        S_BEST_IVW_GET_VALUE.with(|getter| getter.set(fake_ivw_get_value));

        let mut ushort_load_info = LoadInfo {
            axis: 3,
            ..Default::default()
        };
        let mut ushort_slice = slice_create(1, 1, MRC_MODE_USHORT).unwrap();
        ushort_slice.data = 500u16.to_ne_bytes().to_vec();
        let ushort_view = ImodView {
            li: &mut ushort_load_info,
            vm_tdim: 1,
            ushort_store: 1,
            cache_index: vec![0],
            vm_cache: vec![IvwSlice {
                cz: 0,
                ct: 0,
                used: 0,
                sec: ushort_slice,
            }],
            ..Default::default()
        };
        assert_eq!(cache_ivw_get_value(&ushort_view, 0, 0, 0), 500);
    }

    #[test]
    fn z_section_uses_owned_cache_storage_through_a_view_borrow() {
        let mut load_info = LoadInfo {
            axis: 3,
            ..Default::default()
        };
        let mut section = slice_create(2, 2, MRC_MODE_BYTE).unwrap();
        section.data = vec![10, 11, 12, 13];
        let mut view = ImodView {
            fp: Some(ImodFile::Token(1)),
            li: &mut load_info,
            xsize: 2,
            ysize: 2,
            zsize: 1,
            vm_size: 1,
            vm_tdim: 1,
            cache_index: vec![0],
            vm_cache: vec![IvwSlice {
                cz: 0,
                ct: 0,
                used: 0,
                sec: section,
            }],
            ..Default::default()
        };

        let lines = ivw_get_z_section(&mut view, 0);
        assert!(!lines.is_null());
        assert_eq!(view.vm_cache[0].used, 1);
        unsafe {
            assert_eq!(*lines.add(0), view.vm_cache[0].sec.data.as_mut_ptr());
            assert_eq!(*lines.add(1), view.vm_cache[0].sec.data.as_mut_ptr().add(2));
        }
    }

    #[test]
    fn image_copy_uses_bounded_rows_and_owned_range_map() {
        let view = ImodView {
            xsize: 2,
            ysize: 2,
            ..Default::default()
        };
        let top = vec![1, 2];
        let bottom = vec![3, 4];
        let mut copied = vec![0; 4];
        assert_eq!(
            ivw_copy_image_to_byte_buffer(&view, &[&top, &bottom], &mut copied),
            0
        );
        assert_eq!(copied, vec![1, 2, 3, 4]);

        let ushort_view = ImodView {
            xsize: 1,
            ysize: 1,
            ushort_store: 1,
            range_low: 0,
            range_high: 65535,
            ..Default::default()
        };
        let pixel = 500u16.to_ne_bytes().to_vec();
        let map = ivw_ushort_in_range_to_byte_map(&ushort_view);
        let mut mapped = vec![0; 1];
        assert_eq!(
            ivw_copy_image_to_byte_buffer(&ushort_view, &[&pixel], &mut mapped),
            0
        );
        assert_eq!(mapped, vec![map[500]]);
        assert_eq!(
            ivw_copy_image_to_byte_buffer(&view, &[&top], &mut copied),
            1
        );
    }

    #[test]
    fn model_getter_returns_a_typed_model_reference() {
        let mut model = Box::new(Imod::default());
        model.pixsize = 2.5;
        let view = ImodView {
            imod: model.as_mut(),
            ..Default::default()
        };

        assert_eq!(ivw_get_model(Some(&view)).unwrap().pixsize, 2.5);
        assert!(ivw_get_model(None).is_none());
    }

    #[test]
    fn current_pixel_size_uses_owned_image_metadata() {
        let model = Imod {
            units: 1,
            pixsize: 2.5,
            ..Default::default()
        };
        let current_image = ImodImageFile {
            xscale: 20.,
            ..Default::default()
        };
        let view = ImodView {
            num_times: 1,
            cur_time: 1,
            zmouse: 1.,
            pixel_size_varies: 1,
            image_list_storage: vec![ImodImageFile {
                xscale: 40.,
                ..Default::default()
            }],
            pix_size_index: vec![0],
            adoc_pix_sizes: vec![vec![10., 30.]],
            ..Default::default()
        };

        assert_eq!(
            ivw_get_cur_pixel_size(&view, Some(&model), Some(&current_image)),
            Some(3.0e-10)
        );
        assert_eq!(
            ivw_get_cur_pixel_size(&view, None, Some(&current_image)),
            None
        );
    }

    #[test]
    fn pixel_bytes_match_every_mrc_mode() {
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_BYTE), 1);
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_SHORT), 2);
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_USHORT), 2);
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_FLOAT), 4);
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_COMPLEX_SHORT), 4);
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_COMPLEX_FLOAT), 8);
        assert_eq!(ivw_get_pixel_bytes(MRC_MODE_RGB), 3);
        assert_eq!(ivw_get_pixel_bytes(99), 1);
    }

    #[test]
    fn init_for_model_view_stops_before_the_movie_rate() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView::default();
        view.black = 77;
        ivw_init(&mut view, true);
        // `ivwInit` returns before `imcSetMovierate` for a model view, so the
        // members after that point keep whatever they held.
        assert!(calls.borrow().is_empty());
        assert_eq!(view.model_view_vi, 1);
        assert_eq!(view.black, 77);
        assert!(view.undo.is_some());
        assert!(view.timers.is_some());

        let mut view = ImodView::default();
        view.black = 77;
        ivw_init(&mut view, false);
        assert_eq!(*calls.borrow(), vec!["imcSetMovierate(0)".to_owned()]);
        assert_eq!(view.model_view_vi, 0);
        assert_eq!(view.black, 0);
        assert_eq!(view.white, 255);
        assert_eq!(view.range_high, 65535);
        assert_eq!(view.dim, 7);
        assert_eq!(view.ghostmode, IMOD_GHOST_2SHADES);
        assert_eq!(view.movie_interval, 17);
        uninstall();
    }

    #[test]
    fn extra_object_allocation_reuses_a_freed_number() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView::default();
        assert_eq!(view.num_extra_obj, 1);
        let first = ivw_get_free_extra_object_number(&mut view);
        assert_eq!(first, 1);
        let second = ivw_get_free_extra_object_number(&mut view);
        assert_eq!(second, 2);
        assert_eq!(view.num_extra_obj, 3);
        assert_eq!(ivw_free_extra_object(&mut view, 1), 0);
        // Freeing clears the object through vbCleanupVBD, as the source does.
        assert!(calls.borrow().iter().any(|c| c == "vbCleanupVBD"));
        assert_eq!(ivw_free_extra_object(&mut view, 1), 1);
        assert_eq!(ivw_get_free_extra_object_number(&mut view), 1);
        // Object 0 can never be freed.
        assert_eq!(ivw_free_extra_object(&mut view, 0), 1);
        assert!(ivw_get_extra_object(&mut view).is_some());
        assert!(ivw_get_an_extra_object(&mut view, 9).is_none());
        uninstall();
    }

    #[test]
    fn extra_object_starts_from_imod_object_new_not_a_zeroed_struct() {
        let mut view = ImodView::default();
        let object = ivw_get_extra_object(&mut view).unwrap();
        // `imodObjectDefault` sets these; a zeroed `Iobj` would not.
        assert_eq!(object.drawmode, 1);
        assert_eq!(object.red, 0.5);
    }

    #[test]
    fn memreccpy_places_a_tile_at_the_requested_offsets() {
        let mut to = vec![0u8; 16];
        let from: Vec<u8> = (1..=4).collect();
        memreccpy(&mut to, &from, 2, 2, 1, 2, 1, 1, 0, 0, 0);
        assert_eq!(to, vec![0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn mem_line_cpy_writes_through_line_pointers() {
        let mut rows = [[0u8; 4]; 3];
        let mut lines: Vec<&mut [u8]> = rows.iter_mut().map(|row| row.as_mut_slice()).collect();
        let from: Vec<u8> = (1..=9).collect();
        mem_line_cpy(&mut lines, &from, 2, 2, 1, 1, 1, 3, 1, 1);
        assert_eq!(rows[0], [0, 0, 0, 0]);
        assert_eq!(rows[1], [0, 5, 6, 0]);
        assert_eq!(rows[2], [0, 8, 9, 0]);
    }

    #[test]
    fn wild_flag_follows_the_nearest_integer_z() {
        let mut model = Imod {
            obj: vec![Iobj {
                cont: vec![
                    Icont {
                        pts: vec![
                            Ipoint {
                                x: 0.,
                                y: 0.,
                                z: 1.2,
                            },
                            Ipoint {
                                x: 0.,
                                y: 0.,
                                z: 1.4,
                            },
                        ],
                        flags: ICONT_WILD,
                        ..Icont::default()
                    },
                    Icont {
                        pts: vec![
                            Ipoint {
                                x: 0.,
                                y: 0.,
                                z: 1.2,
                            },
                            Ipoint {
                                x: 0.,
                                y: 0.,
                                z: 1.6,
                            },
                        ],
                        ..Icont::default()
                    },
                ],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        ivw_check_wild_flag(&mut model);
        assert_eq!(model.obj[0].cont[0].flags & ICONT_WILD, 0);
        assert_eq!(model.obj[0].cont[1].flags & ICONT_WILD, ICONT_WILD);
    }

    #[test]
    fn set_time_clamps_and_records_the_model_time() {
        let calls = install(RecordingBoundary::default());
        let mut model = Imod::default();
        let mut view = ImodView {
            num_times: 3,
            fake_image: 1,
            imod: &mut model,
            ..Default::default()
        };
        ivw_set_time(&mut view, 9);
        assert_eq!(view.cur_time, 3);
        assert_eq!(model.ctime, 3);
        ivw_set_time(&mut view, -4);
        assert_eq!(view.cur_time, 1);
        assert_eq!(model.ctime, 1);
        assert_eq!(
            *calls.borrow(),
            vec![
                "imodImageScaleUpdate".to_owned(),
                "setWindowTitle(3dmod:)".to_owned(),
                "imodImageScaleUpdate".to_owned(),
                "setWindowTitle(3dmod:)".to_owned(),
            ]
        );

        // With no times at all the source zeroes both and returns at once.
        view.num_times = 0;
        ivw_set_time(&mut view, 2);
        assert_eq!(view.cur_time, 0);
        assert_eq!(model.ctime, 0);
        assert_eq!(calls.borrow().len(), 4);
        uninstall();
    }

    #[test]
    fn set_time_selects_the_owned_image_record() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView {
            num_times: 2,
            cur_time: 1,
            fake_image: 0,
            volume_stack: 1,
            image_list_storage: vec![
                ImodImageFile {
                    description: Some(b"first".to_vec()),
                    ..Default::default()
                },
                ImodImageFile {
                    description: Some(b"second".to_vec()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        ivw_set_time(&mut view, 2);
        assert_eq!(view.cur_time, 2);
        assert_eq!(ivw_get_time_label(&view), b"second");
        assert_eq!(
            view.image,
            view.image_list_storage.as_mut_ptr().wrapping_add(1)
        );
        assert_eq!(view.hdr, view.image);
        assert_eq!(
            *calls.borrow(),
            vec![
                "imodImageScaleUpdate".to_owned(),
                "setWindowTitle(3dmod:)".to_owned(),
            ]
        );
        uninstall();
    }

    #[test]
    fn black_and_white_from_model_scale_between_byte_and_short() {
        let mut model = Imod {
            blacklevel: 10,
            whitelevel: 200,
            ..Imod::default()
        };
        let mut view = ImodView {
            ushort_store: 1,
            ..Default::default()
        };
        ivw_set_black_white_from_model(&mut view, Some(&model));
        assert_eq!((view.black, view.white), (2560, 51200));

        let mut model = Imod {
            blacklevel: 2560,
            whitelevel: 51200,
            ..Imod::default()
        };
        let mut view = ImodView::default();
        ivw_set_black_white_from_model(&mut view, Some(&model));
        assert_eq!((view.black, view.white), (10, 200));
    }

    #[test]
    fn time_mismatch_needs_times_a_time_flag_and_a_contour_time() {
        let mut obj = Iobj {
            flags: 1 << 18,
            ..Iobj::default()
        };
        let mut cont = Icont {
            time: 2,
            ..Icont::default()
        };
        let mut view = ImodView {
            num_times: 3,
            cur_time: 1,
            ..Default::default()
        };
        assert!(ivw_time_mismatch(&view, 0, &obj, &cont));
        assert!(!ivw_time_mismatch(&view, 2, &obj, &cont));
        obj.flags = 0;
        assert!(!ivw_time_mismatch(&view, 0, &obj, &cont));
        assert_eq!(ivw_window_time(&view, 0), 1);
        assert_eq!(ivw_window_time(&view, 7), 7);
    }

    #[test]
    fn contour_creation_and_insertion_use_typed_view_and_model_borrows() {
        let mut model = Imod {
            cindex: Iindex {
                object: 0,
                contour: -1,
                point: -1,
            },
            obj: vec![Iobj {
                flags: 1 << 18,
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        let mut view = ImodView {
            num_times: 3,
            cur_time: 2,
            undo: Some(Box::new(crate::imod::three_dmod::undoredo::UndoRedo::new())),
            ..ImodView::default()
        };

        let contour = ivw_get_or_make_contour(&mut view, &mut model, 0)
            .expect("a selected object gets a contour");
        assert_eq!(contour.time, 2);
        let contour_first = contour.pts.first().map(|point| (point.z, contour.flags));
        assert_eq!(
            ivw_register_insert_point(
                &mut view,
                &mut model,
                contour_first,
                Ipoint {
                    x: 4.,
                    y: 5.,
                    z: 6.,
                },
                0,
            ),
            1
        );
        assert_eq!(
            model.obj[0].cont[0].pts,
            vec![Ipoint {
                x: 4.,
                y: 5.,
                z: 6.
            }]
        );
    }

    #[test]
    fn model_flip_uses_explicit_model_and_load_info_borrows() {
        let mut load_info = zeroed_load_info();
        load_info.axis = 2;
        let mut model = Imod::default();
        ivw_flip_model(&mut model, Some(&load_info), true);
        assert_ne!(model.flags & IMODF_ROT90X, 0);

        load_info.axis = 3;
        ivw_flip_model(&mut model, Some(&load_info), true);
        assert_eq!(model.flags & IMODF_ROT90X, 0);
    }

    #[test]
    fn top_zap_queries_return_the_source_s_no_window_code() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView::default();
        let mut z = 0;
        assert_eq!(ivw_get_top_zap_zslice(&view, &mut z), 1);
        let mut zoom = 0.;
        assert_eq!(ivw_get_top_zap_zoom(&view, &mut zoom), 1);
        assert_eq!(ivw_set_top_zap_zoom(&view, 2., true), 1);
        assert_eq!(ivw_get_top_zap_dev_pixel_ratio(&view), 1.);
        // A zoom outside the source's own limits is refused before the window
        // is even consulted for the draw.
        assert!(!calls.borrow().iter().any(|c| c == "ZapFuncs::draw"));
        uninstall();
    }

    #[test]
    fn snapshot_refuses_an_unsupported_format_and_an_unknown_window() {
        let mut name = String::from("out");
        assert_eq!(snapshot_top_window(&mut name, 99, false, 7, false), -4);
        assert_eq!(
            snapshot_top_window(&mut name, SNAPSHOT_TIF, false, ZAP_WINDOW_TYPE, false),
            -1
        );
    }

    #[test]
    fn adjusted_z_shifts_only_for_a_volume_stack() {
        let mut view = ImodView {
            zsize: 10,
            cur_time: 3,
            ..Default::default()
        };
        assert_eq!(ivw_adjusted_z_if_vol_stack(&view, 4), 4);
        view.volume_stack = 1;
        assert_eq!(ivw_adjusted_z_if_vol_stack(&view, 4), 24);
    }

    #[test]
    fn plist_blank_reports_a_section_with_no_piece() {
        let mut li = zeroed_load_info();
        li.plist = 2;
        li.zmin = 1;
        li.pcoords = Some(vec![0, 0, 5, 0, 0, 7]);
        assert_eq!(ivw_plist_blank(Some(&li), 4), 0);
        assert_eq!(ivw_plist_blank(Some(&li), 6), 0);
        assert_eq!(ivw_plist_blank(Some(&li), 3), 1);
        li.plist = 0;
        assert_eq!(ivw_plist_blank(Some(&li), 3), 0);
    }

    #[test]
    fn line_pointer_allocation_uses_owned_storage() {
        let mut view = ImodView::default();
        assert_eq!(ivw_check_line_ptr_allocation(&mut view, 3), 0);
        assert_eq!(view.line_ptrs.len(), 3);
        assert_eq!(view.line_ptr_max, 3);
        assert_eq!(ivw_check_line_ptr_allocation(&mut view, 2), 0);
        assert_eq!(view.line_ptrs.len(), 3);
        assert_eq!(ivw_check_line_ptr_allocation(&mut view, -1), 1);
    }

    #[test]
    fn cache_initialization_uses_typed_load_info_and_owned_storage() {
        let mut load_info = zeroed_load_info();
        load_info.xmin = 2;
        load_info.xmax = 5;
        load_info.ymin = 3;
        load_info.ymax = 5;
        load_info.zmin = 4;
        load_info.zmax = 5;
        load_info.axis = 3;
        let mut view = ImodView {
            vm_size: 2,
            num_times: 2,
            raw_image_store: MRC_MODE_BYTE as i16,
            ..Default::default()
        };

        assert_eq!(ivw_init_cache(&mut view, &load_info), 0);
        assert_eq!(view.vm_tdim, 2);
        assert_eq!(view.vm_tbase, 1);
        assert_eq!(view.vm_cache.len(), 2);
        assert_eq!(view.cache_index, vec![-1; 4]);
        assert_eq!(view.blank_line.len(), 4);
        assert!(view.vm_cache.iter().all(|slice| slice.sec.data.len() == 12));
    }

    #[test]
    fn line_pointers_step_by_the_pixel_size_of_the_mode() {
        let mut data = vec![0u8; 24];
        let mut view = ImodView::default();
        let lines = ivw_make_line_pointers(
            &mut view.line_ptrs,
            &mut view.line_ptr_max,
            &mut data,
            4,
            3,
            1,
        )
        .unwrap();
        assert_eq!(lines[0], NonNull::new(data.as_mut_ptr()));
        unsafe {
            assert_eq!(
                lines[1]
                    .expect("line pointer was constructed")
                    .as_ptr()
                    .offset_from(data.as_mut_ptr()),
                8
            );
            assert_eq!(
                lines[2]
                    .expect("line pointer was constructed")
                    .as_ptr()
                    .offset_from(data.as_mut_ptr()),
                16
            );
        }
        assert_eq!(view.line_ptr_max, 3);
        // A null data pointer is the source's own "nothing to point at".
        assert!(
            ivw_make_line_pointers(
                &mut view.line_ptrs,
                &mut view.line_ptr_max,
                &mut [],
                4,
                3,
                1,
            )
            .is_none()
        );
        assert!(
            ivw_make_line_pointers(
                &mut view.line_ptrs,
                &mut view.line_ptr_max,
                &mut data,
                0,
                3,
                1,
            )
            .is_none()
        );
    }

    #[test]
    fn read_binned_section_fills_a_blank_when_the_section_is_out_of_range() {
        let mut li = zeroed_load_info();
        li.axis = 3;
        let mut image = ImodImageFile::default();
        image.nx = 4;
        image.ny = 4;
        image.nz = 2;
        image.axis = 3;
        image.llx = 0;
        image.urx = 3;
        image.lly = 0;
        image.ury = 3;
        image.llz = 0;
        image.urz = 1;
        let mut view = ImodView {
            li: &mut li,
            xsize: 4,
            ysize: 4,
            zsize: 2,
            full_xsize: 4,
            full_ysize: 4,
            full_zsize: 2,
            ..Default::default()
        };
        let mut buf = vec![0u8; 16];
        // Section 5 is past `im.nz / zbin`, so `blankZ` is set and the whole
        // plane comes back as 127 without any read.
        let ret =
            unsafe { ivw_read_binned_section_image(&mut view, &mut image, buf.as_mut_ptr(), 5) };
        assert_eq!(ret, 0);
        assert!(buf.iter().all(|&b| b == 127));
    }

    #[test]
    fn fast_access_picks_the_getter_the_source_picks() {
        let mut li = zeroed_load_info();
        li.axis = 3;
        let mut plane0: Vec<u8> = (0..12).collect();
        let mut plane1: Vec<u8> = (100..112).collect();
        let mut planes = [plane0.as_mut_ptr(), plane1.as_mut_ptr()];
        let mut view = ImodView {
            li: &mut li,
            xsize: 4,
            ysize: 3,
            zsize: 2,
            idata: planes.as_mut_ptr(),
            ..Default::default()
        };
        let mut imdata: *mut *mut u8 = ptr::null_mut();
        let mut cache_sum = -1;
        let ret = unsafe { ivw_setup_fast_access(&mut view, &mut imdata, 0, &mut cache_sum, -1) };
        assert_eq!(ret, 0);
        assert_eq!(cache_sum, 0);
        assert_eq!(ivw_fast_get_value(1, 2, 0), 9);
        assert_eq!(ivw_fast_get_value(1, 2, 1), 109);

        // Flipped, unbinned data uses the flipped getter, which indexes idata
        // by Y and steps Z backwards from `xzsizeFac`.
        li.axis = 2;
        let ret = unsafe { ivw_setup_fast_access(&mut view, &mut imdata, 0, &mut cache_sum, -1) };
        assert_eq!(ret, 0);
        assert_eq!(ivw_fast_get_value(1, 0, 1), 1);
        assert_eq!(ivw_fast_get_value(1, 1, 0), 105);
    }

    #[test]
    fn a_missing_host_reports_the_unit_by_name_and_keeps_the_source_return() {
        uninstall();
        let mut view = ImodView::default();
        // `icfGetAutofill` with no cache-fill dialog is the source's 0.
        assert_eq!(with_boundary(|n| n.icf_get_autofill()), 0);
        // `imodCacheFill` with no filler reports and returns the success code
        // the source's own filler returns when there is nothing to do.
        assert_eq!(with_boundary(|n| n.imod_cache_fill(&mut view)), 0);
        assert!(with_boundary(|n| n.app_cvi()).is_null());
    }

    #[test]
    fn valid_scale_matches_the_native_verdict_for_a_real_mrc() {
        // Values read from an MRC written by the reference `raw2mrc`; native
        // `3dmod -Dr` on that file prints
        //   mmvalid 1  meanvalid 0  rmsvalid 0
        //   return min/max 0.017216  199.971176
        let mut image = ImodImageFile::default();
        image.file = IIFILE_MRC;
        image.type_ = IITYPE_FLOAT;
        image.amin = 0.017216;
        image.amax = 199.971176;
        image.amean = 100.451;
        image.rms = 0.;
        image.mrc_header = Some(MrcHeader::default());
        let mut min = 0.0f32;
        let mut max = 0.0f32;
        let ret = unsafe { get_valid_scale(&mut image, Some(&mut min), Some(&mut max)) };
        assert_eq!(ret, 1);
        assert_eq!(min, 0.017216);
        assert_eq!(max, 199.971176);
        assert_eq!(
            format!("return min/max {min:.6}  {max:.6}"),
            "return min/max 0.017216  199.971176"
        );
    }

    #[test]
    fn valid_scale_keeps_the_source_s_or_in_the_rms_flag_test() {
        // `imodview.cpp:3262` writes `hdata->imodFlags | MRC_FLAGS_BAD_RMS_NEG`,
        // an OR where a mask test would use AND, so the flag word never
        // matters and the RMS is valid whenever the IMOD stamp is present.
        let mut header = MrcHeader::default();
        header.imod_stamp = IMOD_MRC_STAMP;
        header.imod_flags = 0;
        let mut image = ImodImageFile::default();
        image.file = IIFILE_MRC;
        image.type_ = IITYPE_FLOAT;
        image.amin = 0.;
        image.amax = 10.;
        image.amean = 5.;
        image.rms = 0.;
        image.mrc_header = Some(header);
        // With a valid min/max the preference decides; the default preference
        // is min/max, so the return is still 1, but mean/SD is now available.
        assert_eq!(unsafe { get_valid_scale(&mut image, None, None) }, 1);
        image.amin = 0.;
        image.amax = 0.;
        image.amean = 5.;
        assert_eq!(unsafe { get_valid_scale(&mut image, None, None) }, 2);
    }

    #[test]
    fn clean_path_matches_qt_normalisation() {
        let mut boundary = ImodviewReportingBoundary;
        assert_eq!(boundary.qdir_clean_path("/a/./b/../c//d"), "/a/c/d");
        assert_eq!(boundary.qdir_clean_path("a/b/../.."), ".");
        assert_eq!(boundary.qdir_clean_path("../x"), "../x");
        assert_eq!(boundary.qdir_to_native_separators("/a/b"), "/a/b");
    }

    #[test]
    fn movie_model_mode_reads_the_model_mouse_mode() {
        let mut model = Imod::default();
        let view = ImodView {
            imod: &mut model,
            ..Default::default()
        };
        assert_eq!(ivw_get_movie_model_mode(Some(&view)), 0);
        model.mousemode = IMOD_MMODEL;
        assert_eq!(ivw_get_movie_model_mode(Some(&view)), 1);
        assert_eq!(ivw_get_movie_model_mode(None), 0);
    }

    #[test]
    fn stipple_and_plug_tracking_track_the_source_counters() {
        let calls = install(RecordingBoundary::default());
        let mut view = ImodView::default();
        ivw_enable_stipple(&mut view, 1);
        assert_eq!(view.draw_stipple, 1);
        ivw_track_mouse_for_plugs(&mut view, 1);
        ivw_track_mouse_for_plugs(&mut view, 1);
        assert_eq!(view.track_mouse_for_plugs, 2);
        ivw_track_mouse_for_plugs(&mut view, 0);
        ivw_track_mouse_for_plugs(&mut view, 0);
        ivw_track_mouse_for_plugs(&mut view, 0);
        assert_eq!(view.track_mouse_for_plugs, 0);
        uninstall();
        let _ = calls;
    }
}
