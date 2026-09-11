//! Translation of `IMOD/libiimod/iilikemrc.c`.
//!
//! Each C definition is retained as one systematic snake-case Rust function.
#![allow(
    non_snake_case,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    unused_variables
)]
use crate::imod::libcfshr::b3dutil::{
    b3d_error, b3d_fread as b3dFread, b3d_fseek as b3dFseek, b3d_rewind as b3dRewind,
    wall_time as wallTime,
};
use crate::imod::libcfshr::ilist::{
    Ilist, ilist_delete as ilistDelete, ilist_insert as ilistInsert, ilist_item as ilistItem,
    ilist_new as ilistNew, ilist_size as ilistSize,
};
pub use crate::imod::libiimod::iimage::ImodImageFile;
use crate::imod::libiimod::iimage::{IiRawCheckFunction, RawImageInfo, ii_sync_from_mrc_header};
use crate::imod::libiimod::iimrc::{ii_mrc_mode_to_format_type, ii_mrc_set_io_funcs};
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_head_new, mrc_set_scale, mrc_swap_longs, mrc_swap_shorts,
};
use core::ffi::CStr;
#[repr(C)]
pub struct _IO_wide_data {
    _private: [u8; 0],
}
#[repr(C)]
pub struct _IO_codecvt {
    _private: [u8; 0],
}
#[repr(C)]
pub struct _IO_marker {
    _private: [u8; 0],
}
unsafe extern "C" {
    static mut stdout: *mut FILE;
    static mut stderr: *mut FILE;
    fn fflush(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn malloc(__size: size_t) -> *mut ::core::ffi::c_void;
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn getenv(__name: *const ::core::ffi::c_char) -> *mut ::core::ffi::c_char;
    fn memcpy(
        __dest: *mut ::core::ffi::c_void,
        __src: *const ::core::ffi::c_void,
        __n: size_t,
    ) -> *mut ::core::ffi::c_void;
    fn strcmp(
        __s1: *const ::core::ffi::c_char,
        __s2: *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    fn strncmp(
        __s1: *const ::core::ffi::c_char,
        __s2: *const ::core::ffi::c_char,
        __n: size_t,
    ) -> ::core::ffi::c_int;
    fn strdup(__s: *const ::core::ffi::c_char) -> *mut ::core::ffi::c_char;
    fn strstr(
        __haystack: *const ::core::ffi::c_char,
        __needle: *const ::core::ffi::c_char,
    ) -> *mut ::core::ffi::c_char;
    fn strlen(__s: *const ::core::ffi::c_char) -> size_t;
    fn stat(__file: *const ::core::ffi::c_char, __buf: *mut stat) -> ::core::ffi::c_int;
}
pub type size_t = usize;
pub type __uint16_t = u16;
pub type __uint32_t = u32;
pub type __uint64_t = u64;
pub type __dev_t = ::core::ffi::c_ulong;
pub type __uid_t = ::core::ffi::c_uint;
pub type __gid_t = ::core::ffi::c_uint;
pub type __ino_t = ::core::ffi::c_ulong;
pub type __mode_t = ::core::ffi::c_uint;
pub type __nlink_t = ::core::ffi::c_ulong;
pub type __off_t = ::core::ffi::c_long;
pub type __off64_t = ::core::ffi::c_long;
pub type __time_t = ::core::ffi::c_long;
pub type __blksize_t = ::core::ffi::c_long;
pub type __blkcnt_t = ::core::ffi::c_long;
pub type __syscall_slong_t = ::core::ffi::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: ::core::ffi::c_int,
    pub _IO_read_ptr: *mut ::core::ffi::c_char,
    pub _IO_read_end: *mut ::core::ffi::c_char,
    pub _IO_read_base: *mut ::core::ffi::c_char,
    pub _IO_write_base: *mut ::core::ffi::c_char,
    pub _IO_write_ptr: *mut ::core::ffi::c_char,
    pub _IO_write_end: *mut ::core::ffi::c_char,
    pub _IO_buf_base: *mut ::core::ffi::c_char,
    pub _IO_buf_end: *mut ::core::ffi::c_char,
    pub _IO_save_base: *mut ::core::ffi::c_char,
    pub _IO_backup_base: *mut ::core::ffi::c_char,
    pub _IO_save_end: *mut ::core::ffi::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1],
    pub _lock: *mut ::core::ffi::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: size_t,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = libc::FILE;
pub type off_t = __off_t;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct timespec {
    pub tv_sec: __time_t,
    pub tv_nsec: __syscall_slong_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct stat {
    pub st_dev: __dev_t,
    pub st_ino: __ino_t,
    pub st_nlink: __nlink_t,
    pub st_mode: __mode_t,
    pub st_uid: __uid_t,
    pub st_gid: __gid_t,
    pub __pad0: ::core::ffi::c_int,
    pub st_rdev: __dev_t,
    pub st_size: __off_t,
    pub st_blksize: __blksize_t,
    pub st_blocks: __blkcnt_t,
    pub st_atim: timespec,
    pub st_mtim: timespec,
    pub st_ctim: timespec,
    pub __glibc_reserved: [__syscall_slong_t; 3],
}
pub type b3dByte = ::core::ffi::c_char;
pub type b3dUByte = ::core::ffi::c_uchar;
pub type b3dInt16 = ::core::ffi::c_short;
pub type b3dUInt16 = ::core::ffi::c_ushort;
pub type b3dInt32 = ::core::ffi::c_int;
pub type b3dFloat = ::core::ffi::c_float;
#[repr(C)]
struct CheckEntry {
    func: IiRawCheckFunction,
    name: *mut ::core::ffi::c_char,
}
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
pub const SEEK_SET: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
pub const SEEK_END: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const MRC_MODE_BYTE: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
pub const MRC_MODE_SHORT: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
pub const MRC_MODE_FLOAT: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const MRC_MODE_COMPLEX_FLOAT: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
pub const MRC_MODE_USHORT: ::core::ffi::c_int = 6 as ::core::ffi::c_int;
pub const MRC_MODE_RGB: ::core::ffi::c_int = 16 as ::core::ffi::c_int;
pub const IIFILE_RAW: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
pub const IIERR_BAD_CALL: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
pub const IIERR_NOT_FORMAT: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
pub const IIERR_IO_ERROR: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const IIERR_MEMORY_ERR: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
pub const IIERR_NO_SUPPORT: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
pub const RAW_MODE_SBYTE: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
pub const RAW_MODE_BYTE: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
pub const RAW_MODE_SHORT: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const RAW_MODE_USHORT: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
pub const RAW_MODE_FLOAT: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
pub const MAX_EM_MACHINES: ::core::ffi::c_int = 20 as ::core::ffi::c_int;
pub const MAX_EM_TYPES: ::core::ffi::c_int = 20 as ::core::ffi::c_int;
pub const MAX_EM_SIZE: ::core::ffi::c_double = 1.6e10f64;
static mut checkList: *mut Ilist = ::core::ptr::null::<Ilist>() as *mut Ilist;
unsafe extern "C" fn init_check_list() -> ::core::ffi::c_int {
    if !checkList.is_null() {
        return 0 as ::core::ffi::c_int;
    }
    checkList = ilistNew(
        ::core::mem::size_of::<CheckEntry>() as ::core::ffi::c_int,
        6 as ::core::ffi::c_int,
    );
    if checkList.is_null() {
        return 1 as ::core::ffi::c_int;
    }
    ii_add_raw_check_function(
        Some(
            check_em
                as unsafe extern "C" fn(
                    *mut FILE,
                    *mut ::core::ffi::c_char,
                    *mut RawImageInfo,
                ) -> ::core::ffi::c_int,
        ),
        b"EM\0" as *const u8 as *const ::core::ffi::c_char,
    );
    ii_add_raw_check_function(
        Some(
            check_dm3
                as unsafe extern "C" fn(
                    *mut FILE,
                    *mut ::core::ffi::c_char,
                    *mut RawImageInfo,
                ) -> ::core::ffi::c_int,
        ),
        b"DM3\0" as *const u8 as *const ::core::ffi::c_char,
    );
    ii_add_raw_check_function(
        Some(
            check_fei_raw
                as unsafe extern "C" fn(
                    *mut FILE,
                    *mut ::core::ffi::c_char,
                    *mut RawImageInfo,
                ) -> ::core::ffi::c_int,
        ),
        b"FEIraw\0" as *const u8 as *const ::core::ffi::c_char,
    );
    ii_add_raw_check_function(
        Some(
            check_winkler
                as unsafe extern "C" fn(
                    *mut FILE,
                    *mut ::core::ffi::c_char,
                    *mut RawImageInfo,
                ) -> ::core::ffi::c_int,
        ),
        b"Winkler\0" as *const u8 as *const ::core::ffi::c_char,
    );
    ii_add_raw_check_function(
        Some(
            check_pif
                as unsafe extern "C" fn(
                    *mut FILE,
                    *mut ::core::ffi::c_char,
                    *mut RawImageInfo,
                ) -> ::core::ffi::c_int,
        ),
        b"PIF\0" as *const u8 as *const ::core::ffi::c_char,
    );
    return 0 as ::core::ffi::c_int;
}
pub unsafe extern "C" fn ii_add_raw_check_function(
    mut func: IiRawCheckFunction,
    mut name: *const ::core::ffi::c_char,
) {
    let mut item: CheckEntry = CheckEntry {
        func: None,
        name: ::core::ptr::null_mut::<::core::ffi::c_char>(),
    };
    item.func = func;
    item.name = strdup(name);
    if init_check_list() != 0 || item.name.is_null() {
        return;
    }
    ilistInsert(
        checkList,
        &raw mut item as *mut ::core::ffi::c_void,
        0 as ::core::ffi::c_int,
    );
}
pub unsafe extern "C" fn ii_delete_raw_check_list() {
    let mut i: ::core::ffi::c_int = 0;
    let mut item: *mut CheckEntry = ::core::ptr::null_mut::<CheckEntry>();
    if checkList.is_null() {
        return;
    }
    i = 0 as ::core::ffi::c_int;
    while i < ilistSize(checkList) {
        item = ilistItem(checkList, i) as *mut CheckEntry;
        if !(*item).name.is_null() {
            free((*item).name as *mut ::core::ffi::c_void);
        }
        i += 1;
    }
    ilistDelete(checkList);
    checkList = ::core::ptr::null_mut::<Ilist>();
}
pub unsafe extern "C" fn ii_like_mrc_check(mut inFile: *mut ImodImageFile) -> ::core::ffi::c_int {
    let mut fp: *mut FILE = ::core::ptr::null_mut::<FILE>();
    let mut i: ::core::ffi::c_int = 0;
    let mut info: RawImageInfo = RawImageInfo {
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
    let mut err: ::core::ffi::c_int = 0;
    let mut item: *mut CheckEntry = ::core::ptr::null_mut::<CheckEntry>();
    info.swap_bytes = 0 as ::core::ffi::c_int;
    info.section_skip = 0 as ::core::ffi::c_int;
    info.y_inverted = 0 as ::core::ffi::c_int;
    info.pixel = 0.0f32;
    info.z_pixel = 0.0f32;
    if inFile.is_null() {
        return IIERR_BAD_CALL;
    }
    fp = (*inFile).fp;
    if fp.is_null() {
        return IIERR_BAD_CALL;
    }
    if init_check_list() != 0 {
        return IIERR_BAD_CALL;
    }
    i = 0 as ::core::ffi::c_int;
    while i < ilistSize(checkList) {
        item = ilistItem(checkList, i) as *mut CheckEntry;
        err = Some((*item).func.expect("non-null function pointer"))
            .expect("non-null function pointer")(
            fp, (*inFile).filename, &raw mut info
        );
        if err == 0 {
            return ii_setup_raw_headers(inFile, &raw mut info);
        }
        if err != IIERR_NOT_FORMAT {
            if err == IIERR_IO_ERROR {
                b3d_error(
                    stderr,
                    format_args!(
                        "ERROR: iiCheckLikeMRC - reading from file {}\n",
                        CStr::from_ptr((*inFile).filename).to_string_lossy()
                    ),
                );
            } else if err == IIERR_NO_SUPPORT {
                b3d_error(
                    stderr,
                    format_args!(
                        "ERROR: iiCheckLikeMRC - unsupported data mode of {}-type file.\n",
                        CStr::from_ptr((*item).name).to_string_lossy()
                    ),
                );
            }
            return err;
        }
        i += 1;
    }
    return IIERR_NOT_FORMAT;
}
pub unsafe extern "C" fn ii_setup_raw_headers(
    mut inFile: *mut ImodImageFile,
    mut info: *mut RawImageInfo,
) -> ::core::ffi::c_int {
    let mut modeTable: [::core::ffi::c_int; 7] = [
        MRC_MODE_BYTE,
        MRC_MODE_BYTE,
        MRC_MODE_SHORT,
        MRC_MODE_USHORT,
        MRC_MODE_FLOAT,
        MRC_MODE_COMPLEX_FLOAT,
        MRC_MODE_RGB,
    ];
    let mut hdr: *mut MrcHeader = ::core::ptr::null_mut::<MrcHeader>();
    hdr = malloc(::core::mem::size_of::<MrcHeader>() as size_t) as *mut MrcHeader;
    if hdr.is_null() {
        b3d_error(
            stderr,
            format_args!("ERROR: iiSetupRawHeaders - Getting memory for header"),
        );
        return IIERR_MEMORY_ERR;
    }
    mrc_head_new(
        &mut *hdr.cast::<MrcHeader>(),
        (*info).nx,
        (*info).ny,
        (*info).nz,
        modeTable[(*info).type_ as usize],
    );
    (*inFile).file = IIFILE_RAW;
    (*hdr).swapped = (*info).swap_bytes;
    (*hdr).header_size = (*info).header_size;
    (*hdr).section_skip = (*info).section_skip;
    (*hdr).y_inverted = (*info).y_inverted;
    (*hdr).bytes_signed = if (*info).type_ == RAW_MODE_SBYTE {
        1 as ::core::ffi::c_int
    } else {
        0 as ::core::ffi::c_int
    };
    (*hdr).packed4bits = 0 as ::core::ffi::c_int;
    (*hdr).half_floats = 0 as ::core::ffi::c_int;
    (*hdr).fp = (*inFile).fp.cast();
    (*hdr).amin = (*info).amin as b3dFloat;
    (*hdr).amax = (*info).amax as b3dFloat;
    (*hdr).amean = (((*info).amin + (*info).amax) as ::core::ffi::c_double / 2.0f64) as b3dFloat;
    if (*info).pixel != 0. {
        mrc_set_scale(
            &mut *hdr.cast::<MrcHeader>(),
            (*info).pixel as ::core::ffi::c_double,
            (*info).pixel as ::core::ffi::c_double,
            (if (*info).z_pixel != 0. {
                (*info).z_pixel
            } else {
                (*info).pixel
            }) as ::core::ffi::c_double,
        );
    }
    (*inFile).header = hdr as *mut ::core::ffi::c_char;
    ii_mrc_mode_to_format_type(
        inFile.cast::<ImodImageFile>(),
        (*hdr).mode as ::core::ffi::c_int,
        (*hdr).bytes_signed,
    );
    ii_sync_from_mrc_header(inFile.cast::<ImodImageFile>(), hdr.cast::<MrcHeader>());
    ii_mrc_set_io_funcs(inFile.cast::<ImodImageFile>(), 1);
    (*inFile).clean_up = Some(ii_like_mrc_delete as unsafe extern "C" fn(*mut ImodImageFile) -> ())
        as Option<unsafe extern "C" fn(*mut ImodImageFile) -> ()>;
    return 0 as ::core::ffi::c_int;
}
pub unsafe extern "C" fn ii_like_mrc_delete(mut inFile: *mut ImodImageFile) {
    if !(*inFile).header.is_null() {
        free((*inFile).header as *mut ::core::ffi::c_void);
    }
}
unsafe extern "C" fn check_winkler(
    mut fp: *mut FILE,
    mut filename: *mut ::core::ffi::c_char,
    mut info: *mut RawImageInfo,
) -> ::core::ffi::c_int {
    let mut svals: [b3dUInt16; 4] = [0; 4];
    let mut ivals: [b3dInt32; 12] = [0; 12];
    b3dRewind(fp);
    if b3dFread(
        &raw mut svals as *mut b3dUInt16 as *mut ::core::ffi::c_void,
        2 as size_t,
        2 as size_t,
        fp,
    ) != 2 as size_t
    {
        return IIERR_IO_ERROR;
    }
    (*info).swap_bytes = 0 as ::core::ffi::c_int;
    if svals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int != 18739 as ::core::ffi::c_int
        || svals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_int
            != 20480 as ::core::ffi::c_int
    {
        mrc_swap_shorts(
            core::slice::from_raw_parts_mut(svals.as_mut_ptr().cast::<i16>(), 2),
            2,
        );
        if svals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int
            != 18739 as ::core::ffi::c_int
            || svals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_int
                != 20480 as ::core::ffi::c_int
        {
            return IIERR_NOT_FORMAT;
        }
        (*info).swap_bytes = 1 as ::core::ffi::c_int;
    }
    if b3dFseek(fp, 16 as ::core::ffi::c_int, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut svals as *mut b3dUInt16 as *mut ::core::ffi::c_void,
        2 as size_t,
        4 as size_t,
        fp,
    ) != 4 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if (*info).swap_bytes != 0 {
        mrc_swap_shorts(
            core::slice::from_raw_parts_mut(svals.as_mut_ptr().cast::<i16>(), 4),
            4,
        );
    }
    if svals[0 as ::core::ffi::c_int as usize] != 0 {
        return IIERR_NO_SUPPORT;
    }
    match svals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_int {
        2 => {
            (*info).type_ = RAW_MODE_BYTE;
        }
        3 => {
            (*info).type_ = RAW_MODE_SHORT;
        }
        15 => {
            (*info).type_ = RAW_MODE_USHORT;
        }
        5 => {
            (*info).type_ = RAW_MODE_FLOAT;
        }
        _ => return IIERR_NO_SUPPORT,
    }
    if svals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int / 2 as ::core::ffi::c_int
        != 1 as ::core::ffi::c_int
    {
        return IIERR_NO_SUPPORT;
    }
    if b3dFseek(fp, 24 as ::core::ffi::c_int, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut ivals as *mut b3dInt32 as *mut ::core::ffi::c_void,
        4 as size_t,
        1 as size_t,
        fp,
    ) != 1 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if (*info).swap_bytes != 0 {
        mrc_swap_longs(core::slice::from_raw_parts_mut(ivals.as_mut_ptr(), 1), 1);
    }
    (*info).header_size = ivals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    if b3dFseek(fp, 64 as ::core::ffi::c_int, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut ivals as *mut b3dInt32 as *mut ::core::ffi::c_void,
        4 as size_t,
        (svals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int * 4 as ::core::ffi::c_int)
            as size_t,
        fp,
    ) != (svals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int
        * 4 as ::core::ffi::c_int) as size_t
    {
        return IIERR_IO_ERROR;
    }
    if (*info).swap_bytes != 0 {
        mrc_swap_longs(core::slice::from_raw_parts_mut(ivals.as_mut_ptr(), 12), 12);
    }
    if svals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int == 3 as ::core::ffi::c_int {
        (*info).nx = ivals[10 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
        (*info).ny = ivals[6 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
        (*info).nz = ivals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    } else {
        (*info).nx = ivals[6 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
        (*info).ny = ivals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
        (*info).nz = 1 as ::core::ffi::c_int;
    }
    (*info).amin = 0.0f32;
    (*info).amax = 0.0f32;
    return 0 as ::core::ffi::c_int;
}
unsafe extern "C" fn check_pif(
    mut fp: *mut FILE,
    mut filename: *mut ::core::ffi::c_char,
    mut info: *mut RawImageInfo,
) -> ::core::ffi::c_int {
    let mut ivals: [b3dInt32; 12] = [0; 12];
    let mut cvals: [b3dByte; 6] = [0; 6];
    if b3dFseek(fp, 32 as ::core::ffi::c_int, SEEK_SET) != 0 as ::core::ffi::c_int {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut cvals as *mut b3dByte as *mut ::core::ffi::c_void,
        1 as size_t,
        5 as size_t,
        fp,
    ) != 5 as size_t
    {
        return IIERR_IO_ERROR;
    }
    cvals[5 as ::core::ffi::c_int as usize] = 0 as b3dByte;
    if strcmp(
        &raw mut cvals as *mut b3dByte,
        b"Bsoft\0" as *const u8 as *const ::core::ffi::c_char,
    ) != 0 as ::core::ffi::c_int
    {
        return IIERR_NOT_FORMAT;
    }
    (*info).header_size = 1024 as ::core::ffi::c_int;
    (*info).section_skip = 512 as ::core::ffi::c_int;
    if b3dFseek(fp, 28 as ::core::ffi::c_int, SEEK_SET) != 0 as ::core::ffi::c_int {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut ivals as *mut b3dInt32 as *mut ::core::ffi::c_void,
        4 as size_t,
        1 as size_t,
        fp,
    ) != 1 as size_t
    {
        return IIERR_IO_ERROR;
    }
    (*info).swap_bytes = 0 as ::core::ffi::c_int;
    #[cfg(target_endian = "little")]
    if ivals[0 as ::core::ffi::c_int as usize] != 0 as ::core::ffi::c_int {
        (*info).swap_bytes = 1 as ::core::ffi::c_int;
    }
    #[cfg(target_endian = "big")]
    if ivals[0 as ::core::ffi::c_int as usize] == 0 as ::core::ffi::c_int {
        (*info).swap_bytes = 1 as ::core::ffi::c_int;
    }
    if b3dFseek(fp, 24 as ::core::ffi::c_int, SEEK_SET) != 0 as ::core::ffi::c_int {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut ivals as *mut b3dInt32 as *mut ::core::ffi::c_void,
        4 as size_t,
        1 as size_t,
        fp,
    ) != 1 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if (*info).swap_bytes != 0 {
        mrc_swap_longs(core::slice::from_raw_parts_mut(ivals.as_mut_ptr(), 1), 1);
    }
    (*info).nz = ivals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    if b3dFseek(fp, 64 as ::core::ffi::c_int, SEEK_SET) != 0 as ::core::ffi::c_int {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut ivals as *mut b3dInt32 as *mut ::core::ffi::c_void,
        4 as size_t,
        5 as size_t,
        fp,
    ) != 5 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if (*info).swap_bytes != 0 {
        mrc_swap_longs(core::slice::from_raw_parts_mut(ivals.as_mut_ptr(), 5), 5);
    }
    if ivals[0 as ::core::ffi::c_int as usize] < 1 as ::core::ffi::c_int
        && (*info).nz > 1 as ::core::ffi::c_int
    {
        return IIERR_NOT_FORMAT;
    }
    (*info).nx = ivals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    (*info).ny = ivals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    match ivals[4 as ::core::ffi::c_int as usize] {
        0 | 6 => {
            (*info).type_ = RAW_MODE_BYTE;
        }
        1 | 7 | 20 | 88 => {
            (*info).type_ = RAW_MODE_SHORT;
        }
        9 => {
            (*info).type_ = RAW_MODE_FLOAT;
        }
        _ => return IIERR_NO_SUPPORT,
    }
    (*info).amin = 0.0f32;
    (*info).amax = 0.0f32;
    return 0 as ::core::ffi::c_int;
}
static mut sAssumeDMmatch: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sLastDMinfo: RawImageInfo = RawImageInfo {
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
static mut sDMinfoSaved: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
pub unsafe extern "C" fn ii_assume_dmfile_matches(mut inVal: ::core::ffi::c_int) {
    sAssumeDMmatch = inVal;
}
pub const DOC_CHECK_BUF: ::core::ffi::c_int = 832 as ::core::ffi::c_int;
unsafe extern "C" fn check_dm3(
    mut fp: *mut FILE,
    mut filename: *mut ::core::ffi::c_char,
    mut info: *mut RawImageInfo,
) -> ::core::ffi::c_int {
    let mut bvals: [::core::ffi::c_uchar; 832] = [0; 832];
    let mut err: ::core::ffi::c_int = 0;
    let mut dmtype: ::core::ffi::c_int = 0;
    let mut dmf: ::core::ffi::c_int = 0;
    let mut i: ::core::ffi::c_int = 0;
    let mut bsave: ::core::ffi::c_uchar = 0;
    let mut testString: *mut ::core::ffi::c_char = b"DocumentObjectList\0" as *const u8
        as *const ::core::ffi::c_char
        as *mut ::core::ffi::c_char;
    let mut testLen: ::core::ffi::c_int = strlen(testString) as ::core::ffi::c_int;
    b3dRewind(fp);
    if b3dFread(
        &raw mut bvals as *mut ::core::ffi::c_uchar as *mut ::core::ffi::c_void,
        1 as size_t,
        4 as size_t,
        fp,
    ) != 4 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if bvals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int != 3 as ::core::ffi::c_int
        && bvals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int != 4 as ::core::ffi::c_int
    {
        return IIERR_NOT_FORMAT;
    }
    dmf = bvals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    if b3dFread(
        &raw mut bvals as *mut ::core::ffi::c_uchar as *mut ::core::ffi::c_void,
        1 as size_t,
        DOC_CHECK_BUF as size_t,
        fp,
    ) != DOC_CHECK_BUF as size_t
    {
        return IIERR_IO_ERROR;
    }
    err = 1 as ::core::ffi::c_int;
    i = 0 as ::core::ffi::c_int;
    while i < DOC_CHECK_BUF - testLen - 4 as ::core::ffi::c_int {
        if bvals[i as usize] as ::core::ffi::c_int
            == *testString.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int
        {
            bsave = bvals[(i + testLen) as usize];
            bvals[(i + testLen) as usize] = 0 as ::core::ffi::c_uchar;
            if strcmp(
                (&raw mut bvals as *mut ::core::ffi::c_uchar).offset(i as isize)
                    as *mut ::core::ffi::c_uchar as *mut ::core::ffi::c_char,
                testString,
            ) == 0
            {
                err = 0 as ::core::ffi::c_int;
                break;
            } else {
                bvals[(i + testLen) as usize] = bsave;
            }
        }
        i += 1;
    }
    if err != 0 {
        return IIERR_NOT_FORMAT;
    }
    if sAssumeDMmatch != 0 && sDMinfoSaved != 0 {
        memcpy(
            info as *mut ::core::ffi::c_void,
            &raw mut sLastDMinfo as *const ::core::ffi::c_void,
            ::core::mem::size_of::<RawImageInfo>() as size_t,
        );
        return 0 as ::core::ffi::c_int;
    }
    err = analyze_dm3(fp, filename, dmf, info, &raw mut dmtype);
    if err != 0 {
        return err;
    }
    match dmtype {
        9 => {
            (*info).type_ = RAW_MODE_SBYTE;
        }
        6 => {
            (*info).type_ = RAW_MODE_BYTE;
        }
        1 => {
            (*info).type_ = RAW_MODE_SHORT;
        }
        10 => {
            (*info).type_ = RAW_MODE_USHORT;
        }
        2 => {
            (*info).type_ = RAW_MODE_FLOAT;
        }
        _ => return IIERR_NO_SUPPORT,
    }
    (*info).amin = 0.0f32;
    (*info).amax = 0.0f32;
    memcpy(
        &raw mut sLastDMinfo as *mut ::core::ffi::c_void,
        info as *const ::core::ffi::c_void,
        ::core::mem::size_of::<RawImageInfo>() as size_t,
    );
    sDMinfoSaved = 1 as ::core::ffi::c_int;
    return 0 as ::core::ffi::c_int;
}
pub const MAX_TYPES: ::core::ffi::c_int = 13 as ::core::ffi::c_int;
pub unsafe extern "C" fn analyze_dm3(
    mut fp: *mut FILE,
    mut filename: *mut ::core::ffi::c_char,
    mut dmformat: ::core::ffi::c_int,
    mut info: *mut RawImageInfo,
    mut dmtype: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut c: ::core::ffi::c_int = 0;
    let mut toffset: ::core::ffi::c_int = 0;
    let mut typeIndex: ::core::ffi::c_int = 0;
    let mut dmind: ::core::ffi::c_int = 0;
    let mut buf: [::core::ffi::c_char; 1000000] = [0; 1000000];
    let mut found: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    let mut lowbyte: ::core::ffi::c_int = 0;
    let mut hibyte: ::core::ffi::c_int = 0;
    let mut loop_0: ::core::ffi::c_int = 0;
    let mut maxUseC: ::core::ffi::c_int = 0;
    let mut matchLast: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
    let mut offset: ::core::ffi::c_int = 0;
    let mut type_: ::core::ffi::c_int = 0;
    let mut xsize: ::core::ffi::c_int = 0;
    let mut ysize: ::core::ffi::c_int = 0;
    let mut zsize: ::core::ffi::c_int = 0;
    let mut gotCal: ::core::ffi::c_int = 0;
    let mut gotDim: ::core::ffi::c_int = 0;
    let mut gotScale: ::core::ffi::c_int = 0;
    let mut gotMeta: ::core::ffi::c_int = 0;
    let mut gotDimInfo: ::core::ffi::c_int = 0;
    let mut scale: ::core::ffi::c_float = 0.;
    let mut tmpPixel: ::core::ffi::c_float = 0.;
    let mut pixel: ::core::ffi::c_float = 0.0f32;
    let mut z_pixel: ::core::ffi::c_float = 0.0f32;
    let mut typeOffset: off_t = 0;
    let mut maxread: off_t = 0;
    let mut plausibleOff: off_t = 0;
    static mut lastMaxRead: off_t = 0 as off_t;
    let mut statbuf: stat = stat {
        st_dev: 0,
        st_ino: 0,
        st_nlink: 0,
        st_mode: 0,
        st_uid: 0,
        st_gid: 0,
        __pad0: 0,
        st_rdev: 0,
        st_size: 0,
        st_blksize: 0,
        st_blocks: 0,
        st_atim: timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        st_mtim: timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        st_ctim: timespec {
            tv_sec: 0,
            tv_nsec: 0,
        },
        __glibc_reserved: [0; 3],
    };
    let mut wallStart: ::core::ffi::c_double = 0.;
    static mut debug: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
    static mut lastDataType: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastDimensions: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastData: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastCalibrations: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastTabDimens: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastDimensInfo: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastUnits: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastOffset: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastZunits: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastMaxEndUsed: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    static mut lastMaxStartUsed: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
    let mut curDataType: ::core::ffi::c_int = 0;
    let mut curDimensions: ::core::ffi::c_int = 0;
    let mut curData: ::core::ffi::c_int = 0;
    let mut curCalibrations: ::core::ffi::c_int = 0;
    let mut curTabDimens: ::core::ffi::c_int = 0;
    let mut curDimensInfo: ::core::ffi::c_int = 0;
    let mut curUnits: ::core::ffi::c_int = 0;
    let mut curZunits: ::core::ffi::c_int = 0;
    let mut curOffset: ::core::ffi::c_int = 0;
    let mut maxCurUsed: ::core::ffi::c_int = 0;
    let mut dataSize: [::core::ffi::c_int; 13] = [
        1 as ::core::ffi::c_int,
        2 as ::core::ffi::c_int,
        4 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
        4 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
        2 as ::core::ffi::c_int,
        4 as ::core::ffi::c_int,
        8 as ::core::ffi::c_int,
    ];
    let mut dimensOff: [::core::ffi::c_int; 2] =
        [15 as ::core::ffi::c_int, 27 as ::core::ffi::c_int];
    let mut xsizeOff: [::core::ffi::c_int; 2] =
        [31 as ::core::ffi::c_int, 59 as ::core::ffi::c_int];
    let mut ysizeOff: [::core::ffi::c_int; 2] =
        [50 as ::core::ffi::c_int, 94 as ::core::ffi::c_int];
    let mut zsizeOff: [::core::ffi::c_int; 2] =
        [69 as ::core::ffi::c_int, 129 as ::core::ffi::c_int];
    let mut dtypeOff: [::core::ffi::c_int; 2] =
        [20 as ::core::ffi::c_int, 36 as ::core::ffi::c_int];
    let mut dataOff: [::core::ffi::c_int; 2] = [24 as ::core::ffi::c_int, 48 as ::core::ffi::c_int];
    let mut scaleOff: [::core::ffi::c_int; 2] =
        [17 as ::core::ffi::c_int, 33 as ::core::ffi::c_int];
    let mut unitsOff: [::core::ffi::c_int; 2] =
        [25 as ::core::ffi::c_int, 49 as ::core::ffi::c_int];
    let mut maxOffset: [::core::ffi::c_int; 2] =
        [90 as ::core::ffi::c_int, 150 as ::core::ffi::c_int];
    if debug < 0 as ::core::ffi::c_int {
        debug = if !getenv(b"ANALYZEDM3_DEBUG\0" as *const u8 as *const ::core::ffi::c_char)
            .is_null()
        {
            1 as ::core::ffi::c_int
        } else {
            0 as ::core::ffi::c_int
        };
    }
    if debug != 0 {
        wallStart = wallTime();
    }
    dmind = dmformat - 3 as ::core::ffi::c_int;
    if dmind < 0 as ::core::ffi::c_int || dmind > 1 as ::core::ffi::c_int {
        b3d_error(
            stderr,
            format_args!("ERROR: analyzeDM3 - DM format {} not supported\n", dmformat),
        );
        return IIERR_NO_SUPPORT;
    }
    if stat(filename, &raw mut statbuf) != 0 {
        b3d_error(
            stderr,
            format_args!(
                "ERROR: analyzeDM3 - Doing stat of {}\n",
                CStr::from_ptr(filename).to_string_lossy()
            ),
        );
        return IIERR_IO_ERROR;
    }
    maxread = (if (statbuf.st_size as ::core::ffi::c_long - 2 as ::core::ffi::c_long)
        < (1000000 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as ::core::ffi::c_long
    {
        statbuf.st_size as ::core::ffi::c_long - 2 as ::core::ffi::c_long
    } else {
        (1000000 as ::core::ffi::c_int - 1 as ::core::ffi::c_int) as ::core::ffi::c_long
    }) as off_t;
    if maxread != lastMaxRead || lastMaxEndUsed == 0 {
        matchLast = 0 as ::core::ffi::c_int;
    }
    curCalibrations = (2 as off_t * maxread) as ::core::ffi::c_int;
    curData = curCalibrations;
    curDimensions = curData;
    curDataType = curDimensions;
    curZunits = (2 as off_t * maxread) as ::core::ffi::c_int;
    curUnits = curZunits;
    curDimensInfo = curUnits;
    curTabDimens = curDimensInfo;
    if debug != 0 {
        printf(
            b"analyze_dm3: Reading up to %d bytes\n\0" as *const u8 as *const ::core::ffi::c_char,
            maxread as ::core::ffi::c_int,
        );
    }
    loop_0 = 1 as ::core::ffi::c_int - matchLast;
    while loop_0 < 2 as ::core::ffi::c_int {
        offset = 0 as ::core::ffi::c_int;
        xsize = 0 as ::core::ffi::c_int;
        ysize = 0 as ::core::ffi::c_int;
        type_ = -(1 as ::core::ffi::c_int);
        typeOffset = 0 as off_t;
        typeIndex = -(1 as ::core::ffi::c_int);
        if loop_0 != 0 {
            c = 0 as ::core::ffi::c_int;
            maxUseC = maxread as ::core::ffi::c_int;
        } else {
            c = if lastDimensions < lastDataType {
                lastDimensions
            } else {
                lastDataType
            };
            maxUseC = lastMaxEndUsed;
        }
        if b3dFseek(
            fp,
            -(maxread as ::core::ffi::c_long - c as ::core::ffi::c_long + 1 as ::core::ffi::c_long)
                as ::core::ffi::c_int,
            SEEK_END,
        ) != 0
        {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: analyzeDM3 - Seeking to end of {}\n",
                    CStr::from_ptr(filename).to_string_lossy()
                ),
            );
            return IIERR_IO_ERROR;
        }
        if b3dFread(
            (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                as *mut ::core::ffi::c_char as *mut ::core::ffi::c_void,
            1 as size_t,
            (maxUseC - c) as size_t,
            fp,
        ) == 0
        {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: analyzeDM3 - Error Reading tail end of {}\n",
                    CStr::from_ptr(filename).to_string_lossy()
                ),
            );
            return IIERR_IO_ERROR;
        }
        buf[(maxUseC - c - 1 as ::core::ffi::c_int) as usize] = 0 as ::core::ffi::c_char;
        if debug != 0 {
            printf(
                b"analyze_dm3: file end loop %d, start at %d, read %d\n\0" as *const u8
                    as *const ::core::ffi::c_char,
                loop_0,
                c,
                maxUseC - c,
            );
        }
        while c < maxUseC
            && (xsize == 0 as ::core::ffi::c_int
                || type_ < 0 as ::core::ffi::c_int
                || c < typeIndex
                    + 64 as ::core::ffi::c_int
                    + (if dmind != 0 {
                        20 as ::core::ffi::c_int
                    } else {
                        0 as ::core::ffi::c_int
                    }))
        {
            if buf[c as usize] as ::core::ffi::c_int == 68 as ::core::ffi::c_int {
                found = strstr(
                    (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                        as *mut ::core::ffi::c_char,
                    b"Dimensions\0" as *const u8 as *const ::core::ffi::c_char,
                );
                if !found.is_null()
                    && (c + ysizeOff[dmind as usize] + 1 as ::core::ffi::c_int) < maxUseC
                {
                    if loop_0 == 0 && c != lastDimensions {
                        matchLast = 0 as ::core::ffi::c_int;
                        break;
                    } else {
                        lowbyte = buf[(c + dimensOff[dmind as usize]) as usize]
                            as ::core::ffi::c_uchar
                            as ::core::ffi::c_int;
                        if lowbyte == 3 as ::core::ffi::c_int {
                            if c + zsizeOff[dmind as usize] + 1 as ::core::ffi::c_int >= maxUseC {
                                matchLast = 0 as ::core::ffi::c_int;
                                break;
                            } else {
                                lowbyte = buf[(c + zsizeOff[dmind as usize]) as usize]
                                    as ::core::ffi::c_uchar
                                    as ::core::ffi::c_int;
                                hibyte =
                                    buf[(c + zsizeOff[dmind as usize] + 1 as ::core::ffi::c_int)
                                        as usize]
                                        as ::core::ffi::c_uchar
                                        as ::core::ffi::c_int;
                                zsize = lowbyte + 256 as ::core::ffi::c_int * hibyte;
                            }
                        } else if lowbyte == 2 as ::core::ffi::c_int {
                            zsize = 1 as ::core::ffi::c_int;
                        } else {
                            b3d_error(
                                stderr,
                                format_args!(
                                    "ERROR: analyzeDM3 - The number of dimensions seemsto be {}, not 2 or 3, in {}\n",
                                    lowbyte,
                                    CStr::from_ptr(filename).to_string_lossy()
                                ),
                            );
                            return IIERR_NO_SUPPORT;
                        }
                        lowbyte = buf[(c + xsizeOff[dmind as usize]) as usize]
                            as ::core::ffi::c_uchar
                            as ::core::ffi::c_int;
                        hibyte = buf
                            [(c + xsizeOff[dmind as usize] + 1 as ::core::ffi::c_int) as usize]
                            as ::core::ffi::c_uchar
                            as ::core::ffi::c_int;
                        xsize = lowbyte + 256 as ::core::ffi::c_int * hibyte;
                        lowbyte = buf[(c + ysizeOff[dmind as usize]) as usize]
                            as ::core::ffi::c_uchar
                            as ::core::ffi::c_int;
                        hibyte = buf
                            [(c + ysizeOff[dmind as usize] + 1 as ::core::ffi::c_int) as usize]
                            as ::core::ffi::c_uchar
                            as ::core::ffi::c_int;
                        ysize = lowbyte + 256 as ::core::ffi::c_int * hibyte;
                        curDimensions = c;
                        if debug != 0 {
                            printf(
                                b"analyze_dm3: Found Dimensions at %d  x %d y %d z %d\n\0"
                                    as *const u8
                                    as *const ::core::ffi::c_char,
                                c,
                                xsize,
                                ysize,
                                zsize,
                            );
                        }
                    }
                } else {
                    found = strstr(
                        (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                            as *mut ::core::ffi::c_char,
                        b"DataType\0" as *const u8 as *const ::core::ffi::c_char,
                    );
                    if !found.is_null()
                        && c + dtypeOff[dmind as usize] < maxUseC
                        && (buf[(c + dtypeOff[dmind as usize]) as usize] as ::core::ffi::c_int)
                            < MAX_TYPES
                    {
                        if loop_0 == 0 && c != lastDataType {
                            matchLast = 0 as ::core::ffi::c_int;
                            break;
                        } else {
                            type_ =
                                buf[(c + dtypeOff[dmind as usize]) as usize] as ::core::ffi::c_int;
                            if typeOffset == 0 {
                                typeOffset = (c as ::core::ffi::c_long
                                    + statbuf.st_size as ::core::ffi::c_long
                                    - (maxread as ::core::ffi::c_long + 1 as ::core::ffi::c_long))
                                    as off_t;
                            }
                            typeIndex = c;
                            curDataType = typeIndex;
                            if debug != 0 {
                                printf(
                                    b"analyze_dm3: Found DataType at %d  type %d\n\0" as *const u8
                                        as *const ::core::ffi::c_char,
                                    c,
                                    type_,
                                );
                            }
                        }
                    }
                }
            }
            c += 1;
        }
        if matchLast != 0 && xsize != 0 && ysize != 0 && type_ >= 0 as ::core::ffi::c_int {
            break;
        }
        loop_0 += 1;
    }
    if xsize == 0 || ysize == 0 || type_ < 0 as ::core::ffi::c_int {
        b3d_error(
            stderr,
            format_args!(
                "ERROR: analyzeDM3 - Dimensions or type not found in {}\n",
                CStr::from_ptr(filename).to_string_lossy()
            ),
        );
        return IIERR_NO_SUPPORT;
    }
    lastMaxEndUsed = (if maxread < (c + maxOffset[dmind as usize]) as ::core::ffi::c_long {
        maxread as ::core::ffi::c_long
    } else {
        (c + maxOffset[dmind as usize]) as ::core::ffi::c_long
    }) as ::core::ffi::c_int;
    plausibleOff = (typeOffset as ::core::ffi::c_long
        - (xsize * ysize) as ::core::ffi::c_long
            * zsize as ::core::ffi::c_long
            * dataSize[type_ as usize] as ::core::ffi::c_long) as off_t;
    loop_0 = 1 as ::core::ffi::c_int - matchLast;
    while loop_0 < 2 as ::core::ffi::c_int {
        gotDimInfo = 0 as ::core::ffi::c_int;
        gotMeta = gotDimInfo;
        gotScale = gotMeta;
        gotDim = gotScale;
        gotCal = gotDim;
        z_pixel = 0.0f32;
        pixel = z_pixel;
        maxCurUsed = maxread as ::core::ffi::c_int;
        if loop_0 != 0 {
            c = 0 as ::core::ffi::c_int;
            maxUseC = maxread as ::core::ffi::c_int;
        } else {
            c = crate::imod::libcfshr::b3dutil::b3d_i_min(&[
                lastDataType,
                lastDimensions,
                lastData,
                lastCalibrations,
                lastTabDimens,
                lastDimensInfo,
                lastUnits,
                lastZunits,
            ]);
            maxUseC = lastMaxStartUsed;
        }
        b3dFseek(fp, c, SEEK_SET);
        if b3dFread(
            (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                as *mut ::core::ffi::c_char as *mut ::core::ffi::c_void,
            1 as size_t,
            (maxUseC - c) as size_t,
            fp,
        ) == 0
        {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: analyzeDM3 - Reading beginning of {}\n",
                    CStr::from_ptr(filename).to_string_lossy()
                ),
            );
            return IIERR_IO_ERROR;
        }
        buf[(maxUseC - 1 as ::core::ffi::c_int) as usize] = 0 as ::core::ffi::c_char;
        if debug != 0 {
            printf(
                b"analyze_dm3: file start loop %d, start at %d, read %d\n\0" as *const u8
                    as *const ::core::ffi::c_char,
                loop_0,
                c,
                maxUseC - c,
            );
        }
        while c < maxUseC {
            if buf[c as usize] as ::core::ffi::c_int == 68 as ::core::ffi::c_int {
                found = found_dm_tag(
                    (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                        as *mut ::core::ffi::c_char,
                    b"Data\0" as *const u8 as *const ::core::ffi::c_char,
                    b"Data%%%%\0" as *const u8 as *const ::core::ffi::c_char,
                    dmind,
                    12 as ::core::ffi::c_int,
                );
                if !found.is_null() {
                    toffset = found
                        .offset(dataOff[dmind as usize] as isize)
                        .offset_from(&raw mut buf as *mut ::core::ffi::c_char)
                        as ::core::ffi::c_long as ::core::ffi::c_int;
                    if offset == 0 || toffset as ::core::ffi::c_long <= plausibleOff {
                        if loop_0 == 0 && (c != lastData || toffset != lastOffset) {
                            matchLast = 0 as ::core::ffi::c_int;
                            break;
                        } else {
                            offset = toffset;
                            curData = c;
                            curOffset = offset;
                            maxCurUsed = c;
                        }
                    }
                    if debug != 0 {
                        printf(
                            b"analyze_dm3: Found Data at %d  offset %d\n\0" as *const u8
                                as *const ::core::ffi::c_char,
                            c,
                            toffset,
                        );
                    }
                } else if gotMeta != 0
                    && !strstr(
                        (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                            as *mut ::core::ffi::c_char,
                        b"Dimension info\0" as *const u8 as *const ::core::ffi::c_char,
                    )
                    .is_null()
                {
                    gotDimInfo = 1 as ::core::ffi::c_int;
                    if loop_0 == 0 && c != lastDimensInfo {
                        matchLast = 0 as ::core::ffi::c_int;
                        break;
                    } else {
                        curDimensInfo = c;
                        maxCurUsed = c;
                        if debug != 0 {
                            printf(
                                b"analyze_dm3: Setting gotDimInfo at %d\n\0" as *const u8
                                    as *const ::core::ffi::c_char,
                                c,
                            );
                        }
                    }
                }
            } else if buf[c as usize] as ::core::ffi::c_int == 67 as ::core::ffi::c_int {
                if !strstr(
                    (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                        as *mut ::core::ffi::c_char,
                    b"Calibrations\0" as *const u8 as *const ::core::ffi::c_char,
                )
                .is_null()
                {
                    gotCal = 1 as ::core::ffi::c_int;
                    gotDimInfo = 0 as ::core::ffi::c_int;
                    gotMeta = gotDimInfo;
                    gotScale = gotMeta;
                    gotDim = gotScale;
                    if loop_0 == 0 && c != lastCalibrations {
                        matchLast = 0 as ::core::ffi::c_int;
                        break;
                    } else {
                        curCalibrations = c;
                        maxCurUsed = c;
                        if debug != 0 {
                            printf(
                                b"analyze_dm3: Found Calibrations at %d\n\0" as *const u8
                                    as *const ::core::ffi::c_char,
                                c,
                            );
                        }
                    }
                }
            } else if buf[c as usize] as ::core::ffi::c_int == '\t' as i32 {
                if gotCal != 0
                    && gotDim == 0
                    && !strstr(
                        (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                            as *mut ::core::ffi::c_char,
                        b"\tDimension\0" as *const u8 as *const ::core::ffi::c_char,
                    )
                    .is_null()
                {
                    gotDim = 1 as ::core::ffi::c_int;
                    if loop_0 == 0 && c != lastTabDimens {
                        matchLast = 0 as ::core::ffi::c_int;
                        break;
                    } else {
                        curTabDimens = c;
                        maxCurUsed = c;
                        if debug != 0 {
                            printf(
                                b"analyze_dm3: Found tabDimension at %d\n\0" as *const u8
                                    as *const ::core::ffi::c_char,
                                c,
                            );
                        }
                    }
                }
                if !strstr(
                    (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                        as *mut ::core::ffi::c_char,
                    b"\tMeta Data\0" as *const u8 as *const ::core::ffi::c_char,
                )
                .is_null()
                {
                    gotMeta = 1 as ::core::ffi::c_int;
                    gotDimInfo = 0 as ::core::ffi::c_int;
                    gotScale = gotDimInfo;
                    gotDim = gotScale;
                    gotCal = gotDim;
                    if debug != 0 {
                        printf(
                            b"analyze_dm3: Found tabMeta Data at %d\n\0" as *const u8
                                as *const ::core::ffi::c_char,
                            c,
                        );
                    }
                }
            } else if (gotDim != 0 || gotDimInfo != 0)
                && gotScale == 0
                && buf[c as usize] as ::core::ffi::c_int == 'S' as i32
            {
                found = found_dm_tag(
                    (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                        as *mut ::core::ffi::c_char,
                    b"Scale\0" as *const u8 as *const ::core::ffi::c_char,
                    b"Scale%%%%\0" as *const u8 as *const ::core::ffi::c_char,
                    dmind,
                    13 as ::core::ffi::c_int,
                );
                if !found.is_null()
                    && (c + scaleOff[dmind as usize] + 3 as ::core::ffi::c_int) < maxUseC
                {
                    gotScale = 1 as ::core::ffi::c_int;
                    memcpy(
                        &raw mut scale as *mut ::core::ffi::c_void,
                        (&raw mut buf as *mut ::core::ffi::c_char).offset(
                            (c + *(&raw mut scaleOff as *mut ::core::ffi::c_int)
                                .offset(dmind as isize)) as isize,
                        ) as *mut ::core::ffi::c_char
                            as *const ::core::ffi::c_void,
                        4 as size_t,
                    );
                    #[cfg(target_endian = "big")]
                    mrc_swap_longs(
                        core::slice::from_raw_parts_mut(
                            (&raw mut scale).cast::<::core::ffi::c_int>(),
                            1,
                        ),
                        1,
                    );
                    if debug != 0 {
                        printf(
                            b"analyze_dm3: Found Scale at %d  %f\n\0" as *const u8
                                as *const ::core::ffi::c_char,
                            c,
                            scale as ::core::ffi::c_double,
                        );
                    }
                }
            } else if gotScale != 0 && buf[c as usize] as ::core::ffi::c_int == 'U' as i32 {
                found = found_dm_tag(
                    (&raw mut buf as *mut ::core::ffi::c_char).offset(c as isize)
                        as *mut ::core::ffi::c_char,
                    b"Units\0" as *const u8 as *const ::core::ffi::c_char,
                    b"Units%%%%\0" as *const u8 as *const ::core::ffi::c_char,
                    dmind,
                    13 as ::core::ffi::c_int,
                );
                if !found.is_null()
                    && (c + unitsOff[dmind as usize] + 2 as ::core::ffi::c_int) < maxUseC
                {
                    toffset = found.offset_from(&raw mut buf as *mut ::core::ffi::c_char)
                        as ::core::ffi::c_long as ::core::ffi::c_int;
                    if gotDim != 0 && pixel == 0.
                        || gotDimInfo != 0 && z_pixel == 0.
                        || toffset as ::core::ffi::c_long <= plausibleOff
                    {
                        if buf[(c + unitsOff[dmind as usize] + 2 as ::core::ffi::c_int) as usize]
                            as ::core::ffi::c_int
                            == 'm' as i32
                        {
                            if buf[(c + unitsOff[dmind as usize]) as usize] as ::core::ffi::c_int
                                == 'n' as i32
                            {
                                tmpPixel = (scale as ::core::ffi::c_double * 10.0f64)
                                    as ::core::ffi::c_float;
                            } else if buf[(c + unitsOff[dmind as usize]) as usize]
                                as ::core::ffi::c_uchar
                                as ::core::ffi::c_int
                                == 181 as ::core::ffi::c_int
                            {
                                tmpPixel = (scale as ::core::ffi::c_double * 10000.0f64)
                                    as ::core::ffi::c_float;
                            }
                            if gotDimInfo != 0 {
                                if loop_0 == 0 && c != lastZunits {
                                    matchLast = 0 as ::core::ffi::c_int;
                                    break;
                                } else {
                                    curZunits = c;
                                    z_pixel = tmpPixel;
                                }
                            } else if loop_0 == 0 && c != lastUnits {
                                matchLast = 0 as ::core::ffi::c_int;
                                break;
                            } else {
                                curUnits = c;
                                pixel = tmpPixel;
                            }
                            maxCurUsed = c;
                            if debug != 0 {
                                printf(
                                    b"analyze_dm3: Assigned %f to %spixel\n\0" as *const u8
                                        as *const ::core::ffi::c_char,
                                    tmpPixel as ::core::ffi::c_double,
                                    if gotDimInfo != 0 {
                                        b"z\0" as *const u8 as *const ::core::ffi::c_char
                                    } else {
                                        b"\0" as *const u8 as *const ::core::ffi::c_char
                                    },
                                );
                            }
                        }
                        if debug != 0 {
                            printf(
                                b"analyze_dm3: Found Units at %d (%p) %d  %d\n\0" as *const u8
                                    as *const ::core::ffi::c_char,
                                c,
                                found,
                                buf[(c + unitsOff[dmind as usize]) as usize] as ::core::ffi::c_uchar
                                    as ::core::ffi::c_int,
                                buf[(c + unitsOff[dmind as usize] + 2 as ::core::ffi::c_int)
                                    as usize]
                                    as ::core::ffi::c_uchar
                                    as ::core::ffi::c_int,
                            );
                        }
                    }
                    gotDimInfo = 0 as ::core::ffi::c_int;
                    gotMeta = gotDimInfo;
                    gotScale = gotMeta;
                    gotDim = gotScale;
                    gotCal = gotDim;
                }
            }
            c += 1;
        }
        if matchLast != 0 && offset != 0 {
            break;
        }
        loop_0 += 1;
    }
    if offset == 0 {
        b3d_error(
            stderr,
            format_args!(
                "ERROR: analyzeDM3 - Data string not found in {}\n",
                CStr::from_ptr(filename).to_string_lossy()
            ),
        );
        return IIERR_NO_SUPPORT;
    }
    if debug != 0 {
        printf(
            b"analyze_dm3: time %.1f\n\0" as *const u8 as *const ::core::ffi::c_char,
            1000.0f64 * (wallTime() - wallStart),
        );
        fflush(stdout);
    }
    lastDataType = curDataType;
    lastData = curData;
    lastCalibrations = curCalibrations;
    lastZunits = curZunits;
    lastUnits = curUnits;
    lastTabDimens = curTabDimens;
    lastDimensions = curDimensions;
    lastDimensInfo = curDimensInfo;
    lastOffset = curOffset;
    lastMaxRead = maxread;
    lastMaxStartUsed = (if maxread < (maxCurUsed + maxOffset[dmind as usize]) as ::core::ffi::c_long
    {
        maxread as ::core::ffi::c_long
    } else {
        (maxCurUsed + maxOffset[dmind as usize]) as ::core::ffi::c_long
    }) as ::core::ffi::c_int;
    (*info).nx = xsize;
    (*info).ny = ysize;
    (*info).nz = zsize;
    (*info).header_size = offset;
    (*info).y_inverted = 1 as ::core::ffi::c_int;
    (*info).pixel = pixel;
    (*info).z_pixel = z_pixel;
    *dmtype = type_;
    #[cfg(target_endian = "little")]
    {
        (*info).swap_bytes = 0 as ::core::ffi::c_int;
    }
    #[cfg(target_endian = "big")]
    {
        (*info).swap_bytes = 1 as ::core::ffi::c_int;
    }
    return 0 as ::core::ffi::c_int;
}
unsafe extern "C" fn found_dm_tag(
    mut buf: *mut ::core::ffi::c_char,
    mut tag: *const ::core::ffi::c_char,
    mut fullTag: *const ::core::ffi::c_char,
    mut dmind: ::core::ffi::c_int,
    mut dm4Offset: ::core::ffi::c_int,
) -> *mut ::core::ffi::c_char {
    let mut found: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    if dmind != 0 {
        found = strstr(buf, tag);
        if !found.is_null()
            && strstr(
                buf.offset(dm4Offset as isize),
                b"%%%%\0" as *const u8 as *const ::core::ffi::c_char,
            )
            .is_null()
        {
            found = ::core::ptr::null_mut::<::core::ffi::c_char>();
        }
    } else {
        found = strstr(buf, fullTag);
    }
    return found;
}
unsafe extern "C" fn check_fei_raw(
    mut fp: *mut FILE,
    mut filename: *mut ::core::ffi::c_char,
    mut info: *mut RawImageInfo,
) -> ::core::ffi::c_int {
    let mut label: [::core::ffi::c_char; 13] = [0; 13];
    let mut ivals: [::core::ffi::c_int; 9] = [0; 9];
    b3dRewind(fp);
    if b3dFread(
        &raw mut label as *mut ::core::ffi::c_char as *mut ::core::ffi::c_void,
        1 as size_t,
        13 as size_t,
        fp,
    ) != 13 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if strncmp(
        &raw mut label as *mut ::core::ffi::c_char,
        b"FEI RawImage\0" as *const u8 as *const ::core::ffi::c_char,
        12 as size_t,
    ) != 0
    {
        return IIERR_NOT_FORMAT;
    }
    if b3dFread(
        &raw mut ivals as *mut ::core::ffi::c_int as *mut ::core::ffi::c_void,
        4 as size_t,
        9 as size_t,
        fp,
    ) != 9 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if ivals[4 as ::core::ffi::c_int as usize] == 16 as ::core::ffi::c_int
        && ivals[5 as ::core::ffi::c_int as usize] == 1 as ::core::ffi::c_int
    {
        (*info).type_ = RAW_MODE_SHORT;
    } else if ivals[4 as ::core::ffi::c_int as usize] == 16 as ::core::ffi::c_int
        && ivals[5 as ::core::ffi::c_int as usize] == 0 as ::core::ffi::c_int
    {
        (*info).type_ = RAW_MODE_USHORT;
    } else if ivals[4 as ::core::ffi::c_int as usize] == 32 as ::core::ffi::c_int
        && ivals[5 as ::core::ffi::c_int as usize] == 2 as ::core::ffi::c_int
    {
        (*info).type_ = RAW_MODE_FLOAT;
    } else {
        return IIERR_NO_SUPPORT;
    }
    (*info).nx = ivals[1 as ::core::ffi::c_int as usize];
    (*info).ny = ivals[2 as ::core::ffi::c_int as usize];
    (*info).nz = 1 as ::core::ffi::c_int;
    #[cfg(target_endian = "little")]
    {
        (*info).swap_bytes = 0 as ::core::ffi::c_int;
    }
    #[cfg(target_endian = "big")]
    {
        (*info).swap_bytes = 1 as ::core::ffi::c_int;
    }
    (*info).header_size = 49 as ::core::ffi::c_int + ivals[6 as ::core::ffi::c_int as usize];
    (*info).y_inverted = 1 as ::core::ffi::c_int;
    (*info).amin = 0.0f32;
    (*info).amax = 0.0f32;
    return 0 as ::core::ffi::c_int;
}
unsafe extern "C" fn check_em(
    mut fp: *mut FILE,
    mut filename: *mut ::core::ffi::c_char,
    mut info: *mut RawImageInfo,
) -> ::core::ffi::c_int {
    let mut bvals: [::core::ffi::c_uchar; 4] = [0; 4];
    let mut ivals: [b3dInt32; 12] = [0; 12];
    b3dRewind(fp);
    if b3dFread(
        &raw mut bvals as *mut ::core::ffi::c_uchar as *mut ::core::ffi::c_void,
        1 as size_t,
        4 as size_t,
        fp,
    ) != 4 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if b3dFread(
        &raw mut ivals as *mut b3dInt32 as *mut ::core::ffi::c_void,
        4 as size_t,
        3 as size_t,
        fp,
    ) != 3 as size_t
    {
        return IIERR_IO_ERROR;
    }
    if ivals[0 as ::core::ffi::c_int as usize] <= 0 as ::core::ffi::c_int
        || ivals[1 as ::core::ffi::c_int as usize] <= 0 as ::core::ffi::c_int
        || ivals[2 as ::core::ffi::c_int as usize] <= 0 as ::core::ffi::c_int
        || ivals[0 as ::core::ffi::c_int as usize] > 65536 as ::core::ffi::c_int
            && ivals[1 as ::core::ffi::c_int as usize] > 65536 as ::core::ffi::c_int
            && ivals[2 as ::core::ffi::c_int as usize] > 65536 as ::core::ffi::c_int
        || bvals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int > MAX_EM_MACHINES
        || bvals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_int == 1 as ::core::ffi::c_int
        || bvals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int > MAX_EM_TYPES
        || (ivals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
            * ivals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
            * ivals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_float)
            as ::core::ffi::c_double
            > MAX_EM_SIZE
    {
        mrc_swap_longs(core::slice::from_raw_parts_mut(ivals.as_mut_ptr(), 3), 3);
        if ivals[0 as ::core::ffi::c_int as usize] <= 0 as ::core::ffi::c_int
            || ivals[1 as ::core::ffi::c_int as usize] <= 0 as ::core::ffi::c_int
            || ivals[2 as ::core::ffi::c_int as usize] <= 0 as ::core::ffi::c_int
            || ivals[0 as ::core::ffi::c_int as usize] > 65536 as ::core::ffi::c_int
                && ivals[1 as ::core::ffi::c_int as usize] > 65536 as ::core::ffi::c_int
                && ivals[2 as ::core::ffi::c_int as usize] > 65536 as ::core::ffi::c_int
            || bvals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int > MAX_EM_MACHINES
            || bvals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_int
                == 1 as ::core::ffi::c_int
            || bvals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int > MAX_EM_TYPES
            || (ivals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
                * ivals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_float
                * ivals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_float)
                as ::core::ffi::c_double
                > MAX_EM_SIZE
        {
            return IIERR_NOT_FORMAT;
        }
        (*info).swap_bytes = 1 as ::core::ffi::c_int;
    }
    match bvals[3 as ::core::ffi::c_int as usize] as ::core::ffi::c_int {
        1 => {
            (*info).type_ = RAW_MODE_BYTE;
        }
        2 => {
            (*info).type_ = RAW_MODE_SHORT;
        }
        5 => {
            (*info).type_ = RAW_MODE_FLOAT;
        }
        _ => return IIERR_NO_SUPPORT,
    }
    (*info).nx = ivals[0 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    (*info).ny = ivals[1 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    (*info).nz = ivals[2 as ::core::ffi::c_int as usize] as ::core::ffi::c_int;
    (*info).header_size = 512 as ::core::ffi::c_int;
    (*info).amin = 0.0f32;
    (*info).amax = 0.0f32;
    return 0 as ::core::ffi::c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::iimage::{IIFILE_RAW, IITYPE_SHORT, ii_new};
    use crate::imod::libiimod::mrcfiles::MrcHeader;

    #[test]
    fn fei_raw_dispatch_constructs_native_mrc_access_state() {
        unsafe {
            ii_delete_raw_check_list();
            let fp = libc::tmpfile();
            assert!(!fp.is_null());
            let mut bytes = b"FEI RawImage\0".to_vec();
            for value in [0_i32, 4, 3, 0, 16, 1, 100, 0, 0] {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                libc::fwrite(bytes.as_ptr().cast(), 1, bytes.len(), fp),
                bytes.len()
            );
            libc::rewind(fp);

            let image = ii_new();
            assert!(!image.is_null());
            (*image).fp = fp;
            (*image).filename = c"synthetic-fei.raw".as_ptr().cast_mut();
            assert_eq!(ii_like_mrc_check(image.cast()), 0);
            let header = (*image).header.cast::<MrcHeader>();
            assert_eq!(((*image).file, (*image).type_), (IIFILE_RAW, IITYPE_SHORT));
            assert_eq!(((*header).nx, (*header).ny, (*header).nz), (4, 3, 1));
            assert_eq!(((*header).header_size, (*header).y_inverted), (149, 1));
            ii_like_mrc_delete(image.cast());
            libc::fclose(fp);
            libc::free(image.cast());
            ii_delete_raw_check_list();
        }
    }

    #[test]
    fn em_checker_accepts_little_endian_dimensions_and_rejects_unknown_type() {
        unsafe {
            let fp = libc::tmpfile();
            assert!(!fp.is_null());
            let mut bytes = vec![6_u8, 0, 0, 2];
            for value in [8_i32, 7, 2] {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                libc::fwrite(bytes.as_ptr().cast(), 1, bytes.len(), fp),
                bytes.len()
            );
            libc::rewind(fp);
            let mut info: RawImageInfo = core::mem::zeroed();
            assert_eq!(check_em(fp, core::ptr::null_mut(), &mut info), 0);
            assert_eq!(
                (info.nx, info.ny, info.nz, info.type_, info.header_size),
                (8, 7, 2, RAW_MODE_SHORT, 512)
            );

            let mut unsupported = bytes;
            unsupported[3] = 3;
            let unsupported_fp = libc::tmpfile();
            assert!(!unsupported_fp.is_null());
            assert_eq!(
                libc::fwrite(
                    unsupported.as_ptr().cast(),
                    1,
                    unsupported.len(),
                    unsupported_fp
                ),
                unsupported.len()
            );
            libc::rewind(unsupported_fp);
            assert_eq!(
                check_em(unsupported_fp, core::ptr::null_mut(), &mut info),
                IIERR_NO_SUPPORT
            );
            libc::fclose(unsupported_fp);
            libc::fclose(fp);
        }
    }

    #[test]
    fn pif_checker_reads_bsoft_header_fields() {
        unsafe {
            let fp = libc::tmpfile();
            assert!(!fp.is_null());
            let mut bytes = vec![0_u8; 84];
            bytes[24..28].copy_from_slice(&3_i32.to_ne_bytes());
            bytes[28..32].copy_from_slice(&0_i32.to_ne_bytes());
            bytes[32..37].copy_from_slice(b"Bsoft");
            for (index, value) in [1_i32, 4, 3, 0, 9].into_iter().enumerate() {
                let start = 64 + 4 * index;
                bytes[start..start + 4].copy_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                libc::fwrite(bytes.as_ptr().cast(), 1, bytes.len(), fp),
                bytes.len()
            );
            let mut info: RawImageInfo = core::mem::zeroed();
            assert_eq!(check_pif(fp, core::ptr::null_mut(), &mut info), 0);
            assert_eq!(
                (
                    info.nx,
                    info.ny,
                    info.nz,
                    info.type_,
                    info.header_size,
                    info.section_skip,
                    info.swap_bytes,
                ),
                (4, 3, 3, RAW_MODE_FLOAT, 1024, 512, 0)
            );
            libc::fclose(fp);
        }
    }
}
