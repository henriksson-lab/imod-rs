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
    fn remove(__filename: *const ::core::ffi::c_char) -> ::core::ffi::c_int;
    fn fflush(__stream: *mut FILE) -> ::core::ffi::c_int;
    fn fprintf(
        __stream: *mut FILE,
        __format: *const ::core::ffi::c_char,
        ...
    ) -> ::core::ffi::c_int;
    fn printf(__format: *const ::core::ffi::c_char, ...) -> ::core::ffi::c_int;
    fn sprintf(
        __s: *mut ::core::ffi::c_char,
        __format: *const ::core::ffi::c_char,
        ...
    ) -> ::core::ffi::c_int;
    fn stat(__file: *const ::core::ffi::c_char, __buf: *mut stat) -> ::core::ffi::c_int;
    fn memset(
        __s: *mut ::core::ffi::c_void,
        __c: ::core::ffi::c_int,
        __n: size_t,
    ) -> *mut ::core::ffi::c_void;
    fn strrchr(
        __s: *const ::core::ffi::c_char,
        __c: ::core::ffi::c_int,
    ) -> *mut ::core::ffi::c_char;
    fn strerror(__errnum: ::core::ffi::c_int) -> *mut ::core::ffi::c_char;
    fn malloc(__size: size_t) -> *mut ::core::ffi::c_void;
    fn realloc(__ptr: *mut ::core::ffi::c_void, __size: size_t) -> *mut ::core::ffi::c_void;
    fn free(__ptr: *mut ::core::ffi::c_void);
    fn exit(__status: ::core::ffi::c_int) -> !;
    fn getenv(__name: *const ::core::ffi::c_char) -> *mut ::core::ffi::c_char;
    fn __errno_location() -> *mut ::core::ffi::c_int;
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
pub type FILE = _IO_FILE;
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
pub type b3dInt32 = ::core::ffi::c_int;
pub type b3dFloat = ::core::ffi::c_float;
pub type fortStrLen_t = ::core::ffi::c_int;
use crate::imod::libcfshr::b3dutil::{
    b3d_error, b3d_get_error as b3dGetError, b3d_milli_sleep as b3dMilliSleep,
    b3d_set_store_error as b3dSetStoreError, f2c_string as f2cString,
    imod_backup_file as imodBackupFile,
};
use crate::imod::libcfshr::ilist::{
    Ilist, ilist_append as ilistAppend, ilist_item as ilistItem, ilist_new as ilistNew,
    ilist_remove as ilistRemove, ilist_size as ilistSize,
};
use crate::imod::libiimod::iihdf::hdf_write_global_adoc as hdfWriteGlobalAdoc;
use crate::imod::libiimod::iimage::{
    IiFileCheckFunction as IIFileCheckFunction, IiSectionFunc as iiSectionFunc, ImodImageFile,
    ii_allow_multi_volume as iiAllowMultiVolume, ii_close as iiClose, ii_delete as iiDelete,
    ii_fill_mrc_header as iiFillMrcHeader, ii_fopen_new_volume as iiFOpenNewVolume,
    ii_fopen_volume as iiFOpenVolume, ii_get_adoc_index as iiGetAdocIndex,
    ii_insert_check_function as iiInsertCheckFunction, ii_open as iiOpen, ii_open_new as iiOpenNew,
    ii_read_section as iiReadSection, ii_read_section_float as iiReadSectionFloat,
    ii_set_chunk_sizes as iiSetChunkSizes, ii_sync_from_mrc_header as iiSyncFromMrcHeader,
    ii_transfer_adoc_sections as iiTransferAdocSections, ii_write_section as iiWriteSection,
    ii_write_section_float as iiWriteSectionFloat,
};
use crate::imod::libiimod::iimrc::ii_mrc_check as iiMRCCheck;
use crate::imod::libiimod::iishrmem::ii_shr_mem_check_size as iiShrMemCheckSize;
use crate::imod::libiimod::iitif::tiff_set_string_tag_to_print as tiffSetStringTagToPrint;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_getdcsize};

#[repr(C)]
pub struct Unit {
    pub ii_file: *mut ImodImageFile,
    pub header: *mut MrcHeader,
    pub current_sec: ::core::ffi::c_int,
    pub current_line: ::core::ffi::c_int,
    pub tail_name: *mut ::core::ffi::c_char,
    pub attribute: ::core::ffi::c_int,
    pub being_used: ::core::ffi::c_uchar,
    pub read_only: ::core::ffi::c_uchar,
    pub no_convert: ::core::ffi::c_uchar,
}
pub const NULL: *mut ::core::ffi::c_void = ::core::ptr::null_mut::<::core::ffi::c_void>();
pub const IIUNIT_SWAPPED: ::core::ffi::c_long =
    (1 as ::core::ffi::c_long) << 0 as ::core::ffi::c_int;
pub const IIUNIT_BYTES_SIGNED: ::core::ffi::c_long =
    (1 as ::core::ffi::c_long) << 1 as ::core::ffi::c_int;
pub const IIFILE_DEFAULT: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
pub const IIFILE_TIFF: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
pub const IIFILE_MRC: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const IIFILE_RAW: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
pub const IIFILE_HDF: ::core::ffi::c_int = 5 as ::core::ffi::c_int;
pub const IIFILE_SHR_MEM: ::core::ffi::c_int = 8 as ::core::ffi::c_int;
pub const IIFORMAT_COMPLEX: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
pub const MAX_UNIT: ::core::ffi::c_int = 1000 as ::core::ffi::c_int;
pub const UNIT_ATBUT_RO: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
pub const UNIT_ATBUT_NEW: ::core::ffi::c_int = 2 as ::core::ffi::c_int;
pub const UNIT_ATBUT_OLD: ::core::ffi::c_int = 3 as ::core::ffi::c_int;
pub const UNIT_ATBUT_SCRATCH: ::core::ffi::c_int = 4 as ::core::ffi::c_int;
static mut sUnitList: *mut Ilist = ::core::ptr::null::<Ilist>() as *mut Ilist;
static mut sUnitMap: *mut ::core::ffi::c_int =
    ::core::ptr::null::<::core::ffi::c_int>() as *mut ::core::ffi::c_int;
static mut sMapSize: ::core::ffi::c_int = 0 as ::core::ffi::c_int;
static mut sNoConvList: *mut Ilist = ::core::ptr::null::<Ilist>() as *mut Ilist;
static mut sBriefHeader: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
static mut sPrintHeader: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
static mut sExitOnError: ::core::ffi::c_int = 1 as ::core::ffi::c_int;
static mut sStoreError: ::core::ffi::c_int = -(1 as ::core::ffi::c_int);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_open(
    mut iunit: ::core::ffi::c_int,
    mut name: *const ::core::ffi::c_char,
    mut attribute: *const ::core::ffi::c_char,
) -> ::core::ffi::c_int {
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    let mut mode: ::core::ffi::c_int = 0;
    let mut errSave: ::core::ffi::c_int = 0;
    let mut modes: [*const ::core::ffi::c_char; 4] = [
        b"rb\0" as *const u8 as *const ::core::ffi::c_char,
        b"rb+\0" as *const u8 as *const ::core::ffi::c_char,
        b"wb\0" as *const u8 as *const ::core::ffi::c_char,
        b"wb+\0" as *const u8 as *const ::core::ffi::c_char,
    ];
    let mut tailback: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    u = find_new_unit(iunit);
    iiu_memory_error(
        u as *mut ::core::ffi::c_void,
        b"ERROR: iiu_open - Allocating new unit\0" as *const u8 as *const ::core::ffi::c_char,
    );
    (*u).being_used = 1 as ::core::ffi::c_uchar;
    (*u).read_only = 0 as ::core::ffi::c_uchar;
    (*u).current_sec = 0 as ::core::ffi::c_int;
    (*u).current_line = 0 as ::core::ffi::c_int;
    if *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'R' as i32
        || *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'r' as i32
    {
        mode = 0 as ::core::ffi::c_int;
        (*u).attribute = UNIT_ATBUT_RO;
        (*u).read_only = 1 as ::core::ffi::c_uchar;
    }
    if name.is_null()
        || *name.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int
            == 0 as ::core::ffi::c_int
    {
        iiInsertCheckFunction(
            Some(iiMRCCheck as unsafe extern "C" fn(*mut ImodImageFile) -> ::core::ffi::c_int),
            0 as ::core::ffi::c_int,
        );
    }
    if *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'N' as i32
        || *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'n' as i32
    {
        if getenv(b"IMOD_NO_IMAGE_BACKUP\0" as *const u8 as *const ::core::ffi::c_char).is_null() {
            *__errno_location() = 0 as ::core::ffi::c_int;
            if imodBackupFile(name) != 0 {
                errSave = *__errno_location();
                fprintf(
                    stdout,
                    b"\nWARNING: iiu_open - Could not rename '%s' to '%s~'\n\0" as *const u8
                        as *const ::core::ffi::c_char,
                    name,
                    name,
                );
                if errSave != 0 {
                    fprintf(
                        stdout,
                        b"WARNING: from system - %s\n\0" as *const u8 as *const ::core::ffi::c_char,
                        strerror(errSave),
                    );
                }
            }
        }
        mode = 3 as ::core::ffi::c_int;
        (*u).attribute = UNIT_ATBUT_NEW;
    }
    if *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'O' as i32
        || *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'o' as i32
    {
        mode = 1 as ::core::ffi::c_int;
        (*u).attribute = UNIT_ATBUT_OLD;
    }
    if *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 'S' as i32
        || *attribute.offset(0 as ::core::ffi::c_int as isize) as ::core::ffi::c_int == 's' as i32
    {
        mode = 3 as ::core::ffi::c_int;
        (*u).attribute = UNIT_ATBUT_SCRATCH;
    }
    if mode == 3 as ::core::ffi::c_int {
        (*u).ii_file = iiOpenNew(name, modes[mode as usize], IIFILE_DEFAULT);
        if (*u).ii_file.is_null() {
            b3d_error(
                stdout.cast(),
                format_args!("\nERROR: iiu_open - Opening new output file\n"),
            );
            if sExitOnError != 0 {
                exit(1 as ::core::ffi::c_int);
            } else {
                return 1 as ::core::ffi::c_int;
            }
        }
        printf(
            b"\n NEW image file on unit %3d : %s\n\0" as *const u8 as *const ::core::ffi::c_char,
            iunit,
            name,
        );
        fflush(stdout);
    } else {
        (*u).ii_file = iiOpen(name, modes[mode as usize]);
        if (*u).ii_file.is_null() {
            b3d_error(
                stdout.cast(),
                format_args!(
                    "\nERROR: iiu_open - Could not open '{}'\n",
                    core::ffi::CStr::from_ptr(name).to_string_lossy()
                ),
            );
            if sExitOnError != 0 {
                exit(1 as ::core::ffi::c_int);
            } else {
                return 1 as ::core::ffi::c_int;
            }
        }
        if !((*(*u).ii_file).write_section.is_some()
            && (*(*u).ii_file).write_section_float.is_some())
            && (*u).read_only == 0
        {
            b3d_error(
                stdout.cast(),
                format_args!(
                    "\nERROR: iiu_open - Non-MRC-type file '{}' with no write function must be opened read-only\n",
                    core::ffi::CStr::from_ptr(name).to_string_lossy()
                ),
            );
            if sExitOnError != 0 {
                exit(1 as ::core::ffi::c_int);
            } else {
                return 1 as ::core::ffi::c_int;
            }
        }
        if (*(*u).ii_file).file == IIFILE_TIFF {
            if (*(*u).ii_file).mode < 0 as ::core::ffi::c_int {
                b3d_error(
                    stdout.cast(),
                    format_args!(
                        "\nERROR: iiu_open - TIFF file '{}' has a data type that is not supported\n",
                        core::ffi::CStr::from_ptr(name).to_string_lossy()
                    ),
                );
                if sExitOnError != 0 {
                    exit(1 as ::core::ffi::c_int);
                } else {
                    return 1 as ::core::ffi::c_int;
                }
            }
        }
    }
    if (*(*u).ii_file).file != IIFILE_MRC
        && (*(*u).ii_file).file != IIFILE_RAW
        && (*(*u).ii_file).file != IIFILE_HDF
        && (*(*u).ii_file).file != IIFILE_SHR_MEM
    {
        (*u).header =
            malloc((1 as size_t).wrapping_mul(::core::mem::size_of::<MrcHeader>() as size_t))
                as *mut MrcHeader;
        iiu_memory_error(
            (*u).header as *mut ::core::ffi::c_void,
            b"ERROR: iiu_open - Allocating MRC header\0" as *const u8 as *const ::core::ffi::c_char,
        );
        if iiFillMrcHeader((*u).ii_file, (*u).header) != 0 {
            b3d_error(
                stdout.cast(),
                format_args!(
                    "\nERROR: iiu_open - file '{}' is not a format that provides an MRC-like header and cannot be read\n",
                    core::ffi::CStr::from_ptr(name).to_string_lossy()
                ),
            );
            if sExitOnError != 0 {
                exit(1 as ::core::ffi::c_int);
            } else {
                return 1 as ::core::ffi::c_int;
            }
        }
    } else {
        (*u).header = (*(*u).ii_file).header as *mut MrcHeader;
    }
    (*u).tail_name = strrchr((*(*u).ii_file).filename, '/' as i32);
    tailback = strrchr((*(*u).ii_file).filename, '\\' as i32);
    if tailback > (*u).tail_name {
        (*u).tail_name = tailback;
    }
    if (*u).tail_name.is_null() {
        (*u).tail_name = (*(*u).ii_file)
            .filename
            .offset(0 as ::core::ffi::c_int as isize)
            as *mut ::core::ffi::c_char;
    } else {
        (*u).tail_name = (*u).tail_name.offset(1);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuopen_(
    mut iunit: *mut ::core::ffi::c_int,
    mut name: *mut ::core::ffi::c_char,
    mut attribute: *mut ::core::ffi::c_char,
    mut name_l: fortStrLen_t,
    mut attr_l: fortStrLen_t,
) -> ::core::ffi::c_int {
    let mut cname: *mut ::core::ffi::c_char = f2cString(name, name_l as ::core::ffi::c_int);
    let mut cattr: *mut ::core::ffi::c_char = f2cString(attribute, attr_l as ::core::ffi::c_int);
    let mut err: ::core::ffi::c_int = 0;
    if cname.is_null() || cattr.is_null() {
        iiu_memory_error(
            NULL,
            b"ERROR: iiuopen - Allocating C strings\0" as *const u8 as *const ::core::ffi::c_char,
        );
    }
    err = iiu_open(*iunit, cname, cattr);
    free(cname as *mut ::core::ffi::c_void);
    free(cattr as *mut ::core::ffi::c_void);
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_close(mut iunit: ::core::ffi::c_int) {
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    let mut trial: ::core::ffi::c_int = 0;
    let mut delay: ::core::ffi::c_int = 500 as ::core::ffi::c_int;
    let mut numTry: ::core::ffi::c_int = 10 as ::core::ffi::c_int;
    let mut unit: ::core::ffi::c_int = iunit - 1 as ::core::ffi::c_int;
    remove_unit_from_list(sNoConvList, unit);
    if unit >= 0 as ::core::ffi::c_int
        && unit < sMapSize
        && *sUnitMap.offset(unit as isize) >= 0 as ::core::ffi::c_int
    {
        u = ilistItem(sUnitList, *sUnitMap.offset(unit as isize)) as *mut Unit;
        if (*u).being_used != 0 {
            (*u).being_used = 0 as ::core::ffi::c_uchar;
            iiClose((*u).ii_file);
            if (*u).attribute == UNIT_ATBUT_SCRATCH {
                trial = 0 as ::core::ffi::c_int;
                while trial < numTry {
                    if remove((*(*u).ii_file).filename) == 0 {
                        break;
                    }
                    if trial < numTry - 1 as ::core::ffi::c_int {
                        b3dMilliSleep(delay);
                    }
                    trial += 1;
                }
            }
            if (*(*u).ii_file).file == IIFILE_TIFF {
                free((*u).header as *mut ::core::ffi::c_void);
            }
            iiDelete((*u).ii_file);
        }
        *sUnitMap.offset(unit as isize) = -(1 as ::core::ffi::c_int);
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imclose_(mut iunit: *mut ::core::ffi::c_int) {
    iiu_close(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuclose_(mut iunit: *mut ::core::ffi::c_int) {
    iiu_close(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_get_ii_file(mut iunit: ::core::ffi::c_int) -> *mut ImodImageFile {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_get_ii_file\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    return (*u).ii_file;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_ret_num_volumes(mut iunit: ::core::ffi::c_int) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_ret_num_volumes\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    return if (*(*u).ii_file).dataset_id != 0 {
        (*(*u).ii_file).num_volumes
    } else {
        0 as ::core::ffi::c_int
    };
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretnumvolumes_(
    mut iunit: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_ret_num_volumes(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_volume_open(
    mut newUnit: ::core::ffi::c_int,
    mut mainUnit: ::core::ffi::c_int,
    mut volIndex: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut fp: *mut libc::FILE = ::core::ptr::null_mut::<libc::FILE>();
    let mut u: *mut Unit = lookup_unit(
        mainUnit,
        b"iiuOpenVolume\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    let mut unew: *mut Unit = find_new_unit(newUnit);
    iiu_memory_error(
        unew as *mut ::core::ffi::c_void,
        b"ERROR:  - Allocating new unit\0" as *const u8 as *const ::core::ffi::c_char,
    );
    (*unew).being_used = 1 as ::core::ffi::c_uchar;
    (*unew).read_only = (*u).read_only;
    (*unew).current_sec = 0 as ::core::ffi::c_int;
    (*unew).current_line = 0 as ::core::ffi::c_int;
    (*unew).attribute = if (*u).attribute == UNIT_ATBUT_SCRATCH {
        UNIT_ATBUT_NEW
    } else {
        (*u).attribute
    };
    if volIndex < 0 as ::core::ffi::c_int {
        fp = iiFOpenNewVolume((*u).ii_file);
        volIndex = (*(*u).ii_file).num_volumes - 1 as ::core::ffi::c_int;
    } else {
        fp = iiFOpenVolume((*u).ii_file, volIndex);
    }
    if fp.is_null() {
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return 1 as ::core::ffi::c_int;
        }
    }
    (*unew).ii_file = *(*(*u).ii_file).ii_volumes.offset(volIndex as isize);
    (*unew).header = (*(*unew).ii_file).header as *mut MrcHeader;
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuvolumeopen_(
    mut newUnit: *mut ::core::ffi::c_int,
    mut mainUnit: *mut ::core::ffi::c_int,
    mut volIndex: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_volume_open(*newUnit, *mainUnit, *volIndex);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_ret_adoc_index(
    mut iunit: ::core::ffi::c_int,
    mut global: ::core::ffi::c_int,
    mut openMdocOrNew: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_ret_adoc_index\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    return iiGetAdocIndex((*u).ii_file, global, openMdocOrNew);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretadocindex_(
    mut iunit: *mut ::core::ffi::c_int,
    mut global: *mut ::core::ffi::c_int,
    mut openMdocOrNew: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = iiu_ret_adoc_index(*iunit, *global, *openMdocOrNew);
    return if err < 0 as ::core::ffi::c_int {
        err
    } else {
        err + 1 as ::core::ffi::c_int
    };
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_trans_adoc_sections(
    mut toUnit: ::core::ffi::c_int,
    mut fromUnit: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut uto: *mut Unit = lookup_unit(
        toUnit,
        b"iiu_trans_adoc_sections\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    let mut ufrom: *mut Unit = lookup_unit(
        fromUnit,
        b"iiu_trans_adoc_sections\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    if (*(*uto).ii_file).adoc_index >= 0 as ::core::ffi::c_int
        && (*(*ufrom).ii_file).adoc_index >= 0 as ::core::ffi::c_int
    {
        return iiTransferAdocSections((*ufrom).ii_file, (*uto).ii_file);
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_write_global_adoc(
    mut iunit: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_write_global_adoc\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    if hdfWriteGlobalAdoc((*u).ii_file) != 0 {
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return 1 as ::core::ffi::c_int;
        }
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwriteglobaladoc_(
    mut iunit: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_write_global_adoc(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_ret_chunk_sizes(
    mut iunit: ::core::ffi::c_int,
    mut xSize: *mut ::core::ffi::c_int,
    mut ySize: *mut ::core::ffi::c_int,
    mut zSize: *mut ::core::ffi::c_int,
) {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiuRetChunkSize\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    *xSize = (*(*u).ii_file).tile_size_x;
    *ySize = (*(*u).ii_file).tile_size_y;
    *zSize = (*(*u).ii_file).z_chunk_size;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretchunksizes_(
    mut iunit: *mut ::core::ffi::c_int,
    mut xSize: *mut ::core::ffi::c_int,
    mut ySize: *mut ::core::ffi::c_int,
    mut zSize: *mut ::core::ffi::c_int,
) {
    iiu_ret_chunk_sizes(*iunit, xSize, ySize, zSize);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_alt_chunk_sizes(
    mut iunit: ::core::ffi::c_int,
    mut xSize: ::core::ffi::c_int,
    mut ySize: ::core::ffi::c_int,
    mut zSize: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiuAltChunkSize\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    return iiSetChunkSizes((*u).ii_file, xSize, ySize, zSize);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiualtchunksizes_(
    mut iunit: *mut ::core::ffi::c_int,
    mut xSize: *mut ::core::ffi::c_int,
    mut ySize: *mut ::core::ffi::c_int,
    mut zSize: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_alt_chunk_sizes(*iunit, *xSize, *ySize, *zSize);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_set_hdf_compression(
    mut iunit: ::core::ffi::c_int,
    mut compression: ::core::ffi::c_int,
) {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_set_hdf_compression\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    (*(*u).ii_file).hdf_compression = compression;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiusethdfcompression_(
    mut iunit: *mut ::core::ffi::c_int,
    mut compression: *mut ::core::ffi::c_int,
) {
    iiu_set_hdf_compression(*iunit, *compression);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_set_position(
    mut iunit: ::core::ffi::c_int,
    mut section: ::core::ffi::c_int,
    mut line: ::core::ffi::c_int,
) {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_set_position\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    (*u).current_sec = section;
    (*u).current_line = line;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn imposn_(
    mut iunit: *mut ::core::ffi::c_int,
    mut section: *mut ::core::ffi::c_int,
    mut line: *mut ::core::ffi::c_int,
) {
    iiu_set_position(*iunit, *section, *line);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiusetposition_(
    mut iunit: *mut ::core::ffi::c_int,
    mut section: *mut ::core::ffi::c_int,
    mut line: *mut ::core::ffi::c_int,
) {
    iiu_set_position(*iunit, *section, *line);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_read_section(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_read_section\0" as *const u8 as *const ::core::ffi::c_char,
        0 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
    );
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    return iiu_read_sec_part(
        iunit,
        array,
        (*(*u).ii_file).nx,
        0 as ::core::ffi::c_int,
        (*(*u).ii_file).nx - 1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
        (*(*u).ii_file).ny - 1 as ::core::ffi::c_int,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiureadsection_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
) -> ::core::ffi::c_int {
    return iiu_read_section(*iunit, array);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_read_sec_part(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: ::core::ffi::c_int,
    mut indX0: ::core::ffi::c_int,
    mut indX1: ::core::ffi::c_int,
    mut indY0: ::core::ffi::c_int,
    mut indY1: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_read_sec_part\0" as *const u8 as *const ::core::ffi::c_char,
        0 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
    );
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    (*(*u).ii_file).llx = indX0;
    (*(*u).ii_file).urx = indX1;
    (*(*u).ii_file).lly = indY0;
    (*(*u).ii_file).ury = indY1;
    (*(*u).ii_file).pad_left = 0 as ::core::ffi::c_int;
    (*(*u).ii_file).pad_right = nxdim - (indX1 + 1 as ::core::ffi::c_int - indX0);
    if (*u).no_convert != 0 {
        err = iiReadSection(
            (*u).ii_file,
            array as *mut ::core::ffi::c_char,
            (*u).current_sec,
        );
    } else {
        err = iiReadSectionFloat(
            (*u).ii_file,
            array as *mut ::core::ffi::c_char,
            (*u).current_sec,
        );
    }
    (*u).current_sec += 1;
    (*u).current_line = 0 as ::core::ffi::c_int;
    if err != 0 && sStoreError < 0 as ::core::ffi::c_int {
        printf(
            b"\n%s\n\0" as *const u8 as *const ::core::ffi::c_char,
            b3dGetError(),
        );
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiureadsecpart_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: *mut ::core::ffi::c_int,
    mut indX0: *mut ::core::ffi::c_int,
    mut indX1: *mut ::core::ffi::c_int,
    mut indY0: *mut ::core::ffi::c_int,
    mut indY1: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_read_sec_part(*iunit, array, *nxdim, *indX0, *indX1, *indY0, *indY1);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_read_lines(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut iz: ::core::ffi::c_int = 0;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_read_lines\0" as *const u8 as *const ::core::ffi::c_char,
        0 as ::core::ffi::c_int,
        1 as ::core::ffi::c_int,
    );
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    iz = (*u).current_sec;
    if setup_current_lines(u, numLines) != 0 {
        return -(2 as ::core::ffi::c_int);
    }
    if (*u).no_convert != 0 {
        err = iiReadSection((*u).ii_file, array as *mut ::core::ffi::c_char, iz);
    } else {
        err = iiReadSectionFloat((*u).ii_file, array as *mut ::core::ffi::c_char, iz);
    }
    if err != 0 && sStoreError < 0 as ::core::ffi::c_int {
        printf(
            b"\n%s\n\0" as *const u8 as *const ::core::ffi::c_char,
            b3dGetError(),
        );
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiureadlines_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_read_lines(*iunit, array, *numLines);
}
#[unsafe(no_mangle)]
pub static mut sWritePartMess: *const ::core::ffi::c_char =
    ::core::ptr::null::<::core::ffi::c_char>();
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_write_section(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_write_section\0" as *const u8 as *const ::core::ffi::c_char,
        sExitOnError,
        2 as ::core::ffi::c_int,
    );
    let mut err: ::core::ffi::c_int = 0;
    let mut mess: [::core::ffi::c_char; 120] = [0; 120];
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    sprintf(
        &raw mut mess as *mut ::core::ffi::c_char,
        b"\nERROR: iiu_write_section - writing section %d to unit %d\n\0" as *const u8
            as *const ::core::ffi::c_char,
        (*u).current_sec - 1 as ::core::ffi::c_int,
        iunit,
    );
    sWritePartMess = &raw mut mess as *mut ::core::ffi::c_char;
    return iiu_write_sec_part(
        iunit,
        array,
        (*(*u).ii_file).nx,
        0 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
        (*(*u).ii_file).nx - 1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
        (*(*u).ii_file).ny - 1 as ::core::ffi::c_int,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritesection_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
) -> ::core::ffi::c_int {
    return iiu_write_section(*iunit, array);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iwrsec_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
) {
    iiu_write_section(*iunit, array);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_write_subarray(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: ::core::ffi::c_int,
    mut ixStart: ::core::ffi::c_int,
    mut iyStart: ::core::ffi::c_int,
    mut iyEnd: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut mess: [::core::ffi::c_char; 120] = [0; 120];
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_write_subarray\0" as *const u8 as *const ::core::ffi::c_char,
        sExitOnError,
        2 as ::core::ffi::c_int,
    );
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    sprintf(
        &raw mut mess as *mut ::core::ffi::c_char,
        b"\nERROR: iiu_write_subarray - writing section %d, lines %d to %d to unit %d\n\0"
            as *const u8 as *const ::core::ffi::c_char,
        (*u).current_sec - 1 as ::core::ffi::c_int,
        iyStart,
        iyEnd,
        iunit,
    );
    sWritePartMess = &raw mut mess as *mut ::core::ffi::c_char;
    return iiu_write_sec_part(
        iunit,
        array,
        nxdim,
        ixStart,
        0 as ::core::ffi::c_int,
        (*(*u).ii_file).nx - 1 as ::core::ffi::c_int,
        iyStart,
        iyEnd,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritesubarray_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: *mut ::core::ffi::c_int,
    mut ixStart: *mut ::core::ffi::c_int,
    mut iyStart: *mut ::core::ffi::c_int,
    mut iyEnd: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_write_subarray(*iunit, array, *nxdim, *ixStart, *iyStart, *iyEnd);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_write_sec_part(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: ::core::ffi::c_int,
    mut ixStart: ::core::ffi::c_int,
    mut indX0: ::core::ffi::c_int,
    mut indX1: ::core::ffi::c_int,
    mut iyStart: ::core::ffi::c_int,
    mut iyEnd: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut useMess: *const ::core::ffi::c_char = sWritePartMess;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_write_sec_part\0" as *const u8 as *const ::core::ffi::c_char,
        sExitOnError,
        2 as ::core::ffi::c_int,
    );
    let mut arrStart: *mut ::core::ffi::c_char = ::core::ptr::null_mut::<::core::ffi::c_char>();
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    sWritePartMess = ::core::ptr::null::<::core::ffi::c_char>();
    if (*(*u).ii_file).file != IIFILE_HDF
        && (indX0 != 0 || indX1 != (*(*u).ii_file).nx - 1 as ::core::ffi::c_int)
    {
        b3d_error(
            stdout.cast(),
            format_args!(
                "\nERROR: iiu_write_sec_part - Attempting to write to a portion of a line for a non-HDF file, unit {}\n",
                iunit
            ),
        );
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return -(2 as ::core::ffi::c_int);
        }
    }
    if (*(*u).ii_file).file == IIFILE_TIFF {
        iiSyncFromMrcHeader((*u).ii_file, (*u).header);
    }
    (*(*u).ii_file).llx = indX0;
    (*(*u).ii_file).urx = indX1;
    (*(*u).ii_file).lly = (*u).current_line;
    (*(*u).ii_file).ury = (*u).current_line + iyEnd - iyStart;
    (*(*u).ii_file).pad_left = ixStart;
    (*(*u).ii_file).pad_right = nxdim - (indX1 + 1 as ::core::ffi::c_int - indX0) - ixStart;
    if (*(*u).ii_file).pad_right < 0 as ::core::ffi::c_int {
        b3d_error(
            stdout.cast(),
            format_args!(
                "\nERROR: iiu_write_sec_part - X dimension of data ({}) is not big enough for specified X indexes in writing to unit {} (xstart {} x0 {} x1 {})\n",
                nxdim, iunit, ixStart, indX0, indX1
            ),
        );
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return -(2 as ::core::ffi::c_int);
        }
    }
    if (*(*u).ii_file).ury >= (*(*u).ii_file).ny {
        b3d_error(
            stdout.cast(),
            format_args!(
                "\nERROR: iiu_write_sec_part - Starting and ending lines ({} to {}) to write to unit {} go past end of end for section from line {}\n",
                iyStart,
                iyEnd,
                iunit,
                (*u).current_line
            ),
        );
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return -(3 as ::core::ffi::c_int);
        }
    }
    arrStart = (array as *mut ::core::ffi::c_char)
        .offset((iyStart * nxdim * iiu_buf_bytes_per_pixel(iunit)) as isize);
    if (*u).no_convert != 0 {
        err = iiWriteSection((*u).ii_file, arrStart, (*u).current_sec);
    } else {
        err = iiWriteSectionFloat((*u).ii_file, arrStart.cast(), (*u).current_sec);
    }
    if err != 0 {
        if !useMess.is_null() {
            b3d_error(
                stdout.cast(),
                format_args!(
                    "\nERROR: iiu_write_sec_part - writing X {} to {}, Y {} to {} to section in unit {}\n",
                    indX0, indX1, iyStart, iyEnd, iunit
                ),
            );
        }
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return err;
        }
    }
    (*u).current_sec += 1;
    (*u).current_line = 0 as ::core::ffi::c_int;
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritesecpart_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut nxdim: *mut ::core::ffi::c_int,
    mut ixStart: *mut ::core::ffi::c_int,
    mut indX0: *mut ::core::ffi::c_int,
    mut indX1: *mut ::core::ffi::c_int,
    mut iyStart: *mut ::core::ffi::c_int,
    mut iyEnd: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_write_sec_part(
        *iunit, array, *nxdim, *ixStart, *indX0, *indX1, *iyStart, *iyEnd,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_write_lines(
    mut iunit: ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut err: ::core::ffi::c_int = 0;
    let mut iz: ::core::ffi::c_int = 0;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiuWriteLine\0" as *const u8 as *const ::core::ffi::c_char,
        sExitOnError,
        2 as ::core::ffi::c_int,
    );
    if u.is_null() {
        return -(1 as ::core::ffi::c_int);
    }
    if (*(*u).ii_file).file == IIFILE_TIFF {
        iiSyncFromMrcHeader((*u).ii_file, (*u).header);
    }
    iz = (*u).current_sec;
    if setup_current_lines(u, numLines) != 0 {
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return -(3 as ::core::ffi::c_int);
        }
    }
    if (*u).no_convert != 0 {
        err = iiWriteSection((*u).ii_file, array as *mut ::core::ffi::c_char, iz);
    } else {
        err = iiWriteSectionFloat((*u).ii_file, array as *mut f32, iz);
    }
    if err != 0 {
        b3d_error(
            stdout.cast(),
            format_args!(
                "\nERROR: iiu_write_lines - writing lines to unit {}.\n",
                iunit
            ),
        );
        if sExitOnError != 0 {
            exit(1 as ::core::ffi::c_int);
        } else {
            return 1 as ::core::ffi::c_int;
        }
    }
    return err;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuwritelines_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    return iiu_write_lines(*iunit, array, *numLines);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iwrlin_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
) {
    iiu_write_lines(*iunit, array, 1 as ::core::ffi::c_int);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iwrsecl_(
    mut iunit: *mut ::core::ffi::c_int,
    mut array: *mut ::core::ffi::c_void,
    mut numLines: *mut ::core::ffi::c_int,
) {
    iiu_write_lines(*iunit, array, *numLines);
}
unsafe extern "C" fn setup_current_lines(
    mut u: *mut Unit,
    mut numLines: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    (*(*u).ii_file).llx = 0 as ::core::ffi::c_int;
    (*(*u).ii_file).urx = (*(*u).ii_file).nx - 1 as ::core::ffi::c_int;
    (*(*u).ii_file).lly = (*u).current_line;
    (*(*u).ii_file).ury = (*u).current_line + numLines - 1 as ::core::ffi::c_int;
    if (*(*u).ii_file).ury >= (*(*u).ii_file).ny {
        b3d_error(
            stdout.cast(),
            format_args!(
                "\nERROR: iiuRead/WriteLines - lines go past end of current section  (cur line {}  #l {}  to {}  ny {}).\n",
                (*u).current_line,
                numLines,
                (*(*u).ii_file).ury,
                (*(*u).ii_file).ny
            ),
        );
        return 1 as ::core::ffi::c_int;
    }
    (*(*u).ii_file).pad_right = 0 as ::core::ffi::c_int;
    (*(*u).ii_file).pad_left = (*(*u).ii_file).pad_right;
    (*u).current_line += numLines;
    if (*u).current_line == (*(*u).ii_file).ny {
        (*u).current_sec += 1;
        (*u).current_line = 0 as ::core::ffi::c_int;
    }
    return 0 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_file_info(
    mut iunit: ::core::ffi::c_int,
    mut fileSize: *mut ::core::ffi::c_int,
    mut fileType: *mut ::core::ffi::c_int,
    mut flags: *mut ::core::ffi::c_int,
) {
    let mut buf: stat = stat {
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
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiuFileSize\0" as *const u8 as *const ::core::ffi::c_char,
        0 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    *flags = 0 as ::core::ffi::c_int;
    *fileType = IIFILE_MRC;
    *fileSize = -(1 as ::core::ffi::c_int);
    if u.is_null() {
        return;
    }
    if (*(*u).ii_file).file == IIFILE_SHR_MEM {
        *fileSize = (iiShrMemCheckSize((*(*u).ii_file).filename) as ::core::ffi::c_double
            / 1024.0f64) as ::core::ffi::c_int;
    } else {
        stat((*(*u).ii_file).filename, &raw mut buf);
        *fileSize = (buf.st_size as ::core::ffi::c_double / 1024.0f64) as ::core::ffi::c_int;
    }
    *fileType = (*(*u).ii_file).file;
    *flags = ((*(*u).header).iiu_flags as ::core::ffi::c_long
        | (if (*(*u).header).swapped != 0 {
            IIUNIT_SWAPPED
        } else {
            0 as ::core::ffi::c_long
        })
        | (if (*(*u).header).bytes_signed != 0 {
            IIUNIT_BYTES_SIGNED
        } else {
            0 as ::core::ffi::c_long
        })) as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiufileinfo_(
    mut iunit: *mut ::core::ffi::c_int,
    mut fileSize: *mut ::core::ffi::c_int,
    mut fileType: *mut ::core::ffi::c_int,
    mut flags: *mut ::core::ffi::c_int,
) {
    iiu_file_info(*iunit, fileSize, fileType, flags);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_exit_on_error(
    mut doExit: ::core::ffi::c_int,
    mut storeError: ::core::ffi::c_int,
) {
    sExitOnError = doExit;
    sStoreError = storeError;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuexitonerror(
    mut doExit: *mut ::core::ffi::c_int,
    mut storeError: *mut ::core::ffi::c_int,
) {
    iiu_exit_on_error(*doExit, *storeError);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_get_exit_on_error() -> ::core::ffi::c_int {
    return sExitOnError;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ialbrief_(mut val: *mut ::core::ffi::c_int) {
    sBriefHeader = *val;
}
/// Matches C `iiuAltBrief` (`unit_fileio.c`), used directly by translated
/// Fortran program units rather than through their underscore ABI wrapper.
pub unsafe fn iiu_alt_brief(val: i32) {
    sBriefHeader = val;
}
/// Matches C `iiuRetBrief` (`unit_fileio.c`).
pub unsafe fn iiu_ret_brief() -> i32 {
    if sBriefHeader >= 0 {
        sBriefHeader
    } else if !getenv(b"IMOD_BRIEF_HEADER\0" as *const u8 as *const ::core::ffi::c_char).is_null() {
        1
    } else {
        0
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretbrief_() -> ::core::ffi::c_int {
    iiu_ret_brief()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiualtprint_(mut val: *mut ::core::ffi::c_int) {
    iiu_alt_print(*val);
}
/// Matches C `iiuAltPrint` (`unit_fileio.c`).
pub unsafe fn iiu_alt_print(val: i32) {
    sPrintHeader = val;
}
/// Matches C `iiuRetPrint` (`unit_fileio.c`).
pub unsafe fn iiu_ret_print() -> i32 {
    sPrintHeader
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiuretprint_() -> ::core::ffi::c_int {
    iiu_ret_print()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_alt_convert(
    mut iunit: ::core::ffi::c_int,
    mut val: ::core::ffi::c_int,
) {
    remove_unit_from_list(sNoConvList, iunit);
    if val == 0 as ::core::ffi::c_int {
        add_unit_to_list(&raw mut sNoConvList, iunit);
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiualtconvert_(
    mut iunit: *mut ::core::ffi::c_int,
    mut val: *mut ::core::ffi::c_int,
) {
    iiu_alt_convert(*iunit, *val);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiallowmultivolume_(mut allow: *mut ::core::ffi::c_int) {
    iiAllowMultiVolume(*allow);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_mrc_header(
    mut iunit: ::core::ffi::c_int,
    mut function: *const ::core::ffi::c_char,
    mut doExit: ::core::ffi::c_int,
    mut checkRW: ::core::ffi::c_int,
) -> *mut MrcHeader {
    let mut u: *mut Unit = lookup_unit(iunit, function, doExit, checkRW);
    if u.is_null() {
        return ::core::ptr::null_mut::<MrcHeader>();
    }
    return (*u).header;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_sync_with_mrc_header(mut iunit: ::core::ffi::c_int) {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_sync_with_mrc_header\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    iiSyncFromMrcHeader((*u).ii_file, (*u).header);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_reassign_header_ptr(mut iunit: ::core::ffi::c_int) {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_reassign_header_ptr\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    if (*(*u).ii_file).file != IIFILE_HDF {
        printf(
            b"ERROR: iiu_reassign_header_ptr - File on unit %d is not HDF\n\0" as *const u8
                as *const ::core::ffi::c_char,
            iunit,
        );
        exit(1 as ::core::ffi::c_int);
    }
    (*u).header = (*(*u).ii_file).header as *mut MrcHeader;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_file_type(mut iunit: ::core::ffi::c_int) -> ::core::ffi::c_int {
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_file_type\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    return (*(*u).ii_file).file;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiufiletype_(mut iunit: *mut ::core::ffi::c_int) -> ::core::ffi::c_int {
    return iiu_file_type(*iunit);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iisettifftagtoprint_(
    mut tag: *mut ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    tiffSetStringTagToPrint(*tag);
    panic!("Reached end of non-void function without returning");
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_buf_bytes_per_pixel(
    mut iunit: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut dsize: ::core::ffi::c_int = 0;
    let mut csize: ::core::ffi::c_int = 0;
    let mut u: *mut Unit = lookup_unit(
        iunit,
        b"iiu_file_type\0" as *const u8 as *const ::core::ffi::c_char,
        1 as ::core::ffi::c_int,
        0 as ::core::ffi::c_int,
    );
    if (*u).no_convert != 0 {
        mrc_getdcsize((*(*u).ii_file).mode, &mut dsize, &mut csize);
        return dsize * csize;
    }
    return 4 as ::core::ffi::c_int;
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn move_(
    mut a: *mut ::core::ffi::c_char,
    mut b: *mut ::core::ffi::c_char,
    mut n: *mut ::core::ffi::c_int,
) {
    mybcopy(a, b, *n);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zero_(mut a: *mut ::core::ffi::c_char, mut n: *mut ::core::ffi::c_int) {
    memset(
        a as *mut ::core::ffi::c_void,
        0 as ::core::ffi::c_int,
        *n as size_t,
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn iiu_memory_error(
    mut ptr: *mut ::core::ffi::c_void,
    mut message: *const ::core::ffi::c_char,
) {
    if !ptr.is_null() {
        return;
    }
    fprintf(
        stdout,
        b"\n%s\n\0" as *const u8 as *const ::core::ffi::c_char,
        message,
    );
    exit(1 as ::core::ffi::c_int);
}
unsafe extern "C" fn find_new_unit(mut iunit: ::core::ffi::c_int) -> *mut Unit {
    let mut newUnit: Unit = Unit {
        ii_file: ::core::ptr::null_mut::<ImodImageFile>(),
        header: ::core::ptr::null_mut::<MrcHeader>(),
        current_sec: 0,
        current_line: 0,
        tail_name: ::core::ptr::null_mut::<::core::ffi::c_char>(),
        attribute: 0,
        being_used: 0,
        read_only: 0,
        no_convert: 0,
    };
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    let mut i: ::core::ffi::c_int = 0;
    let mut newSize: ::core::ffi::c_int = 0;
    if sMapSize == 0 {
        b3dSetStoreError(sStoreError);
    }
    if iunit <= 0 as ::core::ffi::c_int || iunit > MAX_UNIT {
        b3d_error(
            stdout.cast(),
            format_args!(
                "ERROR: iiu_open - A unit number of {} is out of range\n",
                iunit
            ),
        );
        exit(1 as ::core::ffi::c_int);
    }
    if iunit > sMapSize {
        newSize = iunit + 9 as ::core::ffi::c_int;
        if sMapSize == 0 {
            sUnitMap = malloc(
                (newSize as size_t)
                    .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
            ) as *mut ::core::ffi::c_int;
        } else {
            sUnitMap = realloc(
                sUnitMap as *mut ::core::ffi::c_void,
                (newSize as size_t)
                    .wrapping_mul(::core::mem::size_of::<::core::ffi::c_int>() as size_t),
            ) as *mut ::core::ffi::c_int;
        }
        if sUnitMap.is_null() {
            return ::core::ptr::null_mut::<Unit>();
        }
        i = sMapSize;
        while i < newSize {
            *sUnitMap.offset(i as isize) = -(1 as ::core::ffi::c_int);
            i += 1;
        }
        sMapSize = newSize;
    }
    if *sUnitMap.offset((iunit - 1 as ::core::ffi::c_int) as isize) >= 0 as ::core::ffi::c_int {
        b3d_error(
            stdout.cast(),
            format_args!(
                "WARNING: iiu_open - Unit number {} is already in use; closing it\n",
                iunit
            ),
        );
        iiu_close(iunit);
    }
    if sUnitList.is_null() {
        sUnitList = ilistNew(
            ::core::mem::size_of::<Unit>() as ::core::ffi::c_int,
            4 as ::core::ffi::c_int,
        );
    }
    if sUnitList.is_null() {
        return ::core::ptr::null_mut::<Unit>();
    }
    i = 0 as ::core::ffi::c_int;
    while i < ilistSize(sUnitList) {
        u = ilistItem(sUnitList, i) as *mut Unit;
        if (*u).being_used == 0 {
            *sUnitMap.offset((iunit - 1 as ::core::ffi::c_int) as isize) = i;
            return u;
        }
        i += 1;
    }
    newUnit.being_used = 0 as ::core::ffi::c_uchar;
    if ilistAppend(sUnitList, &raw mut newUnit as *mut ::core::ffi::c_void) != 0 {
        return ::core::ptr::null_mut::<Unit>();
    }
    *sUnitMap.offset((iunit - 1 as ::core::ffi::c_int) as isize) =
        ilistSize(sUnitList) - 1 as ::core::ffi::c_int;
    return ilistItem(sUnitList, ilistSize(sUnitList) - 1 as ::core::ffi::c_int) as *mut Unit;
}
unsafe extern "C" fn mybcopy(
    mut a: *mut ::core::ffi::c_char,
    mut b: *mut ::core::ffi::c_char,
    mut n: ::core::ffi::c_int,
) {
    loop {
        let fresh0 = n;
        n = n - 1;
        if !(fresh0 != 0) {
            break;
        }
        let fresh1 = b;
        b = b.offset(1);
        let fresh2 = a;
        a = a.offset(1);
        *fresh2 = *fresh1;
    }
}
unsafe extern "C" fn lookup_unit(
    mut unit: ::core::ffi::c_int,
    mut function: *const ::core::ffi::c_char,
    mut doExit: ::core::ffi::c_int,
    mut checkRW: ::core::ffi::c_int,
) -> *mut Unit {
    let mut u: *mut Unit = ::core::ptr::null_mut::<Unit>();
    if unit <= 0 as ::core::ffi::c_int || unit > MAX_UNIT {
        b3d_error(
            stdout.cast(),
            format_args!(
                "\nERROR: {} - {} is not a legal unit number.\n",
                core::ffi::CStr::from_ptr(function).to_string_lossy(),
                unit
            ),
        );
        return exit_or_null(doExit) as *mut Unit;
    }
    if unit <= sMapSize
        && *sUnitMap.offset((unit - 1 as ::core::ffi::c_int) as isize) >= 0 as ::core::ffi::c_int
    {
        u = ilistItem(
            sUnitList,
            *sUnitMap.offset((unit - 1 as ::core::ffi::c_int) as isize),
        ) as *mut Unit;
        (*u).no_convert = is_unit_on_list(sNoConvList, unit) as ::core::ffi::c_uchar;
        if (*(*u).ii_file).format == IIFORMAT_COMPLEX {
            (*u).no_convert = 1 as ::core::ffi::c_uchar;
        }
        if (*u).being_used != 0 {
            if checkRW > 1 as ::core::ffi::c_int && (*u).read_only as ::core::ffi::c_int != 0 {
                b3d_error(
                    stdout.cast(),
                    format_args!(
                        "\nERROR: {} - Trying to write to unit {}, which was opened read-only.\n",
                        core::ffi::CStr::from_ptr(function).to_string_lossy(),
                        unit
                    ),
                );
                return exit_or_null(doExit) as *mut Unit;
            }
            if checkRW == 1 as ::core::ffi::c_int
                && ((*u).no_convert as ::core::ffi::c_int != 0
                    && (*(*u).ii_file).read_section.is_none()
                    || (*u).no_convert == 0 && (*(*u).ii_file).read_section_float.is_none())
            {
                b3d_error(
                    stdout.cast(),
                    format_args!(
                        "\nERROR: {} - There is no function for reading {} from the type of file on unit {}.\n",
                        core::ffi::CStr::from_ptr(function).to_string_lossy(),
                        if (*u).no_convert != 0 {
                            "raw data"
                        } else {
                            "floats"
                        },
                        unit
                    ),
                );
                return exit_or_null(doExit) as *mut Unit;
            }
            if checkRW > 1 as ::core::ffi::c_int
                && ((*u).no_convert as ::core::ffi::c_int != 0
                    && (*(*u).ii_file).write_section.is_none()
                    || (*u).no_convert == 0 && (*(*u).ii_file).write_section_float.is_none())
            {
                b3d_error(
                    stdout.cast(),
                    format_args!(
                        "\nERROR: {} - There is no function for writing {} to the type of file on unit {}.\n",
                        core::ffi::CStr::from_ptr(function).to_string_lossy(),
                        if (*u).no_convert != 0 {
                            "raw data"
                        } else {
                            "floats"
                        },
                        unit
                    ),
                );
                return exit_or_null(doExit) as *mut Unit;
            }
            return u;
        }
    }
    b3d_error(
        stdout.cast(),
        format_args!(
            "\nERROR: {} - unit {} is not open.\n",
            core::ffi::CStr::from_ptr(function).to_string_lossy(),
            unit
        ),
    );
    return exit_or_null(doExit) as *mut Unit;
}
unsafe extern "C" fn exit_or_null(mut doExit: ::core::ffi::c_int) -> *mut ::core::ffi::c_void {
    if doExit != 0 {
        exit(3 as ::core::ffi::c_int);
    }
    return NULL;
}
unsafe extern "C" fn add_unit_to_list(mut list: *mut *mut Ilist, mut unit: ::core::ffi::c_int) {
    let mut val: b3dInt16 = unit as b3dInt16;
    if (*list).is_null() {
        *list = ilistNew(
            ::core::mem::size_of::<b3dInt16>() as ::core::ffi::c_int,
            4 as ::core::ffi::c_int,
        );
    }
    iiu_memory_error(
        *list as *mut ::core::ffi::c_void,
        b"Allocating a unit list\0" as *const u8 as *const ::core::ffi::c_char,
    );
    if ilistAppend(*list, &raw mut val as *mut ::core::ffi::c_void) != 0 {
        iiu_memory_error(
            NULL,
            b"Appending to a unit list\0" as *const u8 as *const ::core::ffi::c_char,
        );
    }
}
unsafe extern "C" fn remove_unit_from_list(mut list: *mut Ilist, mut unit: ::core::ffi::c_int) {
    let mut valp: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut i: ::core::ffi::c_int = 0;
    i = 0 as ::core::ffi::c_int;
    while i < ilistSize(list) {
        valp = ilistItem(list, i) as *mut b3dInt16;
        if *valp as ::core::ffi::c_int == unit {
            ilistRemove(list, i);
            return;
        }
        i += 1;
    }
}
unsafe extern "C" fn is_unit_on_list(
    mut list: *mut Ilist,
    mut unit: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    let mut valp: *mut b3dInt16 = ::core::ptr::null_mut::<b3dInt16>();
    let mut i: ::core::ffi::c_int = 0;
    i = 0 as ::core::ffi::c_int;
    while i < ilistSize(list) {
        valp = ilistItem(list, i) as *mut b3dInt16;
        if *valp as ::core::ffi::c_int == unit {
            return 1 as ::core::ffi::c_int;
        }
        i += 1;
    }
    return 0 as ::core::ffi::c_int;
}

#[cfg(test)]
mod tests {
    use super::{move_, zero_};

    #[test]
    fn move_and_zero_wrappers_preserve_requested_byte_count() {
        unsafe {
            let mut source = *b"abcdef";
            let mut destination = [0_i8; 6];
            let mut count = 4;
            move_(
                destination.as_mut_ptr(),
                source.as_mut_ptr().cast(),
                &mut count,
            );
            assert_eq!(
                &destination[..4],
                &[b'a' as i8, b'b' as i8, b'c' as i8, b'd' as i8]
            );
            zero_(destination.as_mut_ptr(), &mut count);
            assert_eq!(&destination[..4], &[0; 4]);
            assert_eq!(destination[4], 0);
        }
    }
}
