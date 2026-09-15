//! Translation of `IMOD/libiimod/iilikemrc.c` — check for recognizable formats
//! that can be read like MRC.
//!
//! Each C definition is retained as one systematic snake-case Rust function,
//! with the original C identifier named in its doc comment.
#![allow(dead_code)]

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, SEEK_END, SEEK_SET, b3d_error, b3d_fread, b3d_fseek, b3d_i_min, b3d_rewind,
    c_format_bytes, wall_time,
};
pub use crate::imod::libiimod::iimage::ImodImageFile;
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_IO_ERROR, IIERR_NO_SUPPORT, IIERR_NOT_FORMAT, IIFILE_RAW,
    IiRawCheckFunction, RawImageInfo, ii_sync_from_mrc_header,
};
use crate::imod::libiimod::iimrc::{ii_mrc_mode_to_format_type, ii_mrc_set_io_funcs};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT,
    MRC_MODE_USHORT, MrcHeader, mrc_head_new, mrc_set_scale, mrc_swap_longs, mrc_swap_shorts,
};
use std::cell::{Cell, RefCell};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;

/// C `MAX_EM_MACHINES` (`iilikemrc.c:23`).
pub const MAX_EM_MACHINES: i32 = 20;
/// C `MAX_EM_TYPES` (`iilikemrc.c:24`).
pub const MAX_EM_TYPES: i32 = 20;
/// C `MAX_EM_SIZE` (`iilikemrc.c:25`).
pub const MAX_EM_SIZE: f64 = 1.6e10;

/// C `RAW_MODE_SBYTE` (`iimage.h`).
pub const RAW_MODE_SBYTE: i32 = 0;
/// C `RAW_MODE_BYTE` (`iimage.h`).
pub const RAW_MODE_BYTE: i32 = 1;
/// C `RAW_MODE_SHORT` (`iimage.h`).
pub const RAW_MODE_SHORT: i32 = 2;
/// C `RAW_MODE_USHORT` (`iimage.h`).
pub const RAW_MODE_USHORT: i32 = 3;
/// C `RAW_MODE_FLOAT` (`iimage.h`).
pub const RAW_MODE_FLOAT: i32 = 4;

/// C `CheckEntry` (`iilikemrc.c:39-42`): `IIRawCheckFunction func; char *name;`.
///
/// `name` is the source's `strdup(name)`, owned by the entry and released when
/// the list is dropped, which is what `iiDeleteRawCheckList`'s `free` does.
#[derive(Clone)]
struct CheckEntry {
    func: IiRawCheckFunction,
    name: Vec<u8>,
}

thread_local! {
    /// C `static Ilist *checkList` (`iilikemrc.c:36`).
    ///
    /// A `Vec<CheckEntry>` rather than an [`crate::imod::libcfshr::ilist::Ilist`]:
    /// `Ilist` addresses its elements as raw bytes, and a `CheckEntry` now owns
    /// its `name`, so a bitwise copy into that storage would duplicate the
    /// owner (NATIVE.md §4d.3).  The `Option` is the source's NULL-versus-
    /// allocated distinction, which `initCheckList` and `iiDeleteRawCheckList`
    /// both test; nothing outside this file can observe the list's growth
    /// quantum, which is the only other thing `ilistNew(sizeof(CheckEntry), 6)`
    /// decided.
    static CHECK_LIST: RefCell<Option<Vec<CheckEntry>>> = const { RefCell::new(None) };
}

/// Original `initCheckList` (`iilikemrc.c:46`).
///
/// Initialize check list: if it does not exist, allocate it and place resident
/// functions on it.
fn init_check_list() -> i32 {
    if CHECK_LIST.with_borrow(|list| list.is_some()) {
        return 0;
    }
    // `ilistNew(sizeof(CheckEntry), 6)`; a `Vec` cannot fail to be created, so
    // the source's `if (!checkList) return 1;` has no reachable arm here.
    CHECK_LIST.with_borrow_mut(|list| *list = Some(Vec::new()));
    ii_add_raw_check_function(Some(check_em), b"EM");
    ii_add_raw_check_function(Some(check_dm3), b"DM3");
    ii_add_raw_check_function(Some(check_fei_raw), b"FEIraw");
    ii_add_raw_check_function(Some(check_winkler), b"Winkler");
    ii_add_raw_check_function(Some(check_pif), b"PIF");
    0
}

/// Original `iiAddRawCheckFunction` (`iilikemrc.c:66`).
///
/// Add the given raw-type checking function `func` to the front of the checking
/// list; `name` is a name for the format.
pub fn ii_add_raw_check_function(func: IiRawCheckFunction, name: &[u8]) {
    let mut item = CheckEntry {
        func: None,
        name: Vec::new(),
    };
    item.func = func;
    item.name = name.to_vec();
    if init_check_list() != 0 {
        return;
    }
    CHECK_LIST.with_borrow_mut(|list| {
        if let Some(list) = list.as_mut() {
            list.insert(0, item);
        }
    });
}

/// Original `iiDeleteRawCheckList` (`iilikemrc.c:79`).
///
/// Frees the checking list and all its data to avoid memory leaks.
pub fn ii_delete_raw_check_list() {
    CHECK_LIST.with_borrow_mut(|list| {
        if list.is_none() {
            return;
        }
        // The source's loop over the items `free`ing each `name`, then
        // `ilistDelete`: dropping the vector releases both.
        *list = None;
    });
}

/// Original `iiLikeMRCCheck` (`iilikemrc.c:100`).
///
/// Checks the image file in `inFile` for one known MRC-like (raw-type) format
/// after another.  Returns IIERR codes for errors.  `b3dError` is called with a
/// message for all errors that occur during checking, except for
/// `IIERR_NOT_FORMAT`.
pub unsafe fn ii_like_mrc_check(in_file: *mut ImodImageFile) -> i32 {
    let Some(in_file) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let mut info = RawImageInfo::default();
    // `fp = inFile->fp` copies the handle by value, as the C does.
    let Some(mut fp) = in_file.fp.clone() else {
        return IIERR_BAD_CALL;
    };
    if init_check_list() != 0 {
        return IIERR_BAD_CALL;
    }

    let mut i: i32 = 0;
    while i < CHECK_LIST.with_borrow(|list| list.as_ref().map_or(0, |list| list.len() as i32)) {
        // `ilistItem(checkList, i)`; the entry is cloned out so that the list is
        // not borrowed across the check function's call.
        let Some(item) = CHECK_LIST
            .with_borrow(|list| list.as_ref().and_then(|list| list.get(i as usize).cloned()))
        else {
            break;
        };

        // `inFile->filename`, which is NULL only where the caller never set it;
        // the C would then pass NULL to the checker and on to `stat`.
        let filename = in_file.filename.as_deref().unwrap_or("").as_bytes();
        let err = (item.func.unwrap())(&mut fp, filename, &mut info);
        if err == 0 {
            return ii_setup_raw_headers(in_file, &info);
        }

        if err != IIERR_NOT_FORMAT {
            if err == IIERR_IO_ERROR {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiCheckLikeMRC - reading from file {}\n",
                        String::from_utf8_lossy(filename)
                    ),
                );
            } else if err == IIERR_NO_SUPPORT {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "ERROR: iiCheckLikeMRC - unsupported data mode of {}-type file.\n",
                        String::from_utf8_lossy(&item.name)
                    ),
                );
            }
            return err;
        }
        i += 1;
    }

    IIERR_NOT_FORMAT
}

/// Original `iiSetupRawHeaders` (`iilikemrc.c:148`).
///
/// Creates an MRC header and fills it and the items in `inFile` from the
/// information in `info`; specifically the `nx`, `ny`, `nz`, `swapBytes`,
/// `headerSize`, `sectionSkip`, `yInverted`, and `type` members.
pub fn ii_setup_raw_headers(in_file: &mut ImodImageFile, info: &RawImageInfo) -> i32 {
    let mode_table: [i32; 7] = [
        MRC_MODE_BYTE,
        MRC_MODE_BYTE,
        MRC_MODE_SHORT,
        MRC_MODE_USHORT,
        MRC_MODE_FLOAT,
        MRC_MODE_COMPLEX_FLOAT,
        MRC_MODE_RGB,
    ];

    /* Get an MRC header; set sizes into that header and the iifile header */
    // The crate owns the header slot; the erased `header` field remains only
    // the callback ABI alias for this MRC-compatible backend.
    in_file.mrc_header = Some(Box::new(MrcHeader::default()));
    let file_handle = in_file.fp.clone();
    let (mode, bytes_signed) = {
        let hdr = in_file
            .mrc_header
            .as_deref_mut()
            .expect("header just installed");
        mrc_head_new(
            hdr,
            info.nx,
            info.ny,
            info.nz,
            mode_table[info.type_ as usize],
        );
        hdr.swapped = info.swap_bytes;
        hdr.header_size = info.header_size;
        hdr.section_skip = info.section_skip;
        hdr.y_inverted = info.y_inverted;
        hdr.bytes_signed = if info.type_ == RAW_MODE_SBYTE { 1 } else { 0 };
        hdr.packed4bits = 0;
        hdr.half_floats = 0;
        hdr.fp = file_handle;

        /* Pass on a min and max of 0 as a sign that there is no min/max */
        hdr.amin = info.amin;
        hdr.amax = info.amax;
        hdr.amean = ((info.amin + info.amax) as f64 / 2.) as f32;
        if info.pixel != 0. {
            mrc_set_scale(
                hdr,
                info.pixel as f64,
                info.pixel as f64,
                (if info.z_pixel != 0. {
                    info.z_pixel
                } else {
                    info.pixel
                }) as f64,
            );
        }
        (hdr.mode, hdr.bytes_signed)
    };
    in_file.file = IIFILE_RAW;
    ii_mrc_mode_to_format_type(in_file, mode, bytes_signed);
    let mut hdr = in_file.mrc_header.take().expect("header remains installed");
    ii_sync_from_mrc_header(in_file, &mut hdr);
    in_file.mrc_header = Some(hdr);

    /* Set the access routines; just use the MRC routines */
    unsafe { ii_mrc_set_io_funcs(in_file, 1) };
    in_file.clean_up = Some(ii_like_mrc_delete);
    0
}

/// Original `iiLikeMRCDelete` (`iilikemrc.c:188`).
pub unsafe fn ii_like_mrc_delete(in_file: *mut ImodImageFile) {
    unsafe {
        (*in_file).mrc_header = None;
    }
}

/// Original `checkWinkler` (`iilikemrc.c:197`).
///
/// Check for the Winkler format.
///
/// The source declares `b3dUInt16 svals[4]` and hands it to `mrc_swap_shorts`
/// through a `(b3dInt16 *)` cast; the array is held signed here so that the
/// swap takes it directly, and every *use* converts back with `as u16` to keep
/// the source's unsigned semantics.
fn check_winkler(fp: &mut ImodFile, _filename: &[u8], info: &mut RawImageInfo) -> i32 {
    let mut sbuf = [0u8; 8];
    let mut ibuf = [0u8; 48];

    b3d_rewind(fp);
    if b3d_fread(&mut sbuf[..4], 2, 2, fp) != 2 {
        return IIERR_IO_ERROR;
    }
    let mut svals: [i16; 4] = [
        i16::from_ne_bytes([sbuf[0], sbuf[1]]),
        i16::from_ne_bytes([sbuf[2], sbuf[3]]),
        i16::from_ne_bytes([sbuf[4], sbuf[5]]),
        i16::from_ne_bytes([sbuf[6], sbuf[7]]),
    ];

    info.swap_bytes = 0;
    if svals[0] as u16 as i32 != 18739 || svals[1] as u16 as i32 != 20480 {
        mrc_swap_shorts(&mut svals, 2);
        if svals[0] as u16 as i32 != 18739 || svals[1] as u16 as i32 != 20480 {
            return IIERR_NOT_FORMAT;
        }
        info.swap_bytes = 1;
    }

    if b3d_fseek(fp, 16, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }
    if b3d_fread(&mut sbuf, 2, 4, fp) != 4 {
        return IIERR_IO_ERROR;
    }
    svals = [
        i16::from_ne_bytes([sbuf[0], sbuf[1]]),
        i16::from_ne_bytes([sbuf[2], sbuf[3]]),
        i16::from_ne_bytes([sbuf[4], sbuf[5]]),
        i16::from_ne_bytes([sbuf[6], sbuf[7]]),
    ];
    if info.swap_bytes != 0 {
        mrc_swap_shorts(&mut svals, 4);
    }
    if svals[0] as u16 != 0 {
        return IIERR_NO_SUPPORT;
    }
    match svals[1] as u16 as i32 {
        2 => {
            info.type_ = RAW_MODE_BYTE;
        }
        3 => {
            info.type_ = RAW_MODE_SHORT;
        }
        15 => {
            info.type_ = RAW_MODE_USHORT;
        }
        5 => {
            info.type_ = RAW_MODE_FLOAT;
        }
        _ => {
            return IIERR_NO_SUPPORT;
        }
    }

    /* Dimension must be 2 or 3 */
    if (svals[3] as u16) / 2 != 1 {
        return IIERR_NO_SUPPORT;
    }
    if b3d_fseek(fp, 24, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }

    if b3d_fread(&mut ibuf[..4], 4, 1, fp) != 1 {
        return IIERR_IO_ERROR;
    }
    let mut ivals = [0i32; 12];
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }
    if info.swap_bytes != 0 {
        mrc_swap_longs(&mut ivals, 1);
    }
    info.header_size = ivals[0];
    if b3d_fseek(fp, 64, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }
    let count = (svals[3] as u16 as i32 * 4) as usize;
    if b3d_fread(&mut ibuf[..4 * count], 4, count, fp) != count {
        return IIERR_IO_ERROR;
    }
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }
    if info.swap_bytes != 0 {
        mrc_swap_longs(&mut ivals, 12);
    }
    if svals[3] as u16 as i32 == 3 {
        info.nx = ivals[10];
        info.ny = ivals[6];
        info.nz = ivals[2];
    } else {
        info.nx = ivals[6];
        info.ny = ivals[2];
        info.nz = 1;
    }

    /* Set these to signal that the range is unknown */
    info.amin = 0.;
    info.amax = 0.;
    0
}

/// Original `checkPif` (`iilikemrc.c:276`).
///
/// Check for the pif format.
fn check_pif(fp: &mut ImodFile, _filename: &[u8], info: &mut RawImageInfo) -> i32 {
    let mut ibuf = [0u8; 48];
    let mut cvals = [0u8; 6];

    /* is it a pif file (only reading bsoft pif files) */
    if b3d_fseek(fp, 32, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }

    if b3d_fread(&mut cvals[..5], 1, 5, fp) != 5 {
        return IIERR_IO_ERROR;
    }

    /* recognize file type  */
    cvals[5] = 0;
    // `strcmp(cvals, "Bsoft")`: with the terminator written at `cvals[5]` and
    // no NUL inside "Bsoft", that is exactly a comparison of the five bytes.
    if cvals[..5] != *b"Bsoft" {
        return IIERR_NOT_FORMAT;
    }

    info.header_size = 1024;
    info.section_skip = 512;

    /* set swapBytes */
    if b3d_fseek(fp, 28, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }

    if b3d_fread(&mut ibuf[..4], 4, 1, fp) != 1 {
        return IIERR_IO_ERROR;
    }
    let mut ivals = [0i32; 12];
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }

    info.swap_bytes = 0;

    #[cfg(target_endian = "little")]
    if ivals[0] != 0 {
        info.swap_bytes = 1;
    }
    #[cfg(target_endian = "big")]
    if ivals[0] == 0 {
        info.swap_bytes = 1;
    }

    if b3d_fseek(fp, 24, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }

    if b3d_fread(&mut ibuf[..4], 4, 1, fp) != 1 {
        return IIERR_IO_ERROR;
    }
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }

    if info.swap_bytes != 0 {
        mrc_swap_longs(&mut ivals, 1);
    }

    /* set nz from numimages because nz is 1 */
    info.nz = ivals[0];

    if b3d_fseek(fp, 64, SEEK_SET) != 0 {
        return IIERR_IO_ERROR;
    }

    if b3d_fread(&mut ibuf[..20], 4, 5, fp) != 5 {
        return IIERR_IO_ERROR;
    }
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }

    if info.swap_bytes != 0 {
        mrc_swap_longs(&mut ivals, 5);
    }

    /* If there are images of different sizes and there is more then one image,
    fail. */
    if ivals[0] < 1 && info.nz > 1 {
        return IIERR_NOT_FORMAT;
    }

    info.nx = ivals[1];
    info.ny = ivals[2];

    match ivals[4] {
        0 | 6 => {
            info.type_ = RAW_MODE_BYTE;
        }
        1 | 7 | 20 | 88 => {
            info.type_ = RAW_MODE_SHORT;
        }
        9 => {
            info.type_ = RAW_MODE_FLOAT;
        }
        _ => {
            return IIERR_NO_SUPPORT;
        }
    }

    /* assume dimensions are 2 or 3 */

    /* Set these to signal that the range is unknown */
    info.amin = 0.;
    info.amax = 0.;
    0
}

thread_local! {
    /// C `static int sAssumeDMmatch` (`iilikemrc.c:369`).
    static S_ASSUME_DM_MATCH: Cell<i32> = const { Cell::new(0) };
    /// C `static RawImageInfo sLastDMinfo` (`iilikemrc.c:370`).
    static S_LAST_DM_INFO: RefCell<RawImageInfo> = RefCell::new(RawImageInfo {
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
    });
    /// C `static int sDMinfoSaved` (`iilikemrc.c:371`).
    static S_DM_INFO_SAVED: Cell<i32> = const { Cell::new(0) };
}

/// Original `iiAssumeDMfileMatches` (`iilikemrc.c:373`).
pub fn ii_assume_dmfile_matches(in_val: i32) {
    S_ASSUME_DM_MATCH.set(in_val);
}

/// C `DOC_CHECK_BUF` (`iilikemrc.c:379`): DocumentObjectList so far seen to end
/// before 208.
pub const DOC_CHECK_BUF: i32 = 832;

/// Original `checkDM3` (`iilikemrc.c:383`).
///
/// Check for the DigitalMicrograph format.
fn check_dm3(fp: &mut ImodFile, filename: &[u8], info: &mut RawImageInfo) -> i32 {
    let mut bvals = [0u8; DOC_CHECK_BUF as usize];
    let mut dmtype: i32 = 0;
    let test_string: &[u8] = b"DocumentObjectList";
    let test_len: i32 = test_string.len() as i32;

    /* Check for 3 or 4 in the fourth byte */
    b3d_rewind(fp);
    if b3d_fread(&mut bvals[..4], 1, 4, fp) != 4 {
        return IIERR_IO_ERROR;
    }
    if bvals[3] as i32 != 3 && bvals[3] as i32 != 4 {
        return IIERR_NOT_FORMAT;
    }
    let dmf: i32 = bvals[3] as i32;
    if b3d_fread(&mut bvals, 1, DOC_CHECK_BUF as usize, fp) != DOC_CHECK_BUF as usize {
        return IIERR_IO_ERROR;
    }

    /* Look for the test string */
    let mut err: i32 = 1;
    let mut i: i32 = 0;
    while i < DOC_CHECK_BUF - test_len - 4 {
        if bvals[i as usize] == test_string[0] {
            let bsave = bvals[(i + test_len) as usize];
            bvals[(i + test_len) as usize] = 0x00;
            // `strcmp(&bvals[i], testString)` with the terminator just written
            // and no NUL inside the tag: a comparison of `testLen` bytes.
            if bvals[i as usize..(i + test_len) as usize] == *test_string {
                err = 0;
                break;
            }
            bvals[(i + test_len) as usize] = bsave;
        }
        i += 1;
    }
    if err != 0 {
        return IIERR_NOT_FORMAT;
    }

    /* If a file was already opened and the flag is set to assume they match, just copy
    old info and return */
    if S_ASSUME_DM_MATCH.get() != 0 && S_DM_INFO_SAVED.get() != 0 {
        // `memcpy(info, &sLastDMinfo, sizeof(RawImageInfo))`.
        S_LAST_DM_INFO.with_borrow(|last| {
            *info = *last;
        });
        return 0;
    }

    err = analyze_dm3(fp, filename, dmf, info, &mut dmtype);
    if err != 0 {
        return err;
    }

    match dmtype {
        9 => {
            info.type_ = RAW_MODE_SBYTE;
        }
        6 => {
            info.type_ = RAW_MODE_BYTE;
        }
        1 => {
            info.type_ = RAW_MODE_SHORT;
        }
        10 => {
            info.type_ = RAW_MODE_USHORT;
        }
        2 => {
            info.type_ = RAW_MODE_FLOAT;
        }
        _ => {
            return IIERR_NO_SUPPORT;
        }
    }

    info.amin = 0.;
    info.amax = 0.;
    // `memcpy(&sLastDMinfo, info, sizeof(RawImageInfo))`.
    S_LAST_DM_INFO.with_borrow_mut(|last| *last = *info);
    S_DM_INFO_SAVED.set(1);
    0
}

/// C `BUFSIZE` (`iilikemrc.c:457`).
///
/// 8/3/09: This was 160000, but a file with Data%%%% at 595098 turned up.
pub const BUFSIZE: usize = 1000000;
/// C `MAX_TYPES` (`iilikemrc.c:458`).
pub const MAX_TYPES: i32 = 13;

thread_local! {
    /// C function-static `lastMaxRead` (`iilikemrc.c:485`).
    static LAST_MAX_READ: Cell<i64> = const { Cell::new(0) };
    /// C function-static `debug` (`iilikemrc.c:489`).
    static DEBUG: Cell<i32> = const { Cell::new(-1) };
    /// C function-statics `lastDataType`, `lastDimensions`, `lastData`,
    /// `lastCalibrations` (`iilikemrc.c:490`).
    static LAST_DATA_TYPE: Cell<i32> = const { Cell::new(0) };
    static LAST_DIMENSIONS: Cell<i32> = const { Cell::new(0) };
    static LAST_DATA: Cell<i32> = const { Cell::new(0) };
    static LAST_CALIBRATIONS: Cell<i32> = const { Cell::new(0) };
    /// C function-statics `lastTabDimens`, `lastDimensInfo`, `lastUnits`,
    /// `lastOffset` (`iilikemrc.c:491`).
    static LAST_TAB_DIMENS: Cell<i32> = const { Cell::new(0) };
    static LAST_DIMENS_INFO: Cell<i32> = const { Cell::new(0) };
    static LAST_UNITS: Cell<i32> = const { Cell::new(0) };
    static LAST_OFFSET: Cell<i32> = const { Cell::new(0) };
    /// C function-statics `lastZunits`, `lastMaxEndUsed`, `lastMaxStartUsed`
    /// (`iilikemrc.c:492`).
    static LAST_ZUNITS: Cell<i32> = const { Cell::new(0) };
    static LAST_MAX_END_USED: Cell<i32> = const { Cell::new(0) };
    static LAST_MAX_START_USED: Cell<i32> = const { Cell::new(0) };
}

/// Original `analyzeDM3` (`iilikemrc.c:467`).
///
/// Analyzes a file known to be a DigitalMicrograph version 3 or 4, as indicated
/// in `dmformat`; the file pointer is in `fp` and the filename in `filename`.
/// Returns size, type, and other information in `info`; specifically the `nx`,
/// `ny`, `nz`, `swapBytes`, `headerSize`, and `type` members.  Returns the DM
/// data type number in `dmtype`.  Returns `IIERR_IO_ERROR` for errors reading
/// the file or `IIERR_NO_SUPPORT` for other errors in analyzing the file.
pub fn analyze_dm3(
    fp: &mut ImodFile,
    filename: &[u8],
    dmformat: i32,
    info: &mut RawImageInfo,
    dmtype: &mut i32,
) -> i32 {
    let mut c: i32 = 0;
    let mut toffset: i32;
    let mut type_index: i32;
    // C `char buf[BUFSIZE]`, an uninitialised one-megabyte stack array.  A
    // `Vec` is zeroed where the C's is stack residue (NATIVE.md §4); only the
    // bytes actually read are used, and the source's own NUL writes below are
    // what `strstr` depends on.
    let mut buf = vec![0u8; BUFSIZE];
    let mut lowbyte: i32;
    let mut hibyte: i32;
    let mut loop_: i32;
    let mut max_use_c: i32;
    let mut match_last: i32 = 1;
    let mut offset: i32 = 0;
    let mut type_: i32 = -1;
    let mut xsize: i32 = 0;
    let mut ysize: i32 = 0;
    let mut zsize: i32 = 0;
    let mut got_cal: i32;
    let mut got_dim: i32;
    let mut got_scale: i32;
    let mut got_meta: i32;
    let mut got_dim_info: i32;
    // The source leaves `scale` and `tmpPixel` uninitialised; they are function
    // locals that persist across both scan loops, and the one path that reads
    // `tmpPixel` without having set it is a latent source bug.
    let mut scale: f32 = 0.;
    let mut tmp_pixel: f32 = 0.;
    let mut pixel: f32 = 0.;
    let mut z_pixel: f32 = 0.;

    /* off_t is only 32 bits in Windows!  So have to explicitly define the type for the
    offsets, and make sure the 64-bit stat is used to get 64-bit size */
    let mut type_offset: i64 = 0;
    let maxread: i64;
    let plausible_off: i64;
    let mut wall_start: f64 = 0.;
    let mut cur_data_type: i32;
    let mut cur_dimensions: i32;
    let mut cur_data: i32;
    let mut cur_calibrations: i32;
    let mut cur_tab_dimens: i32;
    let mut cur_dimens_info: i32;
    let mut cur_units: i32;
    let mut cur_zunits: i32;
    let mut cur_offset: i32 = 0;
    let mut max_cur_used: i32 = 0;

    /* The type-dependent values that were found after
    D a t a % % % % 0 0 0 3 0 0 0 24 0 0 0 */
    /*int datacode[MAX_TYPES] = {0, 2, 6, 0, 0, 0, 10, 3, 0, 9, 4, 5, 12}; */

    let data_size: [i32; MAX_TYPES as usize] = [1, 2, 4, 1, 1, 1, 1, 4, 1, 1, 2, 4, 8];
    let dimens_off: [i32; 2] = [15, 27];
    let xsize_off: [i32; 2] = [31, 59];
    let ysize_off: [i32; 2] = [50, 94];
    let zsize_off: [i32; 2] = [69, 129];
    let dtype_off: [i32; 2] = [20, 36];
    let data_off: [i32; 2] = [24, 48];
    let scale_off: [i32; 2] = [17, 33];
    let units_off: [i32; 2] = [25, 49];
    let max_offset: [i32; 2] = [90, 150]; /* Keep this higher than any offsets */

    if DEBUG.get() < 0 {
        DEBUG.set(if std::env::var_os("ANALYZEDM3_DEBUG").is_some() {
            1
        } else {
            0
        });
    }
    let debug = DEBUG.get();
    if debug != 0 {
        wall_start = wall_time();
    }

    let dmind: i32 = dmformat - 3;
    if dmind < 0 || dmind > 1 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: analyzeDM3 - DM format {} not supported\n", dmformat),
        );
        return IIERR_NO_SUPPORT;
    }

    let st_size: i64 = match std::fs::metadata(std::ffi::OsStr::from_bytes(filename)) {
        Ok(metadata) => metadata.len() as i64,
        Err(_) => {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: analyzeDM3 - Doing stat of {}\n",
                    String::from_utf8_lossy(filename)
                ),
            );
            return IIERR_IO_ERROR;
        }
    };

    maxread = if st_size - 2 < (BUFSIZE as i64 - 1) {
        st_size - 2
    } else {
        BUFSIZE as i64 - 1
    };
    if maxread != LAST_MAX_READ.get() || LAST_MAX_END_USED.get() == 0 {
        match_last = 0;
    }

    /* Initialize to big values so that the min of all can be taken even if some aren't
    found */
    cur_calibrations = (2 * maxread) as i32;
    cur_data = cur_calibrations;
    cur_dimensions = cur_data;
    cur_data_type = cur_dimensions;
    cur_zunits = (2 * maxread) as i32;
    cur_units = cur_zunits;
    cur_dimens_info = cur_units;
    cur_tab_dimens = cur_dimens_info;
    if debug != 0 {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "analyzeDM3: Reading up to %d bytes\n",
            &[CArg::Int(maxread as i32 as i64)],
        ));
    }

    /* Read the end of the file first because we need the data type
    before we can be sure we have the right Data%%%% entry */
    loop_ = 1 - match_last;
    while loop_ < 2 {
        offset = 0;
        xsize = 0;
        ysize = 0;
        type_ = -1;
        type_offset = 0;
        type_index = -1;

        /* The first time through loop, if matching last file is possible, just read and
        scan what is needed to find the tags in the same place */
        if loop_ != 0 {
            c = 0;
            max_use_c = maxread as i32;
        } else {
            c = if LAST_DIMENSIONS.get() < LAST_DATA_TYPE.get() {
                LAST_DIMENSIONS.get()
            } else {
                LAST_DATA_TYPE.get()
            };
            max_use_c = LAST_MAX_END_USED.get();
        }

        if b3d_fseek(fp, -((maxread - c as i64) + 1) as i32, SEEK_END) != 0 {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: analyzeDM3 - Seeking to end of {}\n",
                    String::from_utf8_lossy(filename)
                ),
            );
            return IIERR_IO_ERROR;
        }
        if b3d_fread(
            &mut buf[c as usize..max_use_c as usize],
            1,
            (max_use_c - c) as usize,
            fp,
        ) == 0
        {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: analyzeDM3 - Error Reading tail end of {}\n",
                    String::from_utf8_lossy(filename)
                ),
            );
            return IIERR_IO_ERROR;
        }
        buf[(max_use_c - c - 1) as usize] = 0x00;

        if debug != 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "analyzeDM3: file end loop %d, start at %d, read %d\n",
                &[
                    CArg::Int(loop_ as i64),
                    CArg::Int(c as i64),
                    CArg::Int((max_use_c - c) as i64),
                ],
            ));
        }

        /* Look past a DataType enough to see another Dimensions - it is supposed
        to be after it */
        while c < max_use_c
            && (xsize == 0 || type_ < 0 || c < type_index + 64 + (if dmind != 0 { 20 } else { 0 }))
        {
            /* Look for D, then check if it is Dimensions or DataType */
            if buf[c as usize] as i32 == 68 {
                // `strstr(&buf[c], "Dimensions")`: the search stops at the first
                // NUL, which the terminator written above guarantees exists.
                let hay = &buf[c as usize..];
                let hay = &hay[..hay.iter().position(|b| *b == 0).unwrap_or(hay.len())];
                let found = hay.windows(10).position(|w| w == b"Dimensions");
                if found.is_some() && c + ysize_off[dmind as usize] + 1 < max_use_c {
                    if loop_ == 0 && c != LAST_DIMENSIONS.get() {
                        match_last = 0;
                        break;
                    }
                    lowbyte = buf[(c + dimens_off[dmind as usize]) as usize] as i32;
                    if lowbyte == 3 {
                        /* break the scan if this is running off the end for either loop type */
                        if c + zsize_off[dmind as usize] + 1 >= max_use_c {
                            match_last = 0;
                            break;
                        }
                        lowbyte = buf[(c + zsize_off[dmind as usize]) as usize] as i32;
                        hibyte = buf[(c + zsize_off[dmind as usize] + 1) as usize] as i32;
                        zsize = lowbyte + 256 * hibyte;
                    } else if lowbyte == 2 {
                        zsize = 1;
                    } else {
                        b3d_error(
                            Some(&mut ImodFile::Stderr),
                            format_args!(
                                "ERROR: analyzeDM3 - The number of dimensions seemsto be {}, not 2 or 3, in {}\n",
                                lowbyte,
                                String::from_utf8_lossy(filename)
                            ),
                        );
                        return IIERR_NO_SUPPORT;
                    }
                    lowbyte = buf[(c + xsize_off[dmind as usize]) as usize] as i32;
                    hibyte = buf[(c + xsize_off[dmind as usize] + 1) as usize] as i32;
                    xsize = lowbyte + 256 * hibyte;
                    lowbyte = buf[(c + ysize_off[dmind as usize]) as usize] as i32;
                    hibyte = buf[(c + ysize_off[dmind as usize] + 1) as usize] as i32;
                    ysize = lowbyte + 256 * hibyte;
                    cur_dimensions = c;
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Found Dimensions at %d  x %d y %d z %d\n",
                            &[
                                CArg::Int(c as i64),
                                CArg::Int(xsize as i64),
                                CArg::Int(ysize as i64),
                                CArg::Int(zsize as i64),
                            ],
                        ));
                    }
                } else {
                    // `strstr(&buf[c], "DataType")`.
                    let found = hay.windows(8).position(|w| w == b"DataType");
                    // `buf[c + dtypeOff[dmind]]` is a C `char`, which is signed
                    // on this platform: the `< MAX_TYPES` test and the value
                    // stored in `type` are both sign-extended.
                    if found.is_some()
                        && c + dtype_off[dmind as usize] < max_use_c
                        && (buf[(c + dtype_off[dmind as usize]) as usize] as i8 as i32) < MAX_TYPES
                    {
                        if loop_ == 0 && c != LAST_DATA_TYPE.get() {
                            match_last = 0;
                            break;
                        }
                        type_ = buf[(c + dtype_off[dmind as usize]) as usize] as i8 as i32;
                        if type_offset == 0 {
                            type_offset = c as i64 + st_size - (maxread + 1);
                        }
                        type_index = c;
                        cur_data_type = type_index;
                        if debug != 0 {
                            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                "analyzeDM3: Found DataType at %d  type %d\n",
                                &[CArg::Int(c as i64), CArg::Int(type_ as i64)],
                            ));
                        }
                    }
                }
            }
            c += 1;
        }

        /* Break the outer loop if it still matches and size and type found */
        if match_last != 0 && xsize != 0 && ysize != 0 && type_ >= 0 {
            break;
        }
        loop_ += 1;
    }
    if xsize == 0 || ysize == 0 || type_ < 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: analyzeDM3 - Dimensions or type not found in {}\n",
                String::from_utf8_lossy(filename)
            ),
        );
        return IIERR_NO_SUPPORT;
    }
    LAST_MAX_END_USED.set(if maxread < (c + max_offset[dmind as usize]) as i64 {
        maxread as i32
    } else {
        c + max_offset[dmind as usize]
    });

    /* Now look for the Data string in the front of the file and pixel size */
    // `(off_t)(xsize * ysize)`: the product is formed as `int` and only then
    // widened, so it wraps exactly where the C's does.
    plausible_off = type_offset
        - (xsize.wrapping_mul(ysize)) as i64 * zsize as i64 * data_size[type_ as usize] as i64;
    loop_ = 1 - match_last;
    while loop_ < 2 {
        got_dim_info = 0;
        got_meta = got_dim_info;
        got_scale = got_meta;
        got_dim = got_scale;
        got_cal = got_dim;
        z_pixel = 0.;
        pixel = z_pixel;
        max_cur_used = maxread as i32;

        /* Try to read and scan only what is needed if things still match */
        if loop_ != 0 {
            c = 0;
            max_use_c = maxread as i32;
        } else {
            c = b3d_i_min(&[
                LAST_DATA_TYPE.get(),
                LAST_DIMENSIONS.get(),
                LAST_DATA.get(),
                LAST_CALIBRATIONS.get(),
                LAST_TAB_DIMENS.get(),
                LAST_DIMENS_INFO.get(),
                LAST_UNITS.get(),
                LAST_ZUNITS.get(),
            ]);
            max_use_c = LAST_MAX_START_USED.get();
        }

        b3d_fseek(fp, c, SEEK_SET);
        if b3d_fread(
            &mut buf[c as usize..max_use_c as usize],
            1,
            (max_use_c - c) as usize,
            fp,
        ) == 0
        {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: analyzeDM3 - Reading beginning of {}\n",
                    String::from_utf8_lossy(filename)
                ),
            );
            return IIERR_IO_ERROR;
        }
        buf[(max_use_c - 1) as usize] = 0x00;
        if debug != 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "analyzeDM3: file start loop %d, start at %d, read %d\n",
                &[
                    CArg::Int(loop_ as i64),
                    CArg::Int(c as i64),
                    CArg::Int((max_use_c - c) as i64),
                ],
            ));
        }

        while c < max_use_c {
            if buf[c as usize] as i32 == 68 {
                let found = found_dm_tag(&buf[c as usize..], b"Data", b"Data%%%%", dmind, 12);
                if let Some(found) = found {
                    toffset = (c as isize + found as isize + data_off[dmind as usize] as isize)
                        as i64 as i32;

                    /* If this is the first data string, or any data
                    string that could still be far enough in front of the datatype
                    string, save the offset */
                    if offset == 0 || toffset as i64 <= plausible_off {
                        if loop_ == 0 && (c != LAST_DATA.get() || toffset != LAST_OFFSET.get()) {
                            match_last = 0;
                            break;
                        }
                        offset = toffset;
                        cur_data = c;
                        cur_offset = offset;
                        max_cur_used = c;
                    }

                    /* It used to be done with code types but that turned out to be
                    unreliable */
                    /* And if the code type is appropriate, save the
                    offset and break out */
                    /*if (type <= 11 && buf[c + 19] == datacode[type]) {
                    offset = toffset;
                    break;
                    } */
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Found Data at %d  offset %d\n",
                            &[CArg::Int(c as i64), CArg::Int(toffset as i64)],
                        ));
                    }
                } else if got_meta != 0 && {
                    // `strstr(&buf[c], "Dimension info")`.
                    let hay = &buf[c as usize..];
                    let hay = &hay[..hay.iter().position(|b| *b == 0).unwrap_or(hay.len())];
                    hay.windows(14).any(|w| w == b"Dimension info")
                } {
                    got_dim_info = 1;
                    if loop_ == 0 && c != LAST_DIMENS_INFO.get() {
                        match_last = 0;
                        break;
                    }
                    cur_dimens_info = c;
                    max_cur_used = c;
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Setting gotDimInfo at %d\n",
                            &[CArg::Int(c as i64)],
                        ));
                    }
                }

            /* Always look for start of the Calibrations sequence */
            } else if buf[c as usize] as i32 == 67 {
                // `strstr(&buf[c], "Calibrations")`.
                let hay = &buf[c as usize..];
                let hay = &hay[..hay.iter().position(|b| *b == 0).unwrap_or(hay.len())];
                if hay.windows(12).any(|w| w == b"Calibrations") {
                    got_cal = 1;
                    got_dim_info = 0;
                    got_meta = got_dim_info;
                    got_scale = got_meta;
                    got_dim = got_scale;
                    if loop_ == 0 && c != LAST_CALIBRATIONS.get() {
                        match_last = 0;
                        break;
                    }
                    cur_calibrations = c;
                    max_cur_used = c;
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Found Calibrations at %d\n",
                            &[CArg::Int(c as i64)],
                        ));
                    }
                }

            /* Look for \tDimension if have Calibrations, or always look for Meta Data */
            } else if buf[c as usize] == b'\t' {
                let hay = &buf[c as usize..];
                let hay = &hay[..hay.iter().position(|b| *b == 0).unwrap_or(hay.len())];
                // `strstr(&buf[c], "\tDimension")`.
                if got_cal != 0 && got_dim == 0 && hay.windows(10).any(|w| w == b"\tDimension") {
                    got_dim = 1;
                    if loop_ == 0 && c != LAST_TAB_DIMENS.get() {
                        match_last = 0;
                        break;
                    }
                    cur_tab_dimens = c;
                    max_cur_used = c;
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Found tabDimension at %d\n",
                            &[CArg::Int(c as i64)],
                        ));
                    }
                }
                // `strstr(&buf[c], "\tMeta Data")`.
                if hay.windows(10).any(|w| w == b"\tMeta Data") {
                    got_meta = 1;
                    got_dim_info = 0;
                    got_scale = got_dim_info;
                    got_dim = got_scale;
                    got_cal = got_dim;
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Found tabMeta Data at %d\n",
                            &[CArg::Int(c as i64)],
                        ));
                    }
                }

            /* Look for Scale if have Dimension or Dimension info */
            } else if (got_dim != 0 || got_dim_info != 0)
                && got_scale == 0
                && buf[c as usize] == b'S'
            {
                let found = found_dm_tag(&buf[c as usize..], b"Scale", b"Scale%%%%", dmind, 13);
                if found.is_some() && c + scale_off[dmind as usize] + 3 < max_use_c {
                    /* Always copy a scale over but do not keep track of where, because there may
                    be two good scales */
                    got_scale = 1;
                    // `memcpy(&scale, &buf[c + scaleOff[dmind]], 4)`.
                    let base = (c + scale_off[dmind as usize]) as usize;
                    scale = f32::from_ne_bytes([
                        buf[base],
                        buf[base + 1],
                        buf[base + 2],
                        buf[base + 3],
                    ]);
                    #[cfg(target_endian = "big")]
                    {
                        let mut swapped = [scale.to_bits() as i32];
                        mrc_swap_longs(&mut swapped, 1);
                        scale = f32::from_bits(swapped[0] as u32);
                    }
                    if debug != 0 {
                        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                            "analyzeDM3: Found Scale at %d  %f\n",
                            &[CArg::Int(c as i64), CArg::Dbl(scale as f64)],
                        ));
                    }
                }

            /* If we have a scale, make sure it is valid and at a plausible location and
            if so set the pixel size, overriding an earlier one */
            } else if got_scale != 0 && buf[c as usize] == b'U' {
                let found = found_dm_tag(&buf[c as usize..], b"Units", b"Units%%%%", dmind, 13);
                if let Some(found) = found {
                    if c + units_off[dmind as usize] + 2 < max_use_c {
                        toffset = (c as isize + found as isize) as i64 as i32;
                        if (got_dim != 0 && pixel == 0.)
                            || (got_dim_info != 0 && z_pixel == 0.)
                            || toffset as i64 <= plausible_off
                        {
                            if buf[(c + units_off[dmind as usize] + 2) as usize] == b'm' {
                                if buf[(c + units_off[dmind as usize]) as usize] == b'n' {
                                    tmp_pixel = (scale as f64 * 10.) as f32;
                                } else if buf[(c + units_off[dmind as usize]) as usize] as i32
                                    == 181
                                {
                                    tmp_pixel = (scale as f64 * 10000.) as f32;
                                }

                                /* Assign to regular or Z pixel and keep track of location separately */
                                if got_dim_info != 0 {
                                    if loop_ == 0 && c != LAST_ZUNITS.get() {
                                        match_last = 0;
                                        break;
                                    }
                                    cur_zunits = c;
                                    z_pixel = tmp_pixel;
                                } else {
                                    if loop_ == 0 && c != LAST_UNITS.get() {
                                        match_last = 0;
                                        break;
                                    }
                                    cur_units = c;
                                    pixel = tmp_pixel;
                                }
                                max_cur_used = c;
                                if debug != 0 {
                                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                        "analyzeDM3: Assigned %f to %spixel\n",
                                        &[
                                            CArg::Dbl(tmp_pixel as f64),
                                            CArg::Str(if got_dim_info != 0 { "z" } else { "" }),
                                        ],
                                    ));
                                }
                            }
                            if debug != 0 {
                                // C `%p` of `found`, which is `buf + toffset`.
                                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                                    "analyzeDM3: Found Units at %d (%p) %d  %d\n",
                                    &[
                                        CArg::Int(c as i64),
                                        CArg::Ptr(buf.as_ptr() as usize + toffset as usize),
                                        CArg::Int(
                                            buf[(c + units_off[dmind as usize]) as usize] as i64,
                                        ),
                                        CArg::Int(
                                            buf[(c + units_off[dmind as usize] + 2) as usize]
                                                as i64,
                                        ),
                                    ],
                                ));
                            }
                        }
                        got_dim_info = 0;
                        got_meta = got_dim_info;
                        got_scale = got_meta;
                        got_dim = got_scale;
                        got_cal = got_dim;
                    }
                }
            }
            c += 1;
        }
        if match_last != 0 && offset != 0 {
            break;
        }
        loop_ += 1;
    }
    if offset == 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: analyzeDM3 - Data string not found in {}\n",
                String::from_utf8_lossy(filename)
            ),
        );
        return IIERR_NO_SUPPORT;
    }
    if debug != 0 {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "analyzeDM3: time %.1f\n",
            &[CArg::Dbl(1000. * (wall_time() - wall_start))],
        ));
        let _ = ImodFile::Stdout.flush();
    }

    /* Save all the indexes that were found for the next time */
    LAST_DATA_TYPE.set(cur_data_type);
    LAST_DATA.set(cur_data);
    LAST_CALIBRATIONS.set(cur_calibrations);
    LAST_ZUNITS.set(cur_zunits);
    LAST_UNITS.set(cur_units);
    LAST_TAB_DIMENS.set(cur_tab_dimens);
    LAST_DIMENSIONS.set(cur_dimensions);
    LAST_DIMENS_INFO.set(cur_dimens_info);
    LAST_OFFSET.set(cur_offset);
    LAST_MAX_READ.set(maxread);
    LAST_MAX_START_USED.set(
        if maxread < (max_cur_used + max_offset[dmind as usize]) as i64 {
            maxread as i32
        } else {
            max_cur_used + max_offset[dmind as usize]
        },
    );

    /* Set return values in info */
    info.nx = xsize;
    info.ny = ysize;
    info.nz = zsize;
    info.header_size = offset;
    info.y_inverted = 1;
    info.pixel = pixel;
    info.z_pixel = z_pixel;
    *dmtype = type_;
    #[cfg(target_endian = "little")]
    {
        info.swap_bytes = 0;
    }
    #[cfg(target_endian = "big")]
    {
        info.swap_bytes = 1;
    }
    0
}

/// Original `foundDMtag` (`iilikemrc.c:849`).
///
/// The C returns a `char *` into `buf`; here the return is the index of the
/// match within the slice it was given, which is the same pointer arithmetic
/// the two callers do with it.  `buf + dm4Offset` can run past the end of the
/// one-megabyte buffer in the C, which reads adjacent stack; a slice that does
/// not reach that far is treated as no match.
fn found_dm_tag(
    buf: &[u8],
    tag: &[u8],
    full_tag: &[u8],
    dmind: i32,
    dm4_offset: i32,
) -> Option<usize> {
    let mut found: Option<usize>;
    // `strstr` searches only as far as the first NUL.
    let hay = &buf[..buf.iter().position(|b| *b == 0).unwrap_or(buf.len())];
    if dmind != 0 {
        found = hay.windows(tag.len()).position(|w| w == tag);
        if found.is_some() {
            let tail = match buf.get(dm4_offset as usize..) {
                Some(tail) => tail,
                None => return None,
            };
            let tail = &tail[..tail.iter().position(|b| *b == 0).unwrap_or(tail.len())];
            if !tail.windows(4).any(|w| w == b"%%%%") {
                found = None;
            }
        }
    } else {
        found = hay.windows(full_tag.len()).position(|w| w == full_tag);
    }
    found
}

/// Original `checkFEIraw` (`iilikemrc.c:865`).
///
/// Check for the FEI raw format.
fn check_fei_raw(fp: &mut ImodFile, _filename: &[u8], info: &mut RawImageInfo) -> i32 {
    let mut label = [0u8; 13];
    let mut ibuf = [0u8; 36];
    b3d_rewind(fp);
    if b3d_fread(&mut label, 1, 13, fp) != 13 {
        return IIERR_IO_ERROR;
    }
    // `strncmp(label, "FEI RawImage", 12)`: the tag has no NUL in those 12
    // bytes, so the comparison is exactly of the first 12 bytes.
    if label[..12] != *b"FEI RawImage" {
        return IIERR_NOT_FORMAT;
    }
    if b3d_fread(&mut ibuf, 4, 9, fp) != 9 {
        return IIERR_IO_ERROR;
    }
    let mut ivals = [0i32; 9];
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }
    if ivals[4] == 16 && ivals[5] == 1 {
        info.type_ = RAW_MODE_SHORT;
    } else if ivals[4] == 16 && ivals[5] == 0 {
        info.type_ = RAW_MODE_USHORT;
    } else if ivals[4] == 32 && ivals[5] == 2 {
        info.type_ = RAW_MODE_FLOAT;
    } else {
        return IIERR_NO_SUPPORT;
    }
    info.nx = ivals[1];
    info.ny = ivals[2];
    info.nz = 1;
    #[cfg(target_endian = "little")]
    {
        info.swap_bytes = 0;
    }
    #[cfg(target_endian = "big")]
    {
        info.swap_bytes = 1;
    }
    info.header_size = 49 + ivals[6];
    info.y_inverted = 1;
    info.amin = 0.;
    info.amax = 0.;
    0
}

/// Original `checkEM` (`iilikemrc.c:902`).
///
/// Check for the EM format.
fn check_em(fp: &mut ImodFile, _filename: &[u8], info: &mut RawImageInfo) -> i32 {
    let mut bvals = [0u8; 4];
    let mut ibuf = [0u8; 48];
    b3d_rewind(fp);
    if b3d_fread(&mut bvals, 1, 4, fp) != 4 {
        return IIERR_IO_ERROR;
    }
    if b3d_fread(&mut ibuf[..12], 4, 3, fp) != 3 {
        return IIERR_IO_ERROR;
    }
    let mut ivals = [0i32; 12];
    for (index, value) in ivals.iter_mut().enumerate() {
        *value = i32::from_ne_bytes([
            ibuf[4 * index],
            ibuf[4 * index + 1],
            ibuf[4 * index + 2],
            ibuf[4 * index + 3],
        ]);
    }

    /*printf("bvals %d %d %d %d  ivals %d %d %d\n", bvals[0], bvals[1], bvals[2],
    bvals[3], ivals[0], ivals[1], ivals[2]);*/

    /* Not much magic here, put limits on type values and machine numbers and
    product of putative sizes */
    // `((float)ivals[0] * ivals[1]) * ivals[2]`: the whole product is formed in
    // single precision and only the comparison with MAX_EM_SIZE is in double.
    if ivals[0] <= 0
        || ivals[1] <= 0
        || ivals[2] <= 0
        || (ivals[0] > 65536 && ivals[1] > 65536 && ivals[2] > 65536)
        || bvals[0] as i32 > MAX_EM_MACHINES
        || bvals[2] as i32 == 1
        || bvals[3] as i32 > MAX_EM_TYPES
        || (ivals[0] as f32 * ivals[1] as f32 * ivals[2] as f32) as f64 > MAX_EM_SIZE
    {
        mrc_swap_longs(&mut ivals, 3);

        if ivals[0] <= 0
            || ivals[1] <= 0
            || ivals[2] <= 0
            || (ivals[0] > 65536 && ivals[1] > 65536 && ivals[2] > 65536)
            || bvals[0] as i32 > MAX_EM_MACHINES
            || bvals[2] as i32 == 1
            || bvals[3] as i32 > MAX_EM_TYPES
            || (ivals[0] as f32 * ivals[1] as f32 * ivals[2] as f32) as f64 > MAX_EM_SIZE
        {
            return IIERR_NOT_FORMAT;
        }
        info.swap_bytes = 1;
    }

    match bvals[3] as i32 {
        1 => {
            info.type_ = RAW_MODE_BYTE;
        }
        2 => {
            info.type_ = RAW_MODE_SHORT;
        }
        5 => {
            info.type_ = RAW_MODE_FLOAT;
        }
        _ => {
            return IIERR_NO_SUPPORT;
        }
    }

    info.nx = ivals[0];
    info.ny = ivals[1];
    info.nz = ivals[2];
    info.header_size = 512;
    info.amin = 0.;
    info.amax = 0.;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::iimage::{IITYPE_SHORT, ii_new_box};

    fn empty_info() -> RawImageInfo {
        RawImageInfo {
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
        }
    }

    #[test]
    fn raw_header_setup_owns_its_mrc_header_through_safe_references() {
        let mut image = ii_new_box();
        let info = RawImageInfo {
            nx: 8,
            ny: 6,
            nz: 2,
            type_: RAW_MODE_USHORT,
            swap_bytes: 1,
            header_size: 512,
            section_skip: 32,
            y_inverted: 1,
            pixel: 1.5,
            z_pixel: 3.0,
            ..empty_info()
        };

        assert_eq!(ii_setup_raw_headers(&mut image, &info), 0);
        let header = image.mrc_header.as_deref().expect("installed header");
        assert_eq!(
            (image.file, image.nx, image.ny, image.nz),
            (IIFILE_RAW, 8, 6, 2)
        );
        assert_eq!(
            (
                header.mode,
                header.swapped,
                header.header_size,
                header.section_skip
            ),
            (MRC_MODE_USHORT, 1, 512, 32)
        );
        assert_eq!((image.xscale, image.yscale, image.zscale), (1.5, 1.5, 3.0));
    }

    #[test]
    fn fei_raw_dispatch_constructs_native_mrc_access_state() {
        unsafe {
            ii_delete_raw_check_list();
            let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut bytes = b"FEI RawImage\0".to_vec();
            for value in [0_i32, 4, 3, 0, 16, 1, 100, 0, 0] {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&bytes, 1, bytes.len(), &mut fp),
                bytes.len()
            );
            crate::imod::libcfshr::b3dutil::b3d_rewind(&mut fp);

            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            (*image).fp = Some(fp.clone());
            (*image).filename = Some("synthetic-fei.raw".into());
            assert_eq!(ii_like_mrc_check(image.cast()), 0);
            let header = (*image)
                .mrc_header
                .as_deref_mut()
                .expect("raw image has an owned header");
            assert_eq!(((*image).file, (*image).type_), (IIFILE_RAW, IITYPE_SHORT));
            assert_eq!((header.nx, header.ny, header.nz), (4, 3, 1));
            assert_eq!((header.header_size, header.y_inverted), (149, 1));
            ii_like_mrc_delete(image.cast());
            assert!((*image).mrc_header.is_none());
            drop(fp);
            ii_delete_raw_check_list();
        }
    }

    #[test]
    fn em_checker_accepts_little_endian_dimensions_and_rejects_unknown_type() {
        let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
        let mut bytes = vec![6_u8, 0, 0, 2];
        for value in [8_i32, 7, 2] {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        assert_eq!(
            crate::imod::libcfshr::b3dutil::b3d_fwrite(&bytes, 1, bytes.len(), &mut fp),
            bytes.len()
        );
        crate::imod::libcfshr::b3dutil::b3d_rewind(&mut fp);
        let mut info = empty_info();
        assert_eq!(check_em(&mut fp, b"", &mut info), 0);
        assert_eq!(
            (info.nx, info.ny, info.nz, info.type_, info.header_size),
            (8, 7, 2, RAW_MODE_SHORT, 512)
        );

        let mut unsupported = bytes;
        unsupported[3] = 3;
        let mut unsupported_fp = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
        assert_eq!(
            crate::imod::libcfshr::b3dutil::b3d_fwrite(
                &unsupported,
                1,
                unsupported.len(),
                &mut unsupported_fp
            ),
            unsupported.len()
        );
        crate::imod::libcfshr::b3dutil::b3d_rewind(&mut unsupported_fp);
        assert_eq!(
            check_em(&mut unsupported_fp, b"", &mut info),
            IIERR_NO_SUPPORT
        );
    }

    #[test]
    fn pif_checker_reads_bsoft_header_fields() {
        let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
        let mut bytes = vec![0_u8; 84];
        bytes[24..28].copy_from_slice(&3_i32.to_ne_bytes());
        bytes[28..32].copy_from_slice(&0_i32.to_ne_bytes());
        bytes[32..37].copy_from_slice(b"Bsoft");
        for (index, value) in [1_i32, 4, 3, 0, 9].into_iter().enumerate() {
            let start = 64 + 4 * index;
            bytes[start..start + 4].copy_from_slice(&value.to_ne_bytes());
        }
        assert_eq!(
            crate::imod::libcfshr::b3dutil::b3d_fwrite(&bytes, 1, bytes.len(), &mut fp),
            bytes.len()
        );
        let mut info = empty_info();
        assert_eq!(check_pif(&mut fp, b"", &mut info), 0);
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
    }
}
