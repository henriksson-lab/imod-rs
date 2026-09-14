//! Translation of `IMOD/libiimod/iimrc.c`.
//!
//! Function names are a systematic snake-case rendering of the C names.
#![allow(dead_code, unused_variables)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{
    SEEK_CUR, SEEK_SET, b3d_error, b3d_fread, b3d_fseek, extra_is_nbytes_and_flags,
};
use crate::imod::libiimod::iimage::{
    IIFILE_MRC, IIFORMAT_COMPLEX, IIFORMAT_LUMINANCE, IIFORMAT_RGB, IITYPE_BYTE, IITYPE_FLOAT,
    IITYPE_SHORT, IITYPE_UBYTE, IITYPE_USHORT, ImodImageFile, ii_change_call_count,
    ii_sync_from_mrc_header,
};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_head_new, mrc_head_read,
    mrc_init_li, mrc_swap_shorts,
};
use crate::imod::libiimod::mrcsec::{
    mrc_read_section, mrc_read_section_byte, mrc_read_section_float, mrc_read_section_ushort,
    mrc_write_z, mrc_write_z_float,
};

const IIERR_BAD_CALL: i32 = -1;
const IIERR_NOT_FORMAT: i32 = 1;
const IIERR_IO_ERROR: i32 = 2;

/// Matches C `iiMRCCheck(ImodImageFile *)` (`iimrc.c:31`).
pub unsafe extern "C" fn ii_mrc_check(iif: *mut ImodImageFile) -> i32 {
    if iif.is_null() || unsafe { (*iif).fp.is_none() } {
        return IIERR_BAD_CALL;
    }
    // Keep the allocation in the image record; `header` below is only the
    // erased legacy ABI alias used by callbacks.
    let mut hdr = Box::new(MrcHeader::default());
    let err = unsafe { mrc_head_read(&mut (*iif).fp.clone().unwrap(), &mut hdr) };
    if err != 0 {
        return if err < 0 {
            IIERR_IO_ERROR
        } else {
            IIERR_NOT_FORMAT
        };
    }
    unsafe {
        (*iif).mrc_header = Some(hdr);
        let hdr = (*iif).mrc_header.as_deref_mut().unwrap() as *mut MrcHeader;
        (*iif).header = hdr.cast();
        (*iif).file = IIFILE_MRC;
        ii_mrc_mode_to_format_type(iif, (*hdr).mode, (*hdr).bytes_signed);
        ii_sync_from_mrc_header(iif, hdr);
        (*iif).smin = (*iif).amin;
        (*iif).smax = (*iif).amax;
        (*iif).has_piece_coords = ii_mrc_check_pcoord(hdr);
        ii_mrc_set_io_funcs(iif, 0);
    }
    0
}

/// Matches C `iiMRCmodeToFormatType` (`iimrc.c:69`).
pub unsafe fn ii_mrc_mode_to_format_type(iif: *mut ImodImageFile, mode: i32, bytes_signed: i32) {
    unsafe {
        match mode {
            MRC_MODE_BYTE => {
                (*iif).format = IIFORMAT_LUMINANCE;
                (*iif).type_ = if bytes_signed != 0 {
                    IITYPE_BYTE
                } else {
                    IITYPE_UBYTE
                };
            }
            MRC_MODE_SHORT => {
                (*iif).format = IIFORMAT_LUMINANCE;
                (*iif).type_ = IITYPE_SHORT;
            }
            MRC_MODE_USHORT => {
                (*iif).format = IIFORMAT_LUMINANCE;
                (*iif).type_ = IITYPE_USHORT;
            }
            MRC_MODE_FLOAT => {
                (*iif).format = IIFORMAT_LUMINANCE;
                (*iif).type_ = IITYPE_FLOAT;
            }
            MRC_MODE_COMPLEX_SHORT => {
                (*iif).format = IIFORMAT_COMPLEX;
                (*iif).type_ = IITYPE_SHORT;
            }
            MRC_MODE_COMPLEX_FLOAT => {
                (*iif).format = IIFORMAT_COMPLEX;
                (*iif).type_ = IITYPE_FLOAT;
            }
            MRC_MODE_RGB => {
                (*iif).format = IIFORMAT_RGB;
                (*iif).type_ = IITYPE_UBYTE;
            }
            _ => {}
        }
        (*iif).mode = mode;
    }
}

/// Matches C `iiMRCsetIOFuncs(ImodImageFile *, int)` (`iimrc.c:105`).
pub unsafe extern "C" fn ii_mrc_set_io_funcs(in_file: *mut ImodImageFile, raw_file: i32) {
    unsafe {
        (*in_file).read_section = Some(ii_mrc_read_section);
        (*in_file).read_section_byte = Some(ii_mrc_read_section_byte);
        (*in_file).read_section_ushort = Some(ii_mrc_read_section_ushort);
        (*in_file).read_section_float = Some(ii_mrc_read_section_float);
        (*in_file).fill_mrc_header = Some(ii_mrc_fill_header);
        if raw_file != 0 {
            return;
        }
        (*in_file).clean_up = Some(ii_mrc_delete);
        (*in_file).write_section = Some(ii_mrc_write_section);
        (*in_file).write_section_float = Some(ii_mrc_write_section_float);
    }
}
/// Matches C `iiMRCdelete(ImodImageFile *)` (`iimrc.c:118`).
pub unsafe extern "C" fn ii_mrc_delete(in_file: *mut ImodImageFile) {
    unsafe {
        (*in_file).mrc_header = None;
        (*in_file).header = core::ptr::null_mut();
    }
}
/// Matches C `iiMRCopenNew(ImodImageFile *, const char *)` (`iimrc.c:125`).
pub unsafe fn ii_mrc_open_new(in_file: *mut ImodImageFile, mode: &str) -> i32 {
    unsafe {
        let name = (*in_file).filename.clone().unwrap_or_default();
        (*in_file).fp = ImodFile::open(&name, mode);
        if (*in_file).fp.is_none() {
            // `ImodFile::open` is implemented with Rust's file API, so retain
            // the operating-system failure as a Rust error value.
            let error = std::io::Error::last_os_error();
            let message = error.to_string();
            let has_os_error = error.raw_os_error().is_some();
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "ERROR: iiMRCopenNew - Could not open {}{}{}\n",
                    name,
                    if has_os_error {
                        " - system message: "
                    } else {
                        ""
                    },
                    message
                ),
            );
            return 1;
        }
        (*in_file).mrc_header = Some(Box::new(MrcHeader::default()));
        let header = (*in_file).mrc_header.as_deref_mut().unwrap();
        mrc_head_new(header, 1, 1, 1, 0);
        ii_mrc_set_io_funcs(in_file, 0);
        header.fp = (*in_file).fp.clone();
        (*in_file).header = (header as *mut MrcHeader).cast();
        (*in_file).file = IIFILE_MRC;
    }
    0
}
/// Matches C `iiMRCfillHeader(ImodImageFile *, MrcHeader *)` (`iimrc.c:145`).
pub unsafe extern "C" fn ii_mrc_fill_header(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    unsafe {
        if in_file.is_null() || (*in_file).header.is_null() {
            return 1;
        }
        let source = (*in_file).header.cast::<MrcHeader>();
        if hdata != source {
            // `iimrc.c:150` is `*hdata = *(MrcHeader *)inFile->header`, a
            // whole-struct assignment.  It has to be a `clone` here, not a
            // bitwise copy: `MrcHeader.fp` owns an `Rc<File>`, so duplicating
            // its bits leaves two owners at refcount one and the second drop
            // corrupts the heap.
            *hdata = (*source).clone();
        }
    }
    0
}
/// Matches C `iiMRCsetLoadInfo(ImodImageFile *, IloadInfo *)` (`iimrc.c:157`).
pub unsafe extern "C" fn ii_mrc_set_load_info(in_file: *mut ImodImageFile, li: *mut LoadInfo) {
    unsafe {
        mrc_init_li(Some(&mut *li), None);
        (*li).xmin = (*in_file).llx;
        (*li).ymin = (*in_file).lly;
        (*li).zmin = (*in_file).llz;
        (*li).xmax = if (*in_file).urx < 0 {
            (*in_file).nx - 1
        } else {
            (*in_file).urx
        };
        (*li).ymax = if (*in_file).ury < 0 {
            (*in_file).ny - 1
        } else {
            (*in_file).ury
        };
        (*li).zmax = if (*in_file).urz < 0 {
            (*in_file).nz - 1
        } else {
            (*in_file).urz
        };
        (*li).slope = (*in_file).slope;
        (*li).offset = (*in_file).offset;
        (*li).axis = (*in_file).axis;
        (*li).pad_left = (*in_file).pad_left;
        (*li).pad_right = (*in_file).pad_right;
    }
}
/// Matches C static `iiMRCreadSection` (`iimrc.c:192`).
unsafe extern "C" fn ii_mrc_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { read_section_unscaled(in_file, buf, in_section, 0) }
}
/// Matches C static `iiMRCreadSectionFloat` (`iimrc.c:197`).
unsafe extern "C" fn ii_mrc_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { read_section_unscaled(in_file, buf, in_section, 1) }
}
/// Matches C static `readSectionUnscaled` (`iimrc.c:202`).
unsafe fn read_section_unscaled(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    as_float: i32,
) -> i32 {
    let mut li = LoadInfo::default();
    let h = unsafe { (*in_file).header.cast::<MrcHeader>() };
    unsafe {
        ii_mrc_set_load_info(in_file, &mut li);
        li.outmin = (*in_file).smin.clamp(-2_000_000_000.0, 2_000_000_000.0) as i32;
        li.outmax = (*in_file).smax.clamp(-2_000_000_000.0, 2_000_000_000.0) as i32;
        li.black = 0;
        li.white = 255;
        li.mirror_fft = 0;
        (*h).fp = (*in_file).fp.clone();
        ii_change_call_count(1);
        let err = if as_float != 0 {
            mrc_read_section_float(h, &mut li, buf.cast(), in_section)
        } else {
            mrc_read_section(h, &mut li, buf.cast(), in_section)
        };
        ii_change_call_count(-1);
        err
    }
}
/// Matches C static `iiMRCreadSectionByte` (`iimrc.c:223`).
unsafe extern "C" fn ii_mrc_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { read_section_scaled(in_file, buf, in_section, 255) }
}
/// Matches C static `iiMRCreadSectionUShort` (`iimrc.c:227`).
unsafe extern "C" fn ii_mrc_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { read_section_scaled(in_file, buf, in_section, 65535) }
}
/// Matches C static `readSectionScaled` (`iimrc.c:232`).
unsafe fn read_section_scaled(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    outmax: i32,
) -> i32 {
    let mut li = LoadInfo::default();
    let h = unsafe { (*in_file).header.cast::<MrcHeader>() };
    unsafe {
        ii_mrc_set_load_info(in_file, &mut li);
        li.outmin = 0;
        li.outmax = outmax;
        li.mirror_fft = (*in_file).mirror_fft;
        (*h).fp = (*in_file).fp.clone();
        ii_change_call_count(1);
        let err = if outmax > 255 {
            mrc_read_section_ushort(h, &mut li, buf.cast(), in_section)
        } else {
            mrc_read_section_byte(h, &mut li, buf.cast(), in_section)
        };
        ii_change_call_count(-1);
        err
    }
}
/// Matches C static `iiMRCwriteSection` (`iimrc.c:255`).
unsafe extern "C" fn ii_mrc_write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { write_section(in_file, buf, in_section, 0) }
}
/// Matches C static `iiMRCwriteSectionFloat` (`iimrc.c:260`).
unsafe extern "C" fn ii_mrc_write_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe { write_section(in_file, buf, in_section, 1) }
}
/// Matches C static `writeSection` (`iimrc.c:265`).
unsafe fn write_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
    as_float: i32,
) -> i32 {
    let mut li = LoadInfo::default();
    let h = unsafe { (*in_file).header.cast::<MrcHeader>() };
    unsafe {
        ii_mrc_set_load_info(in_file, &mut li);
        (*h).fp = (*in_file).fp.clone();
        if (*in_file).axis != 3 {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: iiMRCwriteSection - attempting to write Y slices\n"),
            );
            return 1;
        }
        ii_change_call_count(1);
        let err = if as_float != 0 {
            mrc_write_z_float(h, &mut li, buf.cast(), in_section)
        } else {
            mrc_write_z(h, &mut li, buf.cast(), in_section)
        };
        ii_change_call_count(-1);
        err
    }
}
/// Matches C `iiMRCcheckPCoord(MrcHeader *)` (`iimrc.c:283`).
pub unsafe extern "C" fn ii_mrc_check_pcoord(hdr: *mut MrcHeader) -> i32 {
    unsafe {
        if (*hdr).next == 0 || ((*hdr).nreal & 2) == 0 || (*hdr).creatid == -16224 {
            return 0;
        }
        extra_is_nbytes_and_flags((*hdr).nint as i32, (*hdr).nreal as i32)
    }
}
/// Matches C `iiMRCLoadPCoord(ImodImageFile *, IloadInfo *, int, int, int)` (`iimrc.c:294`).
pub unsafe extern "C" fn ii_mrc_load_pcoord(
    in_file: *mut ImodImageFile,
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    let mut offset = 1024;
    let mut nread = nz;
    let hdr = unsafe { (*in_file).header.cast::<MrcHeader>() };
    let iflag = unsafe { (*hdr).nreal as i32 };
    let nbytes = unsafe { (*hdr).nint as i32 };
    let nextra = unsafe { (*hdr).next };
    if unsafe { ii_mrc_check_pcoord(hdr) } == 0 {
        return 0;
    }
    if iflag & 1 != 0 {
        offset += 2;
    }
    if nbytes * nz > nextra {
        nread = nextra / nbytes;
        unsafe {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!(
                    "There are piece coordinates for only {} frames in the extra header\n",
                    nread
                ),
            )
        };
    }
    unsafe {
        (*li).pcoords = Some(vec![0i32; 3 * nz as usize]);
        let mut fp = (*in_file).fp.clone().unwrap();
        b3d_fseek(&mut fp, offset, SEEK_SET);
        for i in 0..nread {
            let mut pcoordxy = [0_u8; 4];
            let mut pcoordz = [0_u8; 2];
            let got_xy = b3d_fread(&mut pcoordxy, 2, 2, &mut fp);
            let got_z = b3d_fread(&mut pcoordz, 1, 2, &mut fp);
            let mut xy = [
                u16::from_ne_bytes([pcoordxy[0], pcoordxy[1]]),
                u16::from_ne_bytes([pcoordxy[2], pcoordxy[3]]),
            ];
            let mut z = i16::from_ne_bytes(pcoordz);
            if (*hdr).swapped != 0 {
                mrc_swap_shorts(
                    core::slice::from_raw_parts_mut(xy.as_mut_ptr().cast::<i16>(), 2),
                    2,
                );
                mrc_swap_shorts(core::slice::from_mut(&mut z), 1);
            }
            // `iimrc.c:406` tests `ferror(inFile->fp)`; a short transfer is the
            // same condition reached through the return value, which is what
            // [`b3d_fread`] reports.
            if got_xy != 2 || got_z != 2 {
                nread = i;
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!(
                        "Error reading piece coordinates from extra header after {} frames\n",
                        i
                    ),
                );
                break;
            }
            let pcoords = (&mut (*li).pcoords).as_mut().unwrap();
            pcoords[(i * 3) as usize] = xy[0] as i32;
            pcoords[(i * 3 + 1) as usize] = xy[1] as i32;
            pcoords[(i * 3 + 2) as usize] = z as i32;
            offset = nbytes - 6;
            if offset > 0 {
                b3d_fseek(&mut fp, offset, SEEK_CUR);
            }
        }
        (*li).plist = nread;
    }
    // `mrc_plist_proc` belongs to the complete source file `plist.c`, whose
    // translation supplies this cross-file call.
    crate::imod::libiimod::plist::mrc_plist_proc(li, nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::iimage::ii_new_box;
    use crate::imod::libiimod::mrcfiles::mrc_head_write;

    #[test]
    fn mode_to_format_type_retains_every_source_case_and_unknown_mode() {
        let mut image_owner = ii_new_box();
        let image = image_owner.as_mut() as *mut ImodImageFile;
        unsafe {
            ii_mrc_mode_to_format_type(image, MRC_MODE_BYTE, 1);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_BYTE)
            );
            ii_mrc_mode_to_format_type(image, MRC_MODE_SHORT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_SHORT)
            );
            ii_mrc_mode_to_format_type(image, MRC_MODE_USHORT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_USHORT)
            );
            ii_mrc_mode_to_format_type(image, MRC_MODE_FLOAT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_FLOAT)
            );
            ii_mrc_mode_to_format_type(image, MRC_MODE_COMPLEX_SHORT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_COMPLEX, IITYPE_SHORT)
            );
            ii_mrc_mode_to_format_type(image, MRC_MODE_COMPLEX_FLOAT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_COMPLEX, IITYPE_FLOAT)
            );
            ii_mrc_mode_to_format_type(image, MRC_MODE_RGB, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_RGB, IITYPE_UBYTE)
            );
            (*image).format = 87;
            (*image).type_ = 99;
            ii_mrc_mode_to_format_type(image, 123, 0);
            assert_eq!(
                ((*image).format, (*image).type_, (*image).mode),
                (87, 99, 123)
            );
        }
    }

    #[test]
    fn fill_header_and_piece_coordinate_check_retain_source_shallow_copy_and_rules() {
        unsafe {
            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            let mut source = MrcHeader::default();
            source.nx = 12;
            source.labels[0][0] = b'X';
            (*image).header = (&mut source as *mut MrcHeader).cast();
            let mut copied = MrcHeader::default();
            assert_eq!(ii_mrc_fill_header(image, &mut copied), 0);
            assert_eq!((copied.nx, copied.labels[0][0]), (12, b'X'));
            assert_eq!(ii_mrc_fill_header(image, &mut source), 0);
            (*image).header = core::ptr::null_mut();
            assert_eq!(ii_mrc_fill_header(image, &mut copied), 1);

            let mut header = MrcHeader::default();
            assert_eq!(ii_mrc_check_pcoord(&mut header), 0);
            header.next = 128;
            header.nreal = 2;
            header.nint = 6;
            assert_eq!(ii_mrc_check_pcoord(&mut header), 1);
            header.creatid = -16224;
            assert_eq!(ii_mrc_check_pcoord(&mut header), 0);
        }
    }

    #[test]
    fn set_load_info_retains_source_bounds_defaults_and_overrides() {
        unsafe {
            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            (*image).nx = 4;
            (*image).ny = 5;
            (*image).nz = 6;
            (*image).llx = 1;
            (*image).lly = 2;
            (*image).llz = 3;
            (*image).urx = -1;
            (*image).ury = 8;
            (*image).urz = -1;
            (*image).slope = 2.5;
            (*image).offset = -3.0;
            (*image).axis = 1;
            (*image).pad_left = 7;
            (*image).pad_right = 9;
            let mut li = LoadInfo::default();
            ii_mrc_set_load_info(image, &mut li);
            assert_eq!(
                (li.xmin, li.xmax, li.ymin, li.ymax, li.zmin, li.zmax),
                (1, 3, 2, 8, 3, 5)
            );
            assert_eq!(
                (li.slope, li.offset, li.axis, li.pad_left, li.pad_right),
                (2.5, -3.0, 1, 7, 9)
            );
        }
    }

    #[test]
    fn delete_releases_only_the_owned_mrc_header_and_clears_its_alias() {
        unsafe {
            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            ii_mrc_delete(image);
            (*image).mrc_header = Some(Box::new(MrcHeader::default()));
            (*image).header =
                ((*image).mrc_header.as_deref_mut().unwrap() as *mut MrcHeader).cast();
            assert!((*image).mrc_header.is_some());
            assert!(!(*image).header.is_null());
            ii_mrc_delete(image);
            assert!((*image).mrc_header.is_none());
            assert!((*image).header.is_null());

            // An erased header may belong to another backend; this cleanup
            // must never reconstruct ownership from that alias.
            let mut borrowed = MrcHeader::default();
            (*image).header = (&mut borrowed as *mut MrcHeader).cast();
            ii_mrc_delete(image);
            borrowed.nx = 7;
            assert_eq!(borrowed.nx, 7);
            assert!((*image).header.is_null());
        }
    }

    #[test]
    fn set_io_funcs_observes_raw_file_early_return_and_new_file_initialization() {
        unsafe {
            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            ii_mrc_set_io_funcs(image, 1);
            assert!((*image).read_section.is_some());
            assert!((*image).read_section_byte.is_some());
            assert!((*image).read_section_ushort.is_some());
            assert!((*image).read_section_float.is_some());
            assert!((*image).fill_mrc_header.is_some());
            assert!((*image).clean_up.is_none());
            assert!((*image).write_section.is_none());
            assert!((*image).write_section_float.is_none());

            let path = std::env::temp_dir().join(format!(
                "imod-rs-iimrc-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::File::create(&path).unwrap();
            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            (*image).filename = Some(path.to_string_lossy().into_owned());
            assert_eq!(ii_mrc_open_new(image, "wb+"), 0);
            let header = (*image).header.cast::<MrcHeader>();
            assert_eq!(
                header,
                (*image).mrc_header.as_deref_mut().unwrap() as *mut MrcHeader
            );
            assert_eq!(
                ((*image).file, (*header).nx, (*header).ny, (*header).nz),
                (IIFILE_MRC, 1, 1, 1)
            );
            assert!((*image).clean_up.is_some());
            assert!((*image).write_section.is_some());
            assert!((*image).write_section_float.is_some());
            (*image).fp = None;
            ii_mrc_delete(image);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn mrc_check_reads_real_header_and_installs_section_reader() {
        unsafe {
            let mut fp = crate::imod::libcfshr::b3dutil::ImodFile::tmpfile().unwrap();
            let mut header = MrcHeader::default();
            mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE);
            header.fp = Some(fp.clone());
            assert_eq!(mrc_head_write(&mut fp, &mut header), 0);
            let pixels = [1_u8, 2, 3, 4];
            assert_eq!(
                crate::imod::libcfshr::b3dutil::b3d_fwrite(&pixels, 1, 4, &mut fp),
                4
            );
            crate::imod::libcfshr::b3dutil::b3d_rewind(&mut fp);

            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            (*image).fp = Some(fp.clone());
            assert_eq!(ii_mrc_check(image), 0);
            assert_eq!(
                (*image).header.cast::<MrcHeader>(),
                (*image).mrc_header.as_deref_mut().unwrap() as *mut MrcHeader
            );
            assert_eq!(
                (
                    (*image).file,
                    (*image).nx,
                    (*image).ny,
                    (*image).nz,
                    (*image).type_
                ),
                (IIFILE_MRC, 2, 2, 1, IITYPE_BYTE)
            );
            let mut read = [0_u8; 4];
            assert_eq!(
                (*image).read_section.expect("MRC reader")(image, read.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(read, [129, 130, 131, 132]);
            ii_mrc_delete(image);
            drop(fp);
        }
    }
}
