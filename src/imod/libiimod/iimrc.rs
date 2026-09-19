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
    MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader, mrc_getdcsize, mrc_head_new,
    mrc_head_read, mrc_init_li, mrc_swap_shorts,
};
use crate::imod::libiimod::mrcsec::{
    mrc_read_section, mrc_read_section_byte, mrc_read_section_float, mrc_read_section_ushort,
    mrc_write_z, mrc_write_z_float,
};

const IIERR_BAD_CALL: i32 = -1;
const IIERR_NOT_FORMAT: i32 = 1;
const IIERR_IO_ERROR: i32 = 2;

/// Matches C `iiMRCCheck(ImodImageFile *)` (`iimrc.c:31`).
pub unsafe fn ii_mrc_check(iif: *mut ImodImageFile) -> i32 {
    let Some(image) = (unsafe { iif.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let Some(mut fp) = image.fp.clone() else {
        return IIERR_BAD_CALL;
    };
    // Move the fully owned header directly into the image record after
    // validation; no interim heap allocation is needed.
    let mut hdr = MrcHeader::default();
    let err = mrc_head_read(&mut fp, &mut hdr);
    if err != 0 {
        return if err < 0 {
            IIERR_IO_ERROR
        } else {
            IIERR_NOT_FORMAT
        };
    }
    image.file = IIFILE_MRC;
    ii_mrc_mode_to_format_type(image, hdr.mode, hdr.bytes_signed);
    ii_sync_from_mrc_header(image, &mut hdr);
    image.mrc_header = Some(hdr);
    image.smin = image.amin;
    image.smax = image.amax;
    image.has_piece_coords = image.mrc_header.as_ref().map_or(0, ii_mrc_check_pcoord);
    ii_mrc_set_io_funcs(iif, 0);
    0
}

/// Matches C `iiMRCmodeToFormatType` (`iimrc.c:69`).
pub fn ii_mrc_mode_to_format_type(image: &mut ImodImageFile, mode: i32, bytes_signed: i32) {
    match mode {
        MRC_MODE_BYTE => {
            image.format = IIFORMAT_LUMINANCE;
            image.type_ = if bytes_signed != 0 {
                IITYPE_BYTE
            } else {
                IITYPE_UBYTE
            };
        }
        MRC_MODE_SHORT => {
            image.format = IIFORMAT_LUMINANCE;
            image.type_ = IITYPE_SHORT;
        }
        MRC_MODE_USHORT => {
            image.format = IIFORMAT_LUMINANCE;
            image.type_ = IITYPE_USHORT;
        }
        MRC_MODE_FLOAT => {
            image.format = IIFORMAT_LUMINANCE;
            image.type_ = IITYPE_FLOAT;
        }
        MRC_MODE_COMPLEX_SHORT => {
            image.format = IIFORMAT_COMPLEX;
            image.type_ = IITYPE_SHORT;
        }
        MRC_MODE_COMPLEX_FLOAT => {
            image.format = IIFORMAT_COMPLEX;
            image.type_ = IITYPE_FLOAT;
        }
        MRC_MODE_RGB => {
            image.format = IIFORMAT_RGB;
            image.type_ = IITYPE_UBYTE;
        }
        _ => {}
    }
    image.mode = mode;
}

/// Matches C `iiMRCsetIOFuncs(ImodImageFile *, int)` (`iimrc.c:105`).
pub unsafe fn ii_mrc_set_io_funcs(in_file: *mut ImodImageFile, raw_file: i32) {
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return;
    };
    image.read_section = Some(ii_mrc_read_section);
    image.read_section_byte = Some(ii_mrc_read_section_byte);
    image.read_section_ushort = Some(ii_mrc_read_section_ushort);
    image.read_section_float = Some(ii_mrc_read_section_float);
    image.fill_mrc_header = Some(ii_mrc_fill_header);
    if raw_file == 0 {
        image.clean_up = Some(ii_mrc_delete);
        image.write_section = Some(ii_mrc_write_section);
        image.write_section_float = Some(ii_mrc_write_section_float);
    }
}
/// Matches C `iiMRCdelete(ImodImageFile *)` (`iimrc.c:118`).
pub unsafe fn ii_mrc_delete(in_file: *mut ImodImageFile) {
    if let Some(image) = unsafe { in_file.as_mut() } {
        image.mrc_header = None;
    }
}
/// Matches C `iiMRCopenNew(ImodImageFile *, const char *)` (`iimrc.c:125`).
pub unsafe fn ii_mrc_open_new(in_file: *mut ImodImageFile, mode: &str) -> i32 {
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let name = image.filename.clone().unwrap_or_default();
    image.fp = ImodFile::open(&name, mode);
    if image.fp.is_none() {
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
    image.mrc_header = Some(MrcHeader::default());
    let header = image.mrc_header.as_mut().expect("header just installed");
    mrc_head_new(header, 1, 1, 1, 0);
    header.fp = image.fp.clone();
    image.file = IIFILE_MRC;
    ii_mrc_set_io_funcs(in_file, 0);
    0
}
/// Matches C `iiMRCfillHeader(ImodImageFile *, MrcHeader *)` (`iimrc.c:145`).
pub unsafe fn ii_mrc_fill_header(in_file: *mut ImodImageFile, hdata: *mut MrcHeader) -> i32 {
    let (Some(image), Some(destination)) = (unsafe { in_file.as_ref() }, unsafe { hdata.as_mut() })
    else {
        return 1;
    };
    let Some(source) = image.mrc_header.as_ref() else {
        return 1;
    };
    if !core::ptr::eq(source, destination) {
        // `iimrc.c:150` is a whole-header assignment.  It has to be a clone
        // here: `MrcHeader.fp` owns an `Rc<File>`, so duplicating its bits
        // would corrupt the reference count on drop.
        *destination = source.clone();
    }
    0
}
/// Matches C `iiMRCsetLoadInfo(ImodImageFile *, IloadInfo *)` (`iimrc.c:157`).
pub fn ii_mrc_set_load_info(image: &ImodImageFile, li: &mut LoadInfo) {
    mrc_init_li(Some(li), None);
    li.xmin = image.llx;
    li.ymin = image.lly;
    li.zmin = image.llz;
    li.xmax = if image.urx < 0 {
        image.nx - 1
    } else {
        image.urx
    };
    li.ymax = if image.ury < 0 {
        image.ny - 1
    } else {
        image.ury
    };
    li.zmax = if image.urz < 0 {
        image.nz - 1
    } else {
        image.urz
    };
    li.slope = image.slope;
    li.offset = image.offset;
    li.axis = image.axis;
    li.pad_left = image.pad_left;
    li.pad_right = image.pad_right;
}
/// Matches C static `iiMRCreadSection` (`iimrc.c:192`).
unsafe fn ii_mrc_read_section(in_file: *mut ImodImageFile, buf: *mut u8, in_section: i32) -> i32 {
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(image, &mut li);
    let Some(width): Option<usize> =
        (li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0))
            .try_into()
            .ok()
    else {
        return IIERR_BAD_CALL;
    };
    let Some(rows): Option<usize> = (if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    })
    .try_into()
    .ok() else {
        return IIERR_BAD_CALL;
    };
    let Some(header) = image.mrc_header.as_ref() else {
        return IIERR_BAD_CALL;
    };
    let mut bytes = 0;
    let mut channels = 0;
    if mrc_getdcsize(header.mode, &mut bytes, &mut channels) != 0 {
        return IIERR_BAD_CALL;
    }
    let pixel_bytes = if header.half_floats != 0 && header.mode == MRC_MODE_FLOAT {
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
    let output = unsafe { core::slice::from_raw_parts_mut(buf, length) };
    read_section_unscaled(image, output, in_section, 0)
}
/// Matches C static `iiMRCreadSectionFloat` (`iimrc.c:197`).
unsafe fn ii_mrc_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(image, &mut li);
    let Some(width): Option<usize> =
        (li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0))
            .try_into()
            .ok()
    else {
        return IIERR_BAD_CALL;
    };
    let Some(rows): Option<usize> = (if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    })
    .try_into()
    .ok() else {
        return IIERR_BAD_CALL;
    };
    let Some(length) = width
        .checked_mul(rows)
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return IIERR_BAD_CALL;
    };
    let output = unsafe { core::slice::from_raw_parts_mut(buf, length) };
    read_section_unscaled(image, output, in_section, 1)
}
/// Matches C static `readSectionUnscaled` (`iimrc.c:202`).
fn read_section_unscaled(
    image: &mut ImodImageFile,
    buf: &mut [u8],
    in_section: i32,
    as_float: i32,
) -> i32 {
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(image, &mut li);
    li.outmin = image.smin.clamp(-2_000_000_000.0, 2_000_000_000.0) as i32;
    li.outmax = image.smax.clamp(-2_000_000_000.0, 2_000_000_000.0) as i32;
    li.black = 0;
    li.white = 255;
    li.mirror_fft = 0;
    let Some(header) = image.mrc_header.as_mut() else {
        return IIERR_BAD_CALL;
    };
    header.fp = image.fp.clone();
    ii_change_call_count(1);
    let Some(width): Option<usize> =
        (li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0))
            .try_into()
            .ok()
    else {
        ii_change_call_count(-1);
        return IIERR_BAD_CALL;
    };
    let Some(rows): Option<usize> = (if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    })
    .try_into()
    .ok() else {
        ii_change_call_count(-1);
        return IIERR_BAD_CALL;
    };
    let err = if as_float != 0 {
        let Some(pixels) = width.checked_mul(rows) else {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        };
        let Some(output_len) = pixels.checked_mul(4) else {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        };
        if buf.len() < output_len {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        }
        // `iimrc.c:211` is `mrcReadSectionFloat(h, &li, (b3dFloat *)buf, inSection)`:
        // the caller's buffer *is* the float buffer, so read straight into it.
        // Every caller's buffer originates from a `Vec<f32>`; if one ever is not
        // 4-byte aligned, stage through a vector rather than reinterpret it.
        if buf.as_ptr().align_offset(core::mem::align_of::<f32>()) == 0 {
            // Sound: `f32` has no invalid bit patterns, and the checked alignment
            // plus `output_len == 4 * pixels` makes the middle slice the whole
            // buffer.
            let (_, output, _) = unsafe { buf[..output_len].align_to_mut::<f32>() };
            mrc_read_section_float(header, &mut li, output, in_section)
        } else {
            let mut output = vec![0.0_f32; pixels];
            let err = mrc_read_section_float(header, &mut li, &mut output, in_section);
            if err == 0 {
                for (value, bytes) in output.iter().zip(buf[..output_len].chunks_exact_mut(4)) {
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
            }
            err
        }
    } else {
        let mut bytes = 0;
        let mut channels = 0;
        if mrc_getdcsize(header.mode, &mut bytes, &mut channels) != 0 {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        }
        let pixel_bytes = if header.half_floats != 0 && header.mode == MRC_MODE_FLOAT {
            2
        } else {
            (bytes * channels) as usize
        };
        let Some(output_len) = width
            .checked_mul(rows)
            .and_then(|pixels| pixels.checked_mul(pixel_bytes))
        else {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        };
        if buf.len() < output_len {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        }
        mrc_read_section(header, &mut li, &mut buf[..output_len], in_section)
    };
    ii_change_call_count(-1);
    err
}
/// Matches C static `iiMRCreadSectionByte` (`iimrc.c:223`).
unsafe fn ii_mrc_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(image, &mut li);
    let Some(width): Option<usize> =
        (li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0))
            .try_into()
            .ok()
    else {
        return IIERR_BAD_CALL;
    };
    let Some(rows): Option<usize> = (if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    })
    .try_into()
    .ok() else {
        return IIERR_BAD_CALL;
    };
    let Some(length) = width.checked_mul(rows) else {
        return IIERR_BAD_CALL;
    };
    let output = unsafe { core::slice::from_raw_parts_mut(buf, length) };
    read_section_scaled(image, output, in_section, 255)
}
/// Matches C static `iiMRCreadSectionUShort` (`iimrc.c:227`).
unsafe fn ii_mrc_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(image, &mut li);
    let Some(width): Option<usize> =
        (li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0))
            .try_into()
            .ok()
    else {
        return IIERR_BAD_CALL;
    };
    let Some(rows): Option<usize> = (if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    })
    .try_into()
    .ok() else {
        return IIERR_BAD_CALL;
    };
    let Some(length) = width
        .checked_mul(rows)
        .and_then(|pixels| pixels.checked_mul(2))
    else {
        return IIERR_BAD_CALL;
    };
    let output = unsafe { core::slice::from_raw_parts_mut(buf, length) };
    read_section_scaled(image, output, in_section, 65535)
}
/// Matches C static `readSectionScaled` (`iimrc.c:232`).
fn read_section_scaled(
    image: &mut ImodImageFile,
    buf: &mut [u8],
    in_section: i32,
    outmax: i32,
) -> i32 {
    let mut li = LoadInfo::default();
    ii_mrc_set_load_info(image, &mut li);
    li.outmin = 0;
    li.outmax = outmax;
    li.mirror_fft = image.mirror_fft;
    let Some(header) = image.mrc_header.as_mut() else {
        return IIERR_BAD_CALL;
    };
    header.fp = image.fp.clone();
    ii_change_call_count(1);
    let Some(width): Option<usize> =
        (li.xmax - li.xmin + 1 + li.pad_left.max(0) + li.pad_right.max(0))
            .try_into()
            .ok()
    else {
        ii_change_call_count(-1);
        return IIERR_BAD_CALL;
    };
    let Some(rows): Option<usize> = (if li.axis == 2 {
        li.zmax - li.zmin + 1
    } else {
        li.ymax - li.ymin + 1
    })
    .try_into()
    .ok() else {
        ii_change_call_count(-1);
        return IIERR_BAD_CALL;
    };
    let Some(pixels) = width.checked_mul(rows) else {
        ii_change_call_count(-1);
        return IIERR_BAD_CALL;
    };
    let err = if outmax > 255 {
        let Some(output_len) = pixels.checked_mul(2) else {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        };
        if buf.len() < output_len {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        }
        // `iimrc.c:246` hands `mrcReadSectionUShort` the caller's buffer directly;
        // do the same rather than staging a copy.  See `read_section_unscaled`.
        if buf.as_ptr().align_offset(core::mem::align_of::<u16>()) == 0 {
            let (_, output, _) = unsafe { buf[..output_len].align_to_mut::<u16>() };
            mrc_read_section_ushort(header, &mut li, output, in_section)
        } else {
            let mut output = vec![0_u16; pixels];
            let err = mrc_read_section_ushort(header, &mut li, &mut output, in_section);
            if err == 0 {
                for (value, bytes) in output.iter().zip(buf[..output_len].chunks_exact_mut(2)) {
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
            }
            err
        }
    } else {
        if buf.len() < pixels {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        }
        mrc_read_section_byte(header, &mut li, &mut buf[..pixels], in_section)
    };
    ii_change_call_count(-1);
    err
}
/// Matches C static `iiMRCwriteSection` (`iimrc.c:255`).
unsafe fn ii_mrc_write_section(in_file: *mut ImodImageFile, buf: *mut u8, in_section: i32) -> i32 {
    unsafe { write_section(in_file, buf, in_section, 0) }
}
/// Matches C static `iiMRCwriteSectionFloat` (`iimrc.c:260`).
unsafe fn ii_mrc_write_section_float(
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
    let Some(image) = (unsafe { in_file.as_mut() }) else {
        return IIERR_BAD_CALL;
    };
    ii_mrc_set_load_info(image, &mut li);
    if image.axis != 3 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiMRCwriteSection - attempting to write Y slices\n"),
        );
        return 1;
    }
    let Some(header) = image.mrc_header.as_mut() else {
        return IIERR_BAD_CALL;
    };
    header.fp = image.fp.clone();
    ii_change_call_count(1);
    let err = if as_float != 0 {
        let Some(length) = header
            .nx
            .checked_mul(header.ny)
            .and_then(|length| usize::try_from(length).ok())
        else {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        };
        mrc_write_z_float(
            header,
            &mut li,
            core::slice::from_raw_parts_mut(buf.cast(), length),
            in_section,
        )
    } else {
        let mut bytes = 0;
        let mut channels = 0;
        if mrc_getdcsize(header.mode, &mut bytes, &mut channels) != 0 {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        }
        let pixel_bytes = if header.half_floats != 0 && header.mode == MRC_MODE_FLOAT {
            2
        } else {
            bytes * channels
        };
        let Some(length) = header
            .nx
            .checked_mul(header.ny)
            .and_then(|pixels| pixels.checked_mul(pixel_bytes))
            .and_then(|length| usize::try_from(length).ok())
        else {
            ii_change_call_count(-1);
            return IIERR_BAD_CALL;
        };
        mrc_write_z(
            header,
            &mut li,
            core::slice::from_raw_parts(buf, length),
            in_section,
        )
    };
    ii_change_call_count(-1);
    err
}
/// Matches C `iiMRCcheckPCoord(MrcHeader *)` (`iimrc.c:283`).
pub fn ii_mrc_check_pcoord(header: &MrcHeader) -> i32 {
    if header.next == 0 || (header.nreal & 2) == 0 || header.creatid == -16224 {
        return 0;
    }
    extra_is_nbytes_and_flags(header.nint as i32, header.nreal as i32)
}
/// Matches C `iiMRCLoadPCoord(ImodImageFile *, IloadInfo *, int, int, int)` (`iimrc.c:294`).
pub fn ii_mrc_load_pcoord(
    image: &ImodImageFile,
    li: &mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    let mut offset = 1024;
    let mut nread = nz;
    let Some(header) = image.mrc_header.as_ref() else {
        return IIERR_BAD_CALL;
    };
    let iflag = header.nreal as i32;
    let nbytes = header.nint as i32;
    let nextra = header.next;
    if ii_mrc_check_pcoord(header) == 0 {
        return 0;
    }
    if iflag & 1 != 0 {
        offset += 2;
    }
    if nbytes * nz > nextra {
        nread = nextra / nbytes;
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "There are piece coordinates for only {} frames in the extra header\n",
                nread
            ),
        );
    }
    {
        li.pcoords = Some(vec![0i32; 3 * nz as usize]);
        let Some(mut fp) = image.fp.clone() else {
            return IIERR_BAD_CALL;
        };
        b3d_fseek(&mut fp, offset, SEEK_SET);
        for i in 0..nread {
            let mut pcoordxy = [0_u8; 4];
            let mut pcoordz = [0_u8; 2];
            let got_xy = b3d_fread(&mut pcoordxy, 2, 2, &mut fp);
            let got_z = b3d_fread(&mut pcoordz, 1, 2, &mut fp);
            let mut xy = [
                i16::from_ne_bytes([pcoordxy[0], pcoordxy[1]]),
                i16::from_ne_bytes([pcoordxy[2], pcoordxy[3]]),
            ];
            let mut z = i16::from_ne_bytes(pcoordz);
            if header.swapped != 0 {
                mrc_swap_shorts(&mut xy, 2);
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
            let pcoords = li.pcoords.as_mut().expect("piece coordinates installed");
            pcoords[(i * 3) as usize] = xy[0] as u16 as i32;
            pcoords[(i * 3 + 1) as usize] = xy[1] as u16 as i32;
            pcoords[(i * 3 + 2) as usize] = z as i32;
            offset = nbytes - 6;
            if offset > 0 {
                b3d_fseek(&mut fp, offset, SEEK_CUR);
            }
        }
        li.plist = nread;
    }
    // `mrc_plist_proc` belongs to the complete source file `plist.c`, whose
    // translation supplies this cross-file call.
    crate::imod::libiimod::plist::mrc_plist_proc(li, nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::b3dutil::b3d_fwrite;
    use crate::imod::libiimod::iimage::ii_new_box;
    use crate::imod::libiimod::mrcfiles::mrc_head_write;

    #[test]
    fn mode_to_format_type_retains_every_source_case_and_unknown_mode() {
        let mut image_owner = ii_new_box();
        let image = image_owner.as_mut() as *mut ImodImageFile;
        unsafe {
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_BYTE, 1);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_BYTE)
            );
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_SHORT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_SHORT)
            );
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_USHORT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_USHORT)
            );
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_FLOAT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_LUMINANCE, IITYPE_FLOAT)
            );
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_COMPLEX_SHORT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_COMPLEX, IITYPE_SHORT)
            );
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_COMPLEX_FLOAT, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_COMPLEX, IITYPE_FLOAT)
            );
            ii_mrc_mode_to_format_type(&mut *image, MRC_MODE_RGB, 0);
            assert_eq!(
                ((*image).format, (*image).type_),
                (IIFORMAT_RGB, IITYPE_UBYTE)
            );
            (*image).format = 87;
            (*image).type_ = 99;
            ii_mrc_mode_to_format_type(&mut *image, 123, 0);
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
            (*image).mrc_header = Some(source.clone());
            let mut copied = MrcHeader::default();
            assert_eq!(ii_mrc_fill_header(image, &mut copied), 0);
            assert_eq!((copied.nx, copied.labels[0][0]), (12, b'X'));
            assert_eq!(ii_mrc_fill_header(image, &mut source), 0);
            (*image).mrc_header = None;
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
    fn piece_coordinate_loader_uses_typed_image_and_load_info() {
        let mut file = ImodFile::tmpfile().unwrap();
        assert_eq!(b3d_fwrite(&[0; 1024], 1, 1024, &mut file), 1024);
        assert_eq!(
            b3d_fwrite(&[10, 0, 20, 0, 5, 0, 12, 0, 22, 0, 6, 0], 1, 12, &mut file,),
            12
        );
        let mut header = MrcHeader::default();
        header.next = 12;
        header.nint = 6;
        header.nreal = 2;
        let image = ImodImageFile {
            fp: Some(file),
            mrc_header: Some(header),
            ..ImodImageFile::default()
        };
        let mut load = LoadInfo::default();

        assert_eq!(ii_mrc_load_pcoord(&image, &mut load, 4, 5, 2), 0);
        assert_eq!(load.plist, 2);
        assert_eq!(load.pcoords.as_deref(), Some(&[0, 0, 0, 2, 2, 1][..]));
        assert_eq!(
            (load.opx, load.opy, load.opz, load.px, load.py, load.pz),
            (10., 20., 5., 6., 7., 2.)
        );
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
            ii_mrc_set_load_info(&*image, &mut li);
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
    fn delete_releases_only_the_owned_mrc_header() {
        unsafe {
            let mut image_owner = ii_new_box();
            let image = image_owner.as_mut() as *mut ImodImageFile;
            ii_mrc_delete(image);
            (*image).mrc_header = Some(MrcHeader::default());
            assert!((*image).mrc_header.is_some());
            ii_mrc_delete(image);
            assert!((*image).mrc_header.is_none());
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
            let header = (*image)
                .mrc_header
                .as_mut()
                .expect("new MRC image has an owned header");
            assert_eq!(
                ((*image).file, header.nx, header.ny, header.nz),
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
                (
                    (*image).file,
                    (*image).nx,
                    (*image).ny,
                    (*image).nz,
                    (*image).type_
                ),
                (IIFILE_MRC, 2, 2, 1, IITYPE_BYTE)
            );
            let mut safe_read = [0_u8; 4];
            assert_eq!(read_section_unscaled(&mut *image, &mut safe_read, 0, 0), 0);
            assert_eq!(safe_read, [129, 130, 131, 132]);
            let mut safe_scaled = [0_u8; 4];
            assert_eq!(
                read_section_scaled(&mut *image, &mut safe_scaled, 0, 255),
                0
            );
            let mut read = [0_u8; 4];
            assert_eq!(
                (*image).read_section.expect("MRC reader")(image, read.as_mut_ptr().cast(), 0),
                0
            );
            assert_eq!(read, [129, 130, 131, 132]);
            let mut scaled = [0_u8; 4];
            assert_eq!(
                (*image).read_section_byte.expect("MRC byte reader")(
                    image,
                    scaled.as_mut_ptr().cast(),
                    0,
                ),
                0
            );
            assert_eq!(scaled, safe_scaled);
            ii_mrc_delete(image);
            drop(fp);
        }
    }
}
