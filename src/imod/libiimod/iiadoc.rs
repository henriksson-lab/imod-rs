//! Translation of `IMOD/libiimod/iiadoc.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, adoc_clear, adoc_get_float, adoc_get_integer, adoc_get_number_of_sections,
    adoc_get_section_name, adoc_get_three_floats, adoc_get_two_integers, adoc_open_image_metadata,
    adoc_read,
};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{b3d_error, b3d_get_store_error, b3d_set_store_error};
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIFILE_ADOC, ImodImageFile, MRSA_BYTE,
    MRSA_FLOAT, MRSA_NOPROC, MRSA_USHORT, ii_default_min_max_mean, ii_delete, ii_open,
    ii_read_section_any, ii_simple_fill_mrc_header,
};
use crate::imod::libiimod::iimrc::ii_mrc_mode_to_format_type;
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MRC_MODE_HALF_FLOAT, MrcHeader, fix_title_padding, mrc_getdcsize, mrc_set_scale,
};

/// Matches C `IIADOC_IMAGE` (`iiadoc.c:14`).  `ADOC_GLOBAL_NAME` comes from
/// `autodoc.h` ("PreData"), which `iiadoc.c` includes.
///
const IIADOC_IMAGE: &[u8] = b"Image";

/// Matches C `iiADOCCheck(ImodImageFile *)` (`iiadoc.c:30`).
pub unsafe fn ii_adoc_check(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() || unsafe { (*in_file).fp.is_none() } {
        return IIERR_BAD_CALL;
    }
    unsafe {
        (*in_file).fp = None;

        let save_store = b3d_get_store_error();
        b3d_set_store_error(1);
        let mut montage = 0;
        let mut num_sect = 0;
        let mut sect_type = 0;
        let name = (*in_file).filename.clone().unwrap_or_default();
        (*in_file).adoc_index = adoc_open_image_metadata(
            name.as_bytes(),
            0,
            &mut montage,
            &mut num_sect,
            &mut sect_type,
        );
        if (*in_file).adoc_index >= 0 && sect_type != 2 {
            adoc_clear((*in_file).adoc_index);
            (*in_file).adoc_index = -1;
        }
        b3d_set_store_error(save_store);

        if (*in_file).adoc_index >= 0 {
            if adoc_get_two_integers(
                ADOC_GLOBAL_NAME,
                0,
                b"ImageSize",
                &mut (*in_file).nx,
                &mut (*in_file).ny,
            ) != 0
                || adoc_get_integer(ADOC_GLOBAL_NAME, 0, b"DataMode", &mut (*in_file).mode) != 0
            {
                adoc_clear((*in_file).adoc_index);
                (*in_file).adoc_index = -1;
            }
            (*in_file).nz = num_sect;
        }

        (*in_file).fp = ImodFile::open(&name, &(*in_file).fmode);
        if (*in_file).fp.is_none() {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: iiADOCCheck - reopening file after reading as adoc"),
            );
            adoc_clear((*in_file).adoc_index);
            return IIERR_IO_ERROR;
        }
        if (*in_file).adoc_index < 0 {
            return IIERR_NOT_FORMAT;
        }
        (*in_file).file = IIFILE_ADOC;
        ii_mrc_mode_to_format_type(&mut *in_file, (*in_file).mode, 0);
        (*in_file).has_piece_coords = montage;

        ii_default_min_max_mean(
            (*in_file).mode,
            &mut (*in_file).amin,
            &mut (*in_file).amax,
            &mut (*in_file).amean,
        );
        for ind in 0..num_sect {
            let mut tmin = 0.0;
            let mut tmax = 0.0;
            let mut tmean = 0.0;
            if adoc_get_three_floats(
                IIADOC_IMAGE,
                ind,
                b"MinMaxMean",
                &mut tmin,
                &mut tmax,
                &mut tmean,
            ) == 0
            {
                if ind == 0 {
                    (*in_file).amin = tmin;
                    (*in_file).amax = tmax;
                } else {
                    /* ACCUM_MIN / ACCUM_MAX from b3dutil.h */
                    if tmin < (*in_file).amin {
                        (*in_file).amin = tmin;
                    }
                    if tmax > (*in_file).amax {
                        (*in_file).amax = tmax;
                    }
                }
            }
        }
        (*in_file).amean = ((*in_file).amax + (*in_file).amin) / 2.0;
        (*in_file).smin = (*in_file).amin;
        (*in_file).smax = (*in_file).amax;
        if adoc_get_float(ADOC_GLOBAL_NAME, 0, b"PixelSpacing", &mut (*in_file).xscale) == 0 {
            (*in_file).yscale = (*in_file).xscale;
        }
        (*in_file).fill_mrc_header = Some(adoc_fill_mrc_header);
        (*in_file).read_section = Some(adoc_read_section);
        (*in_file).read_section_ushort = Some(adoc_read_section_ushort);
        (*in_file).read_section_byte = Some(adoc_read_section_byte);
        (*in_file).read_section_float = Some(adoc_read_section_float);
        (*in_file).clean_up = Some(adoc_close);
        (*in_file).close = Some(adoc_close);
        (*in_file).reopen = Some(adoc_reopen);
    }
    0
}

/// Matches C static `adocClose` (`iiadoc.c:106`).
unsafe fn adoc_close(in_file: *mut ImodImageFile) {
    unsafe {
        if !in_file.is_null() && (*in_file).adoc_index >= 0 {
            adoc_clear((*in_file).adoc_index);
        }
        if !in_file.is_null() {
            (*in_file).fp = None;
            (*in_file).adoc_index = -1;
        }
    }
}

/// Matches C static `adocReopen` (`iiadoc.c:116`).
unsafe fn adoc_reopen(in_file: *mut ImodImageFile) -> i32 {
    unsafe {
        let name = (*in_file).filename.clone().unwrap_or_default();
        (*in_file).adoc_index = adoc_read(name.as_bytes());
        (*in_file).fp = ImodFile::open(&name, &(*in_file).fmode);
        if (*in_file).fp.is_none() {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR: adocReopen - reopening file"),
            );
            adoc_clear((*in_file).adoc_index);
        }
        if (*in_file).adoc_index < 0 || (*in_file).fp.is_none() {
            1
        } else {
            0
        }
    }
}

/// Matches C static `adocFillMrcHeader` (`iiadoc.c:129`).
unsafe fn adoc_fill_mrc_header(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    unsafe {
        ii_simple_fill_mrc_header(&*in_file, &mut *hdata);
        mrc_set_scale(
            &mut *hdata,
            (*in_file).xscale as f64,
            (*in_file).yscale as f64,
            (*in_file).zscale as f64,
        );
        (*hdata).nlabl = adoc_get_number_of_sections(b"T");
        for ind in 0..(*hdata).nlabl {
            let mut label = Vec::new();
            if adoc_get_section_name(b"T", ind, &mut label) != 0 {
                return 1;
            }
            // `iiadoc.c:142` is `strncpy(hdata->labels[ind], label,
            // MRC_LABEL_SIZE)`: at most 80 bytes, NUL-padded to that length if
            // the name is shorter, and byte 80 left as it was.
            let count = label.len().min(MRC_LABEL_SIZE);
            let slot = &mut (*hdata).labels[ind as usize];
            slot[..count].copy_from_slice(&label[..count]);
            for byte in slot[count..MRC_LABEL_SIZE].iter_mut() {
                *byte = 0;
            }
            fix_title_padding(&mut (*hdata).labels[ind as usize]);
        }
    }
    0
}

/// Matches C static `readSectionFile` (`iiadoc.c:151`).
fn read_section_file(
    in_file: &mut ImodImageFile,
    buf: &mut [u8],
    section: i32,
    convert_to: i32,
) -> i32 {
    let mut filename = Vec::new();
    if adoc_get_section_name(IIADOC_IMAGE, section, &mut filename) != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiadoc - Getting filename for section {}", section),
        );
        return 1;
    }
    let own_name = in_file.filename.clone().unwrap_or_default();
    // `iiadoc.c:161-167`: the last '/' and the last '\', whichever is later.
    let mut slash_ind = -1_isize;
    let mut bs_ind = -1_isize;
    if let Some(pos) = own_name.bytes().rposition(|b| b == b'/') {
        slash_ind = pos as isize;
    }
    if let Some(pos) = own_name.bytes().rposition(|b| b == b'\\') {
        bs_ind = pos as isize;
    }
    slash_ind = slash_ind.max(bs_ind);
    let use_name = if slash_ind >= 0 {
        // `iiadoc.c:172-177`: the directory part of the idoc name, up to
        // and including the separator, then the section's file name.
        format!(
            "{}{}",
            &own_name[..slash_ind as usize + 1],
            String::from_utf8_lossy(&filename)
        )
    } else {
        String::from_utf8_lossy(&filename).into_owned()
    };
    let sect_file = unsafe { ii_open(use_name.as_bytes(), "rb") };
    let Some(mut sect_file) = (!sect_file.is_null()).then(|| unsafe { Box::from_raw(sect_file) })
    else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: iiadoc - Cannot open file for section {}\n", section),
        );
        return 1;
    };
    if sect_file.nx != in_file.nx || sect_file.ny != in_file.ny || sect_file.mode != in_file.mode {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!(
                "ERROR: iiadoc - File for section {} has wrong size or mode (section: {} x {} mode {}, file: {} x {} mode {})\n",
                section,
                sect_file.nx,
                sect_file.ny,
                sect_file.mode,
                in_file.nx,
                in_file.ny,
                in_file.mode,
            ),
        );
        unsafe { ii_delete(Box::into_raw(sect_file)) };
        return 1;
    }
    sect_file.llx = in_file.llx;
    sect_file.urx = in_file.urx;
    sect_file.lly = in_file.lly;
    sect_file.ury = in_file.ury;
    sect_file.amin = in_file.amin;
    sect_file.amax = in_file.amax;
    sect_file.amean = in_file.amean;
    sect_file.smin = in_file.smin;
    sect_file.smax = in_file.smax;
    sect_file.slope = in_file.slope;
    sect_file.offset = in_file.offset;
    let err = ii_read_section_any(&mut sect_file, buf, 0, convert_to);
    unsafe { ii_delete(Box::into_raw(sect_file)) };
    err
}

/// Matches C static `adocReadSectionByte` (`iiadoc.c:221`).
unsafe fn adoc_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        let Some(in_file) = in_file.as_mut() else {
            return IIERR_BAD_CALL;
        };
        let width =
            (in_file.urx - in_file.llx + 1 + in_file.pad_left.max(0) + in_file.pad_right.max(0))
                .max(0) as usize;
        let rows = (if in_file.axis == 2 {
            in_file.urz - in_file.llz + 1
        } else {
            in_file.ury - in_file.lly + 1
        })
        .max(0) as usize;
        let Some(length) = width.checked_mul(rows) else {
            return IIERR_BAD_CALL;
        };
        let Some(buf) = (!buf.is_null()).then(|| core::slice::from_raw_parts_mut(buf, length))
        else {
            return IIERR_BAD_CALL;
        };
        read_section_file(in_file, buf, in_section, MRSA_BYTE)
    }
}

/// Matches C static `adocReadSectionUShort` (`iiadoc.c:226`).
unsafe fn adoc_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        let Some(in_file) = in_file.as_mut() else {
            return IIERR_BAD_CALL;
        };
        let width =
            (in_file.urx - in_file.llx + 1 + in_file.pad_left.max(0) + in_file.pad_right.max(0))
                .max(0) as usize;
        let rows = (if in_file.axis == 2 {
            in_file.urz - in_file.llz + 1
        } else {
            in_file.ury - in_file.lly + 1
        })
        .max(0) as usize;
        let Some(length) = width
            .checked_mul(rows)
            .and_then(|pixels| pixels.checked_mul(2))
        else {
            return IIERR_BAD_CALL;
        };
        let Some(buf) = (!buf.is_null()).then(|| core::slice::from_raw_parts_mut(buf, length))
        else {
            return IIERR_BAD_CALL;
        };
        read_section_file(in_file, buf, in_section, MRSA_USHORT)
    }
}

/// Matches C static `adocReadSectionFloat` (`iiadoc.c:231`).
unsafe fn adoc_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        let Some(in_file) = in_file.as_mut() else {
            return IIERR_BAD_CALL;
        };
        let width =
            (in_file.urx - in_file.llx + 1 + in_file.pad_left.max(0) + in_file.pad_right.max(0))
                .max(0) as usize;
        let rows = (if in_file.axis == 2 {
            in_file.urz - in_file.llz + 1
        } else {
            in_file.ury - in_file.lly + 1
        })
        .max(0) as usize;
        let Some(length) = width
            .checked_mul(rows)
            .and_then(|pixels| pixels.checked_mul(4))
        else {
            return IIERR_BAD_CALL;
        };
        let Some(buf) = (!buf.is_null()).then(|| core::slice::from_raw_parts_mut(buf, length))
        else {
            return IIERR_BAD_CALL;
        };
        read_section_file(in_file, buf, in_section, MRSA_FLOAT)
    }
}

/// Matches C static `adocReadSection` (`iiadoc.c:236`).
unsafe fn adoc_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut u8,
    in_section: i32,
) -> i32 {
    unsafe {
        let Some(in_file) = in_file.as_mut() else {
            return IIERR_BAD_CALL;
        };
        let mut bytes = 0;
        let mut channels = 0;
        if mrc_getdcsize(in_file.mode, &mut bytes, &mut channels) != 0 {
            return IIERR_BAD_CALL;
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
        let Some(buf) = (!buf.is_null()).then(|| core::slice::from_raw_parts_mut(buf, length))
        else {
            return IIERR_BAD_CALL;
        };
        read_section_file(in_file, buf, in_section, MRSA_NOPROC)
    }
}
