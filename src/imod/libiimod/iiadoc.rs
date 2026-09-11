//! Translation of `IMOD/libiimod/iiadoc.c`.
#![allow(dead_code)]

use crate::imod::libcfshr::autodoc::{
    ADOC_GLOBAL_NAME, adoc_clear, adoc_get_float, adoc_get_integer, adoc_get_number_of_sections,
    adoc_get_section_name, adoc_get_three_floats, adoc_get_two_integers, adoc_open_image_metadata,
    adoc_read,
};
use crate::imod::libcfshr::b3dutil::{b3d_error, b3d_get_store_error, b3d_set_store_error};
use crate::imod::libiimod::iimage::{
    IIERR_BAD_CALL, IIERR_IO_ERROR, IIERR_NOT_FORMAT, IIFILE_ADOC, IiSectionFunc, ImodImageFile,
    ii_default_min_max_mean, ii_delete, ii_open, ii_read_section, ii_read_section_byte,
    ii_read_section_float, ii_read_section_ushort, ii_simple_fill_mrc_header,
};
use crate::imod::libiimod::iimrc::ii_mrc_mode_to_format_type;
use crate::imod::libiimod::mrcfiles::{
    MRC_LABEL_SIZE, MrcHeader, fix_title_padding, mrc_set_scale,
};
use core::ffi::c_char;

/// Matches C `IIADOC_IMAGE` (`iiadoc.c:14`).  `ADOC_GLOBAL_NAME` comes from
/// `autodoc.h` ("PreData"), which `iiadoc.c` includes.
const IIADOC_IMAGE: &core::ffi::CStr = c"Image";

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// Matches C `iiADOCCheck(ImodImageFile *)` (`iiadoc.c:30`).
pub unsafe extern "C" fn ii_adoc_check(in_file: *mut ImodImageFile) -> i32 {
    if in_file.is_null() || unsafe { (*in_file).fp }.is_null() {
        return IIERR_BAD_CALL;
    }
    unsafe {
        libc::fclose((*in_file).fp);
        (*in_file).fp = core::ptr::null_mut();

        let save_store = b3d_get_store_error();
        b3d_set_store_error(1);
        let mut montage = 0;
        let mut num_sect = 0;
        let mut sect_type = 0;
        (*in_file).adoc_index = adoc_open_image_metadata(
            (*in_file).filename,
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
                ADOC_GLOBAL_NAME.as_ptr(),
                0,
                c"ImageSize".as_ptr(),
                &mut (*in_file).nx,
                &mut (*in_file).ny,
            ) != 0
                || adoc_get_integer(
                    ADOC_GLOBAL_NAME.as_ptr(),
                    0,
                    c"DataMode".as_ptr(),
                    &mut (*in_file).mode,
                ) != 0
            {
                adoc_clear((*in_file).adoc_index);
                (*in_file).adoc_index = -1;
            }
            (*in_file).nz = num_sect;
        }

        (*in_file).fp = libc::fopen((*in_file).filename, (*in_file).fmode.as_ptr());
        if (*in_file).fp.is_null() {
            b3d_error(
                stderr,
                format_args!("ERROR: iiADOCCheck - reopening file after reading as adoc"),
            );
            adoc_clear((*in_file).adoc_index);
            return IIERR_IO_ERROR;
        }
        if (*in_file).adoc_index < 0 {
            return IIERR_NOT_FORMAT;
        }
        (*in_file).file = IIFILE_ADOC;
        ii_mrc_mode_to_format_type(in_file, (*in_file).mode, 0);
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
                IIADOC_IMAGE.as_ptr(),
                ind,
                c"MinMaxMean".as_ptr(),
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
        if adoc_get_float(
            ADOC_GLOBAL_NAME.as_ptr(),
            0,
            c"PixelSpacing".as_ptr(),
            &mut (*in_file).xscale,
        ) == 0
        {
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
unsafe extern "C" fn adoc_close(in_file: *mut ImodImageFile) {
    unsafe {
        if !in_file.is_null() && (*in_file).adoc_index >= 0 {
            adoc_clear((*in_file).adoc_index);
        }
        if !in_file.is_null() && !(*in_file).fp.is_null() {
            libc::fclose((*in_file).fp);
        }
        if !in_file.is_null() {
            (*in_file).fp = core::ptr::null_mut();
            (*in_file).adoc_index = -1;
        }
    }
}

/// Matches C static `adocReopen` (`iiadoc.c:116`).
unsafe extern "C" fn adoc_reopen(in_file: *mut ImodImageFile) -> i32 {
    unsafe {
        (*in_file).adoc_index = adoc_read((*in_file).filename);
        (*in_file).fp = libc::fopen((*in_file).filename, (*in_file).fmode.as_ptr());
        if (*in_file).fp.is_null() {
            b3d_error(stderr, format_args!("ERROR: adocReopen - reopening file"));
            adoc_clear((*in_file).adoc_index);
        }
        if (*in_file).adoc_index < 0 || (*in_file).fp.is_null() {
            1
        } else {
            0
        }
    }
}

/// Matches C static `adocFillMrcHeader` (`iiadoc.c:129`).
unsafe extern "C" fn adoc_fill_mrc_header(
    in_file: *mut ImodImageFile,
    hdata: *mut MrcHeader,
) -> i32 {
    unsafe {
        ii_simple_fill_mrc_header(in_file, hdata);
        mrc_set_scale(
            &mut *hdata,
            (*in_file).xscale as f64,
            (*in_file).yscale as f64,
            (*in_file).zscale as f64,
        );
        (*hdata).nlabl = adoc_get_number_of_sections(c"T".as_ptr());
        for ind in 0..(*hdata).nlabl {
            let mut label = core::ptr::null_mut();
            if adoc_get_section_name(c"T".as_ptr(), ind, &mut label) != 0 {
                return 1;
            }
            libc::strncpy(
                (*hdata).labels[ind as usize].as_mut_ptr().cast(),
                label,
                MRC_LABEL_SIZE,
            );
            libc::free(label.cast());
            fix_title_padding(&mut (*hdata).labels[ind as usize]);
        }
    }
    0
}

/// Matches C static `readSectionFile` (`iiadoc.c:151`).
unsafe fn read_section_file(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    section: i32,
    func: IiSectionFunc,
) -> i32 {
    unsafe {
        let mut filename = core::ptr::null_mut();
        if adoc_get_section_name(IIADOC_IMAGE.as_ptr(), section, &mut filename) != 0 {
            b3d_error(
                stderr,
                format_args!("ERROR: iiadoc - Getting filename for section {}", section),
            );
            return 1;
        }
        let slash = libc::strrchr((*in_file).filename, b'/' as i32);
        let back_slash = libc::strrchr((*in_file).filename, b'\\' as i32);
        let mut slash_ind = -1_isize;
        let mut bs_ind = -1_isize;
        if !slash.is_null() {
            slash_ind = slash.offset_from((*in_file).filename);
        }
        if !back_slash.is_null() {
            bs_ind = back_slash.offset_from((*in_file).filename);
        }
        slash_ind = slash_ind.max(bs_ind);
        let mut use_name = filename;
        if slash_ind >= 0 {
            let full_len = slash_ind as usize + libc::strlen(filename) + 4;
            use_name = libc::malloc(full_len).cast();
            if use_name.is_null() {
                b3d_error(
                    stderr,
                    format_args!("ERROR: iiadoc - Allocating memory for filename\n"),
                );
                return 1;
            }
            libc::strncpy(use_name, (*in_file).filename, slash_ind as usize + 1);
            libc::strcpy(use_name.add(slash_ind as usize + 1), filename);
        }
        let sect_file = ii_open(use_name, c"rb".as_ptr());
        libc::free(filename.cast());
        if slash_ind >= 0 {
            libc::free(use_name.cast());
        }
        if sect_file.is_null() {
            b3d_error(
                stderr,
                format_args!("ERROR: iiadoc - Cannot open file for section {}\n", section),
            );
            return 1;
        }
        if (*sect_file).nx != (*in_file).nx
            || (*sect_file).ny != (*in_file).ny
            || (*sect_file).mode != (*in_file).mode
        {
            b3d_error(
                stderr,
                format_args!(
                    "ERROR: iiadoc - File for section {} has wrong size or mode (section: {} x {} mode {}, file: {} x {} mode {})\n",
                    section,
                    (*sect_file).nx,
                    (*sect_file).ny,
                    (*sect_file).mode,
                    (*in_file).nx,
                    (*in_file).ny,
                    (*in_file).mode,
                ),
            );
            ii_delete(sect_file);
            return 1;
        }
        (*sect_file).llx = (*in_file).llx;
        (*sect_file).urx = (*in_file).urx;
        (*sect_file).lly = (*in_file).lly;
        (*sect_file).ury = (*in_file).ury;
        (*sect_file).amin = (*in_file).amin;
        (*sect_file).amax = (*in_file).amax;
        (*sect_file).amean = (*in_file).amean;
        (*sect_file).smin = (*in_file).smin;
        (*sect_file).smax = (*in_file).smax;
        (*sect_file).slope = (*in_file).slope;
        (*sect_file).offset = (*in_file).offset;
        let err = func.map_or(-1, |read| read(sect_file, buf, 0));
        ii_delete(sect_file);
        err
    }
}

/// Matches C static `adocReadSectionByte` (`iiadoc.c:221`).
unsafe extern "C" fn adoc_read_section_byte(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    unsafe { read_section_file(in_file, buf, in_section, Some(ii_read_section_byte)) }
}

/// Matches C static `adocReadSectionUShort` (`iiadoc.c:226`).
unsafe extern "C" fn adoc_read_section_ushort(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    unsafe { read_section_file(in_file, buf, in_section, Some(ii_read_section_ushort)) }
}

/// Matches C static `adocReadSectionFloat` (`iiadoc.c:231`).
unsafe extern "C" fn adoc_read_section_float(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    unsafe { read_section_file(in_file, buf, in_section, Some(ii_read_section_float)) }
}

/// Matches C static `adocReadSection` (`iiadoc.c:236`).
unsafe extern "C" fn adoc_read_section(
    in_file: *mut ImodImageFile,
    buf: *mut c_char,
    in_section: i32,
) -> i32 {
    unsafe { read_section_file(in_file, buf, in_section, Some(ii_read_section)) }
}
