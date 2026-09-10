//! Translation of `IMOD/libcfshr/extraheader.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CStr;
use std::sync::Mutex;

use super::autodoc::{
    ADOC_ZVALUE_NAME, adoc_get_float, adoc_get_integer, adoc_get_string, adoc_get_three_floats,
    adoc_get_three_integers, adoc_get_two_floats, adoc_lookup_by_name_value, adoc_set_current,
};
use super::b3dutil::{
    b3d_error, b3d_get_error, b3d_get_store_error, b3d_set_store_error, extra_is_nbytes_and_flags,
};

const MRC_EXT_TYPE_FEI: i32 = 3;
const RADIANS_PER_DEGREE: f64 = 0.017_453_292_52;
static S_ZERO_DOSE: Mutex<(f32, f32)> = Mutex::new((0., 0.));

/// C static `extraHeaderSizes` (`extraheader.c:915`).
pub unsafe fn extra_header_sizes(
    ext_head: *mut c_void,
    ext_size: i32,
    num_int: i32,
    num_real: i32,
    iz_sect: i32,
    offset: *mut i32,
    size: *mut i32,
    max_size: *mut i32,
) -> i32 {
    if extra_is_nbytes_and_flags(num_int, num_real) != 0 {
        *offset = iz_sect * num_int;
        *size = num_int;
        *max_size = num_int;
        return 0;
    }
    if num_int >= 0 && num_real >= 0 {
        *offset = 4 * iz_sect * (num_int + num_real);
        *size = 4 * (num_int + num_real);
        *max_size = *size;
        return 0;
    }
    *max_size = 0;
    if num_int == -MRC_EXT_TYPE_FEI {
        let mut data = ext_head.cast::<u8>();
        *offset = 0;
        for _ in 0..iz_sect {
            let value = ptr::read_unaligned(data.cast::<i32>());
            if value < 0 || *offset + value > ext_size {
                return 2;
            }
            *offset += value;
            data = data.add(value as usize);
            *max_size = (*max_size).max(value);
        }
        *size = ptr::read_unaligned(data.cast::<i32>());
        *max_size = (*max_size).max(*size);
        return 0;
    }
    1
}

/// C static `freeValStrings` (`extraheader.c:821`).
pub unsafe fn free_val_strings(val_strings: *mut *mut *mut c_char, nz: i32) {
    if (*val_strings).is_null() {
        return;
    }
    for ind in 0..nz - 1 {
        libc::free((*(*val_strings).add(ind as usize)).cast());
    }
    libc::free((*val_strings).cast());
    *val_strings = ptr::null_mut();
}

/// C `getExtraHeaderTilts` (`extraheader.c:74`).
pub unsafe fn get_extra_header_tilts(
    array: *mut c_char,
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    tilt: *mut f32,
    num_tilts: *mut i32,
    max_tilts: i32,
    iz_piece: *mut i32,
) -> i32 {
    get_extra_header_items(
        array,
        num_extra_bytes,
        nbytes,
        iflags,
        nz,
        1,
        tilt,
        tilt,
        num_tilts,
        max_tilts,
        iz_piece,
    )
}

/// C Fortran wrapper `get_extra_header_tilts` (`extraheader.c:83`).
pub unsafe fn get_extra_header_tilts_fortran(
    array: *mut c_char,
    num_extra_bytes: *mut i32,
    nbytes: *mut i32,
    iflags: *mut i32,
    nz: *mut i32,
    tilt: *mut f32,
    num_tilts: *mut i32,
    max_tilts: *mut i32,
    iz_piece: *mut i32,
) {
    b3d_set_store_error(1);
    if get_extra_header_tilts(
        array,
        *num_extra_bytes,
        *nbytes,
        *iflags,
        *nz,
        tilt,
        num_tilts,
        *max_tilts,
        iz_piece,
    ) != 0
    {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
}

/// C `getExtraHeaderItems` (`extraheader.c:117`).
pub unsafe fn get_extra_header_items(
    array: *mut c_char,
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    itype: i32,
    val1: *mut f32,
    val2: *mut f32,
    num_vals: *mut i32,
    max_vals: i32,
    iz_piece: *mut i32,
) -> i32 {
    *num_vals = 0;
    if num_extra_bytes == 0 || (nbytes < 0 && nbytes != -MRC_EXT_TYPE_FEI) {
        return 0;
    }
    let bytes: [i32; 11] = [2, 6, 4, 2, 2, 4, 2, 4, 2, 4, 2];
    let shorts = extra_is_nbytes_and_flags(nbytes, iflags) != 0;
    let (mut ind, mut skip, mut scale, mut fei_offset) = (0_i32, 0_i32, 1_f64, 100_i32);
    if shorts {
        if ((iflags >> (itype - 1)) & 1) == 0 || nbytes == 0 {
            return 0;
        }
        skip = nbytes;
        for i in 1..itype {
            if ((iflags >> (i - 1)) & 1) != 0 {
                ind += bytes[(i - 1) as usize];
            }
        }
    } else if nbytes == -MRC_EXT_TYPE_FEI {
        if itype > 1 && itype != 6 && itype != 10 {
            return 0;
        }
        scale = get_fei_ext_head_angle_scale(array.cast());
        let mut bt = 0;
        let mut st = 0;
        let mut mask = 0;
        let mut ft = 0.;
        let mut angle = 0.;
        if get_extra_header_value(
            array.cast(),
            8,
            3,
            &mut bt,
            &mut st,
            &mut mask,
            &mut ft,
            &mut angle,
        ) != 0
        {
            return 0;
        }
        if (itype == 1 && mask & (1 << 7) == 0)
            || (itype == 6 && mask & (1 << 6) == 0)
            || (itype == 10 && mask & 1 == 0)
        {
            return 0;
        }
        if itype == 6 {
            fei_offset = 92;
            scale = 1.0e-20;
        } else if itype == 10 {
            fei_offset = 12;
            scale = 86400.;
        }
    } else {
        if iflags == 0 || itype > 1 {
            return 0;
        }
        skip = 4 * (nbytes + iflags);
        ind = 4 * nbytes;
    }
    let mut first_stamp = 0.;
    for i in 0..nz {
        let value_index = *iz_piece.add(i as usize);
        if value_index < 0 {
            b3d_error(
                ptr::null_mut(),
                format_args!(
                    "getExtraHeaderItems - Value array not designed for negative Z values"
                ),
            );
            return 1;
        }
        if value_index >= max_vals {
            b3d_error(
                ptr::null_mut(),
                format_args!("getExtraHeaderItems - Array not big enough for data"),
            );
            return 1;
        }
        if shorts {
            let p = array.add(ind as usize).cast::<i16>();
            let value = ptr::read_unaligned(p);
            match itype {
                1 => *val1.add(value_index as usize) = value as f32 / 100.,
                3 => {
                    *val1.add(value_index as usize) = value as f32 / 25.;
                    *val2.add(value_index as usize) = ptr::read_unaligned(p.add(1)) as f32 / 25.;
                }
                4 => *val1.add(value_index as usize) = value as f32 * 100.,
                5 => *val1.add(value_index as usize) = value as f32 / 25000.,
                6 => {
                    *val1.add(value_index as usize) =
                        semshorts_to_float(value, ptr::read_unaligned(p.add(1))) as f32
                }
                _ => {}
            }
        } else if nbytes == -MRC_EXT_TYPE_FEI {
            let mut bt = 0;
            let mut st = 0;
            let mut itemp = 0;
            let mut ft = 0.;
            let mut angle = 0.;
            if get_extra_header_value(
                array.cast(),
                ind,
                3,
                &mut bt,
                &mut st,
                &mut skip,
                &mut ft,
                &mut angle,
            ) != 0
            {
                return 0;
            }
            if get_extra_header_value(
                array.cast(),
                ind + fei_offset,
                4,
                &mut bt,
                &mut st,
                &mut itemp,
                &mut ft,
                &mut angle,
            ) != 0
            {
                *val1.add(value_index as usize) = 0.;
            } else if itype == 10 {
                if i == 0 {
                    first_stamp = angle;
                }
                *val1.add(value_index as usize) = (scale * (angle - first_stamp)) as f32;
            } else {
                *val1.add(value_index as usize) = (scale * angle) as f32;
            }
        } else {
            *val1.add(value_index as usize) =
                ptr::read_unaligned(array.add(ind as usize).cast::<f32>());
        }
        *num_vals = (*num_vals).max(value_index + 1);
        ind += skip;
        if ind > num_extra_bytes {
            return 0;
        }
    }
    0
}

/// C Fortran wrapper `get_extra_header_items` (`extraheader.c:236`).
pub unsafe fn get_extra_header_items_fortran(
    array: *mut c_char,
    num_extra_bytes: *mut i32,
    nbytes: *mut i32,
    iflags: *mut i32,
    nz: *mut i32,
    itype: *mut i32,
    val1: *mut f32,
    val2: *mut f32,
    num_vals: *mut i32,
    max_vals: *mut i32,
    iz_piece: *mut i32,
) {
    b3d_set_store_error(1);
    if get_extra_header_items(
        array,
        *num_extra_bytes,
        *nbytes,
        *iflags,
        *nz,
        *itype,
        val1,
        val2,
        num_vals,
        *max_vals,
        iz_piece,
    ) != 0
    {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
}

/// C `SEMshortsToFloat` (`extraheader.c:252`).
pub fn semshorts_to_float(mut low: i16, mut ihigh: i16) -> f64 {
    let mut value_sign = 1;
    let mut exponent_sign = 1;
    if low < 0 {
        low = -low;
        value_sign = -1;
    }
    if ihigh < 0 {
        ihigh = -ihigh;
        exponent_sign = -1;
    }
    let value = low as i32 * 256 + ihigh as i32 % 256;
    let exponent = ihigh as i32 / 256;
    (value_sign * value) as f64 * 2_f64.powi(exponent * exponent_sign)
}

/// C `getMetadataItems` (`extraheader.c:282`).
pub unsafe fn get_metadata_items(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    data_type: i32,
    val1: *mut f32,
    val2: *mut f32,
    num_vals: *mut i32,
    num_found: *mut i32,
    max_vals: i32,
    iz_piece: *mut i32,
) -> i32 {
    const KEYS: [&[u8]; 9] = [
        b"TiltAngle\0",
        b"N\0",
        b"StagePosition\0",
        b"Magnification\0",
        b"Intensity\0",
        b"ExposureDose\0",
        b"PixelSpacing\0",
        b"Defocus\0",
        b"ExposureTime\0",
    ];
    const WHICH: [i32; 9] = [2, 1, 3, 1, 2, 2, 2, 2, 2];
    *num_vals = 0;
    *num_found = 0;
    if !(1..=9).contains(&data_type) {
        b3d_error(
            ptr::null_mut(),
            format_args!(
                "getMetadataItems - type value {} is outside allowed range",
                adoc_type
            ),
        );
        return 1;
    }
    let mut val3 = 0.;
    get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        KEYS[(data_type - 1) as usize].as_ptr().cast(),
        WHICH[(data_type - 1) as usize],
        val1,
        val2,
        &mut val3,
        ptr::null_mut(),
        num_vals,
        num_found,
        max_vals,
        iz_piece,
    )
}

/// C Fortran wrapper `get_metadata_items` (`extraheader.c:305`).
pub unsafe fn get_metadata_items_fortran(
    ind_adoc: *mut i32,
    adoc_type: *mut i32,
    nz: *mut i32,
    data_type: *mut i32,
    val1: *mut f32,
    val2: *mut f32,
    num_vals: *mut i32,
    num_found: *mut i32,
    max_vals: *mut i32,
    iz_piece: *mut i32,
) {
    b3d_set_store_error(1);
    if get_metadata_items(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        *data_type,
        val1,
        val2,
        num_vals,
        num_found,
        *max_vals,
        iz_piece,
    ) != 0
    {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
}

/// C `getMetadataByKey` (`extraheader.c:345`).
pub unsafe fn get_metadata_by_key(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    key: *const c_char,
    value_type: i32,
    val1: *mut f32,
    val2: *mut f32,
    val3: *mut f32,
    val_string: *mut *mut c_char,
    num_vals: *mut i32,
    num_found: *mut i32,
    max_vals: i32,
    iz_piece: *mut i32,
) -> i32 {
    let names = [
        ADOC_ZVALUE_NAME.as_ptr(),
        c"Image".as_ptr(),
        ADOC_ZVALUE_NAME.as_ptr(),
    ];
    if adoc_set_current(ind_adoc) != 0 {
        b3d_error(
            ptr::null_mut(),
            format_args!("getMetadataByKey - Failed to set autodoc index"),
        );
        return 1;
    }
    *num_vals = 0;
    *num_found = 0;
    if value_type == 0 {
        if val_string.is_null() {
            b3d_error(
                ptr::null_mut(),
                format_args!(
                    "getMetadataByKey - Requested string items but called with NULL pointer to string array"
                ),
            );
            return 1;
        }
        for i in 0..max_vals {
            *val_string.add(i as usize) = ptr::null_mut();
        }
    }
    for i in 0..nz {
        let output = *iz_piece.add(i as usize);
        if output < 0 {
            b3d_error(
                ptr::null_mut(),
                format_args!("getMetadataByKey - Value array not designed for negative Z values"),
            );
            return 1;
        }
        if output >= max_vals {
            b3d_error(
                ptr::null_mut(),
                format_args!("getMetadataByKey - Array not big enough for data"),
            );
            return 1;
        }
        let mut section = i;
        if adoc_type == 3 {
            section = adoc_lookup_by_name_value(names[2], i);
            if section < 0 {
                continue;
            }
        }
        let name = names[(adoc_type - 1) as usize];
        let out = output as usize;
        match value_type {
            0 => {
                let mut string = ptr::null_mut();
                if adoc_get_string(name, section, key, &mut string) == 0 {
                    libc::free((*val_string.add(out)).cast());
                    *val_string.add(out) = string;
                    *val1.add(out) = 0.;
                    *num_found += 1;
                }
            }
            1 => {
                let mut value = 0;
                if adoc_get_integer(name, section, key, &mut value) == 0 {
                    *val1.add(out) = value as f32;
                    *num_found += 1;
                }
            }
            2 => {
                if adoc_get_float(name, section, key, val1.add(out)) == 0 {
                    *num_found += 1;
                }
            }
            3 => {
                if adoc_get_two_floats(name, section, key, val1.add(out), val2.add(out)) == 0 {
                    *num_found += 1;
                }
            }
            4 => {
                if adoc_get_three_floats(
                    name,
                    section,
                    key,
                    val1.add(out),
                    val2.add(out),
                    val3.add(out),
                ) == 0
                {
                    *num_found += 1;
                }
            }
            _ => {}
        }
        *num_vals = (*num_vals).max(output + 1);
    }
    0
}

/// C Fortran wrapper `get_metadata_by_key` (`extraheader.c:421`).  Character conversion is a Fortran ABI boundary.
pub unsafe fn get_metadata_by_key_fortran(
    ind_adoc: *mut i32,
    adoc_type: *mut i32,
    nz: *mut i32,
    key: *mut c_char,
    value_type: *mut i32,
    val1: *mut f32,
    val2: *mut f32,
    val3: *mut f32,
    val_string: *mut c_char,
    num_vals: *mut i32,
    num_found: *mut i32,
    max_vals: *mut i32,
    iz_piece: *mut i32,
    key_size: usize,
    val_size: usize,
) {
    b3d_set_store_error(1);
    *num_vals = 0;
    *num_found = 0;
    let key_bytes = core::slice::from_raw_parts(key.cast::<u8>(), key_size);
    let key_end = key_bytes
        .iter()
        .rposition(|byte| *byte != b' ')
        .map_or(0, |index| index + 1);
    let key_text =
        std::ffi::CString::new(&key_bytes[..key_end]).expect("Fortran key has an interior NUL");
    let mut strings = if *value_type == 0 {
        vec![ptr::null_mut(); *max_vals as usize]
    } else {
        Vec::new()
    };
    let strings_ptr = if *value_type == 0 {
        strings.as_mut_ptr()
    } else {
        ptr::null_mut()
    };
    if get_metadata_by_key(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        key_text.as_ptr(),
        *value_type,
        val1,
        val2,
        val3,
        strings_ptr,
        num_vals,
        num_found,
        *max_vals,
        iz_piece,
    ) != 0
    {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
    if *value_type == 0 {
        for ind in 0..*num_found {
            let destination = val_string.add(ind as usize * val_size);
            ptr::write_bytes(destination, b' ', val_size);
            if !strings[ind as usize].is_null() {
                let source = CStr::from_ptr(strings[ind as usize]).to_bytes();
                ptr::copy_nonoverlapping(
                    source.as_ptr().cast::<c_char>(),
                    destination,
                    source.len().min(val_size),
                );
            }
            libc::free(strings[ind as usize].cast());
        }
    }
}

/// C `getExtraHeaderPieces` (`extraheader.c:472`).
pub unsafe fn get_extra_header_pieces(
    array: *mut c_char,
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    ix_piece: *mut i32,
    iy_piece: *mut i32,
    iz_piece: *mut i32,
    num_pieces: *mut i32,
    max_piece: i32,
) -> i32 {
    *num_pieces = 0;
    if num_extra_bytes == 0 {
        return 0;
    }
    if nz > max_piece {
        b3d_error(
            ptr::null_mut(),
            format_args!("getExtraHeaderPieces - arrays not large enough for piece lists"),
        );
        return 1;
    }
    if nbytes == 0 || extra_is_nbytes_and_flags(nbytes, iflags) == 0 || ((iflags / 2) & 1) == 0 {
        return 0;
    }
    let mut ind = if iflags & 1 != 0 { 2 } else { 0 };
    for i in 0..nz {
        if ind > num_extra_bytes {
            return 0;
        }
        let p = array.add(ind as usize).cast::<u16>();
        *ix_piece.add(i as usize) = ptr::read_unaligned(p) as i32;
        *iy_piece.add(i as usize) = ptr::read_unaligned(p.add(1)) as i32;
        *iz_piece.add(i as usize) = ptr::read_unaligned(p.add(2)) as i32;
        ind += nbytes;
        *num_pieces = i + 1;
    }
    0
}

/// C Fortran wrapper `get_extra_header_pieces` (`extraheader.c:510`).
pub unsafe fn get_extra_header_pieces_fortran(
    array: *mut c_char,
    num_extra_bytes: *mut i32,
    nbytes: *mut i32,
    iflags: *mut i32,
    nz: *mut i32,
    ix_piece: *mut i32,
    iy_piece: *mut i32,
    iz_piece: *mut i32,
    num_pieces: *mut i32,
    max_piece: *mut i32,
) {
    b3d_set_store_error(1);
    if get_extra_header_pieces(
        array,
        *num_extra_bytes,
        *nbytes,
        *iflags,
        *nz,
        ix_piece,
        iy_piece,
        iz_piece,
        num_pieces,
        *max_piece,
    ) != 0
    {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
}

/// C `getMetadataPieces` (`extraheader.c:537`).
pub unsafe fn get_metadata_pieces(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    ix_piece: *mut i32,
    iy_piece: *mut i32,
    iz_piece: *mut i32,
    max_piece: i32,
    num_found: *mut i32,
) -> i32 {
    let names = [
        ADOC_ZVALUE_NAME.as_ptr(),
        c"Image".as_ptr(),
        ADOC_ZVALUE_NAME.as_ptr(),
    ];
    *num_found = 0;
    if nz > max_piece {
        b3d_error(
            ptr::null_mut(),
            format_args!("getMetadataPieces - Arrays not large enough for piece lists"),
        );
        return 1;
    }
    if adoc_set_current(ind_adoc) != 0 {
        b3d_error(
            ptr::null_mut(),
            format_args!("get_metadata_pieces - Failed to set autodoc index"),
        );
        return 1;
    }
    for i in 0..nz {
        let mut section = i;
        if adoc_type == 3 {
            section = adoc_lookup_by_name_value(names[2], i);
            if section < 0 {
                continue;
            }
        }
        if adoc_get_three_integers(
            names[(adoc_type - 1) as usize],
            section,
            c"PieceCoordinates".as_ptr(),
            ix_piece.add(i as usize),
            iy_piece.add(i as usize),
            iz_piece.add(i as usize),
        ) == 0
        {
            *num_found += 1;
        }
    }
    0
}

/// C Fortran wrapper `get_metadata_pieces` (`extraheader.c:567`).
pub unsafe fn get_metadata_pieces_fortran(
    ind_adoc: *mut i32,
    adoc_type: *mut i32,
    nz: *mut i32,
    ix_piece: *mut i32,
    iy_piece: *mut i32,
    iz_piece: *mut i32,
    max_piece: *mut i32,
    num_found: *mut i32,
) {
    b3d_set_store_error(1);
    if get_metadata_pieces(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        ix_piece,
        iy_piece,
        iz_piece,
        *max_piece,
        num_found,
    ) != 0
    {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
}

/// C `getMetadataWeightingDoses` (`extraheader.c:614`).
pub unsafe fn get_metadata_weighting_doses(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    iz_piece: *mut i32,
    bidir_num_invert: i32,
    prior_dose: *mut f32,
    sec_dose: *mut f32,
) -> i32 {
    let (mut min_dose, accum_dose) = {
        let mut zero = S_ZERO_DOSE.lock().unwrap();
        let value = *zero;
        if value.0 > 0. {
            *zero = (0., 0.);
            value
        } else {
            (0., 0.)
        }
    };
    let mut dummy1 = 0.;
    let mut dummy2 = 0.;
    let mut num_values = 0;
    let mut num_found = 0;
    if get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        c"ExposureDose".as_ptr(),
        2,
        sec_dose,
        &mut dummy1,
        &mut dummy2,
        ptr::null_mut(),
        &mut num_values,
        &mut num_found,
        nz,
        iz_piece,
    ) != 0
    {
        return 1;
    }
    if num_found < nz {
        b3d_error(
            ptr::null_mut(),
            format_args!(
                "getMetadataWeightingDoses - {} entries were found in autodoc file for ExposureDose",
                if num_found != 0 { "Not enough" } else { "No" }
            ),
        );
        return 2;
    }
    for i in 0..nz {
        if *sec_dose.add(i as usize) <= 0. && min_dose <= 0. {
            b3d_error(
                ptr::null_mut(),
                format_args!(
                    "getMetadataWeightingDoses - Some sections have 0 for ExposureDose in the autodoc file"
                ),
            );
            return 2;
        }
    }
    if get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        c"PriorRecordDose".as_ptr(),
        2,
        prior_dose,
        &mut dummy1,
        &mut dummy2,
        ptr::null_mut(),
        &mut num_values,
        &mut num_found,
        nz,
        iz_piece,
    ) != 0
    {
        return 1;
    }
    if num_found == nz && (min_dose <= 0. || accum_dose <= 0.01) {
        return 0;
    }
    let bidir = bidir_num_invert > 1 || bidir_num_invert < 0;
    let saved_error = b3d_get_store_error();
    if bidir {
        b3d_set_store_error(1);
    }
    let mut strings = vec![ptr::null_mut(); nz as usize];
    let ret = get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        c"DateTime".as_ptr(),
        0,
        prior_dose,
        &mut dummy1,
        &mut dummy2,
        strings.as_mut_ptr(),
        &mut num_values,
        &mut num_found,
        nz,
        iz_piece,
    );
    if ret != 0 || num_found == 0 || num_found < nz {
        for string in strings {
            libc::free(string.cast());
        }
        if bidir {
            prior_doses_from_image_doses(sec_dose, nz, bidir_num_invert, prior_dose);
            b3d_set_store_error(saved_error);
            return 0;
        }
        if num_found == 0 {
            prior_doses_from_image_doses(sec_dose, nz, 0, prior_dose);
            return -1;
        }
        return if ret != 0 { 1 } else { 2 };
    }
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut times = Vec::with_capacity(nz as usize);
    let mut bad = false;
    for (i, string) in strings.into_iter().enumerate() {
        let text = CStr::from_ptr(string).to_string_lossy();
        libc::free(string.cast());
        let parts: Vec<_> = text.split(|c| c == '-' || c == ' ' || c == ':').collect();
        if parts.len() != 6 {
            bad = true;
            break;
        }
        let month = months.iter().position(|&m| m == parts[1]);
        let parsed = (
            parts[0].parse::<i32>(),
            parts[2].parse::<i32>(),
            parts[3].parse::<i32>(),
            parts[4].parse::<i32>(),
            parts[5].parse::<i32>(),
        );
        if month.is_none()
            || parsed.0.is_err()
            || parsed.1.is_err()
            || parsed.2.is_err()
            || parsed.3.is_err()
            || parsed.4.is_err()
        {
            bad = true;
            break;
        }
        // `mktime` ordering is all the C source needs here; this monotonically sortable tuple is equivalent.
        times.push((
            (
                parsed.1.unwrap(),
                month.unwrap() as i32,
                parsed.0.unwrap(),
                parsed.2.unwrap(),
                parsed.3.unwrap(),
                parsed.4.unwrap(),
            ),
            i,
        ));
    }
    if bad {
        if bidir {
            prior_doses_from_image_doses(sec_dose, nz, bidir_num_invert, prior_dose);
            b3d_set_store_error(saved_error);
            return 0;
        }
        return 2;
    }
    times.sort_by_key(|entry| entry.0);
    *prior_dose.add(times[0].1) = 0.;
    for i in 1..times.len() {
        let dose = *sec_dose.add(times[i - 1].1);
        *prior_dose.add(times[i].1) = *prior_dose.add(times[i - 1].1)
            + if dose < min_dose && accum_dose >= 0.01 {
                accum_dose.max(dose)
            } else {
                dose
            };
    }
    if bidir {
        b3d_set_store_error(saved_error);
    }
    0
}

/// C Fortran wrapper `getmetadataweightingdoses` (`extraheader.c:797`).
pub unsafe fn get_metadata_weighting_doses_fortran(
    ind_adoc: *mut i32,
    adoc_type: *mut i32,
    nz: *mut i32,
    iz_piece: *mut i32,
    bidir_num_invert: *mut i32,
    prior_dose: *mut f32,
    sec_dose: *mut f32,
) -> i32 {
    let error = get_metadata_weighting_doses(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        iz_piece,
        *bidir_num_invert,
        prior_dose,
        sec_dose,
    );
    if error > 0 {
        panic!("ERROR: {:?}", CStr::from_ptr(b3d_get_error()));
    }
    error
}

/// C `setZeroDoseThreshAndAccum` (`extraheader.c:815`).
pub fn set_zero_dose_thresh_and_accum(thresh: f32, accum: f32) {
    *S_ZERO_DOSE.lock().unwrap() = (thresh, accum);
}

/// C `priorDosesFromImageDoses` (`extraheader.c:838`).
pub unsafe fn prior_doses_from_image_doses(
    sec_dose: *mut f32,
    nz: i32,
    mut bidir_num_invert: i32,
    prior_dose: *mut f32,
) {
    if bidir_num_invert > 1 {
        *prior_dose.add((bidir_num_invert - 1) as usize) = 0.;
        for ind in (0..bidir_num_invert - 1).rev() {
            *prior_dose.add(ind as usize) =
                *prior_dose.add((ind + 1) as usize) + *sec_dose.add((ind + 1) as usize);
        }
        *prior_dose.add(bidir_num_invert as usize) = *prior_dose.add(0) + *sec_dose.add(0);
        for ind in bidir_num_invert + 1..nz {
            *prior_dose.add(ind as usize) =
                *prior_dose.add((ind - 1) as usize) + *sec_dose.add((ind - 1) as usize);
        }
    } else if bidir_num_invert < 0 {
        bidir_num_invert = -bidir_num_invert;
        *prior_dose.add(bidir_num_invert as usize) = 0.;
        for ind in bidir_num_invert + 1..nz {
            *prior_dose.add(ind as usize) =
                *prior_dose.add((ind - 1) as usize) + *sec_dose.add((ind - 1) as usize);
        }
        *prior_dose.add((bidir_num_invert - 1) as usize) =
            *prior_dose.add((nz - 1) as usize) + *sec_dose.add((nz - 1) as usize);
        for ind in (0..bidir_num_invert - 1).rev() {
            *prior_dose.add(ind as usize) =
                *prior_dose.add((ind + 1) as usize) + *sec_dose.add((ind + 1) as usize);
        }
    } else {
        *prior_dose = 0.;
        for ind in 1..nz {
            *prior_dose.add(ind as usize) =
                *prior_dose.add((ind - 1) as usize) + *sec_dose.add((ind - 1) as usize);
        }
    }
}

/// C Fortran wrapper `priordosesfromimagedoses` (`extraheader.c:867`).
pub unsafe fn prior_doses_from_image_doses_fortran(
    sec_dose: *mut f32,
    nz: *mut i32,
    bidir_num_invert: *mut i32,
    prior_dose: *mut f32,
) {
    prior_doses_from_image_doses(sec_dose, *nz, *bidir_num_invert, prior_dose)
}

/// C `getExtraHeaderValue` (`extraheader.c:881`).
pub unsafe fn get_extra_header_value(
    ext_head: *mut c_void,
    offset: i32,
    value_type: i32,
    bval: *mut u8,
    sval: *mut i16,
    ival: *mut i32,
    fval: *mut f32,
    dval: *mut f64,
) -> i32 {
    let data = ext_head.cast::<u8>().add(offset as usize);
    match value_type {
        0 => *bval = *data,
        1 => *sval = ptr::read_unaligned(data.cast()),
        2 => *fval = ptr::read_unaligned(data.cast()),
        3 => *ival = ptr::read_unaligned(data.cast()),
        4 => *dval = ptr::read_unaligned(data.cast()),
        _ => return 1,
    }
    0
}

/// C Fortran wrapper `getextraheadervalue` (`extraheader.c:908`).
pub unsafe fn get_extra_header_value_fortran(
    ext_head: *mut c_void,
    offset: *mut i32,
    value_type: *mut i32,
    bval: *mut u8,
    sval: *mut i16,
    ival: *mut i32,
    fval: *mut f32,
    dval: *mut f64,
) -> i32 {
    get_extra_header_value(ext_head, *offset, *value_type, bval, sval, ival, fval, dval)
}

/// C `getExtraHeaderSecOffset` (`extraheader.c:967`).
pub unsafe fn get_extra_header_sec_offset(
    ext_head: *mut c_void,
    ext_size: i32,
    num_int: i32,
    num_real: i32,
    iz_sect: i32,
    offset: *mut i32,
    size: *mut i32,
) -> i32 {
    let mut maximum = 0;
    extra_header_sizes(
        ext_head,
        ext_size,
        num_int,
        num_real,
        iz_sect,
        offset,
        size,
        &mut maximum,
    )
}

/// C Fortran wrapper `getextraheadersecoffset` (`extraheader.c:976`).
pub unsafe fn get_extra_header_sec_offset_fortran(
    ext_head: *mut c_void,
    ext_size: *mut i32,
    num_int: *mut i32,
    num_real: *mut i32,
    iz_sect: *mut i32,
    offset: *mut i32,
    size: *mut i32,
) -> i32 {
    get_extra_header_sec_offset(
        ext_head, *ext_size, *num_int, *num_real, *iz_sect, offset, size,
    )
}

/// C `getExtraHeaderMaxSecSize` (`extraheader.c:988`).
pub unsafe fn get_extra_header_max_sec_size(
    ext_head: *mut c_void,
    ext_size: i32,
    num_int: i32,
    num_real: i32,
    num_sect: i32,
    max_size: *mut i32,
) -> i32 {
    let mut offset = 0;
    let mut size = 0;
    extra_header_sizes(
        ext_head,
        ext_size,
        num_int,
        num_real,
        num_sect - 1,
        &mut offset,
        &mut size,
        max_size,
    )
}

/// C Fortran wrapper `getextraheadermaxsecsize` (`extraheader.c:997`).
pub unsafe fn get_extra_header_max_sec_size_fortran(
    ext_head: *mut c_void,
    ext_size: *mut i32,
    num_int: *mut i32,
    num_real: *mut i32,
    num_sect: *mut i32,
    max_size: *mut i32,
) -> i32 {
    get_extra_header_max_sec_size(
        ext_head, *ext_size, *num_int, *num_real, *num_sect, max_size,
    )
}

/// C `copyExtraHeaderSection` (`extraheader.c:1012`).
pub unsafe fn copy_extra_header_section(
    extra_in: *mut c_void,
    size_in: i32,
    extra_out: *mut c_void,
    size_out: i32,
    num_int: i32,
    num_real: i32,
    iz_sect: i32,
    cumul_bytes_out: *mut i32,
) -> i32 {
    let mut offset = 0;
    let mut size = 0;
    let error = get_extra_header_sec_offset(
        extra_in,
        size_in,
        num_int,
        num_real,
        iz_sect,
        &mut offset,
        &mut size,
    );
    if error != 0 {
        return error;
    }
    if size + *cumul_bytes_out > size_out {
        return 3;
    }
    ptr::copy_nonoverlapping(
        extra_in.cast::<u8>().add(offset as usize),
        extra_out.cast::<u8>().add(*cumul_bytes_out as usize),
        size as usize,
    );
    *cumul_bytes_out += size;
    0
}

/// C Fortran wrapper `copyextraheadersection` (`extraheader.c:1026`).
pub unsafe fn copy_extra_header_section_fortran(
    extra_in: *mut c_void,
    size_in: *mut i32,
    extra_out: *mut c_void,
    size_out: *mut i32,
    num_int: *mut i32,
    num_real: *mut i32,
    iz_sect: *mut i32,
    cumul_bytes_out: *mut i32,
) -> i32 {
    copy_extra_header_section(
        extra_in,
        *size_in,
        extra_out,
        *size_out,
        *num_int,
        *num_real,
        *iz_sect,
        cumul_bytes_out,
    )
}

/// C `getFeiExtHeadAngleScale` (`extraheader.c:1037`).
pub unsafe fn get_fei_ext_head_angle_scale(ext_head: *mut c_void) -> f64 {
    if CStr::from_ptr(ext_head.cast::<c_char>().add(68)).to_bytes() == b"4.4.0.4981" {
        RADIANS_PER_DEGREE
    } else {
        1.
    }
}

/// C Fortran wrapper `getfeiextheadanglescale` (`extraheader.c:1043`).
pub unsafe fn get_fei_ext_head_angle_scale_fortran(ext_head: *mut c_void) -> f64 {
    get_fei_ext_head_angle_scale(ext_head)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sem_short_and_prior_doses_match_source() {
        assert_eq!(semshorts_to_float(1, 0), 256.);
        let mut dose = [1., 2., 3., 4.];
        let mut prior = [0.; 4];
        unsafe {
            prior_doses_from_image_doses(dose.as_mut_ptr(), 4, 2, prior.as_mut_ptr());
        }
        assert_eq!(prior, [2., 0., 3., 6.]);
    }
    #[test]
    fn serialem_offsets_and_copy_match_source() {
        let mut data = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut offset = 0;
        let mut size = 0;
        unsafe {
            assert_eq!(
                get_extra_header_sec_offset(
                    data.as_mut_ptr().cast(),
                    8,
                    2,
                    1,
                    1,
                    &mut offset,
                    &mut size
                ),
                0
            );
        }
        assert_eq!((offset, size), (2, 2));
        let mut out = [0_u8; 2];
        let mut cumulative = 0;
        unsafe {
            assert_eq!(
                copy_extra_header_section(
                    data.as_mut_ptr().cast(),
                    8,
                    out.as_mut_ptr().cast(),
                    2,
                    2,
                    1,
                    0,
                    &mut cumulative
                ),
                0
            );
        }
        assert_eq!(out, [1, 2]);
    }
    #[test]
    fn get_value_preserves_unaligned_representation() {
        let mut data = [0_u8; 12];
        data[1..5].copy_from_slice(&42_i32.to_ne_bytes());
        let mut value = 0;
        unsafe {
            assert_eq!(
                get_extra_header_value(
                    data.as_mut_ptr().cast(),
                    1,
                    3,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut value,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                0
            );
        }
        assert_eq!(value, 42);
    }
}
