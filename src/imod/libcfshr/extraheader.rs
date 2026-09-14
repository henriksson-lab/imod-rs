//! Translation of `IMOD/libcfshr/extraheader.c`.
#![allow(dead_code, unsafe_op_in_unsafe_fn)]

use core::ptr;
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

#[derive(Default)]
struct ZeroDoseSettings {
    threshold: f32,
    accumulated: f32,
}

static S_ZERO_DOSE: Mutex<ZeroDoseSettings> = Mutex::new(ZeroDoseSettings {
    threshold: 0.,
    accumulated: 0.,
});

/// C static `extraHeaderSizes` (`extraheader.c:915`).
pub fn extra_header_sizes(
    ext_head: &[u8],
    ext_size: i32,
    num_int: i32,
    num_real: i32,
    iz_sect: i32,
    offset: &mut i32,
    size: &mut i32,
    max_size: &mut i32,
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
        let Ok(ext_size) = usize::try_from(ext_size) else {
            return 2;
        };
        let Some(ext_head) = ext_head.get(..ext_size) else {
            return 2;
        };
        let mut data_offset = 0_usize;
        *offset = 0;
        for _ in 0..iz_sect {
            let Some(bytes) = ext_head.get(data_offset..data_offset + size_of::<i32>()) else {
                return 2;
            };
            let value = i32::from_ne_bytes(bytes.try_into().unwrap());
            if value < 0 || *offset + value > ext_size as i32 {
                return 2;
            }
            *offset += value;
            data_offset += value as usize;
            *max_size = (*max_size).max(value);
        }
        let Some(bytes) = ext_head.get(data_offset..data_offset + size_of::<i32>()) else {
            return 2;
        };
        *size = i32::from_ne_bytes(bytes.try_into().unwrap());
        *max_size = (*max_size).max(*size);
        return 0;
    }
    1
}

/// C static `freeValStrings` (`extraheader.c:821`).
///
/// Rust owns the strings, so clearing their containing vector drops every
/// value and needs neither a count nor a separate deallocation pass.
pub fn free_val_strings(val_strings: &mut Vec<Option<String>>) {
    val_strings.clear();
}

/// C `getExtraHeaderTilts` (`extraheader.c:74`).
pub fn get_extra_header_tilts(
    array: &[u8],
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    tilt: &mut [f32],
    num_tilts: &mut i32,
    iz_piece: &[i32],
) -> i32 {
    get_extra_header_items(
        array,
        num_extra_bytes,
        nbytes,
        iflags,
        nz,
        1,
        tilt,
        None,
        num_tilts,
        iz_piece,
    )
}

/// C Fortran wrapper `get_extra_header_tilts` (`extraheader.c:83`).
pub unsafe fn get_extra_header_tilts_fortran(
    array: *mut u8,
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
    let array = core::slice::from_raw_parts(array, (*num_extra_bytes).max(0) as usize);
    let tilt = core::slice::from_raw_parts_mut(tilt, (*max_tilts).max(0) as usize);
    let iz_piece = core::slice::from_raw_parts(iz_piece, (*nz).max(0) as usize);
    if get_extra_header_tilts(
        array,
        *num_extra_bytes,
        *nbytes,
        *iflags,
        *nz,
        tilt,
        &mut *num_tilts,
        iz_piece,
    ) != 0
    {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
    }
}

/// C `getExtraHeaderItems` (`extraheader.c:117`).
pub fn get_extra_header_items(
    array: &[u8],
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    itype: i32,
    val1: &mut [f32],
    mut val2: Option<&mut [f32]>,
    num_vals: &mut i32,
    iz_piece: &[i32],
) -> i32 {
    *num_vals = 0;
    let Ok(nz_usize) = usize::try_from(nz) else {
        return 1;
    };
    let Ok(extra_bytes) = usize::try_from(num_extra_bytes) else {
        return 0;
    };
    let Some(array) = array.get(..extra_bytes) else {
        return 0;
    };
    if val1.len() < nz_usize || iz_piece.len() < nz_usize {
        return 1;
    }
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
        // The FEI version string occupies bytes 68 through 78 of the complete
        // extended header.  A truncated header has no compatible version
        // marker, so it keeps the historical unit scale.
        scale = if num_extra_bytes >= 79 {
            get_fei_ext_head_angle_scale(array)
        } else {
            1.
        };
        let mut bt = 0;
        let mut st = 0;
        let mut mask = 0;
        let mut ft = 0.;
        let mut angle = 0.;
        if get_extra_header_value(
            array, 8, 3, &mut bt, &mut st, &mut mask, &mut ft, &mut angle,
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
        let value_index = iz_piece[i as usize];
        if value_index < 0 {
            b3d_error(
                None,
                format_args!(
                    "getExtraHeaderItems - Value array not designed for negative Z values"
                ),
            );
            return 1;
        }
        if value_index as usize >= val1.len() {
            b3d_error(
                None,
                format_args!("getExtraHeaderItems - Array not big enough for data"),
            );
            return 1;
        }
        if shorts {
            let Some(value_bytes) = array.get(ind as usize..ind as usize + 2) else {
                return 0;
            };
            let value = i16::from_ne_bytes(value_bytes.try_into().unwrap());
            match itype {
                1 => val1[value_index as usize] = value as f32 / 100.,
                3 => {
                    let Some(second_bytes) = array.get(ind as usize + 2..ind as usize + 4) else {
                        return 0;
                    };
                    val1[value_index as usize] = value as f32 / 25.;
                    let second = i16::from_ne_bytes(second_bytes.try_into().unwrap()) as f32 / 25.;
                    if let Some(values) = val2.as_deref_mut() {
                        values[value_index as usize] = second;
                    } else {
                        val1[value_index as usize] = second;
                    }
                }
                4 => val1[value_index as usize] = value as f32 * 100.,
                5 => val1[value_index as usize] = value as f32 / 25000.,
                6 => {
                    let Some(second_bytes) = array.get(ind as usize + 2..ind as usize + 4) else {
                        return 0;
                    };
                    val1[value_index as usize] = semshorts_to_float(
                        value,
                        i16::from_ne_bytes(second_bytes.try_into().unwrap()),
                    ) as f32
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
                array,
                ind as usize,
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
                array,
                (ind + fei_offset) as usize,
                4,
                &mut bt,
                &mut st,
                &mut itemp,
                &mut ft,
                &mut angle,
            ) != 0
            {
                val1[value_index as usize] = 0.;
            } else if itype == 10 {
                if i == 0 {
                    first_stamp = angle;
                }
                val1[value_index as usize] = (scale * (angle - first_stamp)) as f32;
            } else {
                val1[value_index as usize] = (scale * angle) as f32;
            }
        } else {
            let Some(value_bytes) = array.get(ind as usize..ind as usize + 4) else {
                return 0;
            };
            val1[value_index as usize] = f32::from_ne_bytes(value_bytes.try_into().unwrap());
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
    array: *mut u8,
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
    let array = core::slice::from_raw_parts(array, (*num_extra_bytes).max(0) as usize);
    let values1 = core::slice::from_raw_parts_mut(val1, (*max_vals).max(0) as usize);
    let iz_piece = core::slice::from_raw_parts(iz_piece, (*nz).max(0) as usize);
    let values2 = if val1 == val2 {
        None
    } else {
        Some(core::slice::from_raw_parts_mut(
            val2,
            (*max_vals).max(0) as usize,
        ))
    };
    if get_extra_header_items(
        array,
        *num_extra_bytes,
        *nbytes,
        *iflags,
        *nz,
        *itype,
        values1,
        values2,
        &mut *num_vals,
        iz_piece,
    ) != 0
    {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
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
pub fn get_metadata_items(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    data_type: i32,
    val1: &mut [f32],
    val2: &mut [f32],
    num_vals: &mut i32,
    num_found: &mut i32,
    iz_piece: &[i32],
) -> i32 {
    const KEYS: [&str; 9] = [
        "TiltAngle",
        "N",
        "StagePosition",
        "Magnification",
        "Intensity",
        "ExposureDose",
        "PixelSpacing",
        "Defocus",
        "ExposureTime",
    ];
    const WHICH: [i32; 9] = [2, 1, 3, 1, 2, 2, 2, 2, 2];
    *num_vals = 0;
    *num_found = 0;
    let Ok(nz_usize) = usize::try_from(nz) else {
        return 1;
    };
    if val1.len() < nz_usize || val2.len() < nz_usize || iz_piece.len() < nz_usize {
        return 1;
    }
    if !(1..=9).contains(&data_type) {
        b3d_error(
            None,
            format_args!(
                "getMetadataItems - type value {} is outside allowed range",
                adoc_type
            ),
        );
        return 1;
    }
    let mut val3 = 0.;
    unsafe {
        get_metadata_by_key(
            ind_adoc,
            adoc_type,
            nz,
            KEYS[(data_type - 1) as usize],
            WHICH[(data_type - 1) as usize],
            val1.as_mut_ptr(),
            val2.as_mut_ptr(),
            &mut val3,
            None,
            num_vals,
            num_found,
            nz,
            iz_piece.as_ptr().cast_mut(),
        )
    }
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
    let val1 = core::slice::from_raw_parts_mut(val1, *max_vals as usize);
    let val2 = core::slice::from_raw_parts_mut(val2, *max_vals as usize);
    let iz_piece = core::slice::from_raw_parts(iz_piece, *nz as usize);
    if get_metadata_items(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        *data_type,
        val1,
        val2,
        &mut *num_vals,
        &mut *num_found,
        iz_piece,
    ) != 0
    {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
    }
}

/// C `getMetadataByKey` (`extraheader.c:345`).
pub unsafe fn get_metadata_by_key(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    key: &str,
    value_type: i32,
    val1: *mut f32,
    val2: *mut f32,
    val3: *mut f32,
    val_string: Option<&mut [Option<String>]>,
    num_vals: *mut i32,
    num_found: *mut i32,
    max_vals: i32,
    iz_piece: *mut i32,
) -> i32 {
    let names: [&[u8]; 3] = [ADOC_ZVALUE_NAME, b"Image", ADOC_ZVALUE_NAME];
    if adoc_set_current(ind_adoc) != 0 {
        b3d_error(
            None,
            format_args!("getMetadataByKey - Failed to set autodoc index"),
        );
        return 1;
    }
    *num_vals = 0;
    *num_found = 0;
    let mut val_string = val_string;
    if value_type == 0 {
        if val_string.is_none() {
            b3d_error(
                None,
                format_args!(
                    "getMetadataByKey - Requested string items but called with NULL pointer to string array"
                ),
            );
            return 1;
        }
        let slots = val_string.as_deref_mut().unwrap();
        for i in 0..max_vals {
            slots[i as usize] = None;
        }
    }
    for i in 0..nz {
        let output = *iz_piece.add(i as usize);
        if output < 0 {
            b3d_error(
                None,
                format_args!("getMetadataByKey - Value array not designed for negative Z values"),
            );
            return 1;
        }
        if output >= max_vals {
            b3d_error(
                None,
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
                let mut bytes = Vec::new();
                if adoc_get_string(name, section, key.as_bytes(), &mut bytes) == 0 {
                    val_string.as_deref_mut().unwrap()[out] =
                        Some(String::from_utf8_lossy(&bytes).into_owned());
                    *val1.add(out) = 0.;
                    *num_found += 1;
                }
            }
            1 => {
                let mut value = 0;
                if adoc_get_integer(name, section, key.as_bytes(), &mut value) == 0 {
                    *val1.add(out) = value as f32;
                    *num_found += 1;
                }
            }
            2 => {
                if adoc_get_float(name, section, key.as_bytes(), &mut *val1.add(out)) == 0 {
                    *num_found += 1;
                }
            }
            3 => {
                if adoc_get_two_floats(
                    name,
                    section,
                    key.as_bytes(),
                    &mut *val1.add(out),
                    &mut *val2.add(out),
                ) == 0
                {
                    *num_found += 1;
                }
            }
            4 => {
                if adoc_get_three_floats(
                    name,
                    section,
                    key.as_bytes(),
                    &mut *val1.add(out),
                    &mut *val2.add(out),
                    &mut *val3.add(out),
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
    key: &[u8],
    value_type: *mut i32,
    val1: *mut f32,
    val2: *mut f32,
    val3: *mut f32,
    val_string: &mut [u8],
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
    let key_bytes = &key[..key_size];
    let key_end = key_bytes
        .iter()
        .rposition(|byte| *byte != b' ')
        .map_or(0, |index| index + 1);
    let key_text = String::from_utf8_lossy(&key_bytes[..key_end]).into_owned();
    let mut strings: Vec<Option<String>> = if *value_type == 0 {
        vec![None; *max_vals as usize]
    } else {
        Vec::new()
    };
    if get_metadata_by_key(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        &key_text,
        *value_type,
        val1,
        val2,
        val3,
        if *value_type == 0 {
            Some(&mut strings[..])
        } else {
            None
        },
        num_vals,
        num_found,
        *max_vals,
        iz_piece,
    ) != 0
    {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
    }
    if *value_type == 0 {
        for ind in 0..*num_found {
            let destination = &mut val_string[ind as usize * val_size..][..val_size];
            destination.fill(b' ');
            if let Some(source) = strings[ind as usize].as_deref() {
                let source = source.as_bytes();
                let n = source.len().min(val_size);
                destination[..n].copy_from_slice(&source[..n]);
            }
        }
    }
}

/// C `getExtraHeaderPieces` (`extraheader.c:472`).
pub fn get_extra_header_pieces(
    array: &[u8],
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    ix_piece: &mut [i32],
    iy_piece: &mut [i32],
    iz_piece: &mut [i32],
    num_pieces: &mut i32,
    max_piece: i32,
) -> i32 {
    *num_pieces = 0;
    if num_extra_bytes == 0 {
        return 0;
    }
    let Ok(nz) = usize::try_from(nz) else {
        return 1;
    };
    if nz > max_piece.max(0) as usize
        || ix_piece.len() < nz
        || iy_piece.len() < nz
        || iz_piece.len() < nz
    {
        b3d_error(
            None,
            format_args!("getExtraHeaderPieces - arrays not large enough for piece lists"),
        );
        return 1;
    }
    if nbytes == 0 || extra_is_nbytes_and_flags(nbytes, iflags) == 0 || ((iflags / 2) & 1) == 0 {
        return 0;
    }
    let Ok(extra_bytes) = usize::try_from(num_extra_bytes) else {
        return 0;
    };
    let Some(array) = array.get(..extra_bytes) else {
        return 0;
    };
    let mut ind = if iflags & 1 != 0 { 2_usize } else { 0 };
    for i in 0..nz {
        let Some(values) = array.get(ind..ind + 6) else {
            return 0;
        };
        ix_piece[i] = u16::from_ne_bytes(values[0..2].try_into().unwrap()) as i32;
        iy_piece[i] = u16::from_ne_bytes(values[2..4].try_into().unwrap()) as i32;
        iz_piece[i] = u16::from_ne_bytes(values[4..6].try_into().unwrap()) as i32;
        ind += nbytes as usize;
        *num_pieces = i as i32 + 1;
    }
    0
}

/// C Fortran wrapper `get_extra_header_pieces` (`extraheader.c:510`).
pub unsafe fn get_extra_header_pieces_fortran(
    array: *mut u8,
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
    let array = core::slice::from_raw_parts(array, (*num_extra_bytes).max(0) as usize);
    let ix_piece = core::slice::from_raw_parts_mut(ix_piece, (*max_piece).max(0) as usize);
    let iy_piece = core::slice::from_raw_parts_mut(iy_piece, (*max_piece).max(0) as usize);
    let iz_piece = core::slice::from_raw_parts_mut(iz_piece, (*max_piece).max(0) as usize);
    if get_extra_header_pieces(
        array,
        *num_extra_bytes,
        *nbytes,
        *iflags,
        *nz,
        ix_piece,
        iy_piece,
        iz_piece,
        &mut *num_pieces,
        *max_piece,
    ) != 0
    {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
    }
}

/// C `getMetadataPieces` (`extraheader.c:537`).
pub fn get_metadata_pieces(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    ix_piece: &mut [i32],
    iy_piece: &mut [i32],
    iz_piece: &mut [i32],
    max_piece: i32,
    num_found: &mut i32,
) -> i32 {
    let names: [&[u8]; 3] = [ADOC_ZVALUE_NAME, b"Image", ADOC_ZVALUE_NAME];
    *num_found = 0;
    let Ok(nz) = usize::try_from(nz) else {
        return 1;
    };
    if nz > max_piece.max(0) as usize
        || ix_piece.len() < nz
        || iy_piece.len() < nz
        || iz_piece.len() < nz
    {
        b3d_error(
            None,
            format_args!("getMetadataPieces - Arrays not large enough for piece lists"),
        );
        return 1;
    }
    if adoc_set_current(ind_adoc) != 0 {
        b3d_error(
            None,
            format_args!("get_metadata_pieces - Failed to set autodoc index"),
        );
        return 1;
    }
    for i in 0..nz {
        let mut section = i as i32;
        if adoc_type == 3 {
            section = adoc_lookup_by_name_value(names[2], i as i32);
            if section < 0 {
                continue;
            }
        }
        if adoc_get_three_integers(
            names[(adoc_type - 1) as usize],
            section,
            b"PieceCoordinates",
            &mut ix_piece[i],
            &mut iy_piece[i],
            &mut iz_piece[i],
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
    let ix_piece = core::slice::from_raw_parts_mut(ix_piece, (*max_piece).max(0) as usize);
    let iy_piece = core::slice::from_raw_parts_mut(iy_piece, (*max_piece).max(0) as usize);
    let iz_piece = core::slice::from_raw_parts_mut(iz_piece, (*max_piece).max(0) as usize);
    if get_metadata_pieces(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        ix_piece,
        iy_piece,
        iz_piece,
        *max_piece,
        &mut *num_found,
    ) != 0
    {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
    }
}

/// C `getMetadataWeightingDoses` (`extraheader.c:614`).
pub fn get_metadata_weighting_doses(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    iz_piece: &[i32],
    bidir_num_invert: i32,
    prior_dose: &mut [f32],
    sec_dose: &mut [f32],
) -> i32 {
    let Ok(nz_usize) = usize::try_from(nz) else {
        return 1;
    };
    if iz_piece.len() < nz_usize || prior_dose.len() < nz_usize || sec_dose.len() < nz_usize {
        return 1;
    }
    let (mut min_dose, accum_dose) = {
        let mut zero = S_ZERO_DOSE.lock().unwrap();
        if zero.threshold > 0. {
            let value = (zero.threshold, zero.accumulated);
            *zero = ZeroDoseSettings::default();
            value
        } else {
            (0., 0.)
        }
    };
    let mut dummy1 = 0.;
    let mut dummy2 = 0.;
    let mut num_values = 0;
    let mut num_found = 0;
    if unsafe {
        get_metadata_by_key(
            ind_adoc,
            adoc_type,
            nz,
            "ExposureDose",
            2,
            sec_dose.as_mut_ptr(),
            &mut dummy1,
            &mut dummy2,
            None,
            &mut num_values,
            &mut num_found,
            nz,
            iz_piece.as_ptr().cast_mut(),
        )
    } != 0
    {
        return 1;
    }
    if num_found < nz {
        b3d_error(
            None,
            format_args!(
                "getMetadataWeightingDoses - {} entries were found in autodoc file for ExposureDose",
                if num_found != 0 { "Not enough" } else { "No" }
            ),
        );
        return 2;
    }
    for dose in &sec_dose[..nz_usize] {
        if *dose <= 0. && min_dose <= 0. {
            b3d_error(
                None,
                format_args!(
                    "getMetadataWeightingDoses - Some sections have 0 for ExposureDose in the autodoc file"
                ),
            );
            return 2;
        }
    }
    if unsafe {
        get_metadata_by_key(
            ind_adoc,
            adoc_type,
            nz,
            "PriorRecordDose",
            2,
            prior_dose.as_mut_ptr(),
            &mut dummy1,
            &mut dummy2,
            None,
            &mut num_values,
            &mut num_found,
            nz,
            iz_piece.as_ptr().cast_mut(),
        )
    } != 0
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
    let mut strings: Vec<Option<String>> = vec![None; nz_usize];
    let ret = unsafe {
        get_metadata_by_key(
            ind_adoc,
            adoc_type,
            nz,
            "DateTime",
            0,
            prior_dose.as_mut_ptr(),
            &mut dummy1,
            &mut dummy2,
            Some(&mut strings[..]),
            &mut num_values,
            &mut num_found,
            nz,
            iz_piece.as_ptr().cast_mut(),
        )
    };
    if ret != 0 || num_found == 0 || num_found < nz {
        if bidir {
            prior_doses_from_image_doses(
                &sec_dose[..nz_usize],
                bidir_num_invert,
                &mut prior_dose[..nz_usize],
            );
            b3d_set_store_error(saved_error);
            return 0;
        }
        if num_found == 0 {
            prior_doses_from_image_doses(&sec_dose[..nz_usize], 0, &mut prior_dose[..nz_usize]);
            return -1;
        }
        return if ret != 0 { 1 } else { 2 };
    }
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut times = Vec::with_capacity(nz_usize);
    let mut bad = false;
    for (i, string) in strings.into_iter().enumerate() {
        let string = string.unwrap_or_default();
        let parts: Vec<_> = string.split(['-', ' ', ':']).collect();
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
            prior_doses_from_image_doses(
                &sec_dose[..nz_usize],
                bidir_num_invert,
                &mut prior_dose[..nz_usize],
            );
            b3d_set_store_error(saved_error);
            return 0;
        }
        return 2;
    }
    times.sort_by_key(|entry| entry.0);
    prior_dose[times[0].1] = 0.;
    for i in 1..times.len() {
        let dose = sec_dose[times[i - 1].1];
        prior_dose[times[i].1] = prior_dose[times[i - 1].1]
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
    let iz_pieces = core::slice::from_raw_parts(iz_piece, *nz as usize);
    let prior_doses = core::slice::from_raw_parts_mut(prior_dose, *nz as usize);
    let section_doses = core::slice::from_raw_parts_mut(sec_dose, *nz as usize);
    let error = get_metadata_weighting_doses(
        *ind_adoc - 1,
        *adoc_type,
        *nz,
        iz_pieces,
        *bidir_num_invert,
        prior_doses,
        section_doses,
    );
    if error > 0 {
        eprintln!("\nERROR: {}", b3d_get_error());
        std::process::exit(1);
    }
    error
}

/// C `setZeroDoseThreshAndAccum` (`extraheader.c:815`).
pub fn set_zero_dose_thresh_and_accum(thresh: f32, accum: f32) {
    *S_ZERO_DOSE.lock().unwrap() = ZeroDoseSettings {
        threshold: thresh,
        accumulated: accum,
    };
}

/// C `priorDosesFromImageDoses` (`extraheader.c:838`).
pub fn prior_doses_from_image_doses(
    sec_dose: &[f32],
    mut bidir_num_invert: i32,
    prior_dose: &mut [f32],
) {
    assert_eq!(sec_dose.len(), prior_dose.len());
    let nz = sec_dose.len() as i32;
    if bidir_num_invert > 1 {
        prior_dose[(bidir_num_invert - 1) as usize] = 0.;
        for ind in (0..bidir_num_invert - 1).rev() {
            prior_dose[ind as usize] =
                prior_dose[(ind + 1) as usize] + sec_dose[(ind + 1) as usize];
        }
        prior_dose[bidir_num_invert as usize] = prior_dose[0] + sec_dose[0];
        for ind in bidir_num_invert + 1..nz {
            prior_dose[ind as usize] =
                prior_dose[(ind - 1) as usize] + sec_dose[(ind - 1) as usize];
        }
    } else if bidir_num_invert < 0 {
        bidir_num_invert = -bidir_num_invert;
        prior_dose[bidir_num_invert as usize] = 0.;
        for ind in bidir_num_invert + 1..nz {
            prior_dose[ind as usize] =
                prior_dose[(ind - 1) as usize] + sec_dose[(ind - 1) as usize];
        }
        prior_dose[(bidir_num_invert - 1) as usize] =
            prior_dose[(nz - 1) as usize] + sec_dose[(nz - 1) as usize];
        for ind in (0..bidir_num_invert - 1).rev() {
            prior_dose[ind as usize] =
                prior_dose[(ind + 1) as usize] + sec_dose[(ind + 1) as usize];
        }
    } else {
        prior_dose[0] = 0.;
        for ind in 1..nz {
            prior_dose[ind as usize] =
                prior_dose[(ind - 1) as usize] + sec_dose[(ind - 1) as usize];
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
    let section_doses = unsafe { core::slice::from_raw_parts(sec_dose, *nz as usize) };
    let prior_doses = unsafe { core::slice::from_raw_parts_mut(prior_dose, *nz as usize) };
    prior_doses_from_image_doses(section_doses, *bidir_num_invert, prior_doses)
}

/// C `getExtraHeaderValue` (`extraheader.c:881`).
pub fn get_extra_header_value(
    ext_head: &[u8],
    offset: usize,
    value_type: i32,
    bval: &mut u8,
    sval: &mut i16,
    ival: &mut i32,
    fval: &mut f32,
    dval: &mut f64,
) -> i32 {
    match value_type {
        0 => {
            let Some(value) = ext_head.get(offset) else {
                return 1;
            };
            *bval = *value;
        }
        1 => {
            let Some(bytes) = ext_head.get(offset..offset + size_of::<i16>()) else {
                return 1;
            };
            *sval = i16::from_ne_bytes(bytes.try_into().unwrap());
        }
        2 => {
            let Some(bytes) = ext_head.get(offset..offset + size_of::<f32>()) else {
                return 1;
            };
            *fval = f32::from_ne_bytes(bytes.try_into().unwrap());
        }
        3 => {
            let Some(bytes) = ext_head.get(offset..offset + size_of::<i32>()) else {
                return 1;
            };
            *ival = i32::from_ne_bytes(bytes.try_into().unwrap());
        }
        4 => {
            let Some(bytes) = ext_head.get(offset..offset + size_of::<f64>()) else {
                return 1;
            };
            *dval = f64::from_ne_bytes(bytes.try_into().unwrap());
        }
        _ => return 1,
    }
    0
}

/// C Fortran wrapper `getextraheadervalue` (`extraheader.c:908`).
pub unsafe fn get_extra_header_value_fortran(
    ext_head: *mut u8,
    offset: *mut i32,
    value_type: *mut i32,
    bval: *mut u8,
    sval: *mut i16,
    ival: *mut i32,
    fval: *mut f32,
    dval: *mut f64,
) -> i32 {
    let data = unsafe { ext_head.add(*offset as usize) };
    unsafe {
        match *value_type {
            0 => *bval = *data,
            1 => *sval = ptr::read_unaligned(data.cast()),
            2 => *fval = ptr::read_unaligned(data.cast()),
            3 => *ival = ptr::read_unaligned(data.cast()),
            4 => *dval = ptr::read_unaligned(data.cast()),
            _ => return 1,
        }
    }
    0
}

/// C `getExtraHeaderSecOffset` (`extraheader.c:967`).
pub fn get_extra_header_sec_offset(
    ext_head: &[u8],
    ext_size: i32,
    num_int: i32,
    num_real: i32,
    iz_sect: i32,
    offset: &mut i32,
    size: &mut i32,
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
    ext_head: *mut u8,
    ext_size: *mut i32,
    num_int: *mut i32,
    num_real: *mut i32,
    iz_sect: *mut i32,
    offset: *mut i32,
    size: *mut i32,
) -> i32 {
    let bytes = unsafe { core::slice::from_raw_parts(ext_head, (*ext_size).max(0) as usize) };
    get_extra_header_sec_offset(
        bytes,
        *ext_size,
        *num_int,
        *num_real,
        *iz_sect,
        unsafe { &mut *offset },
        unsafe { &mut *size },
    )
}

/// C `getExtraHeaderMaxSecSize` (`extraheader.c:988`).
pub fn get_extra_header_max_sec_size(
    ext_head: &[u8],
    ext_size: i32,
    num_int: i32,
    num_real: i32,
    num_sect: i32,
    max_size: &mut i32,
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
    ext_head: *mut u8,
    ext_size: *mut i32,
    num_int: *mut i32,
    num_real: *mut i32,
    num_sect: *mut i32,
    max_size: *mut i32,
) -> i32 {
    let bytes = unsafe { core::slice::from_raw_parts(ext_head, (*ext_size).max(0) as usize) };
    get_extra_header_max_sec_size(bytes, *ext_size, *num_int, *num_real, *num_sect, unsafe {
        &mut *max_size
    })
}

/// C `copyExtraHeaderSection` (`extraheader.c:1012`).
pub fn copy_extra_header_section(
    extra_in: &[u8],
    extra_out: &mut [u8],
    num_int: i32,
    num_real: i32,
    iz_sect: i32,
    cumul_bytes_out: &mut i32,
) -> i32 {
    let mut offset = 0;
    let mut size = 0;
    let error = get_extra_header_sec_offset(
        extra_in,
        extra_in.len() as i32,
        num_int,
        num_real,
        iz_sect,
        &mut offset,
        &mut size,
    );
    if error != 0 {
        return error;
    }
    if size < 0
        || *cumul_bytes_out < 0
        || size as usize + *cumul_bytes_out as usize > extra_out.len()
    {
        return 3;
    }
    extra_out[*cumul_bytes_out as usize..*cumul_bytes_out as usize + size as usize]
        .copy_from_slice(&extra_in[offset as usize..offset as usize + size as usize]);
    *cumul_bytes_out += size;
    0
}

/// C Fortran wrapper `copyextraheadersection` (`extraheader.c:1026`).
pub unsafe fn copy_extra_header_section_fortran(
    extra_in: *mut u8,
    size_in: *mut i32,
    extra_out: *mut u8,
    size_out: *mut i32,
    num_int: *mut i32,
    num_real: *mut i32,
    iz_sect: *mut i32,
    cumul_bytes_out: *mut i32,
) -> i32 {
    let input = unsafe { core::slice::from_raw_parts(extra_in, (*size_in).max(0) as usize) };
    let output = unsafe { core::slice::from_raw_parts_mut(extra_out, (*size_out).max(0) as usize) };
    copy_extra_header_section(input, output, *num_int, *num_real, *iz_sect, unsafe {
        &mut *cumul_bytes_out
    })
}

/// C `getFeiExtHeadAngleScale` (`extraheader.c:1037`).
///
/// The FEI version field is at byte 68 and includes its terminating NUL, so a
/// header shorter than 79 bytes cannot carry the legacy marker.
pub fn get_fei_ext_head_angle_scale(ext_head: &[u8]) -> f64 {
    /* `strcmp((char *)extHead + 68, "4.4.0.4981")`: the version string in the
    FEI extended header, NUL-terminated in the file's own bytes. */
    if ext_head.get(68..79) == Some(b"4.4.0.4981\0") {
        RADIANS_PER_DEGREE
    } else {
        1.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_strings_are_owned_rust_text() {
        let mut strings = vec![Some(String::from("first")), Some(String::from("second"))];
        free_val_strings(&mut strings);
        assert!(strings.is_empty());
    }

    #[test]
    fn sem_short_and_prior_doses_match_source() {
        assert_eq!(semshorts_to_float(1, 0), 256.);
        let mut dose = [1., 2., 3., 4.];
        let mut prior = [0.; 4];
        prior_doses_from_image_doses(&dose, 2, &mut prior);
        assert_eq!(prior, [2., 0., 3., 6.]);
    }
    #[test]
    fn serialem_offsets_and_copy_match_source() {
        let mut data = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut offset = 0;
        let mut size = 0;
        assert_eq!(
            get_extra_header_sec_offset(&data, 8, 2, 1, 1, &mut offset, &mut size),
            0
        );
        assert_eq!((offset, size), (2, 2));
        let mut out = [0_u8; 2];
        let mut cumulative = 0;
        assert_eq!(
            copy_extra_header_section(&data, &mut out, 2, 1, 0, &mut cumulative),
            0
        );
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn fei_section_offsets_reject_truncated_length_records() {
        let truncated = [4_u8, 0, 0];
        let (mut offset, mut size, mut maximum) = (0, 0, 0);
        assert_eq!(
            extra_header_sizes(
                &truncated,
                truncated.len() as i32,
                -MRC_EXT_TYPE_FEI,
                0,
                0,
                &mut offset,
                &mut size,
                &mut maximum,
            ),
            2
        );
    }
    #[test]
    fn get_value_preserves_unaligned_representation() {
        let mut data = [0_u8; 12];
        data[1..5].copy_from_slice(&42_i32.to_ne_bytes());
        let (mut byte, mut short, mut value, mut float, mut double) = (0, 0, 0, 0., 0.);
        assert_eq!(
            get_extra_header_value(
                &data,
                1,
                3,
                &mut byte,
                &mut short,
                &mut value,
                &mut float,
                &mut double,
            ),
            0
        );
        assert_eq!(value, 42);
    }

    #[test]
    fn scalar_reader_rejects_truncated_values() {
        let data = [0_u8; 3];
        let (mut byte, mut short, mut integer, mut float, mut double) = (0, 0, 0, 0., 0.);
        assert_eq!(
            get_extra_header_value(
                &data,
                0,
                3,
                &mut byte,
                &mut short,
                &mut integer,
                &mut float,
                &mut double,
            ),
            1
        );
    }

    #[test]
    fn fei_angle_scale_uses_only_the_complete_legacy_version_field() {
        let mut legacy = [0_u8; 79];
        legacy[68..79].copy_from_slice(b"4.4.0.4981\0");
        assert_eq!(get_fei_ext_head_angle_scale(&legacy), RADIANS_PER_DEGREE);
        assert_eq!(get_fei_ext_head_angle_scale(&legacy[..78]), 1.);

        legacy[78] = b'X';
        assert_eq!(get_fei_ext_head_angle_scale(&legacy), 1.);
    }
}
