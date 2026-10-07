//! Translation of `IMOD/libcfshr/extraheader.c`.

use std::sync::Mutex;

use super::autodoc::{
    ADOC_ZVALUE_NAME, adoc_get_float, adoc_get_integer, adoc_get_string, adoc_get_three_floats,
    adoc_get_three_integers, adoc_get_two_floats, adoc_lookup_by_name_value, adoc_set_current,
};
use super::b3dutil::{
    ImodFile, b3d_error, b3d_get_error, b3d_get_store_error, b3d_set_store_error, exit,
    extra_is_nbytes_and_flags,
};
use std::io::Write;

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
fn extra_header_sizes(
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

/// C Fortran wrapper `get_extra_header_tilts` (`extraheader.c:83`): exits
/// with an `ERROR:` string upon error.
pub fn get_extra_header_tilts_fortran(
    array: &[u8],
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    tilt: &mut [f32],
    num_tilts: &mut i32,
    iz_piece: &[i32],
) {
    b3d_set_store_error(1);
    if get_extra_header_tilts(
        array,
        num_extra_bytes,
        nbytes,
        iflags,
        nz,
        tilt,
        num_tilts,
        iz_piece,
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
    }
}

/// C Fortran wrapper `get_extra_header_items` (`extraheader.c:236`): exits
/// with an `ERROR:` string upon error.  A caller passing the same array as
/// `val1` and `val2` passes `None` for `val2`.
pub fn get_extra_header_items_fortran(
    array: &[u8],
    num_extra_bytes: i32,
    nbytes: i32,
    iflags: i32,
    nz: i32,
    itype: i32,
    val1: &mut [f32],
    val2: Option<&mut [f32]>,
    num_vals: &mut i32,
    iz_piece: &[i32],
) {
    b3d_set_store_error(1);
    if get_extra_header_items(
        array,
        num_extra_bytes,
        nbytes,
        iflags,
        nz,
        itype,
        val1,
        val2,
        num_vals,
        iz_piece,
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
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
    get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        KEYS[(data_type - 1) as usize],
        WHICH[(data_type - 1) as usize],
        val1,
        val2,
        core::slice::from_mut(&mut val3),
        None,
        num_vals,
        num_found,
        nz,
        iz_piece,
    )
}

/// C Fortran wrapper `get_metadata_items` (`extraheader.c:305`): `indAdoc`
/// is 1-based; exits with an `ERROR:` string upon error.
pub fn get_metadata_items_fortran(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    data_type: i32,
    val1: &mut [f32],
    val2: &mut [f32],
    num_vals: &mut i32,
    num_found: &mut i32,
    iz_piece: &[i32],
) {
    b3d_set_store_error(1);
    if get_metadata_items(
        ind_adoc - 1,
        adoc_type,
        nz,
        data_type,
        val1,
        val2,
        num_vals,
        num_found,
        iz_piece,
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
    }
}

/// C `getMetadataByKey` (`extraheader.c:345`).
pub fn get_metadata_by_key(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    key: &str,
    value_type: i32,
    val1: &mut [f32],
    val2: &mut [f32],
    val3: &mut [f32],
    val_string: Option<&mut [Option<String>]>,
    num_vals: &mut i32,
    num_found: &mut i32,
    max_vals: i32,
    iz_piece: &[i32],
) -> i32 {
    let names: [&[u8]; 3] = [ADOC_ZVALUE_NAME, b"Image", ADOC_ZVALUE_NAME];
    if adoc_set_current(ind_adoc).is_err() {
        b3d_error(
            None,
            format_args!("getMetadataByKey - Failed to set autodoc index"),
        );
        return 1;
    }
    *num_vals = 0;
    *num_found = 0;
    let Ok(nz) = usize::try_from(nz) else {
        return 1;
    };
    let Ok(max_vals) = usize::try_from(max_vals) else {
        return 1;
    };
    if iz_piece.len() < nz || val1.len() < max_vals {
        b3d_error(
            None,
            format_args!("getMetadataByKey - Array not big enough for data"),
        );
        return 1;
    }
    if value_type == 3 && val2.len() < max_vals
        || value_type == 4 && (val2.len() < max_vals || val3.len() < max_vals)
    {
        b3d_error(
            None,
            format_args!("getMetadataByKey - Array not big enough for data"),
        );
        return 1;
    }
    let Some(name) = adoc_type
        .checked_sub(1)
        .and_then(|index| names.get(index as usize))
    else {
        b3d_error(
            None,
            format_args!("getMetadataByKey - Invalid autodoc type"),
        );
        return 1;
    };
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
        if slots.len() < max_vals {
            b3d_error(
                None,
                format_args!("getMetadataByKey - Array not big enough for data"),
            );
            return 1;
        }
        for slot in &mut slots[..max_vals] {
            *slot = None;
        }
    }
    for i in 0..nz {
        let output = iz_piece[i];
        if output < 0 {
            b3d_error(
                None,
                format_args!("getMetadataByKey - Value array not designed for negative Z values"),
            );
            return 1;
        }
        if output as usize >= max_vals {
            b3d_error(
                None,
                format_args!("getMetadataByKey - Array not big enough for data"),
            );
            return 1;
        }
        let mut section = i as i32;
        if adoc_type == 3 {
            section = adoc_lookup_by_name_value(names[2], i as i32);
            if section < 0 {
                continue;
            }
        }
        let out = output as usize;
        match value_type {
            0 => {
                let mut bytes = Vec::new();
                if adoc_get_string(name, section, key.as_bytes(), &mut bytes) == 0 {
                    // Take ownership of the valid-UTF-8 buffer (the C keeps
                    // its one `strdup`); lossy conversion only on invalid bytes.
                    val_string.as_deref_mut().unwrap()[out] =
                        Some(String::from_utf8(bytes).unwrap_or_else(|e| {
                            String::from_utf8_lossy(e.as_bytes()).into_owned()
                        }));
                    val1[out] = 0.;
                    *num_found += 1;
                }
            }
            1 => {
                let mut value = 0;
                if adoc_get_integer(name, section, key.as_bytes(), &mut value) == 0 {
                    val1[out] = value as f32;
                    *num_found += 1;
                }
            }
            2 => {
                if adoc_get_float(name, section, key.as_bytes(), &mut val1[out]) == 0 {
                    *num_found += 1;
                }
            }
            3 => {
                if adoc_get_two_floats(
                    name,
                    section,
                    key.as_bytes(),
                    &mut val1[out],
                    &mut val2[out],
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
                    &mut val1[out],
                    &mut val2[out],
                    &mut val3[out],
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

/// C Fortran wrapper `get_metadata_by_key` (`extraheader.c:421`): `indAdoc`
/// is 1-based, an error return is printed and exits, and for string items the
/// first `numFound` returned strings are copied into the character array
/// `valString` (`c2fString`: truncated to `val_size` characters; the caller
/// trims the blank padding) -- by position, so a string beyond `numFound` is
/// dropped and a missing one below it becomes a blank.  Elements past
/// `numFound` are left as they were.
pub fn get_metadata_by_key_fortran(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    key: &str,
    value_type: i32,
    val1: &mut [f32],
    val2: &mut [f32],
    val3: &mut [f32],
    val_string: &mut [String],
    val_size: usize,
    num_vals: &mut i32,
    num_found: &mut i32,
    max_vals: i32,
    iz_piece: &[i32],
) {
    let mut new_strings: Vec<Option<String>> = Vec::new();
    b3d_set_store_error(1);
    *num_vals = 0;
    *num_found = 0;
    // `f2cString` strips the trailing blanks of the Fortran key.
    let ckey = key.trim_end_matches(' ');
    if value_type == 0 {
        new_strings = vec![None; max_vals.max(0) as usize];
    }
    if get_metadata_by_key(
        ind_adoc - 1,
        adoc_type,
        nz,
        ckey,
        value_type,
        val1,
        val2,
        val3,
        if value_type == 0 {
            Some(&mut new_strings[..])
        } else {
            None
        },
        num_vals,
        num_found,
        max_vals,
        iz_piece,
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
    }

    /* Ignore strings not long enough... */
    if value_type == 0 {
        for ind in 0..*num_found as usize {
            match new_strings[ind].take() {
                Some(string) => {
                    let mut end = string.len().min(val_size);
                    while !string.is_char_boundary(end) {
                        end -= 1;
                    }
                    val_string[ind] = string[..end].to_owned();
                }
                None => val_string[ind] = " ".to_owned(),
            }
        }
    }
}

/// C `getExtraHeaderPieces` (`extraheader.c:472`).
///
/// Reads the three unsigned shorts at each section's offset; bytes past the
/// end of `array` read as zero (the source reads whatever follows).
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
    let short_at = |offset: i32| -> i32 {
        let offset = offset as usize;
        match array.get(offset..offset + 2) {
            Some(bytes) => u16::from_ne_bytes([bytes[0], bytes[1]]) as i32,
            None => 0,
        }
    };
    *num_pieces = 0;
    if num_extra_bytes == 0 {
        return 0;
    }
    if nz > max_piece {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!("getExtraHeaderPieces - arrays not large enough for piece lists\n"),
        );
        return 1;
    }

    /* if data are packed as shorts, see if the montage flag is set
     * set starting index based on whether there are tilt angles too */
    let shorts = extra_is_nbytes_and_flags(nbytes, iflags);
    if nbytes == 0 || shorts == 0 || (iflags / 2) % 2 == 0 {
        return 0;
    }
    let mut ind = 0_i32;
    if iflags % 2 != 0 {
        ind = 2;
    }
    for i in 0..nz.max(0) as usize {
        if ind > num_extra_bytes {
            return 0;
        }
        ix_piece[i] = short_at(ind);
        iy_piece[i] = short_at(ind + 2);
        iz_piece[i] = short_at(ind + 4);
        ind += nbytes;
        *num_pieces = i as i32 + 1;
    }
    0
}

/// C Fortran wrapper `get_extra_header_pieces` (`extraheader.c:510`): exits
/// with an `ERROR:` string upon error.
pub fn get_extra_header_pieces_fortran(
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
) {
    b3d_set_store_error(1);
    if get_extra_header_pieces(
        array,
        num_extra_bytes,
        nbytes,
        iflags,
        nz,
        ix_piece,
        iy_piece,
        iz_piece,
        num_pieces,
        max_piece,
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
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
    let sect_names: [&[u8]; 3] = [ADOC_ZVALUE_NAME, b"Image", ADOC_ZVALUE_NAME];
    *num_found = 0;

    if nz > max_piece {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!("getMetadataPieces - Arrays not large enough for piece lists"),
        );
        return 1;
    }
    if adoc_set_current(ind_adoc).is_err() {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!("get_metadata_pieces - Failed to set autodoc index"),
        );
        return 1;
    }
    let name = sect_names[(adoc_type - 1) as usize];
    for i in 0..nz.max(0) {
        let mut ind = i;
        if adoc_type == 3 {
            ind = adoc_lookup_by_name_value(name, i);
            if ind < 0 {
                continue;
            }
        }
        let iu = i as usize;
        let (mut ix, mut iy, mut iz) = (ix_piece[iu], iy_piece[iu], iz_piece[iu]);
        if adoc_get_three_integers(name, ind, b"PieceCoordinates", &mut ix, &mut iy, &mut iz) == 0 {
            *num_found += 1;
        }
        (ix_piece[iu], iy_piece[iu], iz_piece[iu]) = (ix, iy, iz);
    }
    0
}

/// C Fortran wrapper `get_metadata_pieces` (`extraheader.c:567`): `indAdoc`
/// is 1-based; exits with an `ERROR:` string upon error.
pub fn get_metadata_pieces_fortran(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    ix_piece: &mut [i32],
    iy_piece: &mut [i32],
    iz_piece: &mut [i32],
    max_piece: i32,
    num_found: &mut i32,
) {
    b3d_set_store_error(1);
    if get_metadata_pieces(
        ind_adoc - 1,
        adoc_type,
        nz,
        ix_piece,
        iy_piece,
        iz_piece,
        max_piece,
        num_found,
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
    }
}

/// C `getMetadataWeightingDoses` (`extraheader.c:614`).
///
/// `mktime` is the C library's (the conversion depends on the local time zone
/// and normalises out-of-range fields, and the sort compares its results with
/// `difftime`), and the `sscanf("%d-%3s-%d %d:%d:%d")` scan is translated in
/// place: a conversion that fails leaves that field and every later one
/// holding what the previous section's scan put there, as the source's `tim`
/// and `monthBuf` do (their first values are uninitialised in the source;
/// zero and empty here).
pub fn get_metadata_weighting_doses(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    iz_piece: &[i32],
    bidir_num_invert: i32,
    prior_dose: &mut [f32],
    sec_dose: &mut [f32],
) -> i32 {
    let mut dummy1 = 0.0_f32;
    let mut dummy2 = 0.0_f32;
    let mut num_vals = 0_i32;
    let mut num_found = 0_i32;
    let mut ret_val = 0_i32;
    let mut val_strings: Vec<Option<String>>;
    let mut full_times: Vec<libc::time_t> = Vec::new();
    let mut min_dose = 0.0_f32;
    let mut replace_dose = 0.0_f32;
    let mut accum_dose = 0.0_f32;
    let mut time_inds: Vec<i32> = Vec::new();
    let mut month_buf: Vec<u8> = Vec::new();
    let is_bidir = bidir_num_invert > 1 || bidir_num_invert < 0;
    let save_error = b3d_get_store_error();
    // SAFETY: an all-zero `tm` is a valid value (plain integers and, on glibc,
    // a null `tm_zone` pointer that `mktime` does not read).
    let mut tim: libc::tm = unsafe { core::mem::zeroed() };
    let months: [&[u8]; 12] = [
        b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov",
        b"Dec",
    ];
    {
        let mut zero = S_ZERO_DOSE.lock().unwrap();
        if zero.threshold > 0. {
            min_dose = zero.threshold;
            zero.threshold = 0.;
            accum_dose = zero.accumulated;
            zero.accumulated = 0.;
            replace_dose = 0.001_f64.min(zero.threshold as f64 / 10.) as f32;
        }
    }

    /* ExposureDose HAS to be there */
    if get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        "ExposureDose",
        2,
        sec_dose,
        core::slice::from_mut(&mut dummy1),
        core::slice::from_mut(&mut dummy2),
        None,
        &mut num_vals,
        &mut num_found,
        nz,
        iz_piece,
    ) != 0
    {
        return 1;
    }
    if num_found < nz {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "getMetadataWeightingDoses - {} entries were found in autodoc file for \
                 ExposureDose\n",
                if num_found != 0 { "Not enough" } else { "No" }
            ),
        );
        return 2;
    }

    /* And they have to be non-zero */
    for ind in 0..nz as usize {
        if sec_dose[ind] <= 0. && min_dose <= 0. {
            if min_dose > 0. {
                if sec_dose[ind] < replace_dose {
                    sec_dose[ind] = replace_dose;
                }
            } else {
                b3d_error(
                    Some(&mut ImodFile::Stdout),
                    format_args!(
                        "getMetadataWeightingDoses - Some sections have 0 for ExposureDose in \
                         the autodoc file\n"
                    ),
                );
                return 2;
            }
        }
    }

    /* Look for PriorRecordDose; if they are there we are done
    Need to ignore incomplete number of these entries thanks to a bug in SerialEM! */
    if get_metadata_by_key(
        ind_adoc,
        adoc_type,
        nz,
        "PriorRecordDose",
        2,
        prior_dose,
        core::slice::from_mut(&mut dummy1),
        core::slice::from_mut(&mut dummy2),
        None,
        &mut num_vals,
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

    ret_val = 0;

    /* Get DateTime strings to determine section order.
    New 9/6/21: Use this in preference to bidir information, which can be inadvertent
    So from here on, any error causes it to fall back to bidir if present */
    if is_bidir {
        b3d_set_store_error(1);
    }
    val_strings = vec![None; nz.max(0) as usize];

    if ret_val == 0 {
        if get_metadata_by_key(
            ind_adoc,
            adoc_type,
            nz,
            "DateTime",
            0,
            prior_dose,
            core::slice::from_mut(&mut dummy1),
            core::slice::from_mut(&mut dummy2),
            Some(&mut val_strings[..]),
            &mut num_vals,
            &mut num_found,
            nz,
            iz_piece,
        ) != 0
        {
            // `freeValStrings`
            val_strings.clear();
            ret_val = 1;
            if !is_bidir {
                return 1;
            }
        }
    }

    /* If no time stamps, have to rely on images being in order if no bidir information.
    Return -1 in this case */
    if num_found == 0 {
        if is_bidir {
            ret_val = -1;
        } else {
            prior_doses_from_image_doses(
                &sec_dose[..nz as usize],
                0,
                &mut prior_dose[..nz as usize],
            );
            return -1;
        }
    }

    if num_found > 0 && num_found < nz {
        b3d_error(
            Some(&mut ImodFile::Stdout),
            format_args!(
                "getMetadataWeightingDoses - The autodoc file has DateTime entries for only \
                 {} of {} sections\n",
                num_found, nz
            ),
        );
        val_strings.clear();
        ret_val = 1;
        if !is_bidir {
            return 2;
        }
    }

    /* Now convert the date-time stamps with sscanf, look up month, fill tm struct and
    convert to time_t */
    if ret_val == 0 {
        full_times = vec![0; nz as usize];
        time_inds = vec![0; nz as usize];
    }
    if ret_val == 0 {
        for ind in 0..nz as usize {
            let text: &[u8] = val_strings[ind].as_deref().map_or(b"", str::as_bytes);
            // `sscanf(valStrings[ind], "%d-%3s-%d %d:%d:%d", &tim.tm_mday, monthBuf,
            //         &tim.tm_year, &tim.tm_hour, &tim.tm_min, &tim.tm_sec)`
            let mut pos = 0_usize;
            'scan: for conv in 0..6 {
                if conv == 1 || conv == 2 {
                    // literal '-'
                    if text.get(pos) != Some(&b'-') {
                        break 'scan;
                    }
                    pos += 1;
                } else if conv == 3 {
                    // ' ' matches any run of white space, including none
                    while text.get(pos).is_some_and(|c| c.is_ascii_whitespace()) {
                        pos += 1;
                    }
                } else if conv > 3 {
                    if text.get(pos) != Some(&b':') {
                        break 'scan;
                    }
                    pos += 1;
                }
                while text.get(pos).is_some_and(|c| c.is_ascii_whitespace()) {
                    pos += 1;
                }
                if conv == 1 {
                    // `%3s`
                    let start = pos;
                    while pos < text.len() && pos - start < 3 && !text[pos].is_ascii_whitespace() {
                        pos += 1;
                    }
                    if pos == start {
                        break 'scan;
                    }
                    month_buf = text[start..pos].to_vec();
                    continue;
                }
                // `%d`: optional sign then decimal digits, as `strtol` reads them
                let start = pos;
                let mut negative = false;
                if matches!(text.get(pos), Some(b'+') | Some(b'-')) {
                    negative = text[pos] == b'-';
                    pos += 1;
                }
                let digits = pos;
                let mut value = 0_i64;
                while pos < text.len() && text[pos].is_ascii_digit() {
                    value = value
                        .saturating_mul(10)
                        .saturating_add((text[pos] - b'0') as i64);
                    pos += 1;
                }
                if pos == digits {
                    let _ = start;
                    break 'scan;
                }
                let value = (if negative { -value } else { value }) as i32;
                match conv {
                    0 => tim.tm_mday = value,
                    2 => tim.tm_year = value,
                    3 => tim.tm_hour = value,
                    4 => tim.tm_min = value,
                    _ => tim.tm_sec = value,
                }
            }
            tim.tm_year += 100;
            let mut mon_ind = 0_usize;
            while mon_ind < 12 {
                if month_buf == months[mon_ind] {
                    break;
                }
                mon_ind += 1;
            }
            if mon_ind > 11 {
                b3d_error(
                    Some(&mut ImodFile::Stdout),
                    format_args!(
                        "getMetadataWeightingDoses - The DateTime entry {} in the autodoc file \
                         has an improper month string\n",
                        String::from_utf8_lossy(text)
                    ),
                );
                ret_val = 2;
                break;
            }
            tim.tm_mon = mon_ind as i32;
            tim.tm_isdst = -1;
            full_times[ind] = crate::imod::libcfshr::b3dutil::c_mktime(&mut tim);
            time_inds[ind] = ind as i32;
        }
    }

    /* Sort the times by index
    difftime gives the time interval FROM the second time TO the first time */
    if ret_val == 0 {
        for ind in 0..(nz - 1).max(0) as usize {
            for jnd in ind + 1..nz as usize {
                // `difftime(a, b) > 0.` is `(double)a - (double)b > 0.`
                if full_times[time_inds[ind] as usize] as f64
                    - full_times[time_inds[jnd] as usize] as f64
                    > 0.
                {
                    let mon_ind = time_inds[ind];
                    time_inds[ind] = time_inds[jnd];
                    time_inds[jnd] = mon_ind;
                }
            }
        }

        /* Now compute the prior doses */
        prior_dose[time_inds[0] as usize] = 0.;
        for ind in 1..nz as usize {
            dummy1 = sec_dose[time_inds[ind - 1] as usize];
            prior_dose[time_inds[ind] as usize] = prior_dose[time_inds[ind - 1] as usize]
                + if dummy1 < min_dose && accum_dose >= 0.01 {
                    if accum_dose < dummy1 {
                        dummy1
                    } else {
                        accum_dose
                    }
                } else {
                    dummy1
                };
        }
    }

    /* If there was an error but information about bidirectional is present, use it to
    compute priorDose in the right order */
    if ret_val != 0 && is_bidir {
        let mut out = ImodFile::Stdout;
        if ret_val > 0 {
            let _ = out.write_all(format!("WARNING: {}", b3d_get_error()).as_bytes());
        }
        let _ = out.write_all(
            format!(
                "{}alling back to using entry for bidirectional tilt series\n",
                if ret_val > 0 {
                    "F"
                } else {
                    "No date-time information; f"
                }
            )
            .as_bytes(),
        );
        let _ = out.flush();
        ret_val = 0;
        b3d_error(Some(&mut ImodFile::Stdout), format_args!(""));
        prior_doses_from_image_doses(
            &sec_dose[..nz as usize],
            bidir_num_invert,
            &mut prior_dose[..nz as usize],
        );
    }

    /* Clean up */
    if is_bidir {
        b3d_set_store_error(save_error);
    }
    ret_val
}

/// C Fortran wrapper `getmetadataweightingdoses` (`extraheader.c:797`):
/// `indAdoc` is 1-based, and an error return is printed and exits.
pub fn get_metadata_weighting_doses_fortran(
    ind_adoc: i32,
    adoc_type: i32,
    nz: i32,
    iz_piece: &[i32],
    bidir_num_invert: i32,
    prior_dose: &mut [f32],
    sec_dose: &mut [f32],
) -> i32 {
    let err = get_metadata_weighting_doses(
        ind_adoc - 1,
        adoc_type,
        nz,
        iz_piece,
        bidir_num_invert,
        prior_dose,
        sec_dose,
    );
    if err > 0 {
        let _ = ImodFile::Stdout.write_all(format!("\nERROR: {}\n", b3d_get_error()).as_bytes());
        exit(1);
    }
    err
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

    #[test]
    fn metadata_key_reads_owned_typed_output_slices() {
        let index = crate::imod::libcfshr::autodoc::adoc_new();
        assert!(index >= 0);
        assert!(crate::imod::libcfshr::autodoc::adoc_set_current(index).is_ok());
        let section = crate::imod::libcfshr::autodoc::adoc_add_section(ADOC_ZVALUE_NAME, b"0")
            .expect("adoc_add_section");
        assert_eq!(section, 0);
        assert_eq!(
            crate::imod::libcfshr::autodoc::adoc_set_float(
                ADOC_ZVALUE_NAME,
                section,
                b"ExposureDose",
                2.5,
            ),
            0
        );
        let mut first = [0.; 1];
        let mut second = [0.; 1];
        let mut third = [0.; 1];
        let mut num_vals = 0;
        let mut num_found = 0;
        assert_eq!(
            get_metadata_by_key(
                index,
                1,
                1,
                "ExposureDose",
                2,
                &mut first,
                &mut second,
                &mut third,
                None,
                &mut num_vals,
                &mut num_found,
                1,
                &[0],
            ),
            0
        );
        assert_eq!((first, num_vals, num_found), ([2.5], 1, 1));
        crate::imod::libcfshr::autodoc::adoc_clear(index);
    }
}
