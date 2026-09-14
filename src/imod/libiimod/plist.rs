//! Translation of `IMOD/libiimod/plist.c` and its `mrcfiles.h` declarations.
#![allow(dead_code)]

use crate::imod::libcfshr::autodoc::{
    adoc_clear, adoc_get_three_integers, adoc_lookup_by_name_value, adoc_open_image_metadata,
    adoc_set_current,
};
use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::b3d_error;
use crate::imod::libiimod::mrcfiles::{LoadInfo, MrcHeader};
use core::ffi::c_char;
use core::sync::atomic::{AtomicI32, Ordering};
use std::io::Read;

static S_PIECE_KEY_IND: AtomicI32 = AtomicI32::new(0);

/// Matches C `mrc_plist_li(IloadInfo *, MrcHeader *, const char *)` (`plist.c:18`).
pub unsafe fn mrc_plist_li(li: *mut LoadInfo, hdata: *mut MrcHeader, fname: *const c_char) -> i32 {
    if fname.is_null() {
        return 1;
    }
    let Some(mut fin) = ImodFile::open(
        &unsafe { core::ffi::CStr::from_ptr(fname) }.to_string_lossy(),
        "r",
    ) else {
        unsafe {
            (*li).plist = 0;
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("ERROR opening piece list file"),
            );
        }
        return -1;
    };
    let retval = unsafe { mrc_plist_load(li, hdata, &mut fin) };
    drop(fin);
    retval
}

/// Matches C `mrc_plist_load(IloadInfo *, MrcHeader *, FILE *)` (`plist.c:43`).
pub unsafe fn mrc_plist_load(li: *mut LoadInfo, hdata: *mut MrcHeader, fin: &mut ImodFile) -> i32 {
    unsafe { plist_load(fin, li, (*hdata).nx, (*hdata).ny, (*hdata).nz) }
}

/// Matches C static `plist_load` (`plist.c:48`).
unsafe fn plist_load(fin: &mut ImodFile, li: *mut LoadInfo, nx: i32, ny: i32, nz: i32) -> i32 {
    unsafe {
        (*li).plist = nz;
        (*li).pcoords = Some(vec![0i32; 3 * nz as usize]);
        // `plist.c:56` is `fscanf(fin, "%d %d %d", &x, &y, &z)`.  `%d` skips
        // leading whitespace, takes an optional sign and digits, and stops at
        // the first character that cannot extend the number; the call returns
        // how many of the three it filled, or `EOF` when the input ends before
        // any conversion at all.  Reading the file once and walking
        // whitespace-separated tokens is that, with the same two outcomes
        // distinguished, which matters because the source prints its error
        // only when the result is not `EOF`.
        let mut text = String::new();
        let _ = fin.read_to_string(&mut text);
        let mut tokens = text.split_ascii_whitespace();
        for i in 0..nz {
            let mut value = [0i32; 3];
            let mut scanret = 0;
            for field in 0..3 {
                let Some(token) = tokens.next() else {
                    scanret = if field == 0 { -1 } else { field };
                    break;
                };
                let end = token
                    .char_indices()
                    .position(|(at, c)| {
                        !(c.is_ascii_digit() || (at == 0 && (c == '-' || c == '+')))
                    })
                    .unwrap_or(token.len());
                match token[..end].parse::<i32>() {
                    Ok(parsed) => {
                        value[field as usize] = parsed;
                        scanret = field + 1;
                    }
                    Err(_) => {
                        scanret = field;
                        break;
                    }
                }
            }
            if scanret == 3 {
                let pcoords = (&mut (*li).pcoords).as_mut().unwrap();
                pcoords[(i * 3) as usize] = value[0];
                pcoords[(i * 3 + 1) as usize] = value[1];
                pcoords[(i * 3 + 2) as usize] = value[2];
            } else {
                (*li).plist = i;
                if scanret != -1 {
                    b3d_error(
                        Some(&mut ImodFile::Stderr),
                        format_args!("Error reading piece list after {} lines\n", i),
                    );
                }
                break;
            }
        }
    }
    unsafe { mrc_plist_proc(li, nx, ny, nz) }
}

/// Matches C `mrc_plist_proc(IloadInfo *, int, int, int)` (`plist.c:84`).
pub unsafe fn mrc_plist_proc(li: *mut LoadInfo, nx: i32, ny: i32, nz: i32) -> i32 {
    unsafe {
        let pcoords = (&mut (*li).pcoords).as_mut().unwrap();
        let mut pmin = [pcoords[0], pcoords[1], pcoords[2]];
        let mut pmax = [pmin[0] + nx, pmin[1] + ny, pmin[2]];
        for i in 1..(*li).plist {
            let point = (i * 3) as usize;
            if pmin[0] > pcoords[point] {
                pmin[0] = pcoords[point];
            }
            if pmin[1] > pcoords[point + 1] {
                pmin[1] = pcoords[point + 1];
            }
            if pmin[2] > pcoords[point + 2] {
                pmin[2] = pcoords[point + 2];
            }
            if pmax[0] < pcoords[point] + nx {
                pmax[0] = pcoords[point] + nx;
            }
            if pmax[1] < pcoords[point + 1] + ny {
                pmax[1] = pcoords[point + 1] + ny;
            }
            if pmax[2] < pcoords[point + 2] {
                pmax[2] = pcoords[point + 2];
            }
        }
        (*li).px = (pmax[0] - pmin[0]) as f32;
        (*li).py = (pmax[1] - pmin[1]) as f32;
        (*li).pz = (pmax[2] - pmin[2] + 1) as f32;
        (*li).opx = pmin[0] as f32;
        (*li).opy = pmin[1] as f32;
        (*li).opz = pmin[2] as f32;
        for i in 0..(*li).plist {
            let point = (i * 3) as usize;
            pcoords[point] -= pmin[0];
            pcoords[point + 1] -= pmin[1];
            pcoords[point + 2] -= pmin[2];
        }
        let mut zlist = vec![0i32; ((*li).pz as i32 + 1) as usize];
        (*li).pdz = 0;
        for i in 0..(*li).pz as i32 {
            zlist[i as usize] = 0;
        }
        for i in 0..(*li).plist {
            zlist[pcoords[(i * 3 + 2) as usize] as usize] += 1;
        }
        for i in 0..(*li).pz as i32 {
            if zlist[i as usize] != 0 {
                (*li).pdz += 1;
            }
        }
        drop(zlist);
    }
    0
}

/// Matches C `mrc_plist_create(IloadInfo *, int, int, int, int, int, int, int)` (`plist.c:151`).
pub unsafe fn mrc_plist_create(
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
    nfx: i32,
    nfy: i32,
    mut ovx: i32,
    mut ovy: i32,
) -> i32 {
    unsafe {
        (*li).plist = nz;
        (*li).pcoords = Some(vec![0i32; 3 * nz as usize]);
        let (mut x, mut y, mut z) = (0, 0, 0);
        if ovx >= nx {
            ovx = nx - 1;
        }
        if ovy >= ny {
            ovy = ny - 1;
        }
        let pcoords = (&mut (*li).pcoords).as_mut().unwrap();
        for i in 0..nz {
            pcoords[(i * 3) as usize] = x * (nx - ovx);
            pcoords[(i * 3 + 1) as usize] = y * (ny - ovy);
            pcoords[(i * 3 + 2) as usize] = z;
            x += 1;
            if x >= nfx {
                y += 1;
                x = 0;
                if y >= nfy {
                    z += 1;
                    y = 0;
                }
            }
        }
    }
    unsafe { mrc_plist_proc(li, nx, ny, nz) }
}

/// Matches C `iiPlistLoad(const char *, IloadInfo *, int, int, int)` (`plist.c:191`).
pub unsafe fn ii_plist_load(
    filename: *const c_char,
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    if filename.is_null() || nz < 1 || ny < 1 || nx < 1 {
        return 1;
    }
    let Some(mut fin) = ImodFile::open(
        &unsafe { core::ffi::CStr::from_ptr(filename) }.to_string_lossy(),
        "r",
    ) else {
        return 1;
    };
    let retval = unsafe { ii_plist_load_f(&mut fin, li, nx, ny, nz) };
    drop(fin);
    retval
}

/// Matches C `iiPlistLoadF(FILE *, IloadInfo *, int, int, int)` (`plist.c:210`).
pub unsafe fn ii_plist_load_f(
    fin: &mut ImodFile,
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    if nz < 1 || ny < 1 || nx < 1 {
        return 1;
    }
    unsafe { plist_load(fin, li, nx, ny, nz) }
}

/// Matches C `iiPlistFromMetadata` (`plist.c:234`).
pub unsafe fn ii_plist_from_metadata(
    filename: *const c_char,
    add_mdoc: i32,
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
) -> i32 {
    if filename.is_null() || nx < 1 || ny < 1 || nz < 1 {
        return -4;
    }
    let (mut montage, mut num_sect, mut sect_type) = (0, 0, 0);
    let adoc_index = unsafe {
        adoc_open_image_metadata(
            filename,
            add_mdoc,
            &mut montage,
            &mut num_sect,
            &mut sect_type,
        )
    };
    if adoc_index < 0 {
        return adoc_index;
    }
    unsafe { ii_plist_from_autodoc(adoc_index, 1, li, nx, ny, nz, montage, num_sect, sect_type) }
}

/// Matches C `iiPlistFromAutodoc` (`plist.c:259`).
pub unsafe fn ii_plist_from_autodoc(
    adoc_index: i32,
    clear_on_done: i32,
    li: *mut LoadInfo,
    nx: i32,
    ny: i32,
    nz: i32,
    montage: i32,
    num_sect: i32,
    sect_type: i32,
) -> i32 {
    let sect_names = [c"ZValue".as_ptr(), c"Image".as_ptr(), c"ZValue".as_ptr()];
    let keys = [
        c"PieceCoordinates".as_ptr(),
        c"AlignedPieceCoords".as_ptr(),
        c"AlignedPieceCoordsVS".as_ptr(),
    ];
    let mut key_order = [2, 1, 0];
    let piece_key_ind = S_PIECE_KEY_IND.load(Ordering::SeqCst);
    let mut num_keys = 1;
    if piece_key_ind > 0 {
        num_keys = 3;
        if piece_key_ind == 1 {
            key_order[0] = 1;
            key_order[1] = 2;
        }
    } else {
        key_order[0] = 0;
    }
    S_PIECE_KEY_IND.store(0, Ordering::SeqCst);
    if adoc_index < 0 || nx < 1 || ny < 1 || nz < 1 {
        return -4;
    }
    if montage == 0 || num_sect != nz || unsafe { adoc_set_current(adoc_index) } != 0 {
        if clear_on_done != 0 {
            unsafe { adoc_clear(adoc_index) };
        }
        return 1;
    }
    unsafe {
        (*li).pcoords = Some(vec![0i32; 3 * nz as usize]);
        (*li).plist = nz;
        let mut i = 0;
        while i < nz {
            let mut index = i;
            if sect_type == 3 {
                index = adoc_lookup_by_name_value(c"ZValue".as_ptr(), i);
                if index < 0 {
                    break;
                }
            }
            let mut err = 0;
            for key_ind in 0..num_keys {
                err = adoc_get_three_integers(
                    sect_names[(sect_type - 1) as usize],
                    index,
                    keys[key_order[key_ind as usize] as usize],
                    (&mut (*li).pcoords)
                        .as_mut()
                        .unwrap()
                        .as_mut_ptr()
                        .add((i * 3) as usize),
                    (&mut (*li).pcoords)
                        .as_mut()
                        .unwrap()
                        .as_mut_ptr()
                        .add((i * 3 + 1) as usize),
                    (&mut (*li).pcoords)
                        .as_mut()
                        .unwrap()
                        .as_mut_ptr()
                        .add((i * 3 + 2) as usize),
                );
                if err == 0 {
                    break;
                }
            }
            if err != 0 {
                break;
            }
            i += 1;
        }
        if clear_on_done != 0 {
            adoc_clear(adoc_index);
        }
        if i < nz {
            (*li).pcoords = None;
            (*li).plist = 0;
            return 2;
        }
    }
    unsafe { mrc_plist_proc(li, nx, ny, nz) }
}

/// Matches C `iiPlistSetAdocCoordType(int)` (`plist.c:321`).
pub fn ii_plist_set_adoc_coord_type(value: i32) {
    S_PIECE_KEY_IND.store(value.clamp(0, 2), Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn loads_vendored_piece_coordinate_fixture_and_normalizes_origin() {
        let path = CString::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/piece-list.txt"
        ))
        .unwrap();
        let mut load_info = LoadInfo::default();
        assert_eq!(
            unsafe { ii_plist_load(path.as_ptr(), &mut load_info, 100, 80, 9) },
            0
        );
        assert_eq!(load_info.plist, 9);
        assert_eq!(
            (load_info.px, load_info.py, load_info.pz),
            (2020.0, 1040.0, 2.0)
        );
        assert_eq!(
            (load_info.opx, load_info.opy, load_info.opz),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(load_info.pdz, 2);
        let pcoords = load_info.pcoords.as_ref().unwrap();
        assert_eq!(pcoords[12], 960);
        assert_eq!(pcoords[26], 1);
    }

    #[test]
    fn creates_piece_grid_with_source_overlap_clamping() {
        let mut load_info = LoadInfo::default();
        assert_eq!(
            unsafe { mrc_plist_create(&mut load_info, 10, 8, 5, 2, 2, 20, 9) },
            0
        );
        assert_eq!((load_info.px, load_info.py, load_info.pz), (11.0, 9.0, 2.0));
        let pcoords = load_info.pcoords.as_ref().unwrap();
        assert_eq!(pcoords[3], 1);
        assert_eq!(pcoords[7], 1);
        assert_eq!(pcoords[14], 1);
    }
}
