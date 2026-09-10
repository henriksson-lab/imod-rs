//! Direct translation of `IMOD/flib/image/alterheader.f90`.
//!
//! All header access goes through the source program's iiunit interface; no
//! copied `MrcHeader` adapter is used.
#![allow(dead_code, unused_variables)]

use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::b3dutil::{extra_is_nbytes_and_flags, override_invert_mrc_origin};
use crate::imod::libiimod::iimage::ii_read_section_float;
use crate::imod::libiimod::mrcfiles::{MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT};
use crate::imod::libiimod::unit_fileio::{ialbrief_, iiu_close, iiu_get_ii_file, iiu_open};
use crate::imod::libiimod::unit_header::*;
use std::ffi::CString;

/// Original Fortran `program alterheader`.
pub fn alterheader() {
    let words: Vec<String> = std::env::args().skip(1).collect();
    let mut name_at = None;
    let mut num_option_args = 0;
    let mut copy_from = false;
    let mut i = 0;
    while i < words.len() {
        if !words[i].starts_with('-') {
            name_at = Some(i);
        }
        let key = words[i].trim_start_matches('-').to_ascii_lowercase();
        if words[i].starts_with('-') {
            num_option_args += 1;
            copy_from |= matches!(key.as_str(), "copy" | "copyfromimage");
        }
        i += 1 + usize::from(
            matches!(
                key.as_str(),
                "org"
                    | "origin"
                    | "cel"
                    | "cellsize"
                    | "dat"
                    | "datatype"
                    | "del"
                    | "pixelsize"
                    | "map"
                    | "mapindexes"
                    | "sam"
                    | "samplesize"
                    | "tlt"
                    | "tiltcurrent"
                    | "firsttlt"
                    | "tiltoriginal"
                    | "rottlt"
                    | "rotatetilt"
                    | "setmmm"
                    | "feipixel"
                    | "ispg"
                    | "spacegroup"
                    | "volstack"
                    | "volumestack"
                    | "4bit"
                    | "change4bitmode"
                    | "title"
                    | "titletoadd"
                    | "position"
                    | "positionfortitle"
                    | "remove"
                    | "removetitles"
                    | "copy"
                    | "copyfromimage"
                    | "start"
            ) && i < words.len(),
        );
    }
    let Some(name_at) = name_at else {
        eprintln!("ERROR: ALTERHEADER - Image filename must be entered");
        return;
    };
    // `alterheader.f90:95-96` sends CopyFromImage directly to the copy
    // branch, and rejects every combination before opening either image.
    if copy_from && num_option_args > 1 {
        println!("ERROR: ALTERHEADER - No other options can be entered with -copy");
        std::process::exit(1);
    }
    let Ok(name) = CString::new(words[name_at].as_bytes()) else {
        eprintln!("ERROR: ALTERHEADER - Image filename contains a NUL byte");
        return;
    };
    let mut brief = 0;
    unsafe { ialbrief_(&mut brief) };
    unsafe { imopen(2, &words[name_at], "OLD") };
    let image = unsafe { iiu_get_ii_file(2) };
    if image.is_null() {
        eprintln!("ERROR: ALTERHEADER - Could not open '{}'", words[name_at]);
        return;
    }
    let (mut nxyz, mut mxyz, mut nxyzst) = ([0; 3], [0; 3], [0; 3]);
    let (mut mode, mut dmin, mut dmax, mut dmean) = (0, 0., 0., 0.);
    unsafe {
        irdhdr(
            2,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    let (mut version, mut flags, mut imod) = (0, 0, 0);
    unsafe {
        iiu_ret_mrc_version(2, &mut version);
        iiu_ret_imod_flags(2, &mut flags, &mut imod);
    }
    let mut invert = i32::from(version > 0 || (imod > 0 && flags & 4 != 0));
    override_invert_mrc_origin(invert);
    let mut changed = false;
    let mut add_title = -1;
    let mut title_position = None::<usize>;
    i = 0;
    while i < words.len() {
        if !words[i].starts_with('-') {
            i += 1;
            continue;
        }
        let key = words[i].trim_start_matches('-').to_ascii_lowercase();
        let value = words.get(i + 1).cloned().unwrap_or_default();
        match key.as_str() {
            // labels 1 through 9 in the Fortran computed GOTO
            "org" | "origin" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                if v.len() == 3 {
                    unsafe {
                        iiu_alt_origin(2, v[0], v[1], v[2]);
                    }
                    changed = true;
                } else {
                    eprintln!("ERROR: ALTERHEADER - Origin requires three values");
                }
                i += 2;
            }
            "cel" | "cellsize" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                let mut c = [0.; 6];
                unsafe { iiu_ret_cell(2, c.as_mut_ptr()) };
                if v.len() == 3 && v.iter().all(|x| *x > 0.) {
                    c[..3].copy_from_slice(&v);
                    unsafe { iiu_alt_cell(2, c.as_mut_ptr()) };
                    changed = true
                } else {
                    eprintln!("ERROR: ALTERHEADER - The values for the cell entry must positive")
                }
                i += 2;
            }
            "dat" | "datatype" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                if v.len() == 6 && (0. ..=4.).contains(&v[0]) {
                    unsafe {
                        iiu_alt_data_type(
                            2,
                            v[0] as i32,
                            v[1] as i32,
                            v[2] as i32,
                            v[3] as i32,
                            v[4],
                            v[5],
                        )
                    };
                    changed = true
                } else {
                    eprintln!(
                        "ERROR: ALTERHEADER - Data type requires type, lens, n1, n2, v1, v2 (type 0-4)"
                    )
                }
                i += 2;
            }
            "del" | "pixelsize" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                let mut c = [0.; 6];
                unsafe {
                    iiu_ret_sample(2, mxyz.as_mut_ptr());
                    iiu_ret_cell(2, c.as_mut_ptr())
                };
                if v.len() == 3 && v.iter().all(|x| *x > 0.) {
                    for j in 0..3 {
                        c[j] = mxyz[j] as f32 * v[j]
                    }
                    unsafe { iiu_alt_cell(2, c.as_mut_ptr()) };
                    changed = true
                } else {
                    eprintln!("ERROR: ALTERHEADER - The values for the delta entry must positive")
                }
                i += 2;
            }
            "map" | "mapindexes" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<i32>().ok())
                    .collect::<Vec<_>>();
                if v.len() == 3
                    && [1, 2, 3]
                        .iter()
                        .all(|x| v.iter().filter(|y| **y == *x).count() == 1)
                {
                    unsafe { iiu_alt_axis_map(2, v.as_ptr() as *mut i32) };
                    changed = true
                } else {
                    eprintln!(
                        "ERROR: ALTERHEADER - The values for the map entry must be a permutation of 1, 2, 3"
                    )
                }
                i += 2;
            }
            "sam" | "samplesize" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<i32>().ok())
                    .collect::<Vec<_>>();
                if v.len() == 3 && v.iter().all(|x| *x > 0) {
                    unsafe { iiu_alt_sample(2, v.as_ptr() as *mut i32) };
                    changed = true
                } else {
                    eprintln!(
                        "ERROR: ALTERHEADER - The values for the sample entry must be positive"
                    )
                }
                i += 2;
            }
            "tlt" | "tiltcurrent" | "firsttlt" | "tiltoriginal" | "rottlt" | "rotatetilt" => {
                let mut v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                if v.len() != 3 {
                    eprintln!("ERROR: ALTERHEADER - Tilt option requires three values")
                } else {
                    unsafe {
                        if matches!(key.as_str(), "firsttlt" | "tiltoriginal") {
                            iiu_alt_tilt_orig(2, v.as_mut_ptr())
                        } else if matches!(key.as_str(), "rottlt" | "rotatetilt") {
                            iiu_alt_tilt_rot(2, v.as_mut_ptr())
                        } else {
                            iiu_alt_tilt(2, v.as_mut_ptr())
                        }
                    };
                    changed = true
                }
                i += 2;
            }
            // 11/16, source `irdsecl` calculation and iiuAltRMS.
            "mmm" | "minmaxmean" | "rms" | "rootmeansquare" => {
                unsafe {
                    iiu_ret_basic_head(
                        2,
                        nxyz.as_mut_ptr(),
                        mxyz.as_mut_ptr(),
                        &mut mode,
                        &mut dmin,
                        &mut dmax,
                        &mut dmean,
                    )
                };
                println!(" Recomputing min/max/mean of images - takes a while...");
                let mut buf =
                    vec![0.; (nxyz[0].max(0) as usize).saturating_mul(nxyz[1].max(0) as usize)];
                let (mut lo, mut hi, mut sum, mut sumsq, mut total) =
                    (f32::INFINITY, f32::NEG_INFINITY, 0f64, 0f64, 0f64);
                let mut bad = false;
                for z in 0..nxyz[2] {
                    if unsafe { ii_read_section_float(image, buf.as_mut_ptr().cast(), z) } != 0 {
                        bad = true;
                        break;
                    }
                    for x in &buf {
                        lo = lo.min(*x);
                        hi = hi.max(*x);
                        sum += *x as f64;
                        sumsq += *x as f64 * *x as f64;
                        total += 1.
                    }
                }
                if bad || total == 0. {
                    eprintln!("ERROR: ALTERHEADER - reading file")
                } else {
                    dmin = lo;
                    dmax = hi;
                    dmean = (sum / total) as f32;
                    let rms = ((sumsq - total * (sum / total).powi(2)) / total).sqrt() as f32;
                    unsafe { iiu_alt_rms(2, rms) };
                    if matches!(key.as_str(), "rms" | "rootmeansquare") {
                        println!(" New RMS value = {:13.5}", rms)
                    }
                    changed = true
                }
                i += 1;
            }
            // 12, 26, 13 and 17.
            "fixpixel" => {
                let mut c = [0.; 6];
                unsafe { iiu_ret_cell(2, c.as_mut_ptr()) };
                for j in 0..3 {
                    c[j] = nxyz[j] as f32
                }
                unsafe {
                    iiu_alt_cell(2, c.as_mut_ptr());
                    iiu_alt_sample(2, nxyz.as_mut_ptr())
                };
                changed = true;
                i += 1;
            }
            "gridfix" | "fixgrid" => {
                let mut c = [0.; 6];
                unsafe {
                    iiu_ret_cell(2, c.as_mut_ptr());
                    iiu_ret_sample(2, mxyz.as_mut_ptr())
                };
                for j in 0..3 {
                    if mxyz[j] != 0 {
                        c[j] = c[j] / mxyz[j] as f32 * nxyz[j] as f32
                    }
                }
                unsafe {
                    iiu_alt_cell(2, c.as_mut_ptr());
                    iiu_alt_sample(2, nxyz.as_mut_ptr())
                };
                changed = true;
                i += 1;
            }
            "extrafix" | "fixextra" => {
                let (mut nbyte, mut extflags) = (0, 0);
                unsafe { iiu_ret_extended_type(2, &mut nbyte, &mut extflags) };
                if extflags / 2 % 2 > 0 {
                    extflags -= 2
                }
                unsafe { iiu_alt_extended_type(2, nbyte, extflags) };
                changed = true;
                i += 1;
            }
            "modefix" | "fixmode" => {
                if mode == 1 || mode == 6 {
                    mode = 7 - mode;
                    // `alterheader.f90:562-572`: mode conversion is a
                    // visible header operation, including the source's
                    // range-loss warnings before the altered header is
                    // committed below.
                    println!("\nChanging mode to{:2}", mode);
                    if dmax > 32767.0 && mode == 1 {
                        println!(
                            "\nThe file maximum is{:12.1} and numbers bigger than 32767 will not be\n represented correctly in this mode.",
                            dmax
                        );
                    }
                    if dmin < 0.0 && mode == 6 {
                        println!(
                            "\nThe file minimum is{:12.1} and negative numbers will not be\n represented correctly in this mode.",
                            dmin
                        );
                    }
                    unsafe { iiu_alt_mode(2, mode) };
                    changed = true
                } else {
                    eprintln!(
                        "ERROR: ALTERHEADER - Only mode 6 can be changed to mode 1, or 1 to 6"
                    )
                }
                i += 1;
            }
            // 18: only final iiuWriteHeader commits MMM, exactly as Fortran.
            "setmmm" | "setminmaxmean" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<f32>().ok())
                    .collect::<Vec<_>>();
                if v.len() == 3 {
                    (dmin, dmax, dmean) = (v[0], v[1], v[2]);
                    changed = true
                } else {
                    eprintln!("ERROR: ALTERHEADER - SetMinMaxMean requires three values")
                }
                i += 2;
            }
            // 19: use iiuRet* extended-header operations, including source binning rules.
            "feipixel" => {
                let (mut nbytes, mut nint, mut nreal) = (0, 0, 0);
                unsafe {
                    iiu_ret_num_extended(2, &mut nbytes);
                    iiu_ret_extended_type(2, &mut nint, &mut nreal)
                };
                let mut ext = vec![0i32; (nbytes.max(0) as usize + 3) / 4];
                if nbytes <= 0 {
                    eprintln!("ERROR: ALTERHEADER - No extended header information in this file")
                } else if nint < 0 || extra_is_nbytes_and_flags(nint, nreal) != 0 {
                    eprintln!("ERROR: ALTERHEADER - The extended header is not in Agard/FEI format")
                } else if nreal < 12
                    || unsafe { iiu_ret_extended_data(2, &mut nbytes, ext.as_mut_ptr()) } != 0
                {
                    eprintln!(
                        "ERROR: ALTERHEADER - There is no pixel size in this extended header (too few values per section)"
                    )
                } else {
                    let pixel =
                        unsafe { *(ext.as_ptr().add(nint as usize + 11).cast::<f32>()) } * 1.0e10;
                    if pixel <= 0. {
                        eprintln!(
                            "ERROR: ALTERHEADER - Pixel size in extended header is not a usable value: {pixel}"
                        )
                    } else {
                        let mut delta = [0.; 3];
                        unsafe {
                            iiu_ret_delta(2, delta.as_mut_ptr());
                            iiu_ret_imod_flags(2, &mut flags, &mut imod);
                            iiu_ret_sample(2, mxyz.as_mut_ptr())
                        };
                        let mut bins = [1; 3];
                        let no_bin = flags & 2 != 0
                            || (0..3).any(|j| {
                                bins[j] = delta[j].round() as i32;
                                (delta[j] - bins[j] as f32).abs() > 1.0e-6
                                    || !(1..=4).contains(&bins[j])
                            });
                        if no_bin {
                            bins = [1; 3]
                        }
                        let mut c = [0.; 6];
                        unsafe { iiu_ret_cell(2, c.as_mut_ptr()) };
                        for j in 0..3 {
                            c[j] = mxyz[j] as f32 * pixel * bins[j] as f32
                        }
                        flags |= 2;
                        unsafe {
                            iiu_alt_imod_flags(2, flags);
                            iiu_alt_cell(2, c.as_mut_ptr())
                        };
                        changed = true
                    }
                }
                i += 2;
            }
            "invertorg" | "invertorigin" => {
                let (mut x, mut y, mut z) = (0., 0., 0.);
                unsafe {
                    iiu_ret_origin(2, &mut x, &mut y, &mut z);
                    iiu_alt_origin(2, -x, -y, -z)
                };
                println!(
                    "Inverting sign of origin: new origin = {:15.6} {:15.6} {:15.6}",
                    -x, -y, -z
                );
                changed = true;
                i += 1;
            }
            "toggleorg" | "toggleorigin" => {
                invert = 1 - invert;
                override_invert_mrc_origin(invert);
                i += 1;
            }
            "real" | "realmode" => {
                if mode == MRC_MODE_COMPLEX_FLOAT {
                    mode = MRC_MODE_FLOAT;
                    nxyz[0] *= 2;
                    unsafe {
                        iiu_alt_mode(2, mode);
                        iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr())
                    };
                    changed = true
                } else {
                    eprintln!("ERROR: ALTERHEADER - Must be mode 4 to change file to real")
                }
                i += 1;
            }
            "fft" | "complexmode" => {
                if mode == MRC_MODE_FLOAT && nxyz[0] % 2 == 0 {
                    mode = MRC_MODE_COMPLEX_FLOAT;
                    nxyz[0] /= 2;
                    unsafe {
                        iiu_alt_mode(2, mode);
                        iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr())
                    };
                    changed = true
                } else {
                    eprintln!(
                        "ERROR: ALTERHEADER - Must be mode 2 and NX must be even to change file to fft"
                    )
                }
                i += 1;
            }
            "ispg" | "spacegroup" => {
                match value.parse::<i32>() {
                    Ok(x) if x >= 0 => {
                        unsafe { iiu_alt_space_group(2, x) };
                        changed = true
                    }
                    _ => eprintln!("ERROR: ALTERHEADER - Space group entry must be non-negative"),
                };
                i += 2;
            }
            "volstack" | "volumestack" => {
                match value.parse::<i32>() {
                    Ok(x) if x >= 0 => {
                        let mut c = [0.; 6];
                        unsafe {
                            iiu_ret_sample(2, mxyz.as_mut_ptr());
                            iiu_ret_cell(2, c.as_mut_ptr())
                        };
                        if x == 0 {
                            if mxyz[2] != 0 {
                                c[2] = c[2] * nxyz[2] as f32 / mxyz[2] as f32
                            }
                            mxyz[2] = nxyz[2];
                            unsafe {
                                iiu_alt_sample(2, mxyz.as_mut_ptr());
                                iiu_alt_cell(2, c.as_mut_ptr());
                                iiu_alt_space_group(2, 1)
                            };
                            changed = true
                        } else if nxyz[2] / x >= 2 {
                            if nxyz[2] % x > 0 {
                                println!(
                                    "WARNING: The # of sections per volume does not divide evenly into the total # of sections"
                                )
                            }
                            if mxyz[2] != 0 {
                                c[2] = c[2] * x as f32 / mxyz[2] as f32
                            }
                            mxyz[2] = x;
                            unsafe {
                                iiu_alt_sample(2, mxyz.as_mut_ptr());
                                iiu_alt_cell(2, c.as_mut_ptr());
                                iiu_alt_space_group(2, 401)
                            };
                            changed = true
                        } else {
                            eprintln!(
                                "ERROR: ALTERHEADER - Too many sections per volume; there must be at least two volumes in stack"
                            )
                        }
                    }
                    _ => eprintln!(
                        "ERROR: ALTERHEADER - Entry for volume stack must be non-negative"
                    ),
                };
                i += 2;
            }
            "4bit" | "change4bitmode" => {
                match value.parse::<i32>() {
                    Ok(x) => {
                        let ret = unsafe { iiu_alt_4_bit_mode(2, x) };
                        if ret < 0 && x != 0 {
                            eprintln!(
                                "ERROR: ALTERHEADER - No change was made; the file was already in the given mode"
                            )
                        } else if ret > 0 {
                            eprintln!("ERROR: ALTERHEADER - 4-bit mode change failed ({ret})")
                        } else {
                            changed |= x != 0
                        }
                    }
                    _ => eprintln!("ERROR: ALTERHEADER - Change4BitMode requires an integer"),
                };
                i += 2;
            }
            // 10 LAB, 27 START, 28 COPY.
            "remove" | "removetitles" => {
                let list = value
                    .split(',')
                    .filter_map(|x| x.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>();
                let mut titles = [[0i32; 20]; 10];
                let mut num = 0;
                unsafe { iiu_ret_labels(2, titles.as_mut_ptr().cast(), &mut num) };
                let mut out = 0;
                for old in 0..num.max(0) as usize {
                    if !list.contains(&(old + 1)) {
                        titles[out] = titles[old];
                        out += 1
                    }
                }
                unsafe { iiu_alt_labels(2, titles.as_mut_ptr().cast(), out as i32) };
                changed = true;
                i += 2;
            }
            "position" | "positionfortitle" => {
                if let Ok(position) = value.parse::<usize>()
                    && position > 0
                {
                    // `alterheader.f90:238-242` stores 1-j in listdel, then
                    // inserts the title after j-1 existing labels.
                    title_position = Some(position);
                    add_title = -2
                } else {
                    eprintln!("ERROR: ALTERHEADER - Position for title to add must be positive")
                }
                i += 2;
            }
            "title" | "titletoadd" => {
                let mut titles = [[0i32; 20]; 10];
                let mut num = 0;
                unsafe { iiu_ret_labels(2, titles.as_mut_ptr().cast(), &mut num) };
                if num >= 10 {
                    eprintln!(
                        "ERROR: ALTERHEADER - You need to delete some labels before adding any"
                    )
                } else {
                    let pos = title_position
                        .map(|position| position.saturating_sub(1).min(num as usize))
                        .unwrap_or(num as usize);
                    for old in (pos..num as usize).rev() {
                        titles[old + 1] = titles[old]
                    }
                    unsafe {
                        core::ptr::write_bytes(titles[pos].as_mut_ptr().cast::<u8>(), b' ', 80);
                        core::ptr::copy_nonoverlapping(
                            value.as_ptr(),
                            titles[pos].as_mut_ptr().cast(),
                            value.len().min(80),
                        );
                        iiu_alt_labels(2, titles.as_mut_ptr().cast(), num + 1)
                    };
                    changed = true
                }
                i += 2;
            }
            "start" => {
                let v = value
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|x| x.parse::<i32>().ok())
                    .collect::<Vec<_>>();
                if v.len() == 3 {
                    unsafe {
                        iiu_ret_size(2, nxyz.as_mut_ptr(), mxyz.as_mut_ptr(), nxyzst.as_mut_ptr())
                    };
                    nxyzst.copy_from_slice(&v);
                    unsafe { iiu_alt_size(2, nxyz.as_mut_ptr(), nxyzst.as_mut_ptr()) };
                    changed = true
                } else {
                    eprintln!("ERROR: ALTERHEADER - Start requires three values")
                }
                i += 2;
            }
            "copy" | "copyfromimage" => {
                let Ok(copy) = CString::new(value.as_bytes()) else {
                    eprintln!("ERROR: ALTERHEADER - Copy filename contains a NUL byte");
                    break;
                };
                unsafe { iiu_open(3, copy.as_ptr(), c"RO".as_ptr()) };
                let (mut nn, mut mm, mut ss) = ([0; 3], [0; 3], [0; 3]);
                let (mut om, mut lo, mut hi, mut mean) = (0, 0., 0., 0.);
                unsafe {
                    iiu_ret_basic_head(
                        3,
                        nn.as_mut_ptr(),
                        mm.as_mut_ptr(),
                        &mut om,
                        &mut lo,
                        &mut hi,
                        &mut mean,
                    )
                };
                if nn[0] != nxyz[0] || nn[1] != nxyz[1] {
                    eprintln!(
                        "ERROR: ALTERHEADER - The image to copy from must be the same size in X and Y"
                    )
                } else {
                    let (mut c, mut oc) = ([0.; 6], [0.; 6]);
                    let (mut x, mut y, mut z) = (0., 0., 0.);
                    let mut t = [0.; 3];
                    unsafe {
                        iiu_ret_cell(2, c.as_mut_ptr());
                        iiu_ret_cell(3, oc.as_mut_ptr());
                        iiu_ret_size(3, nn.as_mut_ptr(), mm.as_mut_ptr(), ss.as_mut_ptr());
                        iiu_ret_origin(3, &mut x, &mut y, &mut z);
                        iiu_ret_tilt(3, t.as_mut_ptr())
                    };
                    c[0] = oc[0];
                    c[1] = oc[1];
                    mxyz[0] = mm[0];
                    mxyz[1] = mm[1];
                    mxyz[2] = ((mm[2] as f32 * nxyz[2] as f32 / nn[2].max(1) as f32).round()
                        as i32)
                        .max(1);
                    c[2] = mxyz[2] as f32 * oc[2] / mm[2].max(1) as f32;
                    unsafe {
                        iiu_alt_cell(2, c.as_mut_ptr());
                        iiu_alt_sample(2, mxyz.as_mut_ptr());
                        iiu_alt_size(2, nxyz.as_mut_ptr(), ss.as_mut_ptr());
                        iiu_alt_origin(2, x, y, z);
                        iiu_alt_tilt(2, t.as_mut_ptr())
                    };
                    changed = true
                }
                unsafe { iiu_close(3) };
                i += 2;
            }
            "help" | "usage" => {
                println!(
                    "org cel dat del map sam tlt tlt_orig tlt_rot lab mmm rms fixpixel feipixel fixextra fixmode invertorg setmmm real fft ispg volstack 4bit toggleorg fixgrid start help done"
                );
                i += 1;
            }
            _ => {
                eprintln!("ERROR: ALTERHEADER - Unknown option -{key}");
                i += 1;
            }
        }
    }
    if changed {
        let mut titles = [[0i32; 20]; 10];
        let mut num = 0;
        unsafe {
            iiu_ret_labels(2, titles.as_mut_ptr().cast(), &mut num);
            if iiu_write_header(2, titles.as_mut_ptr().cast(), add_title, dmin, dmax, dmean) != 0 {
                eprintln!("ERROR: ALTERHEADER - writing header")
            }
        }
    }
    unsafe {
        iiu_close(2);
    }
    if changed {
        unsafe {
            imopen(3, &words[name_at], "RO");
            irdhdr(
                3,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
            iiu_close(3);
        }
    }
}
