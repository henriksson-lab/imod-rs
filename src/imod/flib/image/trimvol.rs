//! Translation of `IMOD/pysrc/trimvol`.
//!
//! `trimvol` has been a Python command since IMOD 4.1 (the earlier Fortran
//! implementation was removed).  Consequently this unit deliberately maps
//! the Python program's single top-level routine to [`trimvol`], rather than
//! inventing a Rust command framework.  Its MRC/newstack/clip path is native;
//! the two explicitly spawned analysis programs remain process boundaries.
#![allow(dead_code)]

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::libiimod::iimage::{
    IIFILE_DEFAULT, IIFILE_MRC, ii_close, ii_fill_mrc_header, ii_open, ii_open_new,
    ii_read_section_float, ii_sync_from_mrc_header,
};
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_FLOAT, MRC_NLABELS, MrcHeader, mrc_head_new, mrc_head_write,
};
use crate::imod::libiimod::mrcsec::mrc_write_section_any;
use std::ffi::CString;

/// Original Python program top level (`pysrc/trimvol:1`).
pub fn trimvol() -> i32 {
    // `pysrc/trimvol:16-29`: the installed command requires its IMOD runtime
    // root before it imports its command support modules, so this deliberately
    // precedes even the special-case option handling below.
    if std::env::var_os("IMOD_DIR").is_none() {
        println!("ERROR: trimvol -  IMOD_DIR is not defined!");
        return 1;
    }
    let mut x_limits = None::<(i32, i32)>;
    let mut y_limits = None::<(i32, i32)>;
    let mut z_limits = None::<(i32, i32)>;
    let mut x_size = None::<i32>;
    let mut y_size = None::<i32>;
    let mut z_size = None::<i32>;
    let mut find_z = None::<(i32, i32)>;
    let mut find_x = None::<(i32, i32)>;
    let mut find_y = None::<(i32, i32)>;
    let mut contrast = None::<(i32, i32)>;
    let mut integer_min_max = None::<(f32, f32)>;
    let mut mean_sd = None::<(f32, f32)>;
    let mut mode = None::<i32>;
    let mut rotate_x = false;
    let mut flip_yz = false;
    let mut index = false;
    let mut flipped_coordinates = false;
    let mut old_flip = 0_i32;
    let mut keep_origin = false;
    let mut output_format = String::new();
    let mut positional = Vec::<String>::new();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        let opt = argument.trim_start_matches('-').to_ascii_lowercase();
        let pair_i = |value: String| -> Option<(i32, i32)> {
            let mut entries = value.split(',').map(str::trim);
            Some((entries.next()?.parse().ok()?, entries.next()?.parse().ok()?))
        };
        let pair_f = |value: String| -> Option<(f32, f32)> {
            let mut entries = value.split(',').map(str::trim);
            Some((entries.next()?.parse().ok()?, entries.next()?.parse().ok()?))
        };
        match opt.as_str() {
            "x" | "xstartandend" => x_limits = args.next().and_then(pair_i),
            "y" | "ystartandend" => y_limits = args.next().and_then(pair_i),
            "z" | "zstartandend" => z_limits = args.next().and_then(pair_i),
            "nx" | "xsize" => x_size = args.next().and_then(|v| v.parse().ok()),
            "ny" | "ysize" => y_size = args.next().and_then(|v| v.parse().ok()),
            "nz" | "zsize" => z_size = args.next().and_then(|v| v.parse().ok()),
            "sz" | "zfindstartandend" => find_z = args.next().and_then(pair_i),
            "sx" | "xfindstartandend" => find_x = args.next().and_then(pair_i),
            "sy" | "yfindstartandend" => find_y = args.next().and_then(pair_i),
            "c" | "contrastblackwhite" => contrast = args.next().and_then(pair_i),
            "mm" | "integerminmax" => integer_min_max = args.next().and_then(pair_f),
            "meansd" | "scaletomeanandsd" => mean_sd = args.next().and_then(pair_f),
            "mode" | "modetooutput" => mode = args.next().and_then(|v| v.parse().ok()),
            "rx" | "rotatex" => rotate_x = true,
            "yz" | "flipyz" => flip_yz = true,
            "i" | "indexcoordinates" => index = true,
            "f" | "flippedcoordinates" => flipped_coordinates = true,
            "old" | "oldflippedcoordinates" => {
                old_flip = args.next().and_then(|v| v.parse().ok()).unwrap_or(0)
            }
            "k" | "keeporigin" => keep_origin = true,
            "format" | "formatofoutputfile" => output_format = args.next().unwrap_or_default(),
            "pid" => println!("{}", std::process::id()),
            "help" | "usage" => {
                println!("Usage: trimvol [options] input output");
                return 0;
            }
            "s" => {
                println!(
                    "ERROR: trimvol - The -s option has been eliminated; use -sz instead and add -f if coordinates are from a volume loaded with flipping"
                );
                return 1;
            }
            _ if argument.starts_with('-') => {
                println!("ERROR: trimvol - Unknown option {argument}");
                return 1;
            }
            _ => positional.push(argument),
        }
    }
    if positional.len() != 2 {
        println!("ERROR: trimvol - wrong number of arguments");
        return 1;
    }
    // `pysrc/trimvol:65-69` checks the input pathname immediately after PIP
    // has supplied the two positional names, before any option-combination
    // validation.  Keep that ordering because it controls the command's
    // observable first diagnostic.
    if !std::path::Path::new(&positional[0]).exists() {
        println!(
            "ERROR: trimvol - Input file {} does not exist",
            positional[0]
        );
        return 1;
    }
    if x_limits.is_some() && x_size.is_some() {
        println!("ERROR: trimvol - You cannot enter both -x and -nx options");
        return 1;
    }
    if y_limits.is_some() && y_size.is_some() {
        println!("ERROR: trimvol - You cannot enter both -y and -ny options");
        return 1;
    }
    if z_limits.is_some() && z_size.is_some() {
        println!("ERROR: trimvol - You cannot enter both -z and -nz options");
        return 1;
    }
    if rotate_x && flip_yz {
        println!("ERROR: trimvol - You cannot use both -yz and -rx options");
        return 1;
    }
    // `pysrc/trimvol:90-95`: a requested output mode cannot accompany direct
    // contrast scaling, nor findcontrast (unless densmatch supersedes it).
    if mode.is_some() && (contrast.is_some() || (find_z.is_some() && mean_sd.is_none())) {
        println!("ERROR: trimvol - You cannot enter -mode with -c, or when running findcontrast");
        return 1;
    }
    if contrast.is_some() as i32 + integer_min_max.is_some() as i32 + find_z.is_some() as i32 > 1
        || contrast.is_some() as i32 + integer_min_max.is_some() as i32 + mean_sd.is_some() as i32
            > 1
    {
        println!(
            "ERROR: trimvol - You cannot enter -c and -mm with each other or with -sz or -meansd"
        );
        return 1;
    }
    if !output_format.is_empty()
        && !matches!(
            output_format.to_ascii_uppercase().as_str(),
            "MRC" | "HDF" | "TIFF" | "TIF"
        )
    {
        println!("ERROR: trimvol - Output format {output_format} is not a valid format");
        return 1;
    }
    // Source delegates HDF/TIFF production to newstack.  These image writer backends
    // are separate source units; retain this explicit boundary rather than emitting a
    // different file format under the requested extension.
    if matches!(
        output_format.to_ascii_uppercase().as_str(),
        "HDF" | "TIFF" | "TIF"
    ) {
        println!(
            "ERROR: trimvol - requested output format requires the untranslated newstack image-format boundary"
        );
        return 1;
    }
    let input_name = match CString::new(positional[0].as_bytes()) {
        Ok(v) => v,
        Err(_) => {
            println!("ERROR: trimvol - Invalid input file name");
            return 1;
        }
    };
    let output_name = match CString::new(positional[1].as_bytes()) {
        Ok(v) => v,
        Err(_) => {
            println!("ERROR: trimvol - Invalid output file name");
            return 1;
        }
    };
    unsafe {
        let input = ii_open(input_name.as_ptr(), c"rb".as_ptr());
        if input.is_null() {
            println!(
                "ERROR: trimvol - Input file {} does not exist",
                positional[0]
            );
            return 1;
        }
        let mut header = MrcHeader::default();
        if ii_fill_mrc_header(input, &raw mut header) != 0 {
            println!("ERROR: trimvol - Reading input header");
            ii_close(input);
            return 1;
        }
        let (nx, ny, nz) = (header.nx, header.ny, header.nz);
        if nx < 1 || ny < 1 || nz < 1 {
            println!("ERROR: trimvol - Invalid image dimensions");
            ii_close(input);
            return 1;
        }
        if flipped_coordinates {
            std::mem::swap(&mut y_size, &mut z_size);
            std::mem::swap(&mut y_limits, &mut z_limits);
            // `pysrc/trimvol:165-174`: after swapping the coordinate entries,
            // only the even old-flip convention needs Y reversal.  These are
            // still user (one-based unless -i) coordinates at this point.
            if old_flip % 2 == 0 {
                if let Some((a, b)) = y_limits {
                    y_limits = Some(if index {
                        (ny - 1 - b, ny - 1 - a)
                    } else {
                        (ny + 1 - b, ny + 1 - a)
                    });
                }
            }
        }
        let mut choose_axis = |limits: Option<(i32, i32)>,
                               size: Option<i32>,
                               n: i32,
                               letter: &str|
         -> Option<(i32, i32)> {
            if let Some((mut low, mut high)) = limits {
                if !index {
                    low -= 1;
                    high -= 1;
                }
                if low < 0 || high >= n || low > high {
                    println!("ERROR: trimvol - {letter} coordinates out of range");
                    return None;
                }
                Some((low, high))
            } else if let Some(width) = size {
                if width <= 0 || width > n {
                    println!("ERROR: trimvol - Illegal {letter} size {width}");
                    None
                } else {
                    let low = (n - width) / 2;
                    Some((low, low + width - 1))
                }
            } else {
                Some((0, n - 1))
            }
        };
        let Some((x0, x1)) = choose_axis(x_limits, x_size, nx, "X") else {
            ii_close(input);
            return 1;
        };
        let Some((y0, y1)) = choose_axis(y_limits, y_size, ny, "Y") else {
            ii_close(input);
            return 1;
        };
        // Unlike X/Y, source permits Z extraction outside the input and passes -blank.
        let (mut z0, mut z1) = if let Some((mut low, mut high)) = z_limits {
            if !index {
                low -= 1;
                high -= 1;
            }
            if low > high {
                println!("ERROR: trimvol - Z coordinates out of range");
                ii_close(input);
                return 1;
            }
            (low, high)
        } else if let Some(width) = z_size {
            if width <= 0 || width > nz {
                println!("ERROR: trimvol - Illegal Z size {width}");
                ii_close(input);
                return 1;
            }
            let low = (nz - width) / 2;
            (low, low + width - 1)
        } else {
            (0, nz - 1)
        };
        let (out_nx, out_ny, out_nz) = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1);
        let mut output_mode = mode.unwrap_or(header.mode);
        if !matches!(output_mode, 0 | 1 | 2 | 6 | 12) {
            println!("ERROR: trimvol - Invalid output mode");
            ii_close(input);
            return 1;
        }
        if contrast.is_some() {
            output_mode = 0;
        }
        if integer_min_max.is_some() && mode.is_none() {
            output_mode = 1;
        }
        if mean_sd.is_some() || find_z.is_some() {
            println!(
                "ERROR: trimvol - densmatch/findcontrast are direct external program boundaries not yet translated"
            );
            ii_close(input);
            return 1;
        }
        let output = ii_open_new(output_name.as_ptr(), c"wb".as_ptr(), IIFILE_DEFAULT);
        if output.is_null() {
            println!("ERROR: trimvol - Opening output file");
            ii_close(input);
            return 1;
        }
        // `iiOpenNew` resolves `IIFILE_DEFAULT` through `b3dOutputFileType`
        // (`iimage.c`), so `IMOD_OUTPUT_FORMAT` or an earlier `-T` can make
        // this a TIFF or JPEG file -- in which case `ImodImageFile.header` is
        // the libtiff `TIFF *` (`iitif.c:204, 687`) or the JPEG state, not an
        // `MrcHeader`, and everything below (`mrc_head_new`, `mrc_head_write`
        // on `(*output).fp`, `mrc_write_section_any`) would write MRC bytes
        // over it.  The source never reaches this: `trimvol:350-360` delegates
        // to `newstack`, which writes through the unit layer and handles every
        // output type.  This direct route is MRC-only, so refuse the
        // combination by name rather than corrupt the file.
        if (*output).file != IIFILE_MRC {
            println!(
                "ERROR: trimvol - This translation writes the trimmed volume directly and can only write MRC; unset IMOD_OUTPUT_FORMAT"
            );
            ii_close(output);
            ii_close(input);
            return 1;
        }
        let out_header = (*output).header.cast::<MrcHeader>();
        let (final_ny, final_nz) = if rotate_x || flip_yz {
            (out_nz, out_ny)
        } else {
            (out_ny, out_nz)
        };
        mrc_head_new(&mut *out_header, out_nx, final_ny, final_nz, output_mode);
        // `trimvol:350-360` delegates the ordinary path to newstack.  Its
        // `iiuTransHeader` preserves input labels and its final
        // `iiuWriteHeader(..., 1, ...)` appends the NEWSTACK title.  The flip
        // paths then delegate to clip and have their own title lifecycle.
        if !rotate_x && !flip_yz {
            (*out_header).labels = header.labels;
            (*out_header).nlabl = header.nlabl;
            (*out_header).nlabl = ((*out_header).nlabl + 1).min(MRC_NLABELS as i32);
            let mut title = [b' '; 80];
            title[..23].copy_from_slice(b"NEWSTACK: Images copied");
            let mut date = [b' '; 9];
            b3d_date(&mut date);
            title[56..65].copy_from_slice(&date);
            let mut now = 0_i64;
            let mut local = std::mem::zeroed::<libc::tm>();
            libc::time(&raw mut now);
            libc::localtime_r(&raw const now, &raw mut local);
            let mut time = [0_i8; 9];
            libc::strftime(
                time.as_mut_ptr(),
                time.len(),
                c"%H:%M:%S".as_ptr(),
                &raw const local,
            );
            title[67..75].copy_from_slice(std::slice::from_raw_parts(time.as_ptr().cast(), 8));
            (&mut (*out_header).labels[((*out_header).nlabl - 1) as usize])[..80]
                .copy_from_slice(&title);
        }
        (*out_header).xlen = header.xlen * out_nx as f32 / nx as f32;
        (*out_header).ylen = if rotate_x || flip_yz {
            header.zlen * out_nz as f32 / nz as f32
        } else {
            header.ylen * out_ny as f32 / ny as f32
        };
        (*out_header).zlen = if rotate_x || flip_yz {
            header.ylen * out_ny as f32 / ny as f32
        } else {
            header.zlen * out_nz as f32 / nz as f32
        };
        if keep_origin {
            (*out_header).xorg = header.xorg;
            (*out_header).yorg = header.yorg;
            (*out_header).zorg = header.zorg;
        } else {
            let dx = if header.mx != 0 {
                header.xlen / header.mx as f32
            } else {
                0.
            };
            let dy = if header.my != 0 {
                header.ylen / header.my as f32
            } else {
                0.
            };
            let dz = if header.mz != 0 {
                header.zlen / header.mz as f32
            } else {
                0.
            };
            (*out_header).xorg = header.xorg - x0 as f32 * dx;
            (*out_header).yorg = header.yorg - y0 as f32 * dy;
            (*out_header).zorg = header.zorg - z0 as f32 * dz;
        }
        // `trimvol` sends `-rx` through `clip rotx`.  Preserve its direct
        // rotation-header branch after the equivalent newstack crop/origin
        // setup above.
        let pre_rotate_ylen = header.ylen * out_ny as f32 / ny as f32;
        let pre_rotate_zlen = header.zlen * out_nz as f32 / nz as f32;
        if rotate_x
            && out_ny != 0
            && pre_rotate_ylen != 0.0
            && out_nz != 0
            && pre_rotate_zlen != 0.0
        {
            for index in 0..3 {
                (*out_header).tiltangles[index] = header.tiltangles[index + 3];
            }
            (*out_header).tiltangles[3] = header.tiltangles[3] - 90.0;
            (*out_header).tiltangles[4] = header.tiltangles[4];
            (*out_header).tiltangles[5] = header.tiltangles[5];
            let ycen = out_ny as f32 / 2.0 - (*out_header).yorg * out_ny as f32 / pre_rotate_ylen;
            let zcen = out_nz as f32 / 2.0 - (*out_header).zorg * out_nz as f32 / pre_rotate_zlen;
            (*out_header).yorg =
                (final_ny as f32 / 2.0 - zcen) * (*out_header).ylen / final_ny as f32;
            (*out_header).zorg =
                (final_nz as f32 / 2.0 + ycen) * (*out_header).zlen / final_nz as f32;
        }
        ii_sync_from_mrc_header(output, out_header);
        if mrc_head_write(&mut (*output).fp.clone().unwrap(), &mut *out_header) != 0 {
            println!("ERROR: trimvol - Writing output header");
            ii_close(output);
            ii_close(input);
            return 1;
        }
        let mut source = vec![0.0_f32; (nx * ny) as usize];
        let mut volume = vec![0.0_f32; (out_nx * out_ny * out_nz) as usize];
        for oz in 0..out_nz {
            let iz = z0 + oz;
            if iz >= 0
                && iz < nz
                && ii_read_section_float(input, source.as_mut_ptr().cast(), iz) != 0
            {
                println!("ERROR: trimvol - Reading image section");
                ii_close(output);
                ii_close(input);
                return 1;
            }
            for oy in 0..out_ny {
                for ox in 0..out_nx {
                    let value = if iz >= 0 && iz < nz {
                        source[((y0 + oy) * nx + x0 + ox) as usize]
                    } else {
                        0.0
                    };
                    volume[((oz * out_ny + oy) * out_nx + ox) as usize] = value;
                }
            }
        }
        if let Some((black, white)) = contrast {
            let factor = 255.0 / (white - black).max(1) as f32;
            for value in &mut volume {
                *value = ((*value - black as f32) * factor).clamp(0.0, 255.0);
            }
        }
        if let Some((low, high)) = integer_min_max {
            // `trimvol:250-254` constructs `newstack -mode 1 -sca low,high`.
            // Newstack scales the selected input range *to* these limits;
            // they are not input clipping limits.
            let input_min = volume.iter().copied().fold(f32::INFINITY, f32::min);
            let input_max = volume.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let factor = if input_max == input_min {
                1.0
            } else {
                (high - low) / (input_max - input_min)
            };
            for value in &mut volume {
                *value = low + (*value - input_min) * factor;
            }
        }
        let mut written = vec![0.0_f32; (out_nx * final_ny) as usize];
        for final_z in 0..final_nz {
            for final_y in 0..final_ny {
                for ox in 0..out_nx {
                    let (oz, oy) = if rotate_x {
                        // `trimvol` delegates `-rx` to `clip rotx`; its
                        // `clip_flip` route rotates by -90 degrees about X.
                        (final_y, out_ny - 1 - final_z)
                    } else if flip_yz {
                        (final_y, final_z)
                    } else {
                        (final_z, final_y)
                    };
                    written[(final_y * out_nx + ox) as usize] =
                        volume[((oz * out_ny + oy) * out_nx + ox) as usize];
                }
            }
            let mut write_info = LoadInfo::default();
            write_info.xmin = 0;
            write_info.xmax = out_nx - 1;
            write_info.ymin = 0;
            write_info.ymax = final_ny - 1;
            (*out_header).fp = (*output).fp.clone();
            if mrc_write_section_any(
                out_header,
                &raw mut write_info,
                written.as_mut_ptr().cast(),
                final_z,
                MRC_MODE_FLOAT,
            ) != 0
            {
                println!("ERROR: trimvol - Writing image section");
                ii_close(output);
                ii_close(input);
                return 1;
            }
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        for value in &volume {
            min = min.min(*value);
            max = max.max(*value);
            sum += f64::from(*value);
        }
        (*out_header).amin = min;
        (*out_header).amax = max;
        (*out_header).amean = (sum / volume.len() as f64) as f32;
        if mrc_head_write(&mut (*output).fp.clone().unwrap(), &mut *out_header) != 0 {
            println!("ERROR: trimvol - Updating output header");
            ii_close(output);
            ii_close(input);
            return 1;
        }
        ii_close(output);
        ii_close(input);
        let xoffset = x0 + out_nx / 2 - nx / 2;
        let yoffset = y0 + out_ny / 2 - ny / 2;
        println!(" ");
        println!("The newstack command was:");
        println!(
            "newstack -siz {out_nx},{out_ny} -off {xoffset},{yoffset} \"{}\" \"{}\"",
            positional[0], positional[1]
        );
    }
    0
}
