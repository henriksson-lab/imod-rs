//! Translation of `IMOD/clip/file_io.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{
    ClipOptions, IP_APPEND_ADD, IP_APPEND_FALSE, IP_APPEND_OVERWRITE, IP_APPEND_TRUNCATE, IP_BOXSD,
    IP_DEFAULT, show_warning,
};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use crate::imod::libcfshr::islice::{Islice, Istack, slice_create, slice_free, slice_put_val};
use crate::imod::libcfshr::parse_params::exit_error;
use crate::imod::libiimod::iimage::{
    ii_fclose, ii_fopen, ii_limited_tile_size, ii_lookup_file_from_fp, ii_set_chunk_sizes_for_fp,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_RGB,
    MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_NLABELS, MrcHeader, mrc_coord_cp, mrc_get_scale,
    mrc_getdcsize, mrc_head_label_cp, mrc_head_new, mrc_head_read, mrc_head_write, mrc_read_slice,
    mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{mrc_slice_resize, slice_mmm, slice_new_mode};
use std::io::Write as _;

/// C++ `set_mrc_coords` (`file_io.cpp:34`).
///
/// `file_io.cpp:37-38` reads the two headers out of `opt->hin` and `opt->hout`,
/// which `main` set to the addresses of the same objects it passes to every
/// process routine.  Rust cannot hold that second, aliasing path, so the fields
/// are gone from [`ClipOptions`] and the headers come in as arguments; every
/// caller already has both in scope.
pub fn set_mrc_coords(input: &MrcHeader, output: &mut MrcHeader, options: &ClipOptions) -> i32 {
    mrc_coord_cp(output, input);
    let (sx, sy, sz) = mrc_get_scale(output);
    output.nxstart = input.nxstart;
    output.nystart = input.nystart;
    output.nzstart = input.nzstart;
    let mut tx = (options.cx as i32 - options.ix / 2) as f32;
    tx += ((options.ix - options.ox) / 2) as f32;
    if sx != 0. {
        tx *= sx;
    }
    output.xorg -= tx;
    let mut ty = (options.cy as i32 - options.iy / 2) as f32;
    ty += ((options.iy - options.oy) / 2) as f32;
    if sy != 0. {
        ty *= sy;
    }
    output.yorg -= ty;
    if options.dim == 3 {
        let mut tz = (options.cz as f64 - options.iz as f64 / 2.).floor() as i32 as f32;
        tz += ((options.iz - options.oz) / 2) as f32;
        if sz != 0. {
            tz *= sz;
        }
        output.zorg -= tz;
    }
    mrc_head_write(&mut output.fp.clone().unwrap(), output)
}
/// C++ `set_input_options` (`file_io.cpp:79`).
///
/// # Safety
///
/// `opt->secs` is `malloc`ed and `free`d by the source; here it is the
/// `Vec<i32>` of `ClipOptions`, so the `free` at `file_io.cpp:108` is a
/// `clear` and each `malloc` is a fresh allocation of the same length.
pub fn set_input_options(
    options: &mut ClipOptions,
    input: &crate::imod::libiimod::mrcfiles::MrcHeader,
) {
    {
        if options.nofsecs == IP_DEFAULT {
            options.nofsecs = input.nz;
            options.secs = vec![0; input.nz.max(0) as usize];
            for i in 0..input.nz {
                options.secs[i as usize] = i;
            }
        }
        if options.dim == 3 {
            if options.iz == IP_DEFAULT {
                options.iz = input.nz;
            }
            if options.cz == IP_DEFAULT as f32 {
                options.cz = input.nz as f32 * 0.5;
            }
            if options.oz == IP_DEFAULT {
                options.oz = options.iz;
            } else if options.iz > options.oz {
                options.iz = options.oz;
            }
            if options.nofsecs != IP_DEFAULT {
                options.secs.clear();
            }
            let mut zst = (options.cz as f64 - options.iz as f64 / 2.).floor() as i32;
            let znd = (zst + options.iz - 1).min(input.nz - 1).max(0);
            zst = zst.min(input.nz - 1).max(0);
            options.nofsecs = znd + 1 - zst;
            // `file_io.cpp:113-115`.  A negative `nofsecs` makes the C's
            // `sizeof(int) * opt->nofsecs` wrap as a `size_t`, `malloc` fails
            // and the fill loop does not run; here the allocation is empty and
            // `zst..=znd` is likewise empty, since `nofsecs < 0` means
            // `znd < zst`.
            options.secs = vec![0; options.nofsecs.max(0) as usize];
            for i in zst..=znd {
                options.secs[(i - zst) as usize] = i;
            }
            if options.oz != IP_DEFAULT && options.oz > options.nofsecs {
                let ozst = (options.cz as f64 - options.oz as f64 / 2.).floor() as i32;
                options.out_before = (zst - ozst).max(0);
                options.out_after = options.oz - options.nofsecs - options.out_before;
                if options.out_after < 0 {
                    options.out_before += options.out_after;
                    options.out_after = 0;
                }
            }
        } else if options.oz == IP_DEFAULT {
            options.oz = options.nofsecs;
        }
        if options.x != IP_DEFAULT {
            options.x2 = options.x.max(options.x2);
            options.cx = (options.x + options.x2 + 1) as f32 / 2.;
            options.ix = options.x2 + 1 - options.x;
        }
        if options.ix == IP_DEFAULT {
            options.ix = input.nx;
        }
        if options.cx as i32 == IP_DEFAULT {
            options.cx = input.nx as f32 * 0.5;
        }
        if options.y != IP_DEFAULT {
            options.y2 = options.y.max(options.y2);
            options.cy = (options.y + options.y2 + 1) as f32 / 2.;
            options.iy = options.y2 + 1 - options.y;
        }
        if options.iy == IP_DEFAULT {
            options.iy = input.ny;
        }
        if options.cy as i32 == IP_DEFAULT {
            options.cy = input.ny as f32 * 0.5;
        }
        if options.process == IP_BOXSD {
            let divisor = options.val.abs().round() as i32;
            if options.ox == IP_DEFAULT {
                options.ox = options.ix / divisor;
            }
            if options.oy == IP_DEFAULT {
                options.oy = options.iy / divisor;
            }
            if options.mode == IP_DEFAULT {
                options.mode = 2;
            }
        }
        if options.ox == IP_DEFAULT {
            options.ox = options.ix;
        } else if options.ocanresize == 0 {
            show_warning("clip - Process can't change output size.");
            options.ox = options.ix;
        }
        if options.oy == IP_DEFAULT {
            options.oy = options.iy;
        } else if options.ocanresize == 0 {
            show_warning("clip - Process can't change output size.");
            options.oy = options.iy;
        }
        if options.pad == IP_DEFAULT as f32 {
            options.pad = input.amean;
        }
        if options.mode == IP_DEFAULT {
            options.mode = input.mode;
        }
    }
}
/// C++ `set_output_options` (`file_io.cpp:202`).
pub fn set_output_options(options: &mut ClipOptions, output: &mut MrcHeader) -> i32 {
    let mut z = 0;
    let limits = [0i32; 3];
    let mut tile_sizes = [0i32; 3];
    let mut num_tiles = [0i32; 3];
    if options.add2file == IP_APPEND_FALSE {
        mrc_head_new(output, options.ox, options.oy, options.oz, options.mode);
        if set_chunk_output(options, output, &limits, &mut num_tiles, &mut tile_sizes) != 0
            || mrc_head_write(&mut output.fp.clone().unwrap(), output) != 0
        {
            return -1;
        }
    } else {
        if options.ox != output.nx || options.oy != output.ny {
            options.ox = output.nx;
            options.oy = output.ny;
            show_warning("clip - Appended file can't change size.");
        }
        if options.mode != output.mode {
            options.mode = output.mode;
            show_warning("clip - Appended file can't change mode.");
        }
        if options.add2file == IP_APPEND_ADD {
            z = output.nz;
            output.nz += options.oz;
            output.amean = z as f32 * output.amean / output.nz as f32;
        } else {
            z = options.isec;
            if options.add2file == IP_APPEND_OVERWRITE {
                output.nz = output.nz.max(options.oz + options.isec);
            } else {
                output.nz = options.oz + options.isec;
            }
        }
        if mrc_head_write(&mut output.fp.clone().unwrap(), output) != 0 {
            return -1;
        }
    }
    z
}
/// C++ `set_options` (`file_io.cpp:244`).
pub fn set_options(
    options: &mut ClipOptions,
    input: &mut MrcHeader,
    output: &mut MrcHeader,
) -> i32 {
    set_input_options(options, input);
    let z = set_output_options(options, output);
    mrc_head_label_cp(input, output);
    z
}
/// C++ `set_multifile_input_options` (`file_io.cpp:255`).
pub fn set_multifile_input_options(options: &mut ClipOptions, input: &mut MrcHeader) {
    if options.infiles > 1 {
        if options.add2file != 0 {
            exit_error(b"Multiple input files can not be entered with appending or overwriting");
        }
        for file_index in 0..options.infiles {
            let name = &options.fnames[file_index as usize];
            let fp = ii_fopen(name.as_bytes(), "rb");
            let Some(mut fp) = fp else {
                exit_error(c_format("Opening %s.", &[CArg::Str(name)]).as_bytes());
            };
            let mut header = MrcHeader::default();
            if mrc_head_read(&mut fp, &mut header) != 0 {
                exit_error(c_format("Reading header of %s.", &[CArg::Str(name)]).as_bytes());
            }
            if input.nx != header.nx
                || input.ny != header.ny
                || input.nz != header.nz
                || input.mode != header.mode
            {
                exit_error(
                    c_format(
                        "Files must be same mode and same size in X, Y, and Z; %s differs",
                        &[CArg::Str(name)],
                    )
                    .as_bytes(),
                );
            }
            ii_fclose(&mut fp);
        }
    }
    set_input_options(options, input);
    if options.infiles > 1 {
        if options.out_before != IP_DEFAULT || options.out_after != IP_DEFAULT {
            exit_error(b"Blank slices cannot be output with multiple input files");
        }
        options.out_before = -1;
    }
    options.oz *= options.infiles;
}
/// C++ `setChunkOutput` (`file_io.cpp:292`).
pub fn set_chunk_output(
    options: &mut ClipOptions,
    output: &mut MrcHeader,
    limits: &[i32; 3],
    num_tiles: &mut [i32; 3],
    tile_sizes: &mut [i32; 3],
) -> i32 {
    let out_sizes = [output.nx, output.ny, output.nz];
    tile_sizes[0] = options.chunk_x;
    tile_sizes[1] = options.chunk_y;
    tile_sizes[2] = options.chunk_z;
    for ind in 0..3usize {
        if tile_sizes[ind] == IP_DEFAULT || tile_sizes[ind] >= out_sizes[ind] {
            num_tiles[ind] = 1;
            tile_sizes[ind] = if ind == 2 { 1 } else { 0 };
        } else {
            ii_limited_tile_size(
                out_sizes[ind],
                &mut tile_sizes[ind],
                &mut num_tiles[ind],
                2,
                limits[ind],
            );
        }
    }
    if num_tiles[0] * num_tiles[1] * num_tiles[2] == 1 {
        return 0;
    }
    if ii_lookup_file_from_fp(&output.fp.clone().unwrap()).is_none() {
        let _ = ImodFile::Stderr.write_all(b"iiLookupFileFromFP cannot find iiFile from fp\n");
        return -1;
    }
    let mut dsize = 0;
    let mut csize = 0;
    mrc_getdcsize(output.mode, &mut dsize, &mut csize);
    let chunk_xy = (if tile_sizes[0] == 0 {
        out_sizes[0]
    } else {
        tile_sizes[0]
    }) as f32
        * (if tile_sizes[1] == 0 {
            out_sizes[1]
        } else {
            tile_sizes[1]
        }) as f32
        * csize as f32
        * dsize as f32;
    let max_z = (3.0e9_f32 / chunk_xy) as i32;
    if max_z < 1 {
        let _ = ImodFile::Stderr.write_all(
                b"Output tile or image size in X and Y too large to fit in one chunk; enter smaller chunk sizes\n",
            );
        return -1;
    }
    if max_z == 1 {
        tile_sizes[2] = 1;
        num_tiles[2] = output.nz;
    } else if tile_sizes[2] > max_z || (options.chunk_z == IP_DEFAULT && out_sizes[2] > max_z) {
        tile_sizes[2] = max_z;
        // `file_io.cpp:330` passes `outSizes[ind]`, and `ind` is 3 here --
        // one past the end of `int outSizes[3]`, so native reads whatever
        // is next on its stack.  There is no value to reproduce; the third
        // element, which every other line of the routine uses, is passed.
        ii_limited_tile_size(
            out_sizes[2],
            &mut tile_sizes[2],
            &mut num_tiles[2],
            2,
            max_z,
        );
    } else if options.chunk_z == IP_DEFAULT {
        tile_sizes[2] = out_sizes[2];
    }
    if ii_set_chunk_sizes_for_fp(
        &output.fp.clone().unwrap(),
        tile_sizes[0],
        tile_sizes[1],
        tile_sizes[2],
    ) != 0
    {
        return -1;
    }
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "Chunk size set to %d %d %d\n",
            &[
                CArg::Int(tile_sizes[0] as i64),
                CArg::Int(tile_sizes[1] as i64),
                CArg::Int(tile_sizes[2] as i64),
            ],
        )
        .as_bytes(),
    );
    0
}
/// C++ `clipWriteSlice` (`file_io.cpp:344`).
pub fn clip_write_slice(
    slice: &mut Islice,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
    ksec: i32,
    z_write: &mut i32,
    _free_slice: i32,
) -> i32 {
    if ksec < (options.nofsecs - options.oz) / 2
        || ksec >= (options.nofsecs - options.oz) / 2 + options.oz
    {
        return 0;
    }
    let mut blank_before = 0;
    let mut blank_after = 0;
    let mut blank: Option<Islice> = None;
    if options.oz > options.nofsecs && options.out_before != -1 {
        if ksec == 0 {
            blank_before = if options.out_before == IP_DEFAULT {
                (options.oz - options.nofsecs) / 2
            } else {
                options.out_before
            };
        }
        if ksec == options.nofsecs - 1 {
            blank_after = if options.out_after == IP_DEFAULT {
                options.oz - options.nofsecs - (options.oz - options.nofsecs) / 2
            } else {
                options.out_after
            };
        }
        if blank_before != 0 || blank_after != 0 {
            blank = clip_blank_slice(output, options);
            if blank.is_none() {
                return -1;
            }
        }
    }
    for _ in 0..blank_before {
        if mrc_write_slice(
            &blank.as_mut().unwrap().data,
            &mut output.fp.clone().unwrap(),
            output,
            *z_write,
            b'z',
        ) != 0
        {
            return -1;
        }
        *z_write += 1;
    }
    if slice.mode != options.mode && slice_new_mode(slice, options.mode) < 0 {
        return -1;
    }
    if options.ox != slice.xsize || options.oy != slice.ysize {
        slice.mean = options.pad;
        let Some(mut resized) = mrc_slice_resize(slice, options.ox, options.oy) else {
            let _ = ImodFile::Stderr.write_all(b"clipWriteSlice: error resizing slice.\n");
            return -1;
        };
        slice_mmm(resized.as_mut());
        // `file_io.cpp:406-407`: `B3DMIN(hout->amin, s->min)` /
        // `B3DMAX(hout->amax, s->max)` — `a < b ? a : b` keeps the
        // *second* operand when either is NaN, unlike `f32::min`/`max`.
        output.amin = if output.amin < resized.min {
            output.amin
        } else {
            resized.min
        };
        output.amax = if output.amax > resized.max {
            output.amax
        } else {
            resized.max
        };
        if options.add2file != IP_APPEND_OVERWRITE && options.add2file != IP_APPEND_TRUNCATE {
            output.amean += (resized.mean + (blank_before + blank_after) as f32 * options.pad)
                / output.nz as f32;
        }
        if mrc_write_slice(
            &resized.data,
            &mut output.fp.clone().unwrap(),
            output,
            *z_write,
            b'z',
        ) != 0
        {
            return -1;
        }
    } else {
        slice_mmm(slice);
        // `file_io.cpp:406-407`: `B3DMIN(hout->amin, s->min)` /
        // `B3DMAX(hout->amax, s->max)` — `a < b ? a : b` keeps the
        // *second* operand when either is NaN, unlike `f32::min`/`max`.
        output.amin = if output.amin < slice.min {
            output.amin
        } else {
            slice.min
        };
        output.amax = if output.amax > slice.max {
            output.amax
        } else {
            slice.max
        };
        if options.add2file != IP_APPEND_OVERWRITE && options.add2file != IP_APPEND_TRUNCATE {
            output.amean +=
                (slice.mean + (blank_before + blank_after) as f32 * options.pad) / output.nz as f32;
        }
        if mrc_write_slice(
            &slice.data,
            &mut output.fp.clone().unwrap(),
            output,
            *z_write,
            b'z',
        ) != 0
        {
            return -1;
        }
    }
    *z_write += 1;
    for _ in 0..blank_after {
        if mrc_write_slice(
            &blank.as_mut().unwrap().data,
            &mut output.fp.clone().unwrap(),
            output,
            *z_write,
            b'z',
        ) != 0
        {
            return -1;
        }
        *z_write += 1;
    }
    0
}
/// C++ `grap_volume_read` (`file_io.cpp:421`).
pub fn grap_volume_read(input: &mut MrcHeader, options: &mut ClipOptions) -> Option<Istack> {
    if options.dim == 2 {
        if options.iz == IP_DEFAULT {
            options.iz = 0;
        }
        if options.iz2 == IP_DEFAULT {
            options.iz2 = input.nz - 1;
        }
        if options.iz2 == 0 || options.iz2 < options.iz {
            options.iz2 = options.iz;
        }
        options.cz = (options.iz2 + options.iz) as f32 / 2.;
        options.iz = options.iz2 - options.iz + 1;
    }
    if options.ix == IP_DEFAULT {
        options.ix = input.nx;
    }
    if options.iy == IP_DEFAULT {
        options.iy = input.ny;
    }
    if options.iz == IP_DEFAULT {
        options.iz = input.nz;
    }
    if options.cx == IP_DEFAULT as f32 {
        options.cx = input.nx as f32 / 2.;
    }
    if options.cy == IP_DEFAULT as f32 {
        options.cy = input.ny as f32 / 2.;
    }
    if options.cz == IP_DEFAULT as f32 {
        options.cz = input.nz as f32 / 2.;
    }
    let pad = if options.pad == IP_DEFAULT as f32 {
        input.amean
    } else {
        options.pad
    };
    let Ok(zsize) = usize::try_from(options.iz) else {
        return None;
    };
    let mut slices = Vec::new();
    if slices.try_reserve_exact(zsize).is_err() {
        return None;
    }
    for z in 0..options.iz {
        let Some(mut out) = slice_create(options.ix, options.iy, input.mode) else {
            return None;
        };
        for y in 0..options.iy {
            for x in 0..options.ix {
                slice_put_val(out.as_mut(), x, y, [pad; 4]);
            }
        }
        out.mean = input.amean;
        out.max = input.amax;
        out.min = input.amin;
        slices.push(out);
    }
    let mut volume = Istack { slices };
    let Some(mut source) = slice_create(input.nx, input.ny, input.mode) else {
        return None;
    };
    let start_z = (options.cz - options.iz as f32 * 0.5f32).floor() as i32;
    for z in 0..options.iz {
        let file_z = start_z + z;
        if file_z < 0 || file_z >= input.nz {
            continue;
        }
        if mrc_read_slice(
            &mut source.data,
            &mut input.fp.clone().unwrap(),
            input,
            file_z,
            b'z',
        ) != 0
        {
            return None;
        }
        let start_y = (options.cy as f64 - options.iy as f64 / 2.).floor() as i32;
        for y in 0..options.iy {
            let from_y = start_y + y;
            if from_y < 0 || from_y >= input.ny {
                continue;
            }
            let start_x = (options.cx as f64 - options.ix as f64 / 2.).floor() as i32;
            for x in 0..options.ix {
                let from_x = start_x + x;
                if from_x < 0 || from_x >= input.nx {
                    continue;
                }
                let mut value = [0.; 4];
                crate::imod::libcfshr::islice::slice_get_val(&source, from_x, from_y, &mut value);
                slice_put_val(&mut volume.slices[z as usize], x, y, value);
            }
        }
    }
    Some(volume)
}
/// C++ `grap_volume_write` (`file_io.cpp:507`).
pub fn grap_volume_write(
    volume: &mut Istack,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    let Some(first) = volume.slices.first() else {
        return -1;
    };
    if options.mode == IP_DEFAULT {
        options.mode = first.mode;
    }
    if options.ox == IP_DEFAULT {
        options.ox = first.xsize;
    }
    if options.oy == IP_DEFAULT {
        options.oy = first.ysize;
    }
    if options.oz == IP_DEFAULT {
        options.oz = volume.slices.len() as i32;
    }
    if first.mode != options.mode {
        for slice in &mut volume.slices {
            slice_new_mode(slice, options.mode);
        }
    }
    for slice in &mut volume.slices {
        slice_mmm(slice);
    }
    let mut min = volume.slices[0].min;
    let mut max = volume.slices[0].max;
    let mut mean = volume.slices[0].mean;
    for slice in &volume.slices[1..] {
        // `file_io.cpp:541-542`: `B3DMIN(min, v->vol[k]->min)`.
        min = if min < slice.min { min } else { slice.min };
        max = if max > slice.max { max } else { slice.max };
        mean += slice.mean;
    }
    mean /= volume.slices.len() as f32;
    let first_mode = volume.slices[0].mode;
    let ks: i32;
    match options.add2file {
        IP_APPEND_TRUNCATE | IP_APPEND_OVERWRITE => {
            ks = options.isec;
            if options.add2file == IP_APPEND_TRUNCATE {
                output.nz = ks + options.oz;
            } else {
                output.nz = output.nz.max(ks + options.oz);
            }
            options.ox = output.nx;
            options.oy = output.ny;
            if output.mode != first_mode {
                let _ = ImodFile::Stderr
                    .write_all(b"overwriting requires data modes to be the same.\n");
                return -1;
            }
            // `file_io.cpp:566-567` / `:582-583`: `B3DMIN(min, hout->amin)`.
            output.amin = if min < output.amin { min } else { output.amin };
            output.amax = if max > output.amax { max } else { output.amax };
            let zscale = output.zlen / output.mz as f32;
            output.mz = output.nz;
            output.zlen = output.mz as f32 * zscale;
        }
        IP_APPEND_ADD => {
            ks = output.nz;
            output.nz += options.oz;
            options.ox = output.nx;
            options.oy = output.ny;
            if output.mode != first_mode {
                let _ =
                    ImodFile::Stderr.write_all(b"inserting requires data modes to be the same.\n");
                return -1;
            }
            // `file_io.cpp:566-567` / `:582-583`: `B3DMIN(min, hout->amin)`.
            output.amin = if min < output.amin { min } else { output.amin };
            output.amax = if max > output.amax { max } else { output.amax };
            output.amean = (output.amean * ks as f32 + mean * options.oz as f32) / output.nz as f32;
            let zscale = output.zlen / output.mz as f32;
            output.mz = output.nz;
            output.zlen = output.mz as f32 * zscale;
        }
        _ => {
            let labels = output.nlabl;
            mrc_head_new(&mut *output, options.ox, options.oy, options.oz, first_mode);
            output.nlabl = labels;
            output.amin = min;
            output.amax = max;
            output.amean = mean;
            ks = 0;
        }
    }
    if options.pad == IP_DEFAULT as f32 {
        options.pad = output.amean;
    }
    if mrc_head_write(&mut output.fp.clone().unwrap(), output) != 0 {
        return -1;
    }
    let zs = (volume.slices.len() as i32 - options.oz) / 2;
    let mut blank: Option<Islice> = None;
    for (out_z, source_z) in (ks..output.nz).zip(zs..) {
        if source_z < 0 || source_z >= volume.slices.len() as i32 {
            if blank.is_none() {
                blank = clip_blank_slice(output, options);
            }
            let Some(blank) = blank.as_mut() else {
                return -1;
            };
            if mrc_write_slice(
                &blank.data,
                &mut output.fp.clone().unwrap(),
                output,
                out_z,
                b'z',
            ) != 0
            {
                return -1;
            }
        } else {
            let source = &mut volume.slices[source_z as usize];
            let resized = if options.ox != source.xsize || options.oy != source.ysize {
                source.mean = options.pad;
                mrc_slice_resize(source, options.ox, options.oy)
            } else {
                None
            };
            let write_slice = resized.as_ref().unwrap_or(source);
            if mrc_write_slice(
                &write_slice.data,
                &mut output.fp.clone().unwrap(),
                output,
                out_z,
                b'z',
            ) != 0
            {
                return -1;
            }
        }
    }
    0
}
/// C++ static `clipBlankSlice` (`file_io.cpp:626`).
pub fn clip_blank_slice(output: &mut MrcHeader, options: &mut ClipOptions) -> Option<Islice> {
    let Some(mut slice) = slice_create(output.nx, output.ny, output.mode) else {
        let _ = ImodFile::Stderr.write_all(b"clipBlankSlice:  error getting slice\n");
        return None;
    };
    for y in 0..output.ny {
        for x in 0..output.nx {
            slice_put_val(&mut slice, x, y, [options.pad; 4]);
        }
    }
    Some(slice)
}
/// C++ `grap_volume_free` (`file_io.cpp:646`).
pub fn grap_volume_free(_volume: Istack) -> i32 {
    0
}
/// C++ `mrc_head_print` (`file_io.cpp:663`).
pub fn mrc_head_print(data: &MrcHeader) -> i32 {
    {
        // `file_io.cpp:684-689` has no null test; it dereferences `data`
        // immediately.  The pointer-shaped translation had invented an
        // `if (!data) return -1;`, which a `&MrcHeader` makes unreachable
        // anyway -- CLAUDE.md's "an error the source does not have is as wrong
        // as a missing one".
        let _ = ImodFile::Stdout.write_all(b"MRC header info:\n");
        match data.mode {
            MRC_MODE_BYTE => {
                let _ = ImodFile::Stdout.write_all(b"mode = Byte\n");
            }
            MRC_MODE_SHORT => {
                let _ = ImodFile::Stdout.write_all(b"mode = Short\n");
            }
            MRC_MODE_USHORT => {
                let _ = ImodFile::Stdout.write_all(b"mode = Unsigned Short\n");
            }
            MRC_MODE_FLOAT => {
                let _ = ImodFile::Stdout.write_all(b"mode = Float\n");
            }
            MRC_MODE_COMPLEX_SHORT => {
                let _ = ImodFile::Stdout.write_all(b"mode = Complex Short\n");
            }
            MRC_MODE_COMPLEX_FLOAT => {
                let _ = ImodFile::Stdout.write_all(b"mode = Complex Float\n");
            }
            MRC_MODE_RGB => {
                let _ = ImodFile::Stdout.write_all(b"mode = rgb byte\n");
            }
            _ => {
                let _ = ImodFile::Stdout.write_all(
                    c_format("mode is unknown (%d).\n", &[CArg::Int((data.mode) as i64)])
                        .as_bytes(),
                );
            }
        };
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Image size    =  ( %d, %d, %d)\n",
                &[
                    CArg::Int((data.nx) as i64),
                    CArg::Int((data.ny) as i64),
                    CArg::Int((data.nz) as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "minimum value = %g\n",
                &[CArg::Dbl((data.amin as f64) as f64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "maximum value = %g\n",
                &[CArg::Dbl((data.amax as f64) as f64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "mean value    = %g\n",
                &[CArg::Dbl((data.amean as f64) as f64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Start reading image at ( %d, %d, %d).\n",
                &[
                    CArg::Int((data.nxstart) as i64),
                    CArg::Int((data.nystart) as i64),
                    CArg::Int((data.nzstart) as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Read length   =  ( %d, %d, %d).\n",
                &[
                    CArg::Int((data.mx) as i64),
                    CArg::Int((data.my) as i64),
                    CArg::Int((data.mz) as i64),
                ],
            )
            .as_bytes(),
        );
        let mut xscale = 1.0_f32;
        let mut yscale = 1.0_f32;
        let mut zscale = 1.0_f32;
        if data.xlen != 0. {
            xscale = data.xlen / data.mx as f32;
        }
        if data.ylen != 0. {
            yscale = data.ylen / data.my as f32;
        }
        if data.zlen != 0. {
            zscale = data.zlen / data.mz as f32;
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Scale         =  ( %g x %g x %g ) Angstrom.\n",
                &[
                    CArg::Dbl((xscale as f64) as f64),
                    CArg::Dbl((yscale as f64) as f64),
                    CArg::Dbl((zscale as f64) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Cell Rotation =  ( %g, %g, %g).\n",
                &[
                    CArg::Dbl((data.alpha as f64) as f64),
                    CArg::Dbl((data.beta as f64) as f64),
                    CArg::Dbl((data.gamma as f64) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Columns are   = axis %d\n",
                &[CArg::Int((data.mapc) as i64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Rows are      = axis %d\n",
                &[CArg::Int((data.mapr) as i64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Sections are  = axis %d\n",
                &[CArg::Int((data.maps) as i64)],
            )
            .as_bytes(),
        );
        if data.ispg != 0 {
            let _ = ImodFile::Stdout
                .write_all(c_format("ispg =\t\t%d\n", &[CArg::Int((data.ispg) as i64)]).as_bytes());
        }
        if data.next != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format("extra header = \t%d\n", &[CArg::Int((data.next) as i64)]).as_bytes(),
            );
        }
        if data.idtype != 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format("idtype =\t%d\n", &[CArg::Int((data.idtype as i32) as i64)]).as_bytes(),
            );
            let _ = ImodFile::Stdout.write_all(
                c_format("nd1 =\t\t%g\n", &[CArg::Dbl((data.nd1 as f64) as f64)]).as_bytes(),
            );
            let _ = ImodFile::Stdout.write_all(
                c_format("nd2 =\t\t%g\n", &[CArg::Dbl((data.nd2 as f64) as f64)]).as_bytes(),
            );
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "vd1 =\t\t%g\n",
                    &[CArg::Dbl((data.vd1 as f64 / 100.) as f64)],
                )
                .as_bytes(),
            );
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "vd2 =\t\t%g\n",
                    &[CArg::Dbl((data.vd2 as f64 / 100.) as f64)],
                )
                .as_bytes(),
            );
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "angles = ( %g, %g, %g, %g, %g, %g)\n",
                &[
                    CArg::Dbl((data.tiltangles[0] as f64) as f64),
                    CArg::Dbl((data.tiltangles[1] as f64) as f64),
                    CArg::Dbl((data.tiltangles[2] as f64) as f64),
                    CArg::Dbl((data.tiltangles[3] as f64) as f64),
                    CArg::Dbl((data.tiltangles[4] as f64) as f64),
                    CArg::Dbl((data.tiltangles[5] as f64) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "orgin  = ( %g, %g, %g)\n",
                &[
                    CArg::Dbl((data.xorg as f64) as f64),
                    CArg::Dbl((data.yorg as f64) as f64),
                    CArg::Dbl((data.zorg as f64) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "creator id = %d\n",
                &[CArg::Int((data.creatid as i32) as i64)],
            )
            .as_bytes(),
        );
        if data.nlabl > MRC_NLABELS as i32 {
            let _ = ImodFile::Stdout.write_all(b"There are to many labels.\n\n");
            return 0;
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Thare are %d labels.\n\n",
                &[CArg::Int((data.nlabl) as i64)],
            )
            .as_bytes(),
        );
        for index in 0..data.nlabl {
            // `%s` on an MRC label: the 80 bytes come from the file and need
            // not be UTF-8, so this goes through `c_format_bytes`, which does
            // not round-trip them through a lossy `String`.
            let label = &data.labels[index as usize];
            let end = label.iter().position(|b| *b == 0).unwrap_or(label.len());
            let _ =
                ImodFile::Stdout.write_all(&c_format_bytes("%s\n", &[CArg::Bytes(&label[..end])]));
        }
        0
    }
}
