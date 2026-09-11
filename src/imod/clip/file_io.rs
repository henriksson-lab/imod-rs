//! Translation of `IMOD/clip/file_io.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{
    ClipOptions, IP_APPEND_ADD, IP_APPEND_FALSE, IP_APPEND_OVERWRITE, IP_APPEND_TRUNCATE, IP_BOXSD,
    IP_DEFAULT, show_warning,
};
use crate::imod::libcfshr::islice::{Islice, Istack, slice_create, slice_free, slice_put_val};
use crate::imod::libcfshr::parse_params::exit_error;
use crate::imod::libiimod::iimage::{
    ii_fclose, ii_fopen, ii_limited_tile_size, ii_lookup_file_from_fp, ii_set_chunk_sizes,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_RGB,
    MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_NLABELS, MrcHeader, mrc_coord_cp, mrc_get_scale,
    mrc_getdcsize, mrc_head_label_cp, mrc_head_new, mrc_head_read, mrc_head_write, mrc_read_slice,
    mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{mrc_slice_resize, slice_mmm, slice_new_mode};

/// C++ `set_mrc_coords` (`file_io.cpp:34`).
///
/// # Safety
///
/// The options must hold valid input/output headers and the output header's
/// file pointer must be writable, as required by the original routine.
pub unsafe fn set_mrc_coords(options: *mut ClipOptions) -> i32 {
    unsafe {
        let input = (*options).hin;
        let output = (*options).hout;
        mrc_coord_cp(&mut *output, &*input);
        let (sx, sy, sz) = mrc_get_scale(&*output);
        (*output).nxstart = (*input).nxstart;
        (*output).nystart = (*input).nystart;
        (*output).nzstart = (*input).nzstart;
        let mut tx = ((*options).cx as i32 - (*options).ix / 2) as f32;
        tx += (((*options).ix - (*options).ox) / 2) as f32;
        if sx != 0. {
            tx *= sx;
        }
        (*output).xorg -= tx;
        let mut ty = ((*options).cy as i32 - (*options).iy / 2) as f32;
        ty += (((*options).iy - (*options).oy) / 2) as f32;
        if sy != 0. {
            ty *= sy;
        }
        (*output).yorg -= ty;
        if (*options).dim == 3 {
            let mut tz = ((*options).cz as f64 - (*options).iz as f64 / 2.).floor() as i32 as f32;
            tz += (((*options).iz - (*options).oz) / 2) as f32;
            if sz != 0. {
                tz *= sz;
            }
            (*output).zorg -= tz;
        }
        mrc_head_write((*output).fp.cast(), output)
    }
}
/// C++ `set_input_options` (`file_io.cpp:79`).
///
/// # Safety
///
/// `options` and `input` must be valid IMOD layouts.  The section-list pointer
/// follows the original malloc/free ownership convention.
pub unsafe fn set_input_options(
    options: *mut ClipOptions,
    input: *mut crate::imod::libiimod::mrcfiles::MrcHeader,
) {
    unsafe {
        if (*options).nofsecs == IP_DEFAULT {
            (*options).nofsecs = (*input).nz;
            (*options).secs = libc::malloc(
                core::mem::size_of::<i32>().wrapping_mul((*input).nz as isize as usize),
            )
            .cast();
            for i in 0..(*input).nz {
                *(*options).secs.add(i as usize) = i;
            }
        }
        if (*options).dim == 3 {
            if (*options).iz == IP_DEFAULT {
                (*options).iz = (*input).nz;
            }
            if (*options).cz == IP_DEFAULT as f32 {
                (*options).cz = (*input).nz as f32 * 0.5;
            }
            if (*options).oz == IP_DEFAULT {
                (*options).oz = (*options).iz;
            } else if (*options).iz > (*options).oz {
                (*options).iz = (*options).oz;
            }
            if (*options).nofsecs != IP_DEFAULT {
                libc::free((*options).secs.cast());
            }
            let mut zst = ((*options).cz as f64 - (*options).iz as f64 / 2.).floor() as i32;
            let znd = (zst + (*options).iz - 1).min((*input).nz - 1).max(0);
            zst = zst.min((*input).nz - 1).max(0);
            (*options).nofsecs = znd + 1 - zst;
            // `file_io.cpp:113`: sizeof(int) * opt->nofsecs is a size_t product;
            // a negative nofsecs wraps and malloc fails, and the fill loop below
            // then does not run.
            (*options).secs = libc::malloc(
                core::mem::size_of::<i32>().wrapping_mul((*options).nofsecs as isize as usize),
            )
            .cast();
            for i in zst..=znd {
                *(*options).secs.add((i - zst) as usize) = i;
            }
            if (*options).oz != IP_DEFAULT && (*options).oz > (*options).nofsecs {
                let ozst = ((*options).cz as f64 - (*options).oz as f64 / 2.).floor() as i32;
                (*options).out_before = (zst - ozst).max(0);
                (*options).out_after = (*options).oz - (*options).nofsecs - (*options).out_before;
                if (*options).out_after < 0 {
                    (*options).out_before += (*options).out_after;
                    (*options).out_after = 0;
                }
            }
        } else if (*options).oz == IP_DEFAULT {
            (*options).oz = (*options).nofsecs;
        }
        if (*options).x != IP_DEFAULT {
            (*options).x2 = (*options).x.max((*options).x2);
            (*options).cx = ((*options).x + (*options).x2 + 1) as f32 / 2.;
            (*options).ix = (*options).x2 + 1 - (*options).x;
        }
        if (*options).ix == IP_DEFAULT {
            (*options).ix = (*input).nx;
        }
        if (*options).cx as i32 == IP_DEFAULT {
            (*options).cx = (*input).nx as f32 * 0.5;
        }
        if (*options).y != IP_DEFAULT {
            (*options).y2 = (*options).y.max((*options).y2);
            (*options).cy = ((*options).y + (*options).y2 + 1) as f32 / 2.;
            (*options).iy = (*options).y2 + 1 - (*options).y;
        }
        if (*options).iy == IP_DEFAULT {
            (*options).iy = (*input).ny;
        }
        if (*options).cy as i32 == IP_DEFAULT {
            (*options).cy = (*input).ny as f32 * 0.5;
        }
        if (*options).process == IP_BOXSD {
            let divisor = (*options).val.abs().round() as i32;
            if (*options).ox == IP_DEFAULT {
                (*options).ox = (*options).ix / divisor;
            }
            if (*options).oy == IP_DEFAULT {
                (*options).oy = (*options).iy / divisor;
            }
            if (*options).mode == IP_DEFAULT {
                (*options).mode = 2;
            }
        }
        if (*options).ox == IP_DEFAULT {
            (*options).ox = (*options).ix;
        } else if (*options).ocanresize == 0 {
            show_warning("clip - Process can't change output size.");
            (*options).ox = (*options).ix;
        }
        if (*options).oy == IP_DEFAULT {
            (*options).oy = (*options).iy;
        } else if (*options).ocanresize == 0 {
            show_warning("clip - Process can't change output size.");
            (*options).oy = (*options).iy;
        }
        if (*options).pad == IP_DEFAULT as f32 {
            (*options).pad = (*input).amean;
        }
        if (*options).mode == IP_DEFAULT {
            (*options).mode = (*input).mode;
        }
    }
}
/// C++ `set_output_options` (`file_io.cpp:202`).
pub unsafe fn set_output_options(options: *mut ClipOptions, output: *mut MrcHeader) -> i32 {
    unsafe {
        let mut z = 0;
        let mut limits = [0; 3];
        let mut tile_sizes = [0; 3];
        let mut num_tiles = [0; 3];
        if (*options).add2file == IP_APPEND_FALSE {
            mrc_head_new(
                &mut *output,
                (*options).ox,
                (*options).oy,
                (*options).oz,
                (*options).mode,
            );
            if set_chunk_output(
                options,
                output,
                limits.as_mut_ptr(),
                num_tiles.as_mut_ptr(),
                tile_sizes.as_mut_ptr(),
            ) != 0
                || mrc_head_write((*output).fp.cast(), output) != 0
            {
                return -1;
            }
        } else {
            if (*options).ox != (*output).nx || (*options).oy != (*output).ny {
                (*options).ox = (*output).nx;
                (*options).oy = (*output).ny;
                show_warning("clip - Appended file can't change size.");
            }
            if (*options).mode != (*output).mode {
                (*options).mode = (*output).mode;
                show_warning("clip - Appended file can't change mode.");
            }
            if (*options).add2file == IP_APPEND_ADD {
                z = (*output).nz;
                (*output).nz += (*options).oz;
                (*output).amean = z as f32 * (*output).amean / (*output).nz as f32;
            } else {
                z = (*options).isec;
                if (*options).add2file == IP_APPEND_OVERWRITE {
                    (*output).nz = (*output).nz.max((*options).oz + (*options).isec);
                } else {
                    (*output).nz = (*options).oz + (*options).isec;
                }
            }
            if mrc_head_write((*output).fp.cast(), output) != 0 {
                return -1;
            }
        }
        z
    }
}
/// C++ `set_options` (`file_io.cpp:244`).
pub unsafe fn set_options(
    options: *mut ClipOptions,
    input: *mut MrcHeader,
    output: *mut MrcHeader,
) -> i32 {
    unsafe {
        set_input_options(options, input);
        let z = set_output_options(options, output);
        mrc_head_label_cp(&*input, &mut *output);
        z
    }
}
/// C++ `set_multifile_input_options` (`file_io.cpp:255`).
pub unsafe fn set_multifile_input_options(options: *mut ClipOptions, input: *mut MrcHeader) {
    unsafe {
        if (*options).infiles > 1 {
            if (*options).add2file != 0 {
                exit_error(
                    c"Multiple input files can not be entered with appending or overwriting"
                        .as_ptr(),
                );
            }
            for file_index in 0..(*options).infiles {
                let fp = ii_fopen(*(*options).fnames.add(file_index as usize), c"rb".as_ptr());
                if fp.is_null() {
                    let message = std::ffi::CString::new(format!(
                        "Opening {}.",
                        core::ffi::CStr::from_ptr(*(*options).fnames.add(file_index as usize))
                            .to_string_lossy()
                    ))
                    .unwrap();
                    exit_error(message.as_ptr());
                }
                let mut header: MrcHeader = core::mem::zeroed();
                if mrc_head_read(fp, &mut header) != 0 {
                    ii_fclose(fp);
                    let message = std::ffi::CString::new(format!(
                        "Reading header of {}.",
                        core::ffi::CStr::from_ptr(*(*options).fnames.add(file_index as usize))
                            .to_string_lossy()
                    ))
                    .unwrap();
                    exit_error(message.as_ptr());
                }
                if (*input).nx != header.nx
                    || (*input).ny != header.ny
                    || (*input).nz != header.nz
                    || (*input).mode != header.mode
                {
                    ii_fclose(fp);
                    let message = std::ffi::CString::new(format!(
                        "Files must be same mode and same size in X, Y, and Z; {} differs",
                        core::ffi::CStr::from_ptr(*(*options).fnames.add(file_index as usize))
                            .to_string_lossy()
                    ))
                    .unwrap();
                    exit_error(message.as_ptr());
                }
                ii_fclose(fp);
            }
        }
        set_input_options(options, input);
        if (*options).infiles > 1 {
            if (*options).out_before != IP_DEFAULT || (*options).out_after != IP_DEFAULT {
                exit_error(c"Blank slices cannot be output with multiple input files".as_ptr());
            }
            (*options).out_before = -1;
        }
        (*options).oz *= (*options).infiles;
    }
}
/// C++ `setChunkOutput` (`file_io.cpp:292`).
pub unsafe fn set_chunk_output(
    options: *mut ClipOptions,
    output: *mut MrcHeader,
    limits: *mut i32,
    num_tiles: *mut i32,
    tile_sizes: *mut i32,
) -> i32 {
    unsafe {
        let out_sizes = [(*output).nx, (*output).ny, (*output).nz];
        *tile_sizes = (*options).chunk_x;
        *tile_sizes.add(1) = (*options).chunk_y;
        *tile_sizes.add(2) = (*options).chunk_z;
        for ind in 0..3usize {
            if *tile_sizes.add(ind) == IP_DEFAULT || *tile_sizes.add(ind) >= out_sizes[ind] {
                *num_tiles.add(ind) = 1;
                *tile_sizes.add(ind) = if ind == 2 { 1 } else { 0 };
            } else {
                ii_limited_tile_size(
                    out_sizes[ind],
                    &mut *tile_sizes.add(ind),
                    &mut *num_tiles.add(ind),
                    2,
                    *limits.add(ind),
                );
            }
        }
        if *num_tiles * *num_tiles.add(1) * *num_tiles.add(2) == 1 {
            return 0;
        }
        let ii_file = ii_lookup_file_from_fp((*output).fp.cast());
        if ii_file.is_null() {
            eprintln!("iiLookupFileFromFP cannot find iiFile from fp");
            return -1;
        }
        let mut dsize = 0;
        let mut csize = 0;
        mrc_getdcsize((*output).mode, &mut dsize, &mut csize);
        let chunk_xy = (if *tile_sizes == 0 {
            out_sizes[0]
        } else {
            *tile_sizes
        }) as f32
            * (if *tile_sizes.add(1) == 0 {
                out_sizes[1]
            } else {
                *tile_sizes.add(1)
            }) as f32
            * csize as f32
            * dsize as f32;
        let max_z = (3.0e9_f32 / chunk_xy) as i32;
        if max_z < 1 {
            eprintln!(
                "Output tile or image size in X and Y too large to fit in one chunk; enter smaller chunk sizes"
            );
            return -1;
        }
        if max_z == 1 {
            *tile_sizes.add(2) = 1;
            *num_tiles.add(2) = (*output).nz;
        } else if *tile_sizes.add(2) > max_z
            || ((*options).chunk_z == IP_DEFAULT && out_sizes[2] > max_z)
        {
            *tile_sizes.add(2) = max_z;
            ii_limited_tile_size(
                out_sizes[2],
                &mut *tile_sizes.add(2),
                &mut *num_tiles.add(2),
                2,
                max_z,
            );
        } else if (*options).chunk_z == IP_DEFAULT {
            *tile_sizes.add(2) = out_sizes[2];
        }
        if ii_set_chunk_sizes(ii_file, *tile_sizes, *tile_sizes.add(1), *tile_sizes.add(2)) != 0 {
            return -1;
        }
        libc::printf(
            c"Chunk size set to %d %d %d\n".as_ptr(),
            *tile_sizes,
            *tile_sizes.add(1),
            *tile_sizes.add(2),
        );
        0
    }
}
/// C++ `clipWriteSlice` (`file_io.cpp:344`).
pub unsafe fn clip_write_slice(
    slice: *mut Islice,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
    ksec: i32,
    z_write: *mut i32,
    free_slice: i32,
) -> i32 {
    unsafe {
        if ksec < ((*options).nofsecs - (*options).oz) / 2
            || ksec >= ((*options).nofsecs - (*options).oz) / 2 + (*options).oz
        {
            if free_slice != 0 {
                slice_free(slice);
            }
            return 0;
        }
        let mut blank_before = 0;
        let mut blank_after = 0;
        let mut blank: *mut Islice = core::ptr::null_mut();
        if (*options).oz > (*options).nofsecs && (*options).out_before != -1 {
            if ksec == 0 {
                blank_before = if (*options).out_before == IP_DEFAULT {
                    ((*options).oz - (*options).nofsecs) / 2
                } else {
                    (*options).out_before
                };
            }
            if ksec == (*options).nofsecs - 1 {
                blank_after = if (*options).out_after == IP_DEFAULT {
                    (*options).oz - (*options).nofsecs - ((*options).oz - (*options).nofsecs) / 2
                } else {
                    (*options).out_after
                };
            }
            if blank_before != 0 || blank_after != 0 {
                blank = clip_blank_slice(output, options);
                if blank.is_null() {
                    return -1;
                }
            }
        }
        for _ in 0..blank_before {
            if mrc_write_slice(
                (*blank).data.b.cast(),
                (*output).fp.cast(),
                output,
                *z_write,
                b'z' as i8,
            ) != 0
            {
                return -1;
            }
            *z_write += 1;
        }
        if (*slice).mode != (*options).mode && slice_new_mode(slice, (*options).mode) < 0 {
            return -1;
        }
        let mut resized = slice;
        if (*options).ox != (*slice).xsize || (*options).oy != (*slice).ysize {
            (*slice).mean = (*options).pad;
            resized = mrc_slice_resize(slice, (*options).ox, (*options).oy);
            if resized.is_null() {
                eprintln!("clipWriteSlice: error resizing slice.");
                return -1;
            }
        }
        slice_mmm(resized);
        (*output).amin = (*output).amin.min((*resized).min);
        (*output).amax = (*output).amax.max((*resized).max);
        if (*options).add2file != IP_APPEND_OVERWRITE && (*options).add2file != IP_APPEND_TRUNCATE {
            (*output).amean += ((*resized).mean
                + (blank_before + blank_after) as f32 * (*options).pad)
                / (*output).nz as f32;
        }
        if mrc_write_slice(
            (*resized).data.b.cast(),
            (*output).fp.cast(),
            output,
            *z_write,
            b'z' as i8,
        ) != 0
        {
            return -1;
        }
        *z_write += 1;
        if resized != slice {
            slice_free(resized);
        }
        if free_slice != 0 {
            slice_free(slice);
        }
        for _ in 0..blank_after {
            if mrc_write_slice(
                (*blank).data.b.cast(),
                (*output).fp.cast(),
                output,
                *z_write,
                b'z' as i8,
            ) != 0
            {
                return -1;
            }
            *z_write += 1;
        }
        if !blank.is_null() {
            slice_free(blank);
        }
        0
    }
}
/// C++ `grap_volume_read` (`file_io.cpp:421`).
pub unsafe fn grap_volume_read(input: *mut MrcHeader, options: *mut ClipOptions) -> *mut Istack {
    unsafe {
        if (*options).dim == 2 {
            if (*options).iz == IP_DEFAULT {
                (*options).iz = 0;
            }
            if (*options).iz2 == IP_DEFAULT {
                (*options).iz2 = (*input).nz - 1;
            }
            if (*options).iz2 == 0 || (*options).iz2 < (*options).iz {
                (*options).iz2 = (*options).iz;
            }
            (*options).cz = ((*options).iz2 + (*options).iz) as f32 / 2.;
            (*options).iz = (*options).iz2 - (*options).iz + 1;
        }
        if (*options).ix == IP_DEFAULT {
            (*options).ix = (*input).nx;
        }
        if (*options).iy == IP_DEFAULT {
            (*options).iy = (*input).ny;
        }
        if (*options).iz == IP_DEFAULT {
            (*options).iz = (*input).nz;
        }
        if (*options).cx == IP_DEFAULT as f32 {
            (*options).cx = (*input).nx as f32 / 2.;
        }
        if (*options).cy == IP_DEFAULT as f32 {
            (*options).cy = (*input).ny as f32 / 2.;
        }
        if (*options).cz == IP_DEFAULT as f32 {
            (*options).cz = (*input).nz as f32 / 2.;
        }
        let pad = if (*options).pad == IP_DEFAULT as f32 {
            (*input).amean
        } else {
            (*options).pad
        };
        let volume = libc::malloc(core::mem::size_of::<Istack>()).cast::<Istack>();
        if volume.is_null() {
            return volume;
        }
        (*volume).zsize = (*options).iz;
        (*volume).vol =
            libc::malloc((*options).iz as usize * core::mem::size_of::<*mut Islice>()).cast();
        if (*volume).vol.is_null() {
            libc::free(volume.cast());
            return core::ptr::null_mut();
        }
        for z in 0..(*options).iz {
            let out = slice_create((*options).ix, (*options).iy, (*input).mode);
            if out.is_null() {
                return core::ptr::null_mut();
            }
            (*volume).vol.add(z as usize).write(out);
            for y in 0..(*options).iy {
                for x in 0..(*options).ix {
                    slice_put_val(out, x, y, [pad; 4]);
                }
            }
            (*out).mean = (*input).amean;
            (*out).max = (*input).amax;
            (*out).min = (*input).amin;
        }
        let source = slice_create((*input).nx, (*input).ny, (*input).mode);
        if source.is_null() {
            return core::ptr::null_mut();
        }
        let start_z = ((*options).cz - (*options).iz as f32 * 0.5f32).floor() as i32;
        for z in 0..(*options).iz {
            let file_z = start_z + z;
            if file_z < 0 || file_z >= (*input).nz {
                continue;
            }
            if mrc_read_slice(
                (*source).data.b.cast(),
                (*input).fp.cast(),
                input,
                file_z,
                b'z' as i8,
            ) != 0
            {
                return core::ptr::null_mut();
            }
            let start_y = ((*options).cy as f64 - (*options).iy as f64 / 2.).floor() as i32;
            for y in 0..(*options).iy {
                let from_y = start_y + y;
                if from_y < 0 || from_y >= (*input).ny {
                    continue;
                }
                let start_x = ((*options).cx as f64 - (*options).ix as f64 / 2.).floor() as i32;
                for x in 0..(*options).ix {
                    let from_x = start_x + x;
                    if from_x < 0 || from_x >= (*input).nx {
                        continue;
                    }
                    let mut value = [0.; 4];
                    crate::imod::libcfshr::islice::slice_get_val(
                        source, from_x, from_y, &mut value,
                    );
                    slice_put_val(*(*volume).vol.add(z as usize), x, y, value);
                }
            }
        }
        slice_free(source);
        volume
    }
}
/// C++ `grap_volume_write` (`file_io.cpp:507`).
pub unsafe fn grap_volume_write(
    volume: *mut Istack,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
) -> i32 {
    unsafe {
        let first = *(*volume).vol;
        if (*options).mode == IP_DEFAULT {
            (*options).mode = (*first).mode;
        }
        if (*options).ox == IP_DEFAULT {
            (*options).ox = (*first).xsize;
        }
        if (*options).oy == IP_DEFAULT {
            (*options).oy = (*first).ysize;
        }
        if (*options).oz == IP_DEFAULT {
            (*options).oz = (*volume).zsize;
        }
        if (*first).mode != (*options).mode {
            for z in 0..(*volume).zsize {
                slice_new_mode(*(*volume).vol.add(z as usize), (*options).mode);
            }
        }
        slice_mmm(first);
        let mut min = (*first).min;
        let mut max = (*first).max;
        let mut mean = (*first).mean;
        for z in 1..(*volume).zsize {
            let slice = *(*volume).vol.add(z as usize);
            slice_mmm(slice);
            min = min.min((*slice).min);
            max = max.max((*slice).max);
            mean += (*slice).mean;
        }
        mean /= (*volume).zsize as f32;
        let ks: i32;
        match (*options).add2file {
            IP_APPEND_TRUNCATE | IP_APPEND_OVERWRITE => {
                ks = (*options).isec;
                if (*options).add2file == IP_APPEND_TRUNCATE {
                    (*output).nz = ks + (*options).oz;
                } else {
                    (*output).nz = (*output).nz.max(ks + (*options).oz);
                }
                (*options).ox = (*output).nx;
                (*options).oy = (*output).ny;
                if (*output).mode != (*first).mode {
                    eprintln!("overwriting requires data modes to be the same.");
                    return -1;
                }
                (*output).amin = (*output).amin.min(min);
                (*output).amax = (*output).amax.max(max);
                let zscale = (*output).zlen / (*output).mz as f32;
                (*output).mz = (*output).nz;
                (*output).zlen = (*output).mz as f32 * zscale;
            }
            IP_APPEND_ADD => {
                ks = (*output).nz;
                (*output).nz += (*options).oz;
                (*options).ox = (*output).nx;
                (*options).oy = (*output).ny;
                if (*output).mode != (*first).mode {
                    eprintln!("inserting requires data modes to be the same.");
                    return -1;
                }
                (*output).amin = (*output).amin.min(min);
                (*output).amax = (*output).amax.max(max);
                (*output).amean = ((*output).amean * ks as f32 + mean * (*options).oz as f32)
                    / (*output).nz as f32;
                let zscale = (*output).zlen / (*output).mz as f32;
                (*output).mz = (*output).nz;
                (*output).zlen = (*output).mz as f32 * zscale;
            }
            _ => {
                let labels = (*output).nlabl;
                mrc_head_new(
                    &mut *output,
                    (*options).ox,
                    (*options).oy,
                    (*options).oz,
                    (*first).mode,
                );
                (*output).nlabl = labels;
                (*output).amin = min;
                (*output).amax = max;
                (*output).amean = mean;
                ks = 0;
            }
        }
        if (*options).pad == IP_DEFAULT as f32 {
            (*options).pad = (*output).amean;
        }
        if mrc_head_write((*output).fp.cast(), output) != 0 {
            return -1;
        }
        let zs = ((*volume).zsize - (*options).oz) / 2;
        let mut blank: *mut Islice = core::ptr::null_mut();
        for (out_z, source_z) in (ks..(*output).nz).zip(zs..) {
            if source_z < 0 || source_z >= (*volume).zsize {
                if blank.is_null() {
                    blank = clip_blank_slice(output, options);
                    if blank.is_null() {
                        return -1;
                    }
                }
                if mrc_write_slice(
                    (*blank).data.b.cast(),
                    (*output).fp.cast(),
                    output,
                    out_z,
                    b'z' as i8,
                ) != 0
                {
                    return -1;
                }
            } else {
                let source = *(*volume).vol.add(source_z as usize);
                let resized =
                    if (*options).ox != (*source).xsize || (*options).oy != (*source).ysize {
                        (*source).mean = (*options).pad;
                        mrc_slice_resize(source, (*options).ox, (*options).oy)
                    } else {
                        source
                    };
                if resized.is_null() {
                    eprintln!("volume_write: error resizing slice.");
                    return -1;
                }
                if mrc_write_slice(
                    (*resized).data.b.cast(),
                    (*output).fp.cast(),
                    output,
                    out_z,
                    b'z' as i8,
                ) != 0
                {
                    return -1;
                }
                if resized != source {
                    slice_free(resized);
                }
            }
        }
        if !blank.is_null() {
            slice_free(blank);
        }
        0
    }
}
/// C++ static `clipBlankSlice` (`file_io.cpp:626`).
pub unsafe fn clip_blank_slice(output: *mut MrcHeader, options: *mut ClipOptions) -> *mut Islice {
    unsafe {
        let slice = slice_create((*output).nx, (*output).ny, (*output).mode);
        if slice.is_null() {
            eprintln!("clipBlankSlice:  error getting slice");
            return slice;
        }
        for y in 0..(*output).ny {
            for x in 0..(*output).nx {
                slice_put_val(slice, x, y, [(*options).pad; 4]);
            }
        }
        slice
    }
}
/// C++ `grap_volume_free` (`file_io.cpp:646`).
pub unsafe fn grap_volume_free(volume: *mut Istack) -> i32 {
    unsafe {
        if volume.is_null() {
            return -1;
        }
        for z in 0..(*volume).zsize {
            slice_free(*(*volume).vol.add(z as usize));
        }
        libc::free((*volume).vol.cast());
        libc::free(volume.cast());
        0
    }
}
/// C++ `mrc_head_print` (`file_io.cpp:663`).
pub unsafe fn mrc_head_print(data: *const MrcHeader) -> i32 {
    unsafe {
        if data.is_null() {
            return -1;
        }
        libc::printf(c"MRC header info:\n".as_ptr());
        match (*data).mode {
            MRC_MODE_BYTE => libc::printf(c"mode = Byte\n".as_ptr()),
            MRC_MODE_SHORT => libc::printf(c"mode = Short\n".as_ptr()),
            MRC_MODE_USHORT => libc::printf(c"mode = Unsigned Short\n".as_ptr()),
            MRC_MODE_FLOAT => libc::printf(c"mode = Float\n".as_ptr()),
            MRC_MODE_COMPLEX_SHORT => libc::printf(c"mode = Complex Short\n".as_ptr()),
            MRC_MODE_COMPLEX_FLOAT => libc::printf(c"mode = Complex Float\n".as_ptr()),
            MRC_MODE_RGB => libc::printf(c"mode = rgb byte\n".as_ptr()),
            _ => libc::printf(c"mode is unknown (%d).\n".as_ptr(), (*data).mode),
        };
        libc::printf(
            c"Image size    =  ( %d, %d, %d)\n".as_ptr(),
            (*data).nx,
            (*data).ny,
            (*data).nz,
        );
        libc::printf(c"minimum value = %g\n".as_ptr(), (*data).amin as f64);
        libc::printf(c"maximum value = %g\n".as_ptr(), (*data).amax as f64);
        libc::printf(c"mean value    = %g\n".as_ptr(), (*data).amean as f64);
        libc::printf(
            c"Start reading image at ( %d, %d, %d).\n".as_ptr(),
            (*data).nxstart,
            (*data).nystart,
            (*data).nzstart,
        );
        libc::printf(
            c"Read length   =  ( %d, %d, %d).\n".as_ptr(),
            (*data).mx,
            (*data).my,
            (*data).mz,
        );
        let mut xscale = 1.0_f32;
        let mut yscale = 1.0_f32;
        let mut zscale = 1.0_f32;
        if (*data).xlen != 0. {
            xscale = (*data).xlen / (*data).mx as f32;
        }
        if (*data).ylen != 0. {
            yscale = (*data).ylen / (*data).my as f32;
        }
        if (*data).zlen != 0. {
            zscale = (*data).zlen / (*data).mz as f32;
        }
        libc::printf(
            c"Scale         =  ( %g x %g x %g ) Angstrom.\n".as_ptr(),
            xscale as f64,
            yscale as f64,
            zscale as f64,
        );
        libc::printf(
            c"Cell Rotation =  ( %g, %g, %g).\n".as_ptr(),
            (*data).alpha as f64,
            (*data).beta as f64,
            (*data).gamma as f64,
        );
        libc::printf(c"Columns are   = axis %d\n".as_ptr(), (*data).mapc);
        libc::printf(c"Rows are      = axis %d\n".as_ptr(), (*data).mapr);
        libc::printf(c"Sections are  = axis %d\n".as_ptr(), (*data).maps);
        if (*data).ispg != 0 {
            libc::printf(c"ispg =\t\t%d\n".as_ptr(), (*data).ispg);
        }
        if (*data).next != 0 {
            libc::printf(c"extra header = \t%d\n".as_ptr(), (*data).next);
        }
        if (*data).idtype != 0 {
            libc::printf(c"idtype =\t%d\n".as_ptr(), (*data).idtype as i32);
            libc::printf(c"nd1 =\t\t%g\n".as_ptr(), (*data).nd1 as f64);
            libc::printf(c"nd2 =\t\t%g\n".as_ptr(), (*data).nd2 as f64);
            libc::printf(c"vd1 =\t\t%g\n".as_ptr(), (*data).vd1 as f64 / 100.);
            libc::printf(c"vd2 =\t\t%g\n".as_ptr(), (*data).vd2 as f64 / 100.);
        }
        libc::printf(
            c"angles = ( %g, %g, %g, %g, %g, %g)\n".as_ptr(),
            (*data).tiltangles[0] as f64,
            (*data).tiltangles[1] as f64,
            (*data).tiltangles[2] as f64,
            (*data).tiltangles[3] as f64,
            (*data).tiltangles[4] as f64,
            (*data).tiltangles[5] as f64,
        );
        libc::printf(
            c"orgin  = ( %g, %g, %g)\n".as_ptr(),
            (*data).xorg as f64,
            (*data).yorg as f64,
            (*data).zorg as f64,
        );
        libc::printf(c"creator id = %d\n".as_ptr(), (*data).creatid as i32);
        if (*data).nlabl > MRC_NLABELS as i32 {
            libc::printf(c"There are to many labels.\n\n".as_ptr());
            return 0;
        }
        libc::printf(c"Thare are %d labels.\n\n".as_ptr(), (*data).nlabl);
        for index in 0..(*data).nlabl {
            libc::printf(
                c"%s\n".as_ptr(),
                (*data).labels[index as usize].as_ptr().cast::<i8>(),
            );
        }
        0
    }
}
