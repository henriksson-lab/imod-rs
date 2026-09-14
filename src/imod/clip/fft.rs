//! Translation of `IMOD/clip/fft.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, IP_DEFAULT};
use crate::imod::clip::file_io::{
    clip_write_slice, grap_volume_free, grap_volume_read, grap_volume_write, set_input_options,
    set_output_options,
};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libcfshr::islice::{Islice, Istack, slice_create};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MrcHeader, mrc_coord_cp,
    mrc_head_label, mrc_head_label_cp, mrc_head_new, mrc_head_write,
};
use crate::imod::libiimod::mrcslice::{
    slice_box_in, slice_complex_float, slice_float, slice_read_subm, slice_wrap_fft_lines,
};
use std::io::Write as _;

/// C++ `clip_fft` (`fft.cpp:21`).
pub unsafe fn clip_fft(
    input: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    unsafe {
        let complex = input.mode == MRC_MODE_COMPLEX_FLOAT || input.mode == MRC_MODE_COMPLEX_SHORT;
        if complex {
            if options.ix != IP_DEFAULT
                || options.iy != IP_DEFAULT
                || options.cx != IP_DEFAULT as f32
                || options.cy != IP_DEFAULT as f32
            {
                crate::imod::clip::clip::show_warning(
                    "clip inverse fft - input sizes or centers are ignored",
                );
            }
            options.ix = IP_DEFAULT;
            options.iy = IP_DEFAULT;
            options.cx = IP_DEFAULT as f32;
            options.cy = IP_DEFAULT as f32;
            if options.mode == IP_DEFAULT {
                options.mode = MRC_MODE_FLOAT;
            }
        } else {
            if options.ox != IP_DEFAULT
                || options.oy != IP_DEFAULT
                || options.oz != IP_DEFAULT
                || options.mode != IP_DEFAULT
            {
                crate::imod::clip::clip::show_warning(
                    "clip forward fft - output sizes or mode are ignored",
                );
            }
            options.ox = IP_DEFAULT;
            options.oy = IP_DEFAULT;
            options.oz = IP_DEFAULT;
            options.mode = MRC_MODE_COMPLEX_FLOAT;
            options.ocanresize = 0;
        }
        if options.dim == 3 {
            return clip_3dfft(input, output, options);
        }
        if complex && options.ox == IP_DEFAULT {
            options.ox = 2 * (input.nx - 1);
        }
        set_input_options(options, input);
        if !complex
            && (clip_nicesize(options.ix) == 0
                || clip_nicesize(options.iy) == 0
                || options.ix % 2 != 0)
        {
            if crate::imod::libfft::using_fftw() != 0 {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "ERROR: clip - fft input size in X (%d) must be even.\n",
                        &[CArg::Int((options.ix) as i64)],
                    )
                    .as_bytes(),
                );
            } else {
                let _ = ImodFile::Stdout.write_all(c_format("ERROR: clip - fft input size (%d, %d) is odd and/or has factors greater than 19.\n", &[CArg::Int((options.ix) as i64), CArg::Int((options.iy) as i64)]).as_bytes());
            }
            return -1;
        }
        if !complex {
            options.ox = options.ix / 2 + 1;
            options.oy = options.iy;
            options.oz = options.nofsecs;
            if options.add2file != 0
                && (output.mode != options.mode
                    || output.nx != options.ox
                    || output.ny != options.oy)
            {
                crate::imod::clip::clip::show_error(
                    "clip forward 2d fft - cannot append to output file of different size or mode",
                );
                return -1;
            }
        }
        let mut z = set_output_options(options, output);
        if z < 0 {
            return z;
        }
        mrc_head_label_cp(&*input, &mut *output);
        mrc_head_label(&mut *output, b"clip: 2d fft");
        crate::imod::clip::clip::show_status("Doing fast fourier transform...\n");
        for k in 0..options.nofsecs {
            let slice = slice_read_subm(
                input,
                options.secs[(k) as usize],
                b'z',
                options.ix,
                options.iy,
                options.cx as i32,
                options.cy as i32,
            );
            let Some(mut slice) = slice else {
                crate::imod::clip::clip::show_error("fft: Error reading slice.");
                return -1;
            };
            slice_fft(slice.as_mut());
            if clip_write_slice(slice.as_mut(), output, options, k, &mut z, 1) != 0 {
                return -1;
            }
        }
        mrc_coord_cp(&mut *output, &*input);
        if input.nx == input.mx && input.ny == input.my && input.nz == input.mz {
            output.mx = output.nx;
        }
        let mut fp = output.fp.clone().unwrap();
        mrc_head_write(&mut fp, output)
    }
}
/// C++ `slice_fft` (`fft.cpp:111`).  The numerical backend is supplied by
/// IMOD's translated `cfft` layer; this preserves its input/output layout.
pub unsafe fn slice_fft(slice: &mut Islice) -> i32 {
    unsafe {
        let nx2 = slice.xsize + 2;
        let ncs = slice.xsize as usize * core::mem::size_of::<f32>();
        let idir = if slice.mode == MRC_MODE_COMPLEX_FLOAT || slice.mode == MRC_MODE_COMPLEX_SHORT {
            -1
        } else {
            1
        };
        if idir == 1 {
            slice_float(slice);
            let buffer_len = (nx2 * slice.ysize) as usize;
            let mut buffer = Vec::new();
            if buffer.try_reserve_exact(buffer_len).is_err() {
                return -1;
            }
            buffer.resize(buffer_len * core::mem::size_of::<f32>(), 0);
            for row in 0..slice.ysize {
                core::ptr::copy_nonoverlapping(
                    slice.data.as_ptr().add((row * slice.xsize) as usize * ncs),
                    buffer
                        .as_mut_ptr()
                        .add((row * nx2) as usize * core::mem::size_of::<f32>()),
                    ncs,
                );
            }
            mrc_to_dfft(buffer.as_mut_ptr().cast(), slice.xsize, slice.ysize, 0);
            slice.data = buffer;
            slice.xsize = slice.xsize / 2 + 1;
            slice.mode = MRC_MODE_COMPLEX_FLOAT;
            slice.csize = 2;
            slice.dsize = core::mem::size_of::<f32>() as i32;
            slice_wrap_fft_lines(slice, 0);
        } else {
            slice_complex_float(slice);
            slice_wrap_fft_lines(slice, 1);
            let new_xsize = 2 * (slice.xsize - 1);
            mrc_to_dfft(slice.data.as_mut_ptr().cast(), new_xsize, slice.ysize, 1);
            for row in 0..slice.ysize {
                for column in 0..new_xsize {
                    *slice
                        .data
                        .as_mut_ptr()
                        .cast::<f32>()
                        .add((column + row * new_xsize) as usize) = *slice
                        .data
                        .as_ptr()
                        .cast::<f32>()
                        .add((column + row * (new_xsize + 2)) as usize);
                }
            }
            slice.xsize = new_xsize;
            slice.mode = MRC_MODE_FLOAT;
            slice.csize = 1;
        }
        0
    }
}
/// C++ `clip_3dfft` (`fft.cpp:191`).
pub unsafe fn clip_3dfft(
    input: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    unsafe {
        if input.mode != MRC_MODE_COMPLEX_FLOAT {
            if options.ix == IP_DEFAULT {
                options.ix = input.nx;
            }
            if options.iy == IP_DEFAULT {
                options.iy = input.ny;
            }
            if options.iz == IP_DEFAULT {
                options.iz = input.nz;
            }
            if clip_nicesize(options.ix) == 0
                || clip_nicesize(options.iy) == 0
                || clip_nicesize(options.iz) == 0
                || options.ix % 2 != 0
            {
                let _ = ImodFile::Stdout.write_all(c_format("ERROR: clip - fft input size %dx%dx%d is odd and/or has factors greater than 19.\n", &[CArg::Int((options.ix) as i64), CArg::Int((options.iy) as i64), CArg::Int((options.iz) as i64)]).as_bytes());
                return -1;
            }
        }
        let Some(mut volume) = grap_volume_read(input, options) else {
            return -1;
        };
        if input.mode != MRC_MODE_COMPLEX_FLOAT {
            for slice in &mut volume.slices {
                if input.mode == MRC_MODE_COMPLEX_SHORT {
                    slice_complex_float(slice.as_mut());
                } else {
                    slice_float(slice.as_mut());
                }
            }
        }
        crate::imod::clip::clip::show_status("Doing 3d fast fourier transform in core...\n");
        clip_fftvol(&mut volume);
        let Some(first) = volume.slices.first() else {
            return -1;
        };
        mrc_head_new(
            &mut *output,
            first.xsize,
            first.ysize,
            volume.slices.len() as i32,
            first.mode,
        );
        mrc_head_label_cp(&*input, &mut *output);
        mrc_head_label(
            &mut *output,
            if input.mode != MRC_MODE_COMPLEX_FLOAT {
                b"Clip: Forward 3D FFT"
            } else {
                b"Clip: Inverse 3D FFT"
            },
        );
        if grap_volume_write(&mut volume, output, options) != 0 {
            return -1;
        }
        mrc_coord_cp(&mut *output, &*input);
        if input.nx == input.mx && input.ny == input.my && input.nz == input.mz {
            output.mx = output.nx;
        }
        let mut fp = output.fp.clone().unwrap();
        if mrc_head_write(&mut fp, output) != 0 {
            return -1;
        }
        0
    }
}
/// C++ `clip_fftvol` (`fft.cpp:258`).
pub unsafe fn clip_fftvol(volume: &mut Istack) -> i32 {
    unsafe {
        // `fft.cpp:258` guards on `!v`; a reference is never null.
        if volume.slices.is_empty() {
            return -1;
        }
        let first = volume.slices.first().unwrap();
        if first.mode == MRC_MODE_COMPLEX_FLOAT {
            clip_wrapvol(volume, 1);
            clip_fftvol3(volume, -2);
            for slice in &mut volume.slices {
                mrc_to_dfft(
                    slice.data.as_mut_ptr().cast(),
                    (slice.xsize - 1) * 2,
                    slice.ysize,
                    -1,
                );
                slice.mode = MRC_MODE_FLOAT;
                slice.csize = 1;
                slice.xsize *= 2;
                slice_box_in(slice.as_mut(), 0, 0, slice.xsize - 2, slice.ysize);
            }
        } else {
            let nx2 = first.xsize + 2;
            for slice in &mut volume.slices {
                slice_float(slice.as_mut());
                let buffer_len = (nx2 * slice.ysize) as usize;
                let mut buffer = Vec::new();
                if buffer.try_reserve_exact(buffer_len).is_err() {
                    return -1;
                }
                buffer.resize(buffer_len * core::mem::size_of::<f32>(), 0);
                for row in 0..slice.ysize {
                    core::ptr::copy_nonoverlapping(
                        slice.data.as_ptr().add((row * slice.xsize * 4) as usize),
                        buffer.as_mut_ptr().add((row * nx2 * 4) as usize),
                        slice.xsize as usize * 4,
                    );
                }
                slice.data = buffer;
                slice.xsize += 2;
                mrc_to_dfft(
                    slice.data.as_mut_ptr().cast(),
                    slice.xsize - 2,
                    slice.ysize,
                    0,
                );
                slice.xsize /= 2;
                slice.mode = MRC_MODE_COMPLEX_FLOAT;
                slice.csize = 2;
            }
            clip_fftvol3(volume, -1);
            clip_wrapvol(volume, 0);
        }
        0
    }
}
/// C++ `clip_fftvol3` (`fft.cpp:309`).
pub unsafe fn clip_fftvol3(volume: &mut Istack, idir: i32) -> i32 {
    unsafe {
        // `fft.cpp:288` guards on `!v`; a reference is never null.
        if volume.slices.is_empty() {
            return -1;
        }
        let first_xsize = volume.slices[0].xsize;
        let first_ysize = volume.slices[0].ysize;
        let depth = volume.slices.len() as i32;
        let mut work = match slice_create(depth, first_xsize, MRC_MODE_COMPLEX_FLOAT) {
            Some(work) => work,
            None => return -1,
        };
        for y in 0..first_ysize {
            for (z, slice) in volume.slices.iter().enumerate() {
                let input = unsafe {
                    slice
                        .data
                        .as_ptr()
                        .cast::<f32>()
                        .add((2 * y * first_xsize) as usize)
                };
                let out = unsafe { work.data.as_mut_ptr().cast::<f32>().add(2 * z) };
                for x in 0..work.ysize {
                    *out.add((2 * x * work.xsize) as usize) = *input.add((2 * x) as usize);
                    *out.add((2 * x * work.xsize + 1) as usize) = *input.add((2 * x + 1) as usize);
                }
            }
            mrc_odfft(work.data.as_mut_ptr().cast(), work.xsize, work.ysize, idir);
            for (z, slice) in volume.slices.iter_mut().enumerate() {
                let out = slice
                    .data
                    .as_mut_ptr()
                    .cast::<f32>()
                    .add((2 * y * first_xsize) as usize);
                let input = work.data.as_ptr().cast::<f32>().add(2 * z);
                for x in 0..work.ysize {
                    *out.add((2 * x) as usize) = *input.add((2 * x * (*work).xsize) as usize);
                    *out.add((2 * x + 1) as usize) =
                        *input.add((2 * x * (*work).xsize + 1) as usize);
                }
            }
        }
        0
    }
}
/// C++ `clip_wrapvol` (`fft.cpp:350`).
pub unsafe fn clip_wrapvol(volume: &mut Istack, direction: i32) -> i32 {
    unsafe {
        // `fft.cpp:326` guards on `!v`; a reference is never null.
        let Some(first) = volume.slices.first() else {
            return -1;
        };
        let first_xsize = first.xsize;
        let first_ysize = first.ysize;
        let depth = volume.slices.len() as i32;
        let temporary_len = (2 * first_xsize) as usize;
        let mut temporary = Vec::new();
        if temporary.try_reserve_exact(temporary_len).is_err() {
            crate::imod::clip::clip::show_error(
                "fft: Memory error getting array for temporary line of data\n",
            );
            return 1;
        }
        temporary.resize(temporary_len, 0.);
        for slice in &mut volume.slices {
            crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
                core::slice::from_raw_parts_mut(
                    slice.data.as_mut_ptr().cast(),
                    (2 * first_xsize * first_ysize) as usize,
                ),
                temporary.as_mut_slice(),
                first_xsize,
                first_ysize,
                direction,
            );
        }
        let mut output = 0;
        let mut low = 0;
        let mut high = 0;
        let increment = crate::imod::libcfshr::filtxcorr::indices_for_fft_wrap(
            depth,
            direction,
            &mut output,
            &mut low,
            &mut high,
        );
        if depth % 2 != 0 {
            for _ in 0..depth / 2 {
                volume.slices.swap(output as usize, high as usize);
                volume.slices.swap(high as usize, low as usize);
                output += increment;
                high += increment;
                low += increment;
            }
        } else {
            for _ in 0..depth / 2 {
                volume.slices.swap(low as usize, high as usize);
                high += 1;
                low += 1;
            }
        }
        0
    }
}
/// C++ `mrcToDFFT` (`fft.cpp:391`).
pub unsafe fn mrc_to_dfft(buffer: *mut f32, nx: i32, ny: i32, idir: i32) {
    unsafe {
        crate::imod::libfft::todfft_c(buffer, nx, ny, idir);
    }
}
/// C++ static `mrcODFFT` (`fft.cpp:397`).
pub unsafe fn mrc_odfft(buffer: *mut f32, nx: i32, ny: i32, idir: i32) -> i32 {
    unsafe {
        crate::imod::libfft::odfft_c(buffer, nx, ny, idir);
    }
    0
}
/// C++ `clip_nicesize` (`fft.cpp:405`).
pub fn clip_nicesize(mut size: i32) -> i32 {
    if crate::imod::libfft::using_fftw() != 0 {
        return 1;
    }
    for factor in 2..20 {
        while size % factor == 0 {
            size /= factor;
        }
    }
    if size > 1 { 0 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::clip_nicesize;

    #[test]
    fn nice_sizes_match_clip_factor_rule() {
        assert_eq!(clip_nicesize(19 * 19 * 18), 1);
        assert_eq!(clip_nicesize(23), 0);
    }
}
