//! Translation of `IMOD/clip/fft.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, IP_DEFAULT};
use crate::imod::clip::file_io::{
    clip_write_slice, grap_volume_free, grap_volume_read, grap_volume_write, set_input_options,
    set_output_options,
};
use crate::imod::libcfshr::islice::{Istack, slice_create, slice_free, slice_init};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MrcHeader, mrc_coord_cp,
    mrc_head_label, mrc_head_label_cp, mrc_head_new, mrc_head_write,
};
use crate::imod::libiimod::mrcslice::{
    slice_box_in, slice_complex_float, slice_float, slice_read_subm, slice_wrap_fft_lines,
};

/// C++ `clip_fft` (`fft.cpp:21`).
pub unsafe fn clip_fft(
    input: *mut MrcHeader,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
) -> i32 {
    unsafe {
        let complex =
            (*input).mode == MRC_MODE_COMPLEX_FLOAT || (*input).mode == MRC_MODE_COMPLEX_SHORT;
        if complex {
            if (*options).ix != IP_DEFAULT
                || (*options).iy != IP_DEFAULT
                || (*options).cx != IP_DEFAULT as f32
                || (*options).cy != IP_DEFAULT as f32
            {
                crate::imod::clip::clip::show_warning(
                    "clip inverse fft - input sizes or centers are ignored",
                );
            }
            (*options).ix = IP_DEFAULT;
            (*options).iy = IP_DEFAULT;
            (*options).cx = IP_DEFAULT as f32;
            (*options).cy = IP_DEFAULT as f32;
            if (*options).mode == IP_DEFAULT {
                (*options).mode = MRC_MODE_FLOAT;
            }
        } else {
            if (*options).ox != IP_DEFAULT
                || (*options).oy != IP_DEFAULT
                || (*options).oz != IP_DEFAULT
                || (*options).mode != IP_DEFAULT
            {
                crate::imod::clip::clip::show_warning(
                    "clip forward fft - output sizes or mode are ignored",
                );
            }
            (*options).ox = IP_DEFAULT;
            (*options).oy = IP_DEFAULT;
            (*options).oz = IP_DEFAULT;
            (*options).mode = MRC_MODE_COMPLEX_FLOAT;
            (*options).ocanresize = 0;
        }
        if (*options).dim == 3 {
            return clip_3dfft(input, output, options);
        }
        if complex && (*options).ox == IP_DEFAULT {
            (*options).ox = 2 * ((*input).nx - 1);
        }
        set_input_options(options, input);
        if !complex
            && (clip_nicesize((*options).ix) == 0
                || clip_nicesize((*options).iy) == 0
                || (*options).ix % 2 != 0)
        {
            if crate::imod::libfft::using_fftw() != 0 {
                libc::printf(
                    c"ERROR: clip - fft input size in X (%d) must be even.\n".as_ptr(),
                    (*options).ix,
                );
            } else {
                libc::printf(
                    c"ERROR: clip - fft input size (%d, %d) is odd and/or has factors greater than 19.\n"
                        .as_ptr(),
                    (*options).ix,
                    (*options).iy,
                );
            }
            return -1;
        }
        if !complex {
            (*options).ox = (*options).ix / 2 + 1;
            (*options).oy = (*options).iy;
            (*options).oz = (*options).nofsecs;
            if (*options).add2file != 0
                && ((*output).mode != (*options).mode
                    || (*output).nx != (*options).ox
                    || (*output).ny != (*options).oy)
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
        for k in 0..(*options).nofsecs {
            let slice = slice_read_subm(
                input,
                *(*options).secs.add(k as usize),
                b'z' as i8,
                (*options).ix,
                (*options).iy,
                (*options).cx as i32,
                (*options).cy as i32,
            );
            if slice.is_null() {
                crate::imod::clip::clip::show_error("fft: Error reading slice.");
                return -1;
            }
            slice_fft(slice);
            if clip_write_slice(slice, output, options, k, &mut z, 1) != 0 {
                return -1;
            }
        }
        mrc_coord_cp(&mut *output, &*input);
        if (*input).nx == (*input).mx && (*input).ny == (*input).my && (*input).nz == (*input).mz {
            (*output).mx = (*output).nx;
        }
        mrc_head_write((*output).fp.cast(), output)
    }
}
/// C++ `slice_fft` (`fft.cpp:111`).  The numerical backend is supplied by
/// IMOD's translated `cfft` layer; this preserves its input/output layout.
pub unsafe fn slice_fft(slice: *mut crate::imod::libcfshr::islice::Islice) -> i32 {
    unsafe {
        if slice.is_null() {
            return -1;
        }
        let nx2 = (*slice).xsize + 2;
        let ncs = (*slice).xsize as usize * core::mem::size_of::<f32>();
        let idir =
            if (*slice).mode == MRC_MODE_COMPLEX_FLOAT || (*slice).mode == MRC_MODE_COMPLEX_SHORT {
                -1
            } else {
                1
            };
        if idir == 1 {
            slice_float(slice);
            let buffer =
                libc::malloc((nx2 * (*slice).ysize) as usize * core::mem::size_of::<f32>())
                    .cast::<f32>();
            if buffer.is_null() {
                return -1;
            }
            for row in 0..(*slice).ysize {
                core::ptr::copy_nonoverlapping(
                    (*slice).data.f.add((row * (*slice).xsize) as usize),
                    buffer.add((row * nx2) as usize),
                    ncs / core::mem::size_of::<f32>(),
                );
            }
            libc::free((*slice).data.f.cast());
            mrc_to_dfft(buffer, (*slice).xsize, (*slice).ysize, 0);
            slice_init(
                slice,
                (*slice).xsize / 2 + 1,
                (*slice).ysize,
                MRC_MODE_COMPLEX_FLOAT,
                buffer.cast(),
            );
            slice_wrap_fft_lines(slice, 0);
        } else {
            slice_complex_float(slice);
            slice_wrap_fft_lines(slice, 1);
            let new_xsize = 2 * ((*slice).xsize - 1);
            mrc_to_dfft((*slice).data.f, new_xsize, (*slice).ysize, 1);
            for row in 0..(*slice).ysize {
                for column in 0..new_xsize {
                    *(*slice).data.f.add((column + row * new_xsize) as usize) = *(*slice)
                        .data
                        .f
                        .add((column + row * (new_xsize + 2)) as usize);
                }
            }
            (*slice).xsize = new_xsize;
            (*slice).mode = MRC_MODE_FLOAT;
        }
        0
    }
}
/// C++ `clip_3dfft` (`fft.cpp:191`).
pub unsafe fn clip_3dfft(
    input: *mut MrcHeader,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
) -> i32 {
    unsafe {
        if (*input).mode != MRC_MODE_COMPLEX_FLOAT {
            if (*options).ix == IP_DEFAULT {
                (*options).ix = (*input).nx;
            }
            if (*options).iy == IP_DEFAULT {
                (*options).iy = (*input).ny;
            }
            if (*options).iz == IP_DEFAULT {
                (*options).iz = (*input).nz;
            }
            if clip_nicesize((*options).ix) == 0
                || clip_nicesize((*options).iy) == 0
                || clip_nicesize((*options).iz) == 0
                || (*options).ix % 2 != 0
            {
                libc::printf(
                    c"ERROR: clip - fft input size %dx%dx%d is odd and/or has factors greater than 19.\n"
                        .as_ptr(),
                    (*options).ix,
                    (*options).iy,
                    (*options).iz,
                );
                return -1;
            }
        }
        let volume = grap_volume_read(input, options);
        if volume.is_null() {
            return -1;
        }
        if (*input).mode != MRC_MODE_COMPLEX_FLOAT {
            for z in 0..(*volume).zsize {
                let slice = *(*volume).vol.add(z as usize);
                if (*input).mode == MRC_MODE_COMPLEX_SHORT {
                    slice_complex_float(slice);
                } else {
                    slice_float(slice);
                }
            }
        }
        crate::imod::clip::clip::show_status("Doing 3d fast fourier transform in core...\n");
        clip_fftvol(volume);
        mrc_head_new(
            &mut *output,
            (*(*(*volume).vol)).xsize,
            (*(*(*volume).vol)).ysize,
            (*volume).zsize,
            (*(*(*volume).vol)).mode,
        );
        mrc_head_label_cp(&*input, &mut *output);
        mrc_head_label(
            &mut *output,
            if (*input).mode != MRC_MODE_COMPLEX_FLOAT {
                b"Clip: Forward 3D FFT"
            } else {
                b"Clip: Inverse 3D FFT"
            },
        );
        if grap_volume_write(volume, output, options) != 0 {
            return -1;
        }
        mrc_coord_cp(&mut *output, &*input);
        if (*input).nx == (*input).mx && (*input).ny == (*input).my && (*input).nz == (*input).mz {
            (*output).mx = (*output).nx;
        }
        if mrc_head_write((*output).fp.cast(), output) != 0 {
            return -1;
        }
        grap_volume_free(volume);
        0
    }
}
/// C++ `clip_fftvol` (`fft.cpp:258`).
pub unsafe fn clip_fftvol(volume: *mut Istack) -> i32 {
    unsafe {
        if volume.is_null() || (*volume).zsize == 0 {
            return -1;
        }
        let first = *(*volume).vol;
        if (*first).mode == MRC_MODE_COMPLEX_FLOAT {
            clip_wrapvol(volume, 1);
            clip_fftvol3(volume, -2);
            for z in 0..(*volume).zsize {
                let slice = *(*volume).vol.add(z as usize);
                mrc_to_dfft(
                    (*slice).data.f,
                    ((*slice).xsize - 1) * 2,
                    (*slice).ysize,
                    -1,
                );
                (*slice).mode = MRC_MODE_FLOAT;
                (*slice).xsize *= 2;
                slice_box_in(slice, 0, 0, (*slice).xsize - 2, (*slice).ysize);
            }
        } else {
            let nx2 = (*first).xsize + 2;
            for z in 0..(*volume).zsize {
                let slice = *(*volume).vol.add(z as usize);
                slice_float(slice);
                let buffer =
                    libc::malloc((nx2 * (*slice).ysize) as usize * core::mem::size_of::<f32>())
                        .cast::<f32>();
                if buffer.is_null() {
                    return -1;
                }
                for row in 0..(*slice).ysize {
                    core::ptr::copy_nonoverlapping(
                        (*slice).data.f.add((row * (*slice).xsize) as usize),
                        buffer.add((row * nx2) as usize),
                        (*slice).xsize as usize,
                    );
                }
                libc::free((*slice).data.f.cast());
                (*slice).data.f = buffer;
                (*slice).xsize += 2;
                mrc_to_dfft((*slice).data.f, (*slice).xsize - 2, (*slice).ysize, 0);
                (*slice).xsize /= 2;
                (*slice).mode = MRC_MODE_COMPLEX_FLOAT;
            }
            clip_fftvol3(volume, -1);
            clip_wrapvol(volume, 0);
        }
        0
    }
}
/// C++ `clip_fftvol3` (`fft.cpp:309`).
pub unsafe fn clip_fftvol3(volume: *mut Istack, idir: i32) -> i32 {
    unsafe {
        if volume.is_null() || (*volume).zsize < 1 {
            return -1;
        }
        let first = *(*volume).vol;
        let work = slice_create((*volume).zsize, (*first).xsize, MRC_MODE_COMPLEX_FLOAT);
        if work.is_null() {
            return -1;
        }
        for y in 0..(*first).ysize {
            for z in 0..(*volume).zsize {
                let slice = *(*volume).vol.add(z as usize);
                let input = (*slice).data.f.add((2 * y * (*first).xsize) as usize);
                let out = (*work).data.f.add((2 * z) as usize);
                for x in 0..(*work).ysize {
                    *out.add((2 * x * (*work).xsize) as usize) = *input.add((2 * x) as usize);
                    *out.add((2 * x * (*work).xsize + 1) as usize) =
                        *input.add((2 * x + 1) as usize);
                }
            }
            mrc_odfft((*work).data.f, (*work).xsize, (*work).ysize, idir);
            for z in 0..(*volume).zsize {
                let slice = *(*volume).vol.add(z as usize);
                let out = (*slice).data.f.add((2 * y * (*first).xsize) as usize);
                let input = (*work).data.f.add((2 * z) as usize);
                for x in 0..(*work).ysize {
                    *out.add((2 * x) as usize) = *input.add((2 * x * (*work).xsize) as usize);
                    *out.add((2 * x + 1) as usize) =
                        *input.add((2 * x * (*work).xsize + 1) as usize);
                }
            }
        }
        slice_free(work);
        0
    }
}
/// C++ `clip_wrapvol` (`fft.cpp:350`).
pub unsafe fn clip_wrapvol(volume: *mut Istack, direction: i32) -> i32 {
    unsafe {
        if volume.is_null() {
            return -1;
        }
        let first = *(*volume).vol;
        let temporary =
            libc::malloc((2 * (*first).xsize) as usize * core::mem::size_of::<f32>()).cast::<f32>();
        if temporary.is_null() {
            crate::imod::clip::clip::show_error(
                "fft: Memory error getting array for temporary line of data\n",
            );
            return 1;
        }
        for z in 0..(*volume).zsize {
            let slice = *(*volume).vol.add(z as usize);
            crate::imod::libcfshr::filtxcorr::wrap_fft_slice(
                (*slice).data.f,
                temporary,
                (*first).xsize,
                (*first).ysize,
                direction,
            );
        }
        let mut output = 0;
        let mut low = 0;
        let mut high = 0;
        let increment = crate::imod::libcfshr::filtxcorr::indices_for_fft_wrap(
            (*volume).zsize,
            direction,
            &mut output,
            &mut low,
            &mut high,
        );
        if (*volume).zsize % 2 != 0 {
            let temporary_slice = *(*volume).vol.add(output as usize);
            for _ in 0..(*volume).zsize / 2 {
                *(*volume).vol.add(output as usize) = *(*volume).vol.add(high as usize);
                *(*volume).vol.add(high as usize) = *(*volume).vol.add(low as usize);
                output += increment;
                high += increment;
                low += increment;
            }
            *(*volume).vol.add(((*volume).zsize / 2) as usize) = temporary_slice;
        } else {
            for _ in 0..(*volume).zsize / 2 {
                let temporary_slice = *(*volume).vol.add(low as usize);
                *(*volume).vol.add(low as usize) = *(*volume).vol.add(high as usize);
                *(*volume).vol.add(high as usize) = temporary_slice;
                high += 1;
                low += 1;
            }
        }
        libc::free(temporary.cast());
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
