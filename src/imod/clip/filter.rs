//! Translation of `IMOD/clip/filter.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, IP_DEFAULT, show_error, show_status};
use crate::imod::clip::fft::{clip_nicesize, slice_fft};
use crate::imod::clip::file_io::{clip_write_slice, set_options};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MrcHeader, mrc_head_label,
};
use crate::imod::libiimod::mrcslice::{mrc_bandpass_filter, slice_complex_float, slice_read_subm};
use std::io::Write as _;

/// C++ `clip_3dfilter` (`filter.cpp:20`).
pub fn clip_3dfilter(
    input: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    // `filter.cpp:21` guards the print with `(hin) && (hout) && (opt)`; a
    // reference is never null, so the guard is always taken.
    let _ = (input, output, options);
    let _ = ImodFile::Stdout.write_all(b"clip: 3d filter... (non-functional).\n");
    0
}

/// C++ `clip_bandpass_filter` (`filter.cpp:29`).
pub unsafe fn clip_bandpass_filter(
    input: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    unsafe {
        let complex = input.mode == MRC_MODE_COMPLEX_FLOAT || input.mode == MRC_MODE_COMPLEX_SHORT;
        if !complex {
            if options.mode == IP_DEFAULT {
                options.mode = MRC_MODE_FLOAT;
            }
        } else {
            options.ocanchmode = 0;
            options.mode = MRC_MODE_COMPLEX_FLOAT;
            options.ix = IP_DEFAULT;
            options.iy = IP_DEFAULT;
            options.cx = IP_DEFAULT as f32;
            options.cy = IP_DEFAULT as f32;
            options.ocanresize = 0;
            if options.add2file != 0
                && (output.mode != options.mode || output.nx != input.nx || output.ny != input.ny)
            {
                show_error("clip filter - cannot append to output file of different size or mode");
                return -1;
            }
        }
        let mut z = set_options(options, input, output);
        if z < 0 {
            return z;
        }
        if !complex
            && (clip_nicesize(options.ix) == 0
                || clip_nicesize(options.iy) == 0
                || options.ix % 2 != 0)
        {
            if crate::imod::libfft::using_fftw() != 0 {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "ERROR: clip - filter input size in X (%d) must be even: use Mtffilter.\n",
                        &[CArg::Int((options.ix) as i64)],
                    )
                    .as_bytes(),
                );
            } else {
                let _ = ImodFile::Stdout.write_all(c_format("ERROR: clip - filter input size (%d, %d) is odd and/or has factors greater than 19: use Mtffilter.\n", &[CArg::Int((options.ix) as i64), CArg::Int((options.iy) as i64)]).as_bytes());
            }
            return -1;
        }
        if options.low == IP_DEFAULT as f32 {
            options.low = 1.;
        }
        if options.high == IP_DEFAULT as f32 {
            options.high = 0.;
        }
        mrc_head_label(&mut *output, b"clip: fourier filter");
        show_status("Doing bandpass filter...\n");
        for k in 0..options.nofsecs {
            let Some(mut slice) = slice_read_subm(
                input,
                options.secs[(k) as usize],
                b'z',
                options.ix,
                options.iy,
                options.cx as i32,
                options.cy as i32,
            ) else {
                return -1;
            };
            if !complex {
                slice_fft(slice.as_mut());
                mrc_bandpass_filter(slice.as_mut(), options.high as f64, options.low as f64);
                slice_fft(slice.as_mut());
            } else {
                slice_complex_float(slice.as_mut());
                mrc_bandpass_filter(slice.as_mut(), options.high as f64, options.low as f64);
            }
            if clip_write_slice(slice.as_mut(), output, options, k, &mut z, 1) != 0 {
                return -1;
            }
        }
        crate::imod::clip::file_io::set_mrc_coords(input, output, options)
    }
}
