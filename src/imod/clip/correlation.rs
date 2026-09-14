//! Translation of `IMOD/clip/correlation.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, show_error, show_status};
use crate::imod::clip::fft::mrc_to_dfft;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, data_size_for_mode};
use crate::imod::libcfshr::islice::{
    Islice, Istack, slice_create, slice_get_pixel_magnitude, slice_get_val, slice_put_val,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MrcHeader, mrc_head_label, mrc_head_new,
    mrc_head_write, mrc_read_slice, mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{
    corr_conj, slice_add_const, slice_box, slice_box_in, slice_float, slice_mmm,
    slice_reduce_mirrored_fft, slice_resize_in,
};
use std::io::Write as _;

/// Matches C++ `corr_getmax`.
pub unsafe fn corr_getmax(
    islice: &mut Islice,
    sa: i32,
    xm: i32,
    ym: i32,
    x: &mut f32,
    y: &mut f32,
) {
    unsafe {
        let Some(mut slice) = slice_box(islice, xm - sa, ym - sa, xm + sa + 1, ym + sa + 1) else {
            return;
        };
        slice_mmm(slice.as_mut());
        slice_add_const(slice.as_mut(), [-slice.min, 0., 0., 0.]);
        slice_mmm(slice.as_mut());
        slice_add_const(slice.as_mut(), [-slice.max + slice.max * 0.025, 0., 0., 0.]);
        *x = xm as f32;
        *y = ym as f32;
        let mut row = 0.;
        let mut weight = 0.;
        for ix in 0..slice.xsize {
            for iy in 0..slice.ysize {
                let value = slice_get_pixel_magnitude(slice.as_mut(), ix, iy);
                if value > 0. {
                    row += (ix + 1) as f32 * value;
                    weight += value;
                }
            }
        }
        if weight > 0. {
            *x = row / weight + (xm - sa - 1) as f32;
        }
        row = 0.;
        weight = 0.;
        for ix in 0..slice.xsize {
            for iy in 0..slice.ysize {
                let value = slice_get_pixel_magnitude(slice.as_mut(), ix, iy);
                if value > 0. {
                    row += (iy + 1) as f32 * value;
                    weight += value;
                }
            }
        }
        if weight > 0. {
            *y = row / weight + (ym - sa - 1) as f32;
        }
    }
}
/// Matches C++ `clip_padcorr`.
pub unsafe fn clip_padcorr(slice: &mut Islice, pad: i32) {
    unsafe {
        if pad == 0 || slice.mode == MRC_MODE_COMPLEX_FLOAT {
            if pad == 0 {
                let _ = ImodFile::Stdout.write_all(b"no padding\n");
            }
            return;
        }
        slice_box_in(
            slice,
            -slice.xsize / 2,
            -slice.ysize / 2,
            slice.xsize + slice.xsize / 2 + 2,
            slice.ysize + slice.ysize / 2,
        );
    }
}
/// Matches C++ `clip_slice_corr`.
pub unsafe fn clip_slice_corr(
    slice1: &mut Islice,
    slice2: Option<&mut Islice>,
) -> Option<Box<Islice>> {
    unsafe {
        if slice1.mode != MRC_MODE_COMPLEX_FLOAT {
            if slice_float(slice1) < 0 {
                return None;
            }
            slice_mmm(slice1);
            slice_add_const(slice1, [-slice1.mean, 0., 0., 0.]);
            mrc_to_dfft(
                slice1.data.as_mut_ptr().cast(),
                slice1.xsize - 2,
                slice1.ysize,
                0,
            );
            slice1.xsize /= 2;
        } else if slice1.xsize % 2 == 0 {
            slice_reduce_mirrored_fft(slice1);
        }
        if let Some(slice2) = slice2 {
            if slice2.mode != MRC_MODE_COMPLEX_FLOAT {
                if slice_float(slice2) < 0 {
                    return None;
                }
                mrc_to_dfft(
                    slice2.data.as_mut_ptr().cast(),
                    slice2.xsize - 2,
                    slice2.ysize,
                    0,
                );
                slice2.xsize /= 2;
            } else if slice2.xsize % 2 == 0 {
                slice_reduce_mirrored_fft(slice2);
            }
            if slice1.xsize != slice2.xsize || slice1.ysize != slice2.ysize {
                show_error("corr: slices must be same size.\n");
                return None;
            }
            corr_conj(
                slice1.data.as_mut_ptr().cast(),
                slice2.data.as_mut_ptr().cast(),
                slice1.xsize * slice1.ysize,
            );
        } else {
            corr_conj(
                slice1.data.as_mut_ptr().cast(),
                slice1.data.as_mut_ptr().cast(),
                slice1.xsize * slice1.ysize,
            );
        }
        let Some(mut output) = slice_create(2 * slice1.xsize, slice1.ysize, MRC_MODE_FLOAT) else {
            return None;
        };
        mrc_to_dfft(
            slice1.data.as_mut_ptr().cast(),
            2 * slice1.xsize - 2,
            slice1.ysize,
            1,
        );
        slice1.xsize *= 2;
        let xm = output.xsize / 2;
        let ym = output.ysize / 2;
        let mut value = [0.; 4];
        for j in 0..ym {
            for i in 0..xm {
                slice_get_val(slice1, i, j, &mut value);
                slice_put_val(output.as_mut(), xm + i, ym + j, value);
            }
            for (i, x) in ((0..xm).rev()).enumerate() {
                slice_get_val(slice1, slice1.xsize - 3 - i as i32, j, &mut value);
                slice_put_val(output.as_mut(), x as i32, ym + j, value);
            }
        }
        for (offset, j) in (ym..slice1.ysize).rev().enumerate() {
            let y = ym - 1 - offset as i32;
            for x in (0..xm).rev() {
                let i = slice1.xsize - 3 - (xm - 1 - x);
                slice_get_val(slice1, i, j, &mut value);
                slice_put_val(output.as_mut(), x, y, value);
            }
            for x in xm..output.xsize {
                slice_get_val(slice1, x - xm, j, &mut value);
                slice_put_val(output.as_mut(), x, y, value);
            }
        }
        Some(output)
    }
}
/// Matches C++ `clip_corr3d`.
pub unsafe fn clip_corr3d(
    input1: &mut MrcHeader,
    input2: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    unsafe {
        if input1.mode == MRC_MODE_COMPLEX_FLOAT {
            return grap_3dcorr(input1, input2, output, options);
        }
        // `correlation.cpp:229` is `memcpy(&opt2, opt, sizeof(ClipOptions))`,
        // a struct copy that shares `secs` and the name pointers with the
        // original; nothing below writes through either copy's vectors, so a
        // clone is the same thing with Rust's ownership.
        let mut second_options = options.clone();
        let Some(mut first) = crate::imod::clip::file_io::grap_volume_read(input1, options) else {
            return -1;
        };
        let (mut min, mut max, mut mean) = (0_f32, 0_f32, 0_f32);
        let (mut xmax, mut ymax, mut zmax) = (0_i32, 0_i32, 0_i32);
        crate::imod::clip::processing::clip_get_stat3d(
            &mut first, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
        );
        let _ = ImodFile::Stdout.write_all(b"stats on vol 1:\n");
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "max = %g, min = %g, mean = %g\n",
                &[
                    CArg::Dbl((max as core::ffi::c_double) as f64),
                    CArg::Dbl((min as core::ffi::c_double) as f64),
                    CArg::Dbl((mean as core::ffi::c_double) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "location of max pixel = (%d, %d %d)\n",
                &[
                    CArg::Int((xmax) as i64),
                    CArg::Int((ymax) as i64),
                    CArg::Int((zmax) as i64),
                ],
            )
            .as_bytes(),
        );
        if options.val as i32 == crate::imod::clip::clip::IP_DEFAULT {
            options.val = 1.;
        }
        if options.val == 1. && padfloat_volume(&mut first, mean) != 0 {
            return -1;
        }
        let _ = ImodFile::Stdout.write_all(b"\n");
        let autocorrelation = options.infiles != 2;
        let mut second = if autocorrelation {
            let _ = ImodFile::Stdout.write_all(b"Auto-Correlation\n");
            None
        } else {
            let _ = ImodFile::Stdout.write_all(b"Cross-Correlation\n");
            let Some(mut v) =
                crate::imod::clip::file_io::grap_volume_read(input2, &mut second_options)
            else {
                return -1;
            };
            crate::imod::clip::processing::clip_get_stat3d(
                &mut v, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
            );
            let _ = ImodFile::Stdout.write_all(b"stats on vol 2:\n");
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "max = %g, min = %g, mean = %g\n",
                    &[
                        CArg::Dbl((max as core::ffi::c_double) as f64),
                        CArg::Dbl((min as core::ffi::c_double) as f64),
                        CArg::Dbl((mean as core::ffi::c_double) as f64),
                    ],
                )
                .as_bytes(),
            );
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "location of max pixel = (%d, %d %d)\n",
                    &[
                        CArg::Int((xmax) as i64),
                        CArg::Int((ymax) as i64),
                        CArg::Int((zmax) as i64),
                    ],
                )
                .as_bytes(),
            );
            if options.val == 1. && padfloat_volume(&mut v, mean) != 0 {
                return -1;
            }
            let _ = ImodFile::Stdout.write_all(b"\n");
            Some(v)
        };
        let _ = ImodFile::Stdout.write_all(b"Calculating fft 1");
        if crate::imod::clip::fft::clip_fftvol(&mut first) != 0 {
            return -1;
        }
        let fp = output.fp.clone();
        let first_slice = first.slices.first().unwrap();
        mrc_head_new(
            &mut *output,
            first_slice.xsize,
            first_slice.ysize,
            first.slices.len() as i32,
            first_slice.mode,
        );
        output.fp = fp.clone();
        if !autocorrelation {
            let _ = ImodFile::Stdout.write_all(b"\rCalculating fft 2");
            if crate::imod::clip::fft::clip_fftvol(second.as_mut().unwrap()) != 0 {
                return -1;
            }
        }
        let size = first.slices[0].xsize * first.slices[0].ysize;
        for z in 0..first.slices.len() {
            let first_data = first.slices[z].data.as_mut_ptr().cast::<f32>();
            let second_data = if autocorrelation {
                first_data
            } else {
                second.as_mut().unwrap().slices[z]
                    .data
                    .as_mut_ptr()
                    .cast::<f32>()
            };
            corr_conj(first_data, second_data, size);
        }
        let _ = ImodFile::Stdout.write_all(b"\rCalculating inverse fft");
        if crate::imod::clip::fft::clip_fftvol(&mut first) != 0
            || clip_cor_scalevol(&mut first) != 0
        {
            return -1;
        }
        let _ = ImodFile::Stdout.write_all(b"\n");
        crate::imod::clip::processing::clip_get_stat3d(
            &mut first, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
        );
        let mut peak_x = 0.;
        let mut peak_y = 0.;
        let mut peak_z = 0.;
        crate::imod::clip::processing::clip_parxyz(
            &mut first,
            xmax,
            ymax,
            zmax,
            &mut peak_x,
            &mut peak_y,
            &mut peak_z,
        );
        let first_slice = first.slices.first().unwrap();
        if peak_x > (first_slice.xsize / 2) as f32 {
            peak_x -= first_slice.xsize as f32;
        }
        if peak_y > (first_slice.ysize / 2) as f32 {
            peak_y -= first_slice.ysize as f32;
        }
        if peak_z > (first.slices.len() as i32 / 2) as f32 {
            peak_z -= first.slices.len() as f32;
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "max = %g  min = %g  mean = %g\n",
                &[
                    CArg::Dbl((max as core::ffi::c_double) as f64),
                    CArg::Dbl((min as core::ffi::c_double) as f64),
                    CArg::Dbl((mean as core::ffi::c_double) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "location of max pixel ( %d, %d, %d) is \n",
                &[
                    CArg::Int((xmax) as i64),
                    CArg::Int((ymax) as i64),
                    CArg::Int((zmax) as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "( %.2f, %.2f, %.2f)\n",
                &[
                    CArg::Dbl((peak_x as core::ffi::c_double) as f64),
                    CArg::Dbl((peak_y as core::ffi::c_double) as f64),
                    CArg::Dbl((peak_z as core::ffi::c_double) as f64),
                ],
            )
            .as_bytes(),
        );
        mrc_head_new(
            &mut *output,
            first_slice.xsize,
            first_slice.ysize,
            first.slices.len() as i32,
            first_slice.mode,
        );
        output.fp = fp.clone();
        mrc_head_label(&mut *output, b"Clip: 3D Correlation");
        crate::imod::clip::file_io::grap_volume_write(&mut first, output, options)
    }
}
/// C++ `grap_3dcorr` (`correlation.cpp:319`).
pub unsafe fn grap_3dcorr(
    input1: &mut MrcHeader,
    input2: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    unsafe {
        show_status("Doing 3d correlation...\n");
        if options.infiles > 2 {
            show_error("3dcorr, Only two input files allowed.\n");
            return -1;
        }
        let autocorr = options.infiles == 1;
        if input1.mode != MRC_MODE_COMPLEX_FLOAT {
            show_error("corr, Input file must be complex float.\n");
            return -1;
        }
        if !autocorr {
            if input2.mode != MRC_MODE_COMPLEX_FLOAT || input1.mode != MRC_MODE_COMPLEX_FLOAT {
                show_error("corr, Both input files must be complex float.\n");
                return -1;
            }
            if input1.nx != input2.nx || input1.ny != input2.ny {
                show_error("corr, input files must be same size.\n");
                return -1;
            }
        }
        mrc_head_new(&mut *output, input1.nx, input1.ny, input1.nz, input1.mode);
        mrc_head_label(
            &mut *output,
            if autocorr {
                b"clip: Auto  correlation."
            } else {
                b"clip: Cross correlation."
            },
        );
        let size = input1.nx * input1.ny;
        let Some(mut slice1) = slice_create(input1.nx, input1.ny, MRC_MODE_COMPLEX_FLOAT) else {
            return -1;
        };
        let mut slice2 = if autocorr {
            None
        } else {
            let Some(slice) = slice_create(input1.nx, input1.ny, MRC_MODE_COMPLEX_FLOAT) else {
                return -1;
            };
            Some(slice)
        };
        for z in 0..input1.nz {
            if mrc_read_slice(
                slice1.data.as_mut_ptr().cast(),
                &mut input1.fp.clone().unwrap(),
                input1,
                z,
                b'z',
            ) != 0
            {
                return -1;
            }
            if let Some(slice2) = slice2.as_mut() {
                if mrc_read_slice(
                    slice2.data.as_mut_ptr().cast(),
                    &mut input2.fp.clone().unwrap(),
                    input2,
                    z,
                    b'z',
                ) != 0
                {
                    return -1;
                }
            }
            let second_data = match slice2.as_mut() {
                Some(slice) => slice.data.as_mut_ptr(),
                None => slice1.data.as_mut_ptr(),
            };
            corr_conj(slice1.data.as_mut_ptr().cast(), second_data.cast(), size);
            if mrc_write_slice(
                slice1.data.as_mut_ptr().cast(),
                &mut output.fp.clone().unwrap(),
                output,
                z,
                b'z',
            ) != 0
            {
                return -1;
            }
        }
        mrc_head_write(&mut output.fp.clone().unwrap(), output)
    }
}
/// Matches C++ `grap_corr`.
pub unsafe fn grap_corr(
    input1: &mut MrcHeader,
    input2: &mut MrcHeader,
    output: &mut MrcHeader,
    options: &mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_APPEND_ADD, IP_APPEND_OVERWRITE, IP_DEFAULT};
        if options.dim == 3 {
            return clip_corr3d(input1, input2, output, options);
        }
        if options.infiles > 2 {
            show_error("corr, Only two input files are allowed.\n");
            return -1;
        }
        let mut autocorrelation = options.infiles != 2;
        let mut z1 = 0;
        let mut z2 = 0;
        if options.nofsecs > 0 {
            z1 = options.secs[0];
        }
        if options.nofsecs > 1 {
            autocorrelation = false;
            z2 = options.secs[(1) as usize];
        }
        if options.val == IP_DEFAULT as f32 {
            options.val = 1.;
        }
        if input1.mode == MRC_MODE_COMPLEX_FLOAT || input1.mode == MRC_MODE_COMPLEX_FLOAT {
            show_error("corr, fourier transform input not allowed");
            return -1;
        }
        if options.add2file != 0 {
            if output.mode != MRC_MODE_FLOAT {
                show_error("corr, add or append to float file only.\n");
                return -1;
            }
        } else {
            mrc_head_new(&mut *output, input1.nx, input1.ny, 1, MRC_MODE_FLOAT);
        }
        let mut data_size = 0;
        let mut channels = 0;
        if data_size_for_mode(input1.mode, &mut data_size, &mut channels) != 0 {
            show_error("corr, error getting slice 1.\n");
            return -1;
        }
        let Some(first_bytes_len) = (input1.nx as usize)
            .checked_mul(input1.ny as usize)
            .and_then(|pixels| pixels.checked_mul(data_size as usize))
            .and_then(|bytes| bytes.checked_mul(channels as usize))
        else {
            show_error("corr, error getting slice 1.\n");
            return -1;
        };
        let mut first_bytes = Vec::new();
        if first_bytes.try_reserve_exact(first_bytes_len).is_err() {
            show_error("corr, error getting slice 1.\n");
            return -1;
        }
        first_bytes.resize(first_bytes_len, 0);
        if mrc_read_slice(
            first_bytes.as_mut_ptr().cast(),
            &mut input1.fp.clone().unwrap(),
            input1,
            z1,
            b'z',
        ) != 0
        {
            show_error("corr, error getting slice 1.\n");
            return -1;
        }
        let mut fp2 = if autocorrelation {
            input1.fp.clone().unwrap()
        } else {
            input2.fp.clone().unwrap()
        };
        let (second_nx, second_ny, second_mode) = if autocorrelation {
            (input1.nx, input1.ny, input1.mode)
        } else {
            (input2.nx, input2.ny, input2.mode)
        };
        if data_size_for_mode(second_mode, &mut data_size, &mut channels) != 0 {
            show_error("corr, error getting slice 2.\n");
            return -1;
        }
        let Some(second_bytes_len) = (second_nx as usize)
            .checked_mul(second_ny as usize)
            .and_then(|pixels| pixels.checked_mul(data_size as usize))
            .and_then(|bytes| bytes.checked_mul(channels as usize))
        else {
            show_error("corr, error getting slice 2.\n");
            return -1;
        };
        let mut second_bytes = Vec::new();
        if second_bytes.try_reserve_exact(second_bytes_len).is_err() {
            show_error("corr, error getting slice 2.\n");
            return -1;
        }
        second_bytes.resize(second_bytes_len, 0);
        let second_read = if autocorrelation {
            mrc_read_slice(second_bytes.as_mut_ptr().cast(), &mut fp2, input1, z1, b'z')
        } else {
            mrc_read_slice(second_bytes.as_mut_ptr().cast(), &mut fp2, input2, z2, b'z')
        };
        if second_read != 0 {
            show_error("corr, error getting slice 2.\n");
            return -1;
        }
        if !autocorrelation {
            show_status("Clip: Doing 2D correlation...\n");
        } else {
            show_status("Clip: Doing 2D auto-correlation...\n");
        }
        if options.ix == IP_DEFAULT {
            options.ix = input1.nx;
        }
        if options.iy == IP_DEFAULT {
            options.iy = input1.ny;
        }
        if options.cx == IP_DEFAULT as f32 {
            options.cx = input1.nx as f32 / 2.;
        }
        if options.cy == IP_DEFAULT as f32 {
            options.cy = input1.ny as f32 / 2.;
        }
        let (llx, lly) = (
            options.cx as i32 - options.ix / 2,
            options.cy as i32 - options.iy / 2,
        );
        let (urx, ury) = (llx + options.ix, lly + options.iy);
        let Some(mut first) = slice_create(input1.nx, input1.ny, input1.mode) else {
            return -1;
        };
        first.data.copy_from_slice(&first_bytes);
        let Some(mut second) = slice_create(second_nx, second_ny, second_mode) else {
            return -1;
        };
        second.data.copy_from_slice(&second_bytes);
        slice_mmm(first.as_mut());
        slice_box_in(first.as_mut(), llx, lly, urx, ury);
        slice_mmm(first.as_mut());
        slice_mmm(second.as_mut());
        slice_box_in(second.as_mut(), llx, lly, urx, ury);
        slice_mmm(second.as_mut());
        if options.pad != IP_DEFAULT as f32 {
            first.mean = options.pad;
            second.mean = options.pad;
        }
        clip_padcorr(first.as_mut(), options.val as i32);
        clip_padcorr(second.as_mut(), options.val as i32);
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "image 1 size %d by %d\n",
                &[CArg::Int(first.xsize as i64), CArg::Int(first.ysize as i64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "image 2 size %d by %d\n",
                &[
                    CArg::Int(second.xsize as i64),
                    CArg::Int(second.ysize as i64),
                ],
            )
            .as_bytes(),
        );
        let Some(mut correlation) = clip_slice_corr(first.as_mut(), Some(second.as_mut())) else {
            return -1;
        };
        let crop_x = (correlation.xsize - 2) / 4;
        let crop_y = correlation.ysize / 4;
        slice_box_in(
            correlation.as_mut(),
            crop_x,
            crop_y,
            crop_x + (correlation.xsize - 2) / 2,
            crop_y + correlation.ysize / 2,
        );
        slice_mmm(correlation.as_mut());
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "image c size %d by %d\n",
                &[
                    CArg::Int(correlation.xsize as i64),
                    CArg::Int(correlation.ysize as i64),
                ],
            )
            .as_bytes(),
        );
        if options.add2file == IP_APPEND_ADD {
            slice_resize_in(correlation.as_mut(), output.nx, output.ny);
            output.nz += 1;
            if mrc_write_slice(
                correlation.data.as_mut_ptr().cast(),
                &mut output.fp.clone().unwrap(),
                output,
                output.nz - 1,
                b'z',
            ) != 0
            {
                return -1;
            }
            output.amin = output.amin.min(correlation.min);
            output.amax = output.amax.max(correlation.max);
            output.amean =
                (output.amean * (output.nz - 1) as f32 + correlation.mean) / output.nz as f32;
            if mrc_head_write(&mut output.fp.clone().unwrap(), output) != 0 {
                return -1;
            }
        } else if options.add2file == IP_APPEND_OVERWRITE {
            slice_resize_in(correlation.as_mut(), output.nx, output.ny);
            if mrc_write_slice(
                correlation.data.as_mut_ptr().cast(),
                &mut output.fp.clone().unwrap(),
                output,
                output.nz - 1,
                b'z',
            ) != 0
            {
                return -1;
            }
        } else {
            output.nz = 1;
            output.amin = correlation.min;
            output.amax = correlation.max;
            output.amean = correlation.mean;
            if options.ox != IP_DEFAULT || options.oy != IP_DEFAULT {
                if options.ox != IP_DEFAULT {
                    options.ox = correlation.xsize;
                }
                if options.oy != IP_DEFAULT {
                    options.oy = correlation.ysize;
                }
                slice_resize_in(correlation.as_mut(), options.ox, options.oy);
                output.nx = options.ox;
                output.ny = options.oy;
            } else {
                output.nx = correlation.xsize;
                output.ny = correlation.ysize;
            }
            mrc_head_label(&mut *output, b"clip: 2d correlation calculated.");
            if mrc_head_write(&mut output.fp.clone().unwrap(), output) != 0
                || mrc_write_slice(
                    correlation.data.as_mut_ptr().cast(),
                    &mut output.fp.clone().unwrap(),
                    output,
                    0,
                    b'z',
                ) != 0
            {
                return -1;
            }
        }
        let (mut xmax, mut ymax) = (0, 0);
        let mut maximum = slice_get_pixel_magnitude(correlation.as_mut(), 0, 0);
        for y in 0..correlation.ysize {
            for x in 0..correlation.xsize {
                let value = slice_get_pixel_magnitude(correlation.as_mut(), x, y);
                if value > maximum {
                    maximum = value;
                    xmax = x;
                    ymax = y;
                }
            }
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "pixel max at ( %d, %d)\n",
                &[CArg::Int((xmax) as i64), CArg::Int((ymax) as i64)],
            )
            .as_bytes(),
        );
        let mut patch = [[0_f64; 3]; 3];
        for dy in -1..=1 {
            for dx in -1..=1 {
                patch[(dy + 1) as usize][(dx + 1) as usize] =
                    slice_get_pixel_magnitude(correlation.as_mut(), xmax + dx, ymax + dy) as f64;
            }
        }
        let (mut cx, mut cy) = (0_f64, 0_f64);
        parabolic_fit(&mut cx, &mut cy, &patch);
        // `correlation.cpp:401` declares `float x, y`, so `x = cx + xmax`
        // rounds the double result to f32, and `correlation.cpp:613-615` then
        // subtracts in three separate float steps with a float `0.5f`.
        // Evaluating the whole thing in f64 as one expression rounds
        // differently.
        let mut x = (cx + xmax as f64) as f32;
        let mut y = (cy + ymax as f64) as f32;
        x -= correlation.xsize as f32 * 0.5;
        x -= 1.0;
        y -= correlation.ysize as f32 * 0.5;
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Maximum at ( %.2f, %.2f), transformation ( %.2f, %.2f)\n",
                &[
                    CArg::Dbl((x as core::ffi::c_double) as f64),
                    CArg::Dbl((y as core::ffi::c_double) as f64),
                    CArg::Dbl((-x as core::ffi::c_double) as f64),
                    CArg::Dbl((-y as core::ffi::c_double) as f64),
                ],
            )
            .as_bytes(),
        );
        0
    }
}
/// C++ `padfloat_volume` (`correlation.cpp:627`).
pub unsafe fn padfloat_volume(volume: &mut Istack, pad: f32) -> i32 {
    unsafe {
        let old_zsize = volume.slices.len();
        if old_zsize == 0 {
            return -1;
        }
        let low_z = old_zsize / 2;
        let high_z = low_z + old_zsize;
        let zsize = old_zsize * 2;
        let first = volume.slices.first().unwrap();
        let xsize = first.xsize * 2;
        let ysize = first.ysize * 2;
        let xysize = xsize * ysize;
        let Ok(new_len) = usize::try_from(zsize) else {
            return -1;
        };
        let mut low_padding = Vec::new();
        let mut high_padding = Vec::new();
        if low_padding.try_reserve_exact(low_z as usize).is_err()
            || high_padding
                .try_reserve_exact((zsize - high_z) as usize)
                .is_err()
        {
            return -1;
        }
        for slice in &mut volume.slices {
            slice.mean = pad;
            if slice_resize_in(slice.as_mut(), xsize, ysize) != 0
                || slice_float(slice.as_mut()) != 0
            {
                return -1;
            }
        }
        for _ in 0..low_z {
            let Some(mut slice) = slice_create(xsize, ysize, MRC_MODE_FLOAT) else {
                return -1;
            };
            for i in 0..xysize {
                *slice.data.as_mut_ptr().cast::<f32>().add(i as usize) = pad;
            }
            low_padding.push(slice);
        }
        for _ in high_z..zsize {
            let Some(mut slice) = slice_create(xsize, ysize, MRC_MODE_FLOAT) else {
                return -1;
            };
            for i in 0..xysize {
                *slice.data.as_mut_ptr().cast::<f32>().add(i as usize) = pad;
            }
            high_padding.push(slice);
        }
        let mut new_slices = Vec::new();
        if new_slices.try_reserve_exact(new_len).is_err() {
            return -1;
        }
        new_slices.extend(low_padding);
        new_slices.append(&mut volume.slices);
        new_slices.extend(high_padding);
        debug_assert_eq!(new_slices.len(), zsize);
        volume.slices = new_slices;
        0
    }
}
/// C++ `clip_cor_scalevol` (`correlation.cpp:670`).
///
pub fn clip_cor_scalevol(volume: &mut Istack) -> i32 {
    let Some(first_slice) = volume.slices.first() else {
        return -1;
    };
    let xysize = first_slice.xsize * first_slice.ysize;
    let zsize = volume.slices.len() as i32;
    let scale = (xysize * zsize) as f32;
    for slice in &mut volume.slices {
        for i in 0..xysize {
            unsafe {
                *slice.data.as_mut_ptr().cast::<f32>().add(i as usize) /= scale;
            }
        }
    }
    0
}
/// Matches C++ `parabolic_fit`.
pub fn parabolic_fit(out_x: &mut f64, out_y: &mut f64, input: &[[f64; 3]; 3]) -> f64 {
    let c = (26. * input[0][0] - input[1][0] + 2. * input[2][0]
        - input[0][1]
        - 19. * input[1][1]
        - 7. * input[2][1]
        + 2. * input[0][2]
        - 7. * input[1][2]
        + 14. * input[2][2])
        / 9.;
    let y = (8. * input[0][0] - 8. * input[1][0] + 5. * input[0][1] - 8. * input[1][1]
        + 3. * input[2][1]
        + 2. * input[0][2]
        - 8. * input[1][2]
        + 6. * input[2][2])
        / -6.;
    let yy = (input[0][0] - 2. * input[1][0] + input[2][0] + input[0][1] - 2. * input[1][1]
        + input[2][1]
        + input[0][2]
        - 2. * input[1][2]
        + input[2][2])
        / 6.;
    let x = (8. * input[0][0] + 5. * input[1][0] + 2. * input[2][0]
        - 8. * input[0][1]
        - 8. * input[1][1]
        - 8. * input[2][1]
        + 3. * input[1][2]
        + 6. * input[2][2])
        / -6.;
    let xy = (input[0][0] - input[2][0] - input[0][2] + input[2][2]) / 4.;
    let xx = (input[0][0] + input[1][0] + input[2][0]
        - 2. * input[0][1]
        - 2. * input[1][1]
        - 2. * input[2][1]
        + input[0][2]
        + input[1][2]
        + input[2][2])
        / 6.;
    let d = 4. * yy * xx - xy * xy;
    if d == 0. {
        return input[1][1];
    }
    let p = (4. * c * yy * xx - c * xy * xy - y * y * xx + y * x * xy - x * x * yy) / d;
    *out_y = ((x * xy - 2. * y * xx) / d - 2.).clamp(-1., 1.);
    *out_x = ((y * xy - 2. * x * yy) / d - 2.).clamp(-1., 1.);
    p
}

#[cfg(test)]
mod tests {
    use super::{clip_padcorr, parabolic_fit};
    use crate::imod::libcfshr::islice::{slice_create, slice_get_val, slice_put_val};
    use crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;

    #[test]
    fn parabolic_fit_preserves_center_peak() {
        let mut x = 0.;
        let mut y = 0.;
        let value = parabolic_fit(&mut x, &mut y, &[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]);
        assert_eq!(x, 0.);
        assert_eq!(y, 0.);
        assert_eq!(value, 5. / 9.);
    }

    #[test]
    fn padding_updates_an_owned_slice_in_place() {
        let mut slice = slice_create(2, 2, MRC_MODE_FLOAT).unwrap();
        slice.mean = -3.5;
        unsafe {
            slice_put_val(slice.as_mut(), 0, 0, [1., 0., 0., 0.]);
            slice_put_val(slice.as_mut(), 1, 0, [2., 0., 0., 0.]);
            slice_put_val(slice.as_mut(), 0, 1, [3., 0., 0., 0.]);
            slice_put_val(slice.as_mut(), 1, 1, [4., 0., 0., 0.]);
        }
        unsafe { clip_padcorr(slice.as_mut(), 1) };
        // correlation.cpp boxes [-x/2, x+x/2+2) by [-y/2, y+y/2), hence a
        // 2 by 2 image becomes 6 by 4.  The original pixels are offset by
        // one in both directions and out-of-bounds pixels get the slice mean.
        assert_eq!((slice.xsize, slice.ysize), (6, 4));
        assert_eq!(slice.data.len(), 6 * 4 * size_of::<f32>());
        let mut value = [0.; 4];
        unsafe {
            slice_get_val(slice.as_mut(), 0, 0, &mut value);
        }
        assert_eq!(value[0], -3.5);
        unsafe {
            slice_get_val(slice.as_mut(), 1, 1, &mut value);
        }
        assert_eq!(value[0], 1.);
        unsafe {
            slice_get_val(slice.as_mut(), 2, 2, &mut value);
        }
        assert_eq!(value[0], 4.);
    }
}
