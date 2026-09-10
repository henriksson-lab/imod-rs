//! Translation of `IMOD/clip/correlation.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, show_error, show_status};
use crate::imod::clip::fft::mrc_to_dfft;
use crate::imod::libcfshr::islice::{
    Islice, Istack, slice_create, slice_free, slice_get_pixel_magnitude, slice_get_val, slice_init,
    slice_put_val,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, MrcHeader, mrc_head_label, mrc_head_new,
    mrc_head_write, mrc_read_slice, mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{
    corr_conj, slice_add_const, slice_box, slice_box_in, slice_float, slice_mmm,
    slice_reduce_mirrored_fft, slice_resize_in,
};

/// Matches C++ `corr_getmax`.
pub unsafe fn corr_getmax(
    islice: *mut Islice,
    sa: i32,
    xm: i32,
    ym: i32,
    x: *mut f32,
    y: *mut f32,
) {
    unsafe {
        let slice = slice_box(islice, xm - sa, ym - sa, xm + sa + 1, ym + sa + 1);
        if slice.is_null() {
            return;
        }
        slice_mmm(slice);
        slice_add_const(slice, [-(*slice).min, 0., 0., 0.]);
        slice_mmm(slice);
        slice_add_const(slice, [-(*slice).max + (*slice).max * 0.025, 0., 0., 0.]);
        *x = xm as f32;
        *y = ym as f32;
        let mut row = 0.;
        let mut weight = 0.;
        for ix in 0..(*slice).xsize {
            for iy in 0..(*slice).ysize {
                let value = slice_get_pixel_magnitude(slice, ix, iy);
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
        for ix in 0..(*slice).xsize {
            for iy in 0..(*slice).ysize {
                let value = slice_get_pixel_magnitude(slice, ix, iy);
                if value > 0. {
                    row += (iy + 1) as f32 * value;
                    weight += value;
                }
            }
        }
        if weight > 0. {
            *y = row / weight + (ym - sa - 1) as f32;
        }
        slice_free(slice);
    }
}
/// Matches C++ `clip_padcorr`.
pub unsafe fn clip_padcorr(slice: *mut Islice, pad: i32) {
    unsafe {
        if slice.is_null() || pad == 0 || (*slice).mode == MRC_MODE_COMPLEX_FLOAT {
            if !slice.is_null() && pad == 0 {
                println!("no padding");
            }
            return;
        }
        slice_box_in(
            slice,
            -(*slice).xsize / 2,
            -(*slice).ysize / 2,
            (*slice).xsize + (*slice).xsize / 2 + 2,
            (*slice).ysize + (*slice).ysize / 2,
        );
    }
}
/// Matches C++ `clip_slice_corr`.
pub unsafe fn clip_slice_corr(mut slice1: *mut Islice, mut slice2: *mut Islice) -> *mut Islice {
    unsafe {
        if slice1.is_null() {
            return core::ptr::null_mut();
        }
        if slice2.is_null() {
            slice2 = slice1;
        }
        let autocorr = slice1 == slice2;
        if (*slice1).mode != MRC_MODE_COMPLEX_FLOAT {
            if slice_float(slice1) < 0 {
                return core::ptr::null_mut();
            }
            slice_mmm(slice1);
            slice_add_const(slice1, [-(*slice1).mean, 0., 0., 0.]);
            mrc_to_dfft((*slice1).data.f, (*slice1).xsize - 2, (*slice1).ysize, 0);
            (*slice1).xsize /= 2;
        } else if (*slice1).xsize % 2 == 0 {
            slice_reduce_mirrored_fft(slice1);
        }
        if !autocorr {
            if (*slice2).mode != MRC_MODE_COMPLEX_FLOAT {
                if slice_float(slice2) < 0 {
                    return core::ptr::null_mut();
                }
                mrc_to_dfft((*slice2).data.f, (*slice2).xsize - 2, (*slice2).ysize, 0);
                (*slice2).xsize /= 2;
            } else if (*slice2).xsize % 2 == 0 {
                slice_reduce_mirrored_fft(slice2);
            }
        }
        if (*slice1).xsize != (*slice2).xsize || (*slice1).ysize != (*slice2).ysize {
            show_error("corr: slices must be same size.\n");
            return core::ptr::null_mut();
        }
        let output = slice_create(2 * (*slice1).xsize, (*slice1).ysize, MRC_MODE_FLOAT);
        if output.is_null() {
            return core::ptr::null_mut();
        }
        corr_conj(
            (*slice1).data.f,
            (*slice2).data.f,
            (*slice1).xsize * (*slice1).ysize,
        );
        mrc_to_dfft(
            (*slice1).data.f,
            2 * (*slice1).xsize - 2,
            (*slice1).ysize,
            1,
        );
        (*slice1).xsize *= 2;
        let xm = (*output).xsize / 2;
        let ym = (*output).ysize / 2;
        let mut value = [0.; 4];
        for j in 0..ym {
            for i in 0..xm {
                slice_get_val(slice1, i, j, &mut value);
                slice_put_val(output, xm + i, ym + j, value);
            }
            for (i, x) in ((0..xm).rev()).enumerate() {
                slice_get_val(slice1, (*slice1).xsize - 3 - i as i32, j, &mut value);
                slice_put_val(output, x as i32, ym + j, value);
            }
        }
        for (offset, j) in (ym..(*slice1).ysize).rev().enumerate() {
            let y = ym - 1 - offset as i32;
            for x in (0..xm).rev() {
                let i = (*slice1).xsize - 3 - (xm - 1 - x);
                slice_get_val(slice1, i, j, &mut value);
                slice_put_val(output, x, y, value);
            }
            for x in xm..(*output).xsize {
                slice_get_val(slice1, x - xm, j, &mut value);
                slice_put_val(output, x, y, value);
            }
        }
        output
    }
}
/// Matches C++ `clip_corr3d`.
pub unsafe fn clip_corr3d(
    input1: *mut MrcHeader,
    input2: *mut MrcHeader,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
) -> i32 {
    unsafe {
        if (*input1).mode == MRC_MODE_COMPLEX_FLOAT {
            return grap_3dcorr(input1, input2, output, options);
        }
        // C++ performs a bytewise option copy; ManuallyDrop preserves its ownership model.
        let mut second_options = core::mem::ManuallyDrop::new(core::ptr::read(options));
        let first = crate::imod::clip::file_io::grap_volume_read(input1, options);
        if first.is_null() {
            return -1;
        }
        let (mut min, mut max, mut mean) = (0_f32, 0_f32, 0_f32);
        let (mut xmax, mut ymax, mut zmax) = (0_i32, 0_i32, 0_i32);
        crate::imod::clip::processing::clip_get_stat3d(
            first, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
        );
        println!("stats on vol 1:");
        println!("max = {max}, min = {min}, mean = {mean}");
        println!("location of max pixel = ({xmax}, {ymax} {zmax})");
        if (*options).val as i32 == crate::imod::clip::clip::IP_DEFAULT {
            (*options).val = 1.;
        }
        if (*options).val == 1. && padfloat_volume(first, mean) != 0 {
            crate::imod::clip::file_io::grap_volume_free(first);
            return -1;
        }
        println!();
        let autocorrelation = (*options).infiles != 2;
        let second = if autocorrelation {
            println!("Auto-Correlation");
            first
        } else {
            println!("Cross-Correlation");
            let v = crate::imod::clip::file_io::grap_volume_read(input2, &mut *second_options);
            if v.is_null() {
                crate::imod::clip::file_io::grap_volume_free(first);
                return -1;
            }
            crate::imod::clip::processing::clip_get_stat3d(
                v, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
            );
            println!("stats on vol 2:");
            println!("max = {max}, min = {min}, mean = {mean}");
            println!("location of max pixel = ({xmax}, {ymax} {zmax})");
            if (*options).val == 1. && padfloat_volume(v, mean) != 0 {
                crate::imod::clip::file_io::grap_volume_free(v);
                crate::imod::clip::file_io::grap_volume_free(first);
                return -1;
            }
            println!();
            v
        };
        print!("Calculating fft 1");
        if crate::imod::clip::fft::clip_fftvol(first) != 0 {
            return -1;
        }
        let fp = (*output).fp;
        mrc_head_new(
            &mut *output,
            (*(*(*first).vol)).xsize,
            (*(*(*first).vol)).ysize,
            (*first).zsize,
            (*(*(*first).vol)).mode,
        );
        (*output).fp = fp;
        if !autocorrelation {
            print!("\rCalculating fft 2");
            if crate::imod::clip::fft::clip_fftvol(second) != 0 {
                return -1;
            }
        }
        let size = (*(*(*first).vol)).xsize * (*(*(*first).vol)).ysize;
        for z in 0..(*first).zsize {
            corr_conj(
                (*(*(*first).vol.add(z as usize))).data.f,
                (*(*(*second).vol.add(z as usize))).data.f,
                size,
            );
        }
        print!("\rCalculating inverse fft");
        if crate::imod::clip::fft::clip_fftvol(first) != 0 || clip_cor_scalevol(first) != 0 {
            return -1;
        }
        println!();
        crate::imod::clip::processing::clip_get_stat3d(
            first, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
        );
        let mut peak_x = 0.;
        let mut peak_y = 0.;
        let mut peak_z = 0.;
        crate::imod::clip::processing::clip_parxyz(
            first,
            xmax,
            ymax,
            zmax,
            &mut peak_x,
            &mut peak_y,
            &mut peak_z,
        );
        if peak_x > ((*(*(*first).vol)).xsize / 2) as f32 {
            peak_x -= (*(*(*first).vol)).xsize as f32;
        }
        if peak_y > ((*(*(*first).vol)).ysize / 2) as f32 {
            peak_y -= (*(*(*first).vol)).ysize as f32;
        }
        if peak_z > ((*first).zsize / 2) as f32 {
            peak_z -= (*first).zsize as f32;
        }
        println!("max = {max}  min = {min}  mean = {mean}");
        println!("location of max pixel ( {xmax}, {ymax}, {zmax}) is ");
        println!("( {peak_x:.2}, {peak_y:.2}, {peak_z:.2})");
        mrc_head_new(
            &mut *output,
            (*(*(*first).vol)).xsize,
            (*(*(*first).vol)).ysize,
            (*first).zsize,
            (*(*(*first).vol)).mode,
        );
        (*output).fp = fp;
        mrc_head_label(&mut *output, b"Clip: 3D Correlation");
        let result = crate::imod::clip::file_io::grap_volume_write(first, output, options);
        crate::imod::clip::file_io::grap_volume_free(first);
        if !autocorrelation {
            crate::imod::clip::file_io::grap_volume_free(second);
        }
        result
    }
}
/// C++ `grap_3dcorr` (`correlation.cpp:319`).
pub unsafe fn grap_3dcorr(
    input1: *mut MrcHeader,
    input2: *mut MrcHeader,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
) -> i32 {
    unsafe {
        show_status("Doing 3d correlation...\n");
        if (*options).infiles > 2 {
            show_error("3dcorr, Only two input files allowed.\n");
            return -1;
        }
        let autocorr = (*options).infiles == 1;
        if (*input1).mode != MRC_MODE_COMPLEX_FLOAT {
            show_error("corr, Input file must be complex float.\n");
            return -1;
        }
        if !autocorr {
            if (*input2).mode != MRC_MODE_COMPLEX_FLOAT || (*input1).mode != MRC_MODE_COMPLEX_FLOAT
            {
                show_error("corr, Both input files must be complex float.\n");
                return -1;
            }
            if (*input1).nx != (*input2).nx || (*input1).ny != (*input2).ny {
                show_error("corr, input files must be same size.\n");
                return -1;
            }
        }
        mrc_head_new(
            &mut *output,
            (*input1).nx,
            (*input1).ny,
            (*input1).nz,
            (*input1).mode,
        );
        mrc_head_label(
            &mut *output,
            if autocorr {
                b"clip: Auto  correlation."
            } else {
                b"clip: Cross correlation."
            },
        );
        let size = (*input1).nx * (*input1).ny;
        let slice1 = slice_create((*input1).nx, (*input1).ny, MRC_MODE_COMPLEX_FLOAT);
        if slice1.is_null() {
            return -1;
        }
        let slice2 = if autocorr {
            slice1
        } else {
            slice_create((*input1).nx, (*input1).ny, MRC_MODE_COMPLEX_FLOAT)
        };
        if slice2.is_null() {
            slice_free(slice1);
            return -1;
        }
        for z in 0..(*input1).nz {
            if mrc_read_slice(
                (*slice1).data.b.cast(),
                (*input1).fp.cast(),
                input1,
                z,
                b'z' as i8,
            ) != 0
            {
                if !autocorr {
                    slice_free(slice2);
                }
                slice_free(slice1);
                return -1;
            }
            if !autocorr
                && mrc_read_slice(
                    (*slice2).data.b.cast(),
                    (*input2).fp.cast(),
                    input2,
                    z,
                    b'z' as i8,
                ) != 0
            {
                slice_free(slice2);
                slice_free(slice1);
                return -1;
            }
            corr_conj((*slice1).data.f, (*slice2).data.f, size);
            if mrc_write_slice(
                (*slice1).data.b.cast(),
                (*output).fp.cast(),
                output,
                z,
                b'z' as i8,
            ) != 0
            {
                if !autocorr {
                    slice_free(slice2);
                }
                slice_free(slice1);
                return -1;
            }
        }
        if !autocorr {
            slice_free(slice2);
        }
        slice_free(slice1);
        mrc_head_write((*output).fp.cast(), output)
    }
}
/// Matches C++ `grap_corr`.
pub unsafe fn grap_corr(
    input1: *mut MrcHeader,
    input2: *mut MrcHeader,
    output: *mut MrcHeader,
    options: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_APPEND_ADD, IP_APPEND_OVERWRITE, IP_DEFAULT};
        if (*options).dim == 3 {
            return clip_corr3d(input1, input2, output, options);
        }
        if (*options).infiles > 2 {
            show_error("corr, Only two input files are allowed.\n");
            return -1;
        }
        let mut autocorrelation = (*options).infiles != 2;
        let mut z1 = 0;
        let mut z2 = 0;
        if (*options).nofsecs > 0 {
            z1 = *(*options).secs;
        }
        if (*options).nofsecs > 1 {
            autocorrelation = false;
            z2 = *(*options).secs.add(1);
        }
        if (*options).val as i32 == IP_DEFAULT {
            (*options).val = 1.;
        }
        if (*input1).mode == MRC_MODE_COMPLEX_FLOAT || (*input1).mode == MRC_MODE_COMPLEX_FLOAT {
            show_error("corr, fourier transform input not allowed");
            return -1;
        }
        if (*options).add2file != 0 {
            if (*output).mode != MRC_MODE_FLOAT {
                show_error("corr, add or append to float file only.\n");
                return -1;
            }
        } else {
            mrc_head_new(&mut *output, (*input1).nx, (*input1).ny, 1, MRC_MODE_FLOAT);
        }
        let buffer1 = crate::imod::libiimod::mrcfiles::mrc_mread_slice(
            (*input1).fp.cast(),
            input1,
            z1,
            b'z' as i8,
        );
        if buffer1.is_null() {
            show_error("corr, error getting slice 1.\n");
            return -1;
        }
        let buffer2 = crate::imod::libiimod::mrcfiles::mrc_mread_slice(
            if autocorrelation {
                (*input1).fp.cast()
            } else {
                (*input2).fp.cast()
            },
            if autocorrelation { input1 } else { input2 },
            if autocorrelation { z1 } else { z2 },
            b'z' as i8,
        );
        if buffer2.is_null() {
            show_error("corr, error getting slice 2.\n");
            libc::free(buffer1);
            return -1;
        }
        if !autocorrelation {
            show_status("Clip: Doing 2D correlation...\n");
        } else {
            show_status("Clip: Doing 2D auto-correlation...\n");
        }
        if (*options).ix == IP_DEFAULT {
            (*options).ix = (*input1).nx;
        }
        if (*options).iy == IP_DEFAULT {
            (*options).iy = (*input1).ny;
        }
        if (*options).cx as i32 == IP_DEFAULT {
            (*options).cx = (*input1).nx as f32 / 2.;
        }
        if (*options).cy as i32 == IP_DEFAULT {
            (*options).cy = (*input1).ny as f32 / 2.;
        }
        let (llx, lly) = (
            (*options).cx as i32 - (*options).ix / 2,
            (*options).cy as i32 - (*options).iy / 2,
        );
        let (urx, ury) = (llx + (*options).ix, lly + (*options).iy);
        let mut first: Islice = core::mem::zeroed();
        let mut second: Islice = core::mem::zeroed();
        if slice_init(
            &mut first,
            (*input1).nx,
            (*input1).ny,
            (*input1).mode,
            buffer1.cast(),
        ) != 0
            || slice_init(
                &mut second,
                if autocorrelation {
                    (*input1).nx
                } else {
                    (*input2).nx
                },
                if autocorrelation {
                    (*input1).ny
                } else {
                    (*input2).ny
                },
                if autocorrelation {
                    (*input1).mode
                } else {
                    (*input2).mode
                },
                buffer2.cast(),
            ) != 0
        {
            libc::free(buffer1);
            libc::free(buffer2);
            return -1;
        }
        slice_mmm(&mut first);
        slice_box_in(&mut first, llx, lly, urx, ury);
        slice_mmm(&mut first);
        slice_mmm(&mut second);
        slice_box_in(&mut second, llx, lly, urx, ury);
        slice_mmm(&mut second);
        if (*options).pad as i32 != IP_DEFAULT {
            first.mean = (*options).pad;
            second.mean = (*options).pad;
        }
        clip_padcorr(&mut first, (*options).val as i32);
        clip_padcorr(&mut second, (*options).val as i32);
        println!("image 1 size {} by {}", first.xsize, first.ysize);
        println!("image 2 size {} by {}", second.xsize, second.ysize);
        let correlation = clip_slice_corr(&mut first, &mut second);
        if correlation.is_null() {
            libc::free(first.data.b.cast());
            libc::free(second.data.b.cast());
            return -1;
        }
        let crop_x = ((*correlation).xsize - 2) / 4;
        let crop_y = (*correlation).ysize / 4;
        slice_box_in(
            correlation,
            crop_x,
            crop_y,
            crop_x + ((*correlation).xsize - 2) / 2,
            crop_y + (*correlation).ysize / 2,
        );
        slice_mmm(correlation);
        println!(
            "image c size {} by {}",
            (*correlation).xsize,
            (*correlation).ysize
        );
        if (*options).add2file == IP_APPEND_ADD {
            slice_resize_in(correlation, (*output).nx, (*output).ny);
            (*output).nz += 1;
            if mrc_write_slice(
                (*correlation).data.f.cast(),
                (*output).fp.cast(),
                output,
                (*output).nz - 1,
                b'z' as i8,
            ) != 0
            {
                return -1;
            }
            (*output).amin = (*output).amin.min((*correlation).min);
            (*output).amax = (*output).amax.max((*correlation).max);
            (*output).amean = ((*output).amean * ((*output).nz - 1) as f32 + (*correlation).mean)
                / (*output).nz as f32;
            if mrc_head_write((*output).fp.cast(), output) != 0 {
                return -1;
            }
        } else if (*options).add2file == IP_APPEND_OVERWRITE {
            slice_resize_in(correlation, (*output).nx, (*output).ny);
            if mrc_write_slice(
                (*correlation).data.f.cast(),
                (*output).fp.cast(),
                output,
                (*output).nz - 1,
                b'z' as i8,
            ) != 0
            {
                return -1;
            }
        } else {
            (*output).nz = 1;
            (*output).amin = (*correlation).min;
            (*output).amax = (*correlation).max;
            (*output).amean = (*correlation).mean;
            if (*options).ox != IP_DEFAULT || (*options).oy != IP_DEFAULT {
                if (*options).ox != IP_DEFAULT {
                    (*options).ox = (*correlation).xsize;
                }
                if (*options).oy != IP_DEFAULT {
                    (*options).oy = (*correlation).ysize;
                }
                slice_resize_in(correlation, (*options).ox, (*options).oy);
                (*output).nx = (*options).ox;
                (*output).ny = (*options).oy;
            } else {
                (*output).nx = (*correlation).xsize;
                (*output).ny = (*correlation).ysize;
            }
            mrc_head_label(&mut *output, b"clip: 2d correlation calculated.");
            if mrc_head_write((*output).fp.cast(), output) != 0
                || mrc_write_slice(
                    (*correlation).data.f.cast(),
                    (*output).fp.cast(),
                    output,
                    0,
                    b'z' as i8,
                ) != 0
            {
                return -1;
            }
        }
        let (mut xmax, mut ymax) = (0, 0);
        let mut maximum = slice_get_pixel_magnitude(correlation, 0, 0);
        for y in 0..(*correlation).ysize {
            for x in 0..(*correlation).xsize {
                let value = slice_get_pixel_magnitude(correlation, x, y);
                if value > maximum {
                    maximum = value;
                    xmax = x;
                    ymax = y;
                }
            }
        }
        println!("pixel max at ( {xmax}, {ymax})");
        let mut patch = [[0_f64; 3]; 3];
        for dy in -1..=1 {
            for dx in -1..=1 {
                patch[(dy + 1) as usize][(dx + 1) as usize] =
                    slice_get_pixel_magnitude(correlation, xmax + dx, ymax + dy) as f64;
            }
        }
        let (mut x, mut y) = (0_f64, 0_f64);
        parabolic_fit(&mut x, &mut y, &patch);
        x += xmax as f64 - (*correlation).xsize as f64 * 0.5 - 1.;
        y += ymax as f64 - (*correlation).ysize as f64 * 0.5;
        println!(
            "Maximum at ( {x:.2}, {y:.2}), transformation ( {:.2}, {:.2})",
            -x, -y
        );
        slice_free(correlation);
        libc::free(first.data.b.cast());
        libc::free(second.data.b.cast());
        0
    }
}
/// C++ `padfloat_volume` (`correlation.cpp:627`).
pub unsafe fn padfloat_volume(volume: *mut Istack, pad: f32) -> i32 {
    unsafe {
        if volume.is_null() || (*volume).vol.is_null() || (*volume).zsize <= 0 {
            return -1;
        }
        let old_zsize = (*volume).zsize;
        let low_z = old_zsize / 2;
        let high_z = low_z + old_zsize;
        let zsize = old_zsize * 2;
        let first = *(*volume).vol;
        let xsize = (*first).xsize * 2;
        let ysize = (*first).ysize * 2;
        let xysize = xsize * ysize;
        let new_volume = libc::malloc(zsize as usize * core::mem::size_of::<*mut Islice>())
            .cast::<*mut Islice>();
        if new_volume.is_null() {
            return -1;
        }
        for k in 0..old_zsize {
            let slice = *(*volume).vol.add(k as usize);
            (*slice).mean = pad;
            if slice_resize_in(slice, xsize, ysize) != 0 || slice_float(slice) != 0 {
                libc::free(new_volume.cast());
                return -1;
            }
        }
        for k in 0..low_z {
            let slice = slice_create(xsize, ysize, MRC_MODE_FLOAT);
            if slice.is_null() {
                libc::free(new_volume.cast());
                return -1;
            }
            for i in 0..xysize {
                *(*slice).data.f.add(i as usize) = pad;
            }
            *new_volume.add(k as usize) = slice;
        }
        for k in low_z..high_z {
            *new_volume.add(k as usize) = *(*volume).vol.add((k - low_z) as usize);
        }
        for k in high_z..zsize {
            let slice = slice_create(xsize, ysize, MRC_MODE_FLOAT);
            if slice.is_null() {
                libc::free(new_volume.cast());
                return -1;
            }
            for i in 0..xysize {
                *(*slice).data.f.add(i as usize) = pad;
            }
            *new_volume.add(k as usize) = slice;
        }
        libc::free((*volume).vol.cast());
        (*volume).vol = new_volume;
        (*volume).zsize = zsize;
        0
    }
}
/// C++ `clip_cor_scalevol` (`correlation.cpp:670`).
///
/// # Safety
///
/// `volume` must be an IMOD `Istack` with `zsize` valid slices, each holding
/// a contiguous floating-point plane.  This is the same ownership and layout
/// requirement as the C++ routine.
pub unsafe fn clip_cor_scalevol(volume: *mut Istack) -> i32 {
    unsafe {
        let first_slice = *(*volume).vol;
        let xysize = (*first_slice).xsize * (*first_slice).ysize;
        let zsize = (*volume).zsize;
        let scale = (xysize * zsize) as f32;
        for k in 0..zsize {
            let slice = *(*volume).vol.add(k as usize);
            for i in 0..xysize {
                *(*slice).data.f.add(i as usize) /= scale;
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
    use super::parabolic_fit;

    #[test]
    fn parabolic_fit_preserves_center_peak() {
        let mut x = 0.;
        let mut y = 0.;
        let value = parabolic_fit(&mut x, &mut y, &[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]);
        assert_eq!(x, 0.);
        assert_eq!(y, 0.);
        assert_eq!(value, 5. / 9.);
    }
}
