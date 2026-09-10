//! Translation of `IMOD/clip/processing.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::ClipOptions;
use crate::imod::libcfshr::islice::{
    Islice, Istack, slice_free, slice_get_pixel_magnitude, slice_get_val, slice_init, slice_put_val,
};
use crate::imod::libiimod::mrcfiles::MrcHeader;
use crate::imod::libiimod::mrcslice::{slice_box, slice_mmm};

/// Matches C++ `clip_scaling`.
pub unsafe fn clip_scaling(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    // This deliberately follows the pixel loop in processing.cpp rather than using a
    // Rust image abstraction: its round-to-mode and complex-pixel behavior is part of
    // CLIP's observable output.
    unsafe {
        use crate::imod::clip::clip::*;
        // This condition is intentionally evaluated before set_multifile_input_options,
        // which turns the default section list into the full input list.
        let copy_extra =
            (*opt).process == IP_UNWRAP && (*opt).nofsecs == IP_DEFAULT && (*opt).oz == IP_DEFAULT;
        crate::imod::clip::file_io::set_multifile_input_options(opt, hin);
        let mut z = crate::imod::clip::file_io::set_output_options(opt, hout);
        if z < 0 {
            return z;
        }
        crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
        let mut base = if (*opt).val != IP_DEFAULT as f32 {
            (*opt).val
        } else {
            0.
        };
        let mut min = 0_f64;
        let mut threshold_low = 0_f32;
        let mut threshold_high = 255_f32;
        let mut trunc_low = false;
        let mut trunc_high = false;
        let mut trunc_mean = false;
        let (mut polarity, mut radius_center, mut radius_inner, mut radius_outer, mut border) =
            (1_f32, 0_f32, 0_f32, 0_f32, 0_i32);
        let mut sd_binning = 0_i32;
        let mut sd_arr = core::ptr::null_mut::<f32>();
        let mut sum_arr = core::ptr::null_mut::<f32>();
        let mut sqr_arr = core::ptr::null_mut::<f32>();
        let mut point_fp = core::ptr::null_mut();
        match (*opt).process {
            IP_BRIGHTNESS => {
                crate::imod::clip::clip::show_status("Brightness...\n");
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: brightness");
                min = (*hin).amin as f64;
            }
            IP_SHADOW => {
                crate::imod::clip::clip::show_status("Shadow...\n");
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: shadow");
                min = (*hin).amax as f64;
            }
            IP_CONTRAST => {
                crate::imod::clip::clip::show_status("Contrast...\n");
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: contrast");
                min = (*hin).amean as f64;
            }
            IP_RESIZE => {
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: resized image");
            }
            IP_THRESHOLD => {
                crate::imod::clip::clip::show_status("Threshold...\n");
                if (*opt).sano != 0 {
                    threshold_low = (*hin).amin;
                    threshold_high = (*hin).amax;
                }
                if (*opt).low != IP_DEFAULT as f32 {
                    threshold_low = (*opt).low;
                }
                if (*opt).high != IP_DEFAULT as f32 {
                    threshold_high = (*opt).high;
                }
                if (*opt).thresh == IP_DEFAULT as f32 {
                    crate::imod::clip::clip::show_error(
                        "clip threshold: You must enter a threshold value",
                    );
                    return -1;
                }
                if (*opt).min_size != IP_DEFAULT {
                    return crate::imod::clip::threshminsize::threshold_with_min_size(
                        hin,
                        hout,
                        opt,
                        threshold_low,
                        threshold_high,
                        z,
                    );
                }
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: thresholded");
                if !(*opt).point_out_name.is_null() {
                    crate::imod::libcfshr::b3dutil::imod_backup_file((*opt).point_out_name);
                    point_fp = libc::fopen((*opt).point_out_name, c"w".as_ptr());
                    if point_fp.is_null() {
                        libc::printf(
                            c"ERROR: clip threshold: Error opening output file for points %s\n"
                                .as_ptr(),
                            (*opt).point_out_name,
                        );
                        return -1;
                    }
                }
            }
            IP_TRUNCATE => {
                crate::imod::clip::clip::show_status("Truncate...\n");
                trunc_low = (*opt).low != IP_DEFAULT as f32;
                trunc_high = (*opt).high != IP_DEFAULT as f32;
                trunc_mean = (*opt).sano != 0;
                if !trunc_low && !trunc_high {
                    crate::imod::clip::clip::show_error(
                        "clip truncate: You must enter a low or a high limit",
                    );
                    return -1;
                }
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: truncated");
            }
            IP_UNWRAP => {
                crate::imod::clip::clip::show_status("Unwrap...\n");
                if !matches!((*hin).mode, 1 | 6) {
                    crate::imod::clip::clip::show_error(
                        "clip truncate: Mode must be short or unsigned short integers",
                    );
                    return -1;
                }
                if (*hin).mode == 6 && (*opt).val == IP_DEFAULT as f32 {
                    crate::imod::clip::clip::show_error(
                        "clip truncate: You must enter a value to add with -n for mode 6 input",
                    );
                    return -1;
                }
                if (*opt).val == IP_DEFAULT as f32 {
                    (*opt).val = 32768.;
                }
                crate::imod::libiimod::mrcfiles::mrc_head_label(
                    &mut *hout,
                    b"clip: unwrapped integer values",
                );
                if copy_extra
                    && crate::imod::libiimod::mrcfiles::mrc_copy_extra_header(hin, hout) != 0
                {
                    crate::imod::clip::clip::show_warning(
                        "clip warning: failed to copy extra header data",
                    );
                }
            }
            IP_LOGARITHM => {
                crate::imod::clip::clip::show_status("Logarithm...\n");
                let mut title = [0_i8; 80];
                libc::snprintf(
                    title.as_mut_ptr(),
                    title.len(),
                    c"clip: logarithm after adding %g".as_ptr(),
                    base as f64,
                );
                crate::imod::libiimod::mrcfiles::mrc_head_label(
                    &mut *hout,
                    core::ffi::CStr::from_ptr(title.as_ptr()).to_bytes(),
                );
            }
            IP_SQROOT => {
                crate::imod::clip::clip::show_status("Square root...\n");
                let mut title = [0_i8; 80];
                libc::snprintf(
                    title.as_mut_ptr(),
                    title.len(),
                    c"clip: square root after adding %g".as_ptr(),
                    base as f64,
                );
                crate::imod::libiimod::mrcfiles::mrc_head_label(
                    &mut *hout,
                    core::ffi::CStr::from_ptr(title.as_ptr()).to_bytes(),
                );
            }
            IP_INTEGRAL => {
                polarity = if (*opt).low == IP_DEFAULT as f32 {
                    1.
                } else {
                    -1.
                };
                base = if (*opt).low == IP_DEFAULT as f32 {
                    (*opt).high
                } else {
                    (*opt).low
                };
                radius_center = (*opt).val;
                radius_inner = radius_center + 1.;
                radius_outer = radius_inner + 1_f32.max(0.5 * radius_center);
                if (*opt).new_xoverlap != IP_DEFAULT {
                    radius_inner = (*opt).new_xoverlap as f32;
                }
                if (*opt).new_yoverlap != IP_DEFAULT {
                    radius_outer = (*opt).new_yoverlap as f32;
                }
                crate::imod::clip::clip::show_status("Local integral...\n");
                border = radius_outer.ceil() as i32 + 1;
                let mut title = [0_i8; 80];
                libc::snprintf(
                    title.as_mut_ptr(),
                    title.len(),
                    c"clip: integral, threshold %g, radius %g".as_ptr(),
                    base as f64,
                    radius_center as f64,
                );
                crate::imod::libiimod::mrcfiles::mrc_head_label(
                    &mut *hout,
                    core::ffi::CStr::from_ptr(title.as_ptr()).to_bytes(),
                );
            }
            IP_BOXSD => {
                sd_binning = (*opt).val.abs().round() as i32;
                let bin_x = (*opt).ix / sd_binning;
                let bin_y = (*opt).iy / sd_binning;
                let bin_size = (bin_x * bin_y) as usize;
                sum_arr = libc::malloc(bin_size * core::mem::size_of::<f32>()).cast();
                sqr_arr = libc::malloc(bin_size * core::mem::size_of::<f32>()).cast();
                sd_arr = libc::malloc(bin_size * core::mem::size_of::<f32>()).cast();
                if sd_binning <= 0 || sum_arr.is_null() || sd_arr.is_null() || sqr_arr.is_null() {
                    libc::free(sd_arr.cast());
                    libc::free(sum_arr.cast());
                    libc::free(sqr_arr.cast());
                    return -1;
                }
                let (sx, sy, sz) = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hin);
                crate::imod::libiimod::mrcfiles::mrc_set_scale(
                    &mut *hout,
                    (sx * sd_binning as f32) as f64,
                    (sy * sd_binning as f32) as f64,
                    sz as f64,
                );
                let _ = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hout);
            }
            _ => return -1,
        }
        if (*opt).val == IP_DEFAULT as f32 {
            (*opt).val = 1.;
        }
        let alpha = (*opt).val as f64;
        let min_for_log = 1.0e-20_f32.max(1.0e-5 * ((*hin).amax - (*hin).amin));
        // Keep the source's file-major order.  `set_multifile_input_options`
        // has already checked every header, but each nonfirst input is opened
        // again here and its own header is passed to `sliceReadSubm`.
        for f in 0..(*opt).infiles {
            let mut hdr: MrcHeader = core::mem::zeroed();
            let input_header = if f != 0 {
                hdr.fp = crate::imod::libiimod::iimage::ii_fopen(
                    *(*opt).fnames.add(f as usize),
                    c"rb".as_ptr(),
                )
                .cast();
                if hdr.fp.is_null()
                    || crate::imod::libiimod::mrcfiles::mrc_head_read(hdr.fp.cast(), &mut hdr) != 0
                {
                    if !hdr.fp.is_null() {
                        crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
                    }
                    return -1;
                }
                &mut hdr as *mut MrcHeader
            } else {
                hin
            };
            for k in 0..(*opt).nofsecs {
                let slice = crate::imod::libiimod::mrcslice::slice_read_subm(
                    input_header,
                    *(*opt).secs.add(k as usize),
                    b'z' as i8,
                    (*opt).ix,
                    (*opt).iy,
                    (*opt).cx as i32,
                    (*opt).cy as i32,
                );
                if slice.is_null() {
                    crate::imod::clip::clip::show_error("clip: Error reading slice.");
                    return -1;
                }
                if ((*opt).process == IP_LOGARITHM || (*opt).process == IP_SQROOT)
                    && (*hout).mode == 2
                    && crate::imod::libiimod::mrcslice::slice_float(slice) < 0
                {
                    crate::imod::clip::clip::show_error(
                        "clip: Error getting memory to convert slice to float.",
                    );
                    slice_free(slice);
                    return -1;
                }
                if (*opt).process == IP_BOXSD {
                    if crate::imod::libiimod::mrcslice::slice_float(slice) < 0 {
                        crate::imod::clip::clip::show_error(
                            "clip: Error getting memory to convert slice to float.",
                        );
                        slice_free(slice);
                        return -1;
                    }
                    let nx_bin = (*opt).ix / sd_binning;
                    let ny_bin = (*opt).iy / sd_binning;
                    if nx_bin <= 0 || ny_bin <= 0 {
                        slice_free(slice);
                        return -1;
                    }
                    let size = (nx_bin * ny_bin) as usize;
                    let (mut x_offset, mut y_offset) = (0_i32, 0_i32);
                    crate::imod::libcfshr::multibinstat::make_standard_dev_map(
                        (*slice).data.f,
                        (*opt).ix,
                        0,
                        ((*opt).ix - 1) * if (*opt).sano != 0 { -1 } else { 1 },
                        0,
                        (*opt).iy - 1,
                        sd_binning * if (*opt).val < 0. { 1 } else { -1 },
                        ((*opt).low.round() as i32) / sd_binning,
                        sd_arr,
                        sum_arr,
                        sqr_arr,
                        &mut x_offset,
                        &mut y_offset,
                    );
                    (*slice).xsize = nx_bin;
                    (*slice).ysize = ny_bin;
                    core::ptr::copy_nonoverlapping(sd_arr, (*slice).data.f, size);
                }
                if ((*opt).dim == 2 && (*opt).process != IP_RESIZE)
                    || ((*opt).process == IP_TRUNCATE && trunc_mean)
                {
                    crate::imod::libiimod::mrcslice::slice_mmm(slice);
                    min = match (*opt).process {
                        IP_BRIGHTNESS => (*slice).min as f64,
                        IP_SHADOW => (*slice).max as f64,
                        _ => (*slice).mean as f64,
                    };
                }
                let mut fl_slice: Islice = core::mem::zeroed();
                if (*opt).process == IP_INTEGRAL {
                    slice_init(
                        &mut fl_slice,
                        (*slice).xsize,
                        (*slice).ysize,
                        (*slice).mode,
                        (*slice).data.f.cast(),
                    );
                    if crate::imod::libiimod::mrcslice::slice_float_ex(&mut fl_slice, 0) != 0 {
                        crate::imod::clip::clip::show_error(
                            "clip: Error getting memory to convert slice to float.",
                        );
                        slice_free(slice);
                        return -1;
                    }
                }
                if (*opt).process != IP_BOXSD && (*opt).process != IP_RESIZE {
                    for j in 0..(*opt).iy {
                        for i in 0..(*opt).ix {
                            let mut val = [0_f32; 4];
                            slice_get_val(slice, i, j, &mut val);
                            match (*opt).process {
                                IP_THRESHOLD => {
                                    for (l, item) in
                                        val.iter_mut().take((*slice).csize as usize).enumerate()
                                    {
                                        *item = if *item <= (*opt).thresh {
                                            threshold_low
                                        } else {
                                            if l == 0 && !point_fp.is_null() {
                                                libc::fprintf(
                                                    point_fp,
                                                    c"%6d %6d %4d  %g\n".as_ptr(),
                                                    i,
                                                    j,
                                                    *(*opt).secs.add(k as usize),
                                                    *item as f64,
                                                );
                                            }
                                            threshold_high
                                        };
                                    }
                                }
                                IP_TRUNCATE => {
                                    for item in val.iter_mut().take((*slice).csize as usize) {
                                        if trunc_low && *item < (*opt).low {
                                            *item =
                                                if trunc_mean { min as f32 } else { (*opt).low };
                                        }
                                        if trunc_high && *item > (*opt).high {
                                            *item =
                                                if trunc_mean { min as f32 } else { (*opt).high };
                                        }
                                    }
                                }
                                IP_UNWRAP => {
                                    val[0] += (*opt).val;
                                    let high = if (*hin).mode == 6 { 65535. } else { 32767. };
                                    let low = high - 65535.;
                                    if val[0] > high {
                                        val[0] -= 65536.;
                                    } else if val[0] < low {
                                        val[0] += 65536.;
                                    }
                                }
                                IP_LOGARITHM => {
                                    for item in val.iter_mut().take((*slice).csize as usize) {
                                        *item = min_for_log.max(*item + base).log10();
                                    }
                                }
                                IP_SQROOT => {
                                    for item in val.iter_mut().take((*slice).csize as usize) {
                                        *item = 0_f32.max(*item + base).sqrt();
                                    }
                                }
                                IP_INTEGRAL => {
                                    if i < border
                                        || i >= (*opt).ix - border
                                        || j < border
                                        || j >= (*opt).iy - border
                                        || ((*opt).low != IP_DEFAULT as f32 && val[0] > (*opt).low)
                                        || ((*opt).high != IP_DEFAULT as f32
                                            && val[0] > (*opt).high)
                                    {
                                        val[0] = 0.;
                                    } else {
                                        let mut center_mean = 0.;
                                        let mut annulus_mean = 0.;
                                        val[0] = (polarity
                                            * crate::imod::libcfshr::beadutil::bead_integral(
                                                fl_slice.data.f,
                                                fl_slice.xsize,
                                                fl_slice.xsize,
                                                fl_slice.ysize,
                                                radius_center,
                                                radius_inner,
                                                radius_outer,
                                                i as f32 + 0.5,
                                                j as f32 + 0.5,
                                                &mut center_mean,
                                                &mut annulus_mean,
                                                core::ptr::null_mut(),
                                                0.,
                                                &mut base,
                                            ) as f32)
                                            .max(0.);
                                    }
                                }
                                IP_RESIZE | IP_BOXSD => {}
                                _ => {}
                            }
                            slice_put_val(slice, i, j, val);
                        }
                    }
                }
                if (*opt).process == IP_INTEGRAL && (*slice).mode != 2 {
                    libc::free(fl_slice.data.f.cast());
                }
                if !matches!(
                    (*opt).process,
                    IP_THRESHOLD
                        | IP_TRUNCATE
                        | IP_UNWRAP
                        | IP_LOGARITHM
                        | IP_SQROOT
                        | IP_INTEGRAL
                        | IP_BOXSD
                        | IP_RESIZE
                ) {
                    crate::imod::libiimod::mrcslice::mrc_slice_lie(slice, min, alpha);
                }
                if (*opt).read_defects != 0
                    && correct_defects(slice, (*hin).nx, (*hin).ny, opt) != 0
                {
                    slice_free(slice);
                    return -1;
                }
                if crate::imod::clip::file_io::clip_write_slice(slice, hout, opt, k, &mut z, 1) != 0
                {
                    slice_free(slice);
                    if f != 0 {
                        crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
                    }
                    return -1;
                }
            }
            if f != 0 {
                crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
            }
        }
        if !point_fp.is_null() {
            libc::fclose(point_fp);
        }
        libc::free(sd_arr.cast());
        libc::free(sum_arr.cast());
        libc::free(sqr_arr.cast());
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clipEdge`.
pub unsafe fn clip_edge(hin: *mut MrcHeader, hout: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_DEFAULT, IP_GRADIENT, IP_GRAHAM, IP_PREWITT, IP_SOBEL};
        if (*opt).mode == IP_DEFAULT {
            (*opt).mode = if (*opt).process == IP_GRADIENT {
                (*hin).mode
            } else {
                0
            };
        }
        if !matches!((*hin).mode, 0 | 1 | 6 | 2)
            && !((*hin).mode == 4 && (*opt).process != IP_GRADIENT)
        {
            crate::imod::clip::clip::show_error(
                "clip edge: only byte, integer and float modes can be used",
            );
            return -1;
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
        if z < 0 {
            return z;
        }
        let (message, title): (&str, &[u8]) = match (*opt).process {
            IP_GRADIENT => ("Taking gradient of", b"clip: gradient"),
            IP_PREWITT => ("Applying Prewitt filter to", b"clip: Prewitt filter"),
            IP_GRAHAM => ("Applying Graham filter to", b"clip: Graham filter"),
            IP_SOBEL => ("Applying Sobel filter to", b"clip: Sobel filter"),
            _ => return -1,
        };
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, title);
        println!("clip: {message} {} slices...", (*opt).nofsecs);
        for k in 0..(*opt).nofsecs {
            let source = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if source.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            let out = if (*opt).process == IP_GRADIENT {
                let result = crate::imod::libiimod::mrcslice::slice_gradient(source);
                slice_free(source);
                result
            } else {
                if (*source).mode != 0 {
                    crate::imod::libcfshr::islice::slice_min_max(source);
                    let mut scale = 1.;
                    if (*source).max > (*source).min {
                        scale = 255. / ((*source).max - (*source).min) as f64;
                    }
                    if (0.95..=1.05).contains(&scale) {
                        scale = 0.95;
                    }
                    crate::imod::libiimod::mrcslice::mrc_slice_lie(
                        source,
                        (*source).min as f64 * scale / (scale - 1.),
                        scale,
                    );
                    crate::imod::libiimod::mrcslice::slice_new_mode(source, 0);
                }
                crate::imod::libcfshr::islice::slice_min_max(source);
                if (*opt).process == IP_GRAHAM {
                    crate::imod::libiimod::sliceproc::slice_byte_graham(source);
                } else if (*opt).process == IP_SOBEL {
                    crate::imod::libiimod::sliceproc::slice_byte_edge_sobel(source);
                } else {
                    crate::imod::libiimod::sliceproc::slice_byte_edge_prewitt(source);
                }
                source
            };
            if out.is_null()
                || crate::imod::clip::file_io::clip_write_slice(out, hout, opt, k, &mut z, 1) != 0
            {
                return -1;
            }
        }
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clip_convolve`.
pub unsafe fn clip_convolve(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::*;
        let mut smooth_kernel = [
            0.0625_f32, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
        ];
        let mut sharpen_kernel = [-1_f32, -1., -1., -1., 9., -1., -1., -1., -1.];
        let mut laplacian_kernel = [1_f32, 1., 1., 1., -4., 1., 1., 1., 1.];
        let mut gaussian_kernel = [0_f32; 49];
        let mut dim = 3_i32;
        let mut niter = 1_i32;
        let mut smooth_3d = false;
        let mut title = [0_i8; 100];
        let message: *const libc::c_char;
        let blur: *mut f32;
        if (*opt).mode == IP_DEFAULT {
            (*opt).mode = if (*opt).process == IP_SMOOTH {
                (*hin).mode
            } else {
                crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT
            };
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
        if z < 0 {
            return z;
        }
        match (*opt).process {
            IP_SMOOTH => {
                if (*opt).val < 0. {
                    smooth_3d = true;
                    if (*opt).dim == 2 {
                        crate::imod::clip::clip::show_error(
                            "clip smooth: the -2d option cannot be entered with smoothing in 3D",
                        );
                        return -1;
                    }
                    if (*opt).low == IP_DEFAULT as f32 {
                        (*opt).low = 0.85;
                    }
                }
                if (*opt).val > 1. {
                    niter = (*opt).val.round() as i32;
                }
                if (*opt).low > 0. {
                    crate::imod::libcfshr::filtxcorr::scaled_gaussian_kernel(
                        gaussian_kernel.as_mut_ptr(),
                        &mut dim,
                        7,
                        (*opt).low,
                    );
                    if smooth_3d {
                        if (*hin).nz < dim {
                            libc::snprintf(
                                title.as_mut_ptr(), title.len(),
                                c"clip smooth: 3D smoothing with sigma %.2f requires %d slices; input has only %d".as_ptr(),
                                (*opt).low as f64, dim, (*hin).nz,
                            );
                            crate::imod::clip::clip::show_error(
                                core::ffi::CStr::from_ptr(title.as_ptr()).to_str().unwrap(),
                            );
                            return -1;
                        }
                        message = c"Gaussian kernel 3D smoothing".as_ptr();
                        libc::snprintf(
                            title.as_mut_ptr(),
                            title.len(),
                            c"clip: Gaussian 3D smoothing, sigma %.2f".as_ptr(),
                            (*opt).low as f64,
                        );
                    } else {
                        message = c"Gaussian kernel smoothing".as_ptr();
                        libc::snprintf(
                            title.as_mut_ptr(),
                            title.len(),
                            c"clip: Gaussian smoothing, sigma %.2f, %d iterations".as_ptr(),
                            (*opt).low as f64,
                            niter,
                        );
                    }
                    blur = gaussian_kernel.as_mut_ptr();
                } else {
                    message = c"Smoothing".as_ptr();
                    libc::snprintf(
                        title.as_mut_ptr(),
                        title.len(),
                        c"clip: Standard smoothing, %d iterations".as_ptr(),
                        niter,
                    );
                    blur = smooth_kernel.as_mut_ptr();
                }
            }
            IP_SHARPEN => {
                message = c"Sharpening".as_ptr();
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: sharpen");
                blur = sharpen_kernel.as_mut_ptr();
            }
            IP_LAPLACIAN => {
                message = c"Applying Laplacian to".as_ptr();
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: Laplacian");
                blur = laplacian_kernel.as_mut_ptr();
            }
            _ => return -1,
        }
        if (*opt).process == IP_SMOOTH {
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut *hout,
                core::ffi::CStr::from_ptr(title.as_ptr()).to_bytes(),
            );
        }
        libc::printf(c"clip: %s %d slices...\n".as_ptr(), message, (*opt).nofsecs);
        if smooth_3d {
            return clip_median(hin, hout, opt, blur, dim, z);
        }
        for k in 0..(*opt).nofsecs {
            let mut s = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if s.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            for _ in 0..niter {
                (*s).mean = (*hin).amean;
                if (*opt).process == IP_SMOOTH {
                    crate::imod::libiimod::mrcslice::slice_mmm(s);
                }
                let slice = crate::imod::libcfshr::islice::slice_mat_filter(s, blur, dim);
                if slice.is_null() {
                    crate::imod::clip::clip::show_error(
                        "clip: Error getting new slice for filtering.",
                    );
                    return -1;
                }
                slice_free(s);
                s = slice;
            }
            if crate::imod::clip::file_io::clip_write_slice(s, hout, opt, k, &mut z, 1) != 0 {
                return -1;
            }
        }
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clipMedian`.
pub unsafe fn clip_median(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
    kernel: *mut f32,
    size_in: i32,
    mut z: i32,
) -> i32 {
    unsafe {
        let mut size = size_in;
        let mut z_kernel = [0_f32; 7];
        let mut zsum = 0_f32;
        if kernel.is_null() {
            if !matches!((*hin).mode, 0 | 1 | 6 | 2) {
                crate::imod::clip::clip::show_error(
                    "clip median: only byte, integer and float modes can be used",
                );
                return -1;
            }
            z = crate::imod::clip::file_io::set_options(opt, hin, hout);
            if z < 0 {
                return z;
            }
            if (*opt).val as i32 == crate::imod::clip::clip::IP_DEFAULT {
                (*opt).val = 3.;
            }
            if (*opt).mode != 0 && (*opt).mode != 1 && (*hin).mode != 6 && (*opt).mode != 2 {
                (*opt).mode = (*hin).mode;
            }
            size = 2.max((*opt).val as i32);
            let mut title = [0_i8; 40];
            libc::snprintf(
                title.as_mut_ptr(),
                title.len(),
                c"clip: %dD median filter, size %d".as_ptr(),
                (*opt).dim,
                size,
            );
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut *hout,
                core::ffi::CStr::from_ptr(title.as_ptr()).to_bytes(),
            );
            libc::printf(
                c"clip: median filtering %d slices...\n".as_ptr(),
                (*opt).nofsecs,
            );
        } else {
            for k in 0..size {
                z_kernel[k as usize] = *kernel.add((k + size * (size / 2)) as usize);
                zsum += z_kernel[k as usize];
            }
            for k in 0..size {
                z_kernel[k as usize] /= zsum;
            }
        }
        let depth = if (*opt).dim == 2 { 1 } else { size };
        let vol =
            libc::calloc(depth as usize, core::mem::size_of::<*mut Islice>()).cast::<*mut Islice>();
        if vol.is_null() {
            return -1;
        }
        let mut stack = Istack { vol, zsize: 0 };
        let mut first = 0_i32;
        let mut last = -1_i32;
        let mut output: *mut Islice = core::ptr::null_mut();
        for k in 0..(*opt).nofsecs {
            if k == 0 || !kernel.is_null() {
                output = crate::imod::libcfshr::islice::slice_create(
                    (*opt).ix,
                    (*opt).iy,
                    if kernel.is_null() { (*opt).mode } else { 2 },
                );
                if output.is_null() {
                    libc::free(vol.cast());
                    return -1;
                }
            }
            let (mut needed_first, mut needed_last) = if (*opt).dim == 2 {
                let sec = *(*opt).secs.add(k as usize);
                (sec, sec)
            } else {
                let sec = *(*opt).secs.add(k as usize);
                let a = 0.max(sec - size / 2);
                let b = ((*hin).nz - 1).min(if kernel.is_null() {
                    sec + (size - 1) / 2
                } else {
                    a + size - 1
                });
                (0.max(b + 1 - size), b)
            };
            let mut kept = 0;
            for old in 0..stack.zsize {
                let sec = first + old;
                let item = *stack.vol.add(old as usize);
                if sec < needed_first || sec > needed_last {
                    slice_free(item);
                } else {
                    *stack.vol.add(kept as usize) = item;
                    kept += 1;
                }
            }
            stack.zsize = kept;
            if kept > 0 {
                first = last + 1 - kept;
            }
            for sec in needed_first..=needed_last {
                if stack.zsize == 0 || sec > last {
                    let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                        hin,
                        sec,
                        b'z' as i8,
                        (*opt).ix,
                        (*opt).iy,
                        (*opt).cx as i32,
                        (*opt).cy as i32,
                    );
                    if s.is_null() {
                        return -1;
                    }
                    let s = if kernel.is_null() {
                        s
                    } else {
                        let f = crate::imod::libcfshr::islice::slice_mat_filter(s, kernel, size);
                        if f.is_null() {
                            crate::imod::clip::clip::show_error(
                                "clip: Error getting filtered slice.",
                            );
                            return -1;
                        }
                        slice_free(s);
                        f
                    };
                    if stack.zsize == 0 {
                        first = sec;
                    }
                    *stack.vol.add(stack.zsize as usize) = s;
                    stack.zsize += 1;
                    last = sec;
                }
            }
            if !kernel.is_null() {
                core::ptr::write_bytes((*output).data.f, 0, ((*opt).ix * (*opt).iy) as usize);
                for n in 0..size {
                    let ind = (n + k - size / 2 - first).clamp(0, size - 1);
                    let p = (*(*stack.vol.add(ind as usize))).data.f;
                    for pix in 0..(*opt).ix * (*opt).iy {
                        *(*output).data.f.add(pix as usize) +=
                            z_kernel[n as usize] * *p.add(pix as usize);
                    }
                }
            } else if crate::imod::libiimod::sliceproc::slice_median_filter(
                output, &mut stack, size,
            ) != 0
            {
                return -1;
            }
            if crate::imod::clip::file_io::clip_write_slice(
                output,
                hout,
                opt,
                k,
                &mut z,
                if kernel.is_null() { 0 } else { 1 },
            ) != 0
            {
                return -1;
            }
        }
        if kernel.is_null() {
            slice_free(output);
        }
        for n in 0..stack.zsize {
            slice_free(*stack.vol.add(n as usize));
        }
        libc::free(vol.cast());
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clipBlankFile`.
pub unsafe fn clip_blank_file(hout: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        let setup_z = crate::imod::clip::file_io::set_output_options(opt, hout);
        if setup_z < 0 {
            return setup_z;
        }
        let mut title = [0_i8; 80];
        libc::snprintf(
            title.as_mut_ptr(),
            title.len(),
            c"clip blankfile: constant value %g".as_ptr(),
            (*opt).pad as f64,
        );
        crate::imod::libiimod::mrcfiles::mrc_head_label(
            &mut *hout,
            core::ffi::CStr::from_ptr(title.as_ptr()).to_bytes(),
        );
        let slice = crate::imod::libcfshr::islice::slice_create((*opt).ox, (*opt).oy, (*opt).mode);
        if slice.is_null() {
            crate::imod::libcfshr::parse_params::exit_error(
                c"Creating slice structure with data array".as_ptr(),
            );
        }
        let val = [(*opt).pad, (*opt).pad, (*opt).pad, 0.];
        for iy in 0..(*opt).oy {
            for ix in 0..(*opt).ox {
                slice_put_val(slice, ix, iy, val);
            }
        }
        let mut z = 0;
        (*opt).nofsecs = (*opt).oz;
        for iz in 0..(*opt).oz {
            if crate::imod::clip::file_io::clip_write_slice(slice, hout, opt, iz, &mut z, 0) != 0 {
                return -1;
            }
        }
        slice_free(slice);
        (*hout).xorg = 0.;
        (*hout).yorg = 0.;
        (*hout).zorg = 0.;
        (*hout).amin = (*opt).pad;
        (*hout).amax = (*opt).pad;
        (*hout).amean = (*opt).pad;
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
            return -1;
        }
        0
    }
}
/// Matches C++ `clipDiffusion`.
pub unsafe fn clip_diffusion(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        if !matches!((*hin).mode, 0 | 1 | 6 | 2) {
            crate::imod::clip::clip::show_error(
                "clip diffusion: only byte, integer and float modes can be used",
            );
            return -1;
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
        if z < 0 {
            return z;
        }
        if (*opt).val == crate::imod::clip::clip::IP_DEFAULT as f32 {
            (*opt).val = 5.;
        }
        let iterations = 1.max((*opt).val as i32);
        if (*opt).thresh == crate::imod::clip::clip::IP_DEFAULT as f32 {
            (*opt).thresh = 2.;
        }
        let cc = 1.max(3.min((*opt).thresh as i32));
        if (*opt).weight == crate::imod::clip::clip::IP_DEFAULT as f32 {
            (*opt).weight = 2.;
        }
        let kk = 0_f64.max((*opt).weight as f64);
        if (*opt).low == crate::imod::clip::clip::IP_DEFAULT as f32 {
            (*opt).low = 0.2;
        }
        let lambda = 0.001_f64.max((*opt).low as f64);
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: diffusion");
        libc::printf(
            c"clip: anistropic diffusion %d slices...\n".as_ptr(),
            (*opt).nofsecs,
        );
        for k in 0..(*opt).nofsecs {
            let slice = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if slice.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            if crate::imod::libiimod::sliceproc::slice_aniso_diff(
                slice,
                (*opt).mode,
                cc,
                kk,
                lambda,
                iterations,
                crate::imod::libiimod::sliceproc::ANISO_CLEAR_AT_END,
            ) != 0
            {
                slice_free(slice);
                return -1;
            }
            if crate::imod::clip::file_io::clip_write_slice(slice, hout, opt, k, &mut z, 1) != 0 {
                return -1;
            }
        }
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clip_flip`.
pub unsafe fn clip_flip(hin: *mut MrcHeader, hout: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        let command = if (*opt).command.is_null() {
            return -1;
        } else {
            core::ffi::CStr::from_ptr((*opt).command).to_bytes()
        };
        (*hout).mode = if (*opt).mode == crate::imod::clip::clip::IP_DEFAULT {
            (*hin).mode
        } else {
            (*opt).mode
        };
        let changed_mode = (*hout).mode != (*hin).mode;
        let (kind, axis) = if command.starts_with(b"flipx") {
            (b"clip: flipx".as_slice(), b'x')
        } else if command.starts_with(b"flipy") {
            (b"clip: flipy".as_slice(), b'y')
        } else if command.starts_with(b"flipz") {
            (b"clip: flipz".as_slice(), b'z')
        } else {
            (&[][..], 0)
        };
        if axis != 0
            && !command.starts_with(b"flipxy")
            && !command.starts_with(b"flipyx")
            && !command.starts_with(b"flipxz")
            && !command.starts_with(b"flipzx")
            && !command.starts_with(b"flipyz")
            && !command.starts_with(b"flipzy")
        {
            (*hout).nx = (*hin).nx;
            (*hout).ny = (*hin).ny;
            (*hout).nz = (*hin).nz;
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, kind);
            if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
                return -1;
            }
            for k in 0..(*hin).nz {
                let sl =
                    crate::imod::libcfshr::islice::slice_create((*hin).nx, (*hin).ny, (*hin).mode);
                if sl.is_null() {
                    return -1;
                }
                let input = if axis == b'z' { (*hin).nz - k - 1 } else { k };
                if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                    (*sl).data.b.cast(),
                    (*hin).fp.cast(),
                    hin,
                    input,
                    b'z' as i8,
                ) != 0
                {
                    slice_free(sl);
                    return -1;
                }
                if changed_mode
                    && crate::imod::libiimod::mrcslice::slice_new_mode(sl, (*hout).mode) < 0
                {
                    slice_free(sl);
                    return -1;
                }
                if axis != b'z' {
                    crate::imod::libiimod::mrcslice::slice_mirror(sl, axis as i8);
                }
                if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                    (*sl).data.b.cast(),
                    (*hout).fp.cast(),
                    hout,
                    k,
                    b'z' as i8,
                ) != 0
                {
                    slice_free(sl);
                    return -1;
                }
                slice_free(sl);
            }
            libc::puts(c" Done!".as_ptr());
            return 0;
        }
        if command.starts_with(b"flipxy") || command.starts_with(b"flipyx") {
            (*hout).nx = (*hin).ny;
            (*hout).ny = (*hin).nx;
            (*hout).nz = (*hin).nz;
            (*hout).mx = (*hin).my;
            (*hout).my = (*hin).mx;
            (*hout).mz = (*hin).mz;
            (*hout).xlen = (*hin).ylen;
            (*hout).ylen = (*hin).xlen;
            (*hout).zlen = (*hin).zlen;
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: flipxy");
            let mut limits = [0; 3];
            let mut num_tiles = [0; 3];
            let mut tile_sizes = [0; 3];
            if crate::imod::clip::file_io::set_chunk_output(
                opt,
                hout,
                limits.as_mut_ptr(),
                num_tiles.as_mut_ptr(),
                tile_sizes.as_mut_ptr(),
            ) != 0
            {
                return -1;
            }
            if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
                return -1;
            }
            for x in 0..(*hin).nx {
                let sl = crate::imod::libcfshr::islice::slice_create(
                    (*hout).nx,
                    (*hout).nz,
                    (*hin).mode,
                );
                if sl.is_null() {
                    return -1;
                }
                if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                    (*sl).data.b.cast(),
                    (*hin).fp.cast(),
                    hin,
                    x,
                    b'x' as i8,
                ) != 0
                {
                    slice_free(sl);
                    return -1;
                }
                if changed_mode
                    && crate::imod::libiimod::mrcslice::slice_new_mode(sl, (*hout).mode) < 0
                {
                    slice_free(sl);
                    return -1;
                }
                if (*opt).sano != 0 {
                    crate::imod::libiimod::mrcslice::slice_mirror(sl, b'y' as i8);
                }
                if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                    (*sl).data.b.cast(),
                    (*hout).fp.cast(),
                    hout,
                    x,
                    b'y' as i8,
                ) != 0
                {
                    slice_free(sl);
                    return -1;
                }
                slice_free(sl);
            }
            libc::puts(c" Done!".as_ptr());
            return 0;
        }
        if command.starts_with(b"flipxz") || command.starts_with(b"flipzx") {
            (*hout).nx = (*hin).nz;
            (*hout).ny = (*hin).ny;
            (*hout).nz = (*hin).nx;
            (*hout).mx = (*hin).mz;
            (*hout).mz = (*hin).mx;
            (*hout).xlen = (*hin).zlen;
            (*hout).zlen = (*hin).xlen;
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: flipxz");
            let mut limits = [0; 3];
            let mut num_tiles = [0; 3];
            let mut tile_sizes = [0; 3];
            if crate::imod::clip::file_io::set_chunk_output(
                opt,
                hout,
                limits.as_mut_ptr(),
                num_tiles.as_mut_ptr(),
                tile_sizes.as_mut_ptr(),
            ) != 0
            {
                return -1;
            }
            if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
                return -1;
            }
            let source =
                crate::imod::libcfshr::islice::slice_create((*hin).ny, (*hin).nz, (*hin).mode);
            let transposed =
                crate::imod::libcfshr::islice::slice_create((*hout).nx, (*hout).ny, (*hout).mode);
            if source.is_null() || transposed.is_null() {
                return -1;
            }
            for x in 0..(*hin).nx {
                if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                    (*source).data.b.cast(),
                    (*hin).fp.cast(),
                    hin,
                    x,
                    b'x' as i8,
                ) != 0
                {
                    return -1;
                }
                for y in 0..(*hin).ny {
                    for z in 0..(*hin).nz {
                        let mut v = [0.; 4];
                        slice_get_val(source, y, z, &mut v);
                        slice_put_val(transposed, z, y, v);
                    }
                }
                if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                    (*transposed).data.b.cast(),
                    (*hout).fp.cast(),
                    hout,
                    x,
                    b'z' as i8,
                ) != 0
                {
                    return -1;
                }
            }
            slice_free(source);
            slice_free(transposed);
            libc::puts(c" Done!".as_ptr());
            return 0;
        }
        if command.starts_with(b"flipyz")
            || command.starts_with(b"flipzy")
            || command.starts_with(b"rotx")
        {
            let rotate = command.starts_with(b"rotx");
            (*hout).mode = (*hin).mode;
            (*hout).nx = (*hin).nx;
            (*hout).mx = (*hin).mx;
            (*hout).xlen = (*hin).xlen;
            (*hout).ny = (*hin).nz;
            (*hout).my = (*hin).mz;
            (*hout).ylen = (*hin).zlen;
            (*hout).nz = (*hin).ny;
            (*hout).mz = (*hin).my;
            (*hout).zlen = (*hin).ylen;
            if rotate && (*hin).my != 0 && (*hin).ylen != 0. && (*hin).mz != 0 && (*hin).zlen != 0.
            {
                for i in 0..3 {
                    (*hout).tiltangles[i] = (*hout).tiltangles[i + 3];
                }
                (*hout).tiltangles[3] -= 90.;
                let ycen = (*hin).ny as f32 / 2. - (*hin).yorg * (*hin).my as f32 / (*hin).ylen;
                let zcen = (*hin).nz as f32 / 2. - (*hin).zorg * (*hin).mz as f32 / (*hin).zlen;
                (*hout).yorg = ((*hout).ny as f32 / 2. - zcen) * (*hout).ylen / (*hout).my as f32;
                (*hout).zorg = ((*hout).nz as f32 / 2. + ycen) * (*hout).zlen / (*hout).mz as f32;
            }
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut *hout,
                if rotate {
                    b"clip: rotx - rotation by -90 around X"
                } else {
                    b"clip: flipyz"
                },
            );
            let mut limits = [0; 3];
            let mut num_tiles = [0; 3];
            let mut tile_sizes = [0; 3];
            if crate::imod::clip::file_io::set_chunk_output(
                opt,
                hout,
                limits.as_mut_ptr(),
                num_tiles.as_mut_ptr(),
                tile_sizes.as_mut_ptr(),
            ) != 0
            {
                return -1;
            }
            if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
                return -1;
            }
            let input_slice =
                crate::imod::libcfshr::islice::slice_create((*hin).nx, (*hin).ny, (*hin).mode);
            let output_slice =
                crate::imod::libcfshr::islice::slice_create((*hout).nx, (*hout).ny, (*hout).mode);
            if input_slice.is_null() || output_slice.is_null() {
                return -1;
            }
            for k in 0..(*hout).nz {
                let y = if rotate { (*hin).ny - k - 1 } else { k };
                for z in 0..(*hin).nz {
                    if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                        (*input_slice).data.b.cast(),
                        (*hin).fp.cast(),
                        hin,
                        z,
                        b'z' as i8,
                    ) != 0
                    {
                        slice_free(input_slice);
                        slice_free(output_slice);
                        return -1;
                    }
                    for x in 0..(*hin).nx {
                        let mut value = [0.; 4];
                        slice_get_val(input_slice, x, y, &mut value);
                        slice_put_val(output_slice, x, z, value);
                    }
                }
                if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                    (*output_slice).data.b.cast(),
                    (*hout).fp.cast(),
                    hout,
                    k,
                    b'z' as i8,
                ) != 0
                {
                    slice_free(input_slice);
                    slice_free(output_slice);
                    return -1;
                }
            }
            slice_free(input_slice);
            slice_free(output_slice);
            libc::puts(c" Done!".as_ptr());
            return 0;
        }
        crate::imod::clip::clip::show_warning("clip flip - no flipping was done.");
        -1
    }
}
/// Matches C++ `clip_quadrant`.
pub unsafe fn clip_quadrant(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        let (nx, ny, nz) = ((*hin).nx, (*hin).ny, (*hin).nz);
        let mut width = 20;
        let mut group_size = 1;
        let mut user_base = 0_f64;
        let mut num_todo = nz;
        if (*opt).ix != crate::imod::clip::clip::IP_DEFAULT
            || (*opt).iy != crate::imod::clip::clip::IP_DEFAULT
            || (*opt).ox != crate::imod::clip::clip::IP_DEFAULT
            || (*opt).oy != crate::imod::clip::clip::IP_DEFAULT
            || (*opt).oz != crate::imod::clip::clip::IP_DEFAULT
        {
            crate::imod::clip::clip::show_warning(
                "clip quadrant - input and output sizes ignored.",
            );
        }
        if (*opt).nofsecs != crate::imod::clip::clip::IP_DEFAULT {
            num_todo = (*opt).nofsecs;
            for iz in 0..num_todo - 1 {
                for jz in iz + 1..num_todo {
                    if *(*opt).secs.add(jz as usize) < *(*opt).secs.add(iz as usize) {
                        core::ptr::swap((*opt).secs.add(iz as usize), (*opt).secs.add(jz as usize));
                    }
                }
            }
            let mut jz = 0;
            for iz in 0..num_todo {
                if jz == 0 || *(*opt).secs.add((jz - 1) as usize) != *(*opt).secs.add(iz as usize) {
                    *(*opt).secs.add(jz as usize) = *(*opt).secs.add(iz as usize);
                    jz += 1;
                }
            }
            num_todo = jz;
        } else {
            crate::imod::clip::file_io::set_input_options(opt, hin);
        }
        (*opt).nofsecs = nz;
        if (*opt).val != crate::imod::clip::clip::IP_DEFAULT as f32 {
            group_size = (*opt).val.round() as i32;
        }
        if (*opt).high != crate::imod::clip::clip::IP_DEFAULT as f32 {
            width = (*opt).high.round() as i32;
        }
        if (*opt).low != crate::imod::clip::clip::IP_DEFAULT as f32 {
            user_base = (*opt).low as f64;
        }
        if width < 2 || width > nx / 4 || width > ny / 4 {
            crate::imod::clip::clip::show_error("clip: width entry too small or too large.");
            return -1;
        }
        group_size = group_size.clamp(1, nz);
        let num_groups = 1.max(num_todo / group_size);
        (*hout).mode = (*hin).mode;
        let mut new_mode = -1;
        if (*opt).mode != crate::imod::clip::clip::IP_DEFAULT && (*opt).mode != (*hin).mode {
            new_mode = crate::imod::libcfshr::islice::slice_mode_if_real((*opt).mode);
            if new_mode < 0 {
                crate::imod::clip::clip::show_error("clip: Inappropriate new mode entry.");
                return -1;
            }
            (*hout).mode = (*opt).mode;
        }
        (*hout).amean = 0.;
        (*hout).amin = 1.0e37;
        (*hout).amax = -1.0e37;
        if crate::imod::libiimod::mrcfiles::mrc_copy_extra_header(hin, hout) != 0 {
            crate::imod::clip::clip::show_warning("clip warning: failed to copy extra header data");
        }
        let vx3 = nx / 2 + 1;
        let vx4 = vx3 + width;
        let vx2 = vx3 - 2;
        let vx1 = vx2 - width;
        let hx1 = nx / 20;
        let hx2 = nx / 2 - 10;
        let hx3 = nx / 2 + 10;
        let hx4 = 19 * nx / 20;
        let hy3 = ny / 2 + 1;
        let hy4 = hy3 + width;
        let hy2 = hy3 - 2;
        let hy1 = hy2 - width;
        let vy1 = ny / 20;
        let vy2 = ny / 2 - 10;
        let vy3 = ny / 2 + 10;
        let vy4 = 19 * ny / 20;
        let mut start = 0_i32;
        let mut last_out = -1;
        for group in 0..num_groups {
            let nin_group = group_size + if group < num_todo % group_size { 1 } else { 0 };
            let end = start + nin_group;
            let mut d = [0_f64; 8];
            let mut iz = 0;
            for ind in start..end {
                iz = *(*opt).secs.add(ind as usize);
                let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                    hin,
                    iz,
                    b'z' as i8,
                    nx,
                    ny,
                    nx / 2,
                    ny / 2,
                );
                if s.is_null() {
                    crate::imod::clip::clip::show_error("clip: Error reading slice.");
                    return -1;
                }
                let coords = [
                    (hx3, hy3, hx4, hy4),
                    (vx3, vy3, vx4, vy4),
                    (hx1, hy3, hx2, hy4),
                    (vx1, vy3, vx2, vy4),
                    (hx1, hy1, hx2, hy2),
                    (vx1, vy1, vx2, vy2),
                    (hx3, hy1, hx4, hy2),
                    (vx3, vy1, vx4, vy2),
                ];
                for n in 0..8 {
                    let mut mean = 0.;
                    if quadrant_sample(
                        s,
                        coords[n].0,
                        coords[n].1,
                        coords[n].2,
                        coords[n].3,
                        &mut mean,
                    ) != 0
                    {
                        slice_free(s);
                        return -1;
                    }
                    d[n] += mean;
                }
                slice_free(s);
            }
            for v in &mut d {
                *v /= nin_group as f64;
            }
            let qmin = d.iter().copied().fold(f64::INFINITY, f64::min);
            let qmax = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let base =
                (if user_base != 0. { 0.01 } else { 0.05 }) * (qmax - qmin) - qmin - user_base;
            if base > 0. {
                crate::imod::clip::clip::show_warning(
                    "clip - intensities being adjusted to avoid taking log of small or negative values",
                );
            }
            let base = base.max(0.) + user_base;
            let t = d.map(|v| (v + base).log10());
            let c2 = t[0] - t[6] + 2. * t[1] - 2. * t[3] + t[4] - t[2];
            let c3 = t[0] - t[6] + t[1] - t[3] + t[2] - t[4] + t[7] - t[5];
            let c4 = 2. * t[0] - 2. * t[6] + t[1] - t[3] + t[5] - t[7];
            let det = 64.;
            let g2 = (c2 * 20. - 2. * (c3 * 6. - 2. * c4) + 4. * (c3 * 2. - 4. * c4)) / det;
            let g3 = (6. * (c3 * 6. - 2. * c4) - c2 * 4. + 4. * (2. * c4 - c3 * 4.)) / det;
            let g4 = (6. * (4. * c4 - c3 * 2.) - 2. * (2. * c4 - c3 * 4.)
                + c2 * (4. * 2. - 4. * 4.))
                / det;
            let gain = [
                10_f64.powf(-(g2 + g3 + g4)),
                10_f64.powf(g2),
                10_f64.powf(g3),
                10_f64.powf(g4),
            ];
            if nin_group > 1 {
                libc::printf(
                    c"Group from %d to %d:".as_ptr(),
                    *(*opt).secs.add(start as usize),
                    iz,
                );
            } else {
                libc::printf(c"Section %d:".as_ptr(), iz);
            }
            libc::printf(
                c" scale factors %.4f %.4f %.4f %.4f\n".as_ptr(),
                gain[0],
                gain[1],
                gain[2],
                gain[3],
            );
            libc::printf(
                c"Boundary diffs before: %6.1f %6.1f %6.1f %6.1f\n".as_ptr(),
                d[0] - d[6],
                d[3] - d[1],
                d[4] - d[2],
                d[7] - d[5],
            );
            libc::printf(
                c"Boundary diffs after: %6.1f %6.1f %6.1f %6.1f\n".as_ptr(),
                gain[0] * (d[0] + base) - gain[3] * (d[6] + base),
                gain[1] * (d[3] + base) - gain[0] * (d[1] + base),
                gain[2] * (d[4] + base) - gain[1] * (d[2] + base),
                gain[3] * (d[7] + base) - gain[2] * (d[5] + base),
            );
            let mut end_out = iz;
            if group == num_groups - 1 {
                end_out = nz - 1;
            }
            for iz in last_out + 1..=end_out {
                let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                    hin,
                    iz,
                    b'z' as i8,
                    nx,
                    ny,
                    nx / 2,
                    ny / 2,
                );
                if s.is_null() {
                    crate::imod::clip::clip::show_error("clip: Error reading slice.");
                    return -1;
                }
                if new_mode >= 0 && crate::imod::libiimod::mrcslice::slice_new_mode(s, new_mode) < 0
                {
                    crate::imod::clip::clip::show_error(
                        "clip: Error converting slice to new mode.",
                    );
                    return -1;
                }
                if (start..end).any(|ind| iz == *(*opt).secs.add(ind as usize)) {
                    correct_quadrant(s, nx / 2, ny / 2, nx, ny, gain[0], base);
                    correct_quadrant(s, 0, ny / 2, nx / 2, ny, gain[1], base);
                    correct_quadrant(s, 0, 0, nx / 2, ny / 2, gain[2], base);
                    correct_quadrant(s, nx / 2, 0, nx, ny / 2, gain[3], base);
                }
                crate::imod::libiimod::mrcslice::slice_mmm(s);
                (*hout).amin = (*hout).amin.min((*s).min);
                (*hout).amax = (*hout).amax.max((*s).max);
                (*hout).amean += (*s).mean / nz as f32;
                if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                    (*s).data.b.cast(),
                    (*hout).fp.cast(),
                    hout,
                    iz,
                    b'z' as i8,
                ) != 0
                {
                    crate::imod::clip::clip::show_error("clip: Error writing slice.");
                    return -1;
                }
                slice_free(s);
                last_out = iz;
            }
            start += nin_group;
        }
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: quadrant correction");
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
            return -1;
        }
        0
    }
}
/// Matches C++ `quadrantSample`.
pub unsafe fn quadrant_sample(
    slice: *mut Islice,
    llx: i32,
    lly: i32,
    urx: i32,
    ury: i32,
    mean: *mut f64,
) -> i32 {
    unsafe {
        let box_slice = slice_box(slice, llx, lly, urx, ury);
        if box_slice.is_null() {
            crate::imod::clip::clip::show_error("clip: Error extracting subslice.");
            return -1;
        }
        slice_mmm(box_slice);
        *mean = (*box_slice).mean as f64;
        slice_free(box_slice);
        0
    }
}
/// Matches C++ `correctQuadrant`.
pub unsafe fn correct_quadrant(
    slice: *mut Islice,
    llx: i32,
    lly: i32,
    urx: i32,
    ury: i32,
    gain: f64,
    base: f64,
) {
    unsafe {
        for iy in lly..ury {
            for ix in llx..urx {
                let mut val = [0.; 4];
                slice_get_val(slice, ix, iy, &mut val);
                let corrected = gain * (val[0] as f64 + base) - base;
                val[0] = if (*slice).mode == 2 {
                    corrected as f32
                } else {
                    (corrected + 0.5).floor() as f32
                };
                slice_put_val(slice, ix, iy, val);
            }
        }
    }
}
/// Matches C++ `fillDriftCorrectedEdges`.
pub unsafe fn fill_drift_corrected_edges(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        let mut width = if (*hin).nx < 3000 { 30 } else { 60 };
        let mut length = 1024.max((*hin).nx / 4);
        let mut crit = 9.;
        if (*opt).low != crate::imod::clip::clip::IP_DEFAULT as f32 {
            length = (*opt).low.round() as i32;
        }
        if (*opt).high != crate::imod::clip::clip::IP_DEFAULT as f32 {
            crit = (*opt).high;
        }
        if (*opt).val != crate::imod::clip::clip::IP_DEFAULT as f32 {
            width = (*opt).val.round() as i32;
        }
        if !matches!((*hin).mode, 1 | 6) {
            crate::imod::clip::clip::show_error(
                "clip fill: only signed or unsigned short integer modes can be used",
            );
            return -1;
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
        if z < 0 {
            return z;
        }
        let mut defects = crate::imod::clip::clip::CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: Vec::new(),
            bad_column_width: Vec::new(),
            partial_bad_col: Vec::new(),
            partial_bad_width: Vec::new(),
            partial_bad_start_y: Vec::new(),
            partial_bad_end_y: Vec::new(),
            bad_row_start: Vec::new(),
            bad_row_height: Vec::new(),
            partial_bad_row: Vec::new(),
            partial_bad_height: Vec::new(),
            partial_bad_start_x: Vec::new(),
            partial_bad_end_x: Vec::new(),
            bad_pixel_x: Vec::new(),
            bad_pixel_y: Vec::new(),
            pix_use_mean: Vec::new(),
        };
        defects.was_scaled = 0;
        defects.rotation_flip = 0;
        defects.k2_type = 0;
        for k in 0..(*opt).nofsecs {
            let slice = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if slice.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            libc::printf(c"\nSLICE %d\n".as_ptr(), *(*opt).secs.add(k as usize));
            if crate::imod::clip::correct_defects::cor_def_find_drift_corr_edges(
                (*slice).data.b.cast(),
                (*hin).mode,
                (*opt).ix,
                (*opt).iy,
                length,
                width,
                crit,
                &mut defects.usable_left,
                &mut defects.usable_right,
                &mut defects.usable_top,
                &mut defects.usable_bottom,
            ) != 0
            {
                let mut message = [0_i8; 80];
                libc::snprintf(
                    message.as_mut_ptr(),
                    message.len(),
                    c"clip: error from CorDefFindDriftCorrEdges for slice %d.".as_ptr(),
                    k,
                );
                crate::imod::clip::clip::show_error(
                    core::ffi::CStr::from_ptr(message.as_ptr())
                        .to_str()
                        .unwrap(),
                );
                return -1;
            }
            crate::imod::clip::correct_defects::cor_def_correct_defects(
                &defects,
                (*slice).data.b.cast(),
                (*hin).mode,
                1,
                0,
                0,
                (*hin).ny,
                (*hin).nx,
            );
            if crate::imod::clip::file_io::clip_write_slice(slice, hout, opt, k, &mut z, 1) != 0 {
                return -1;
            }
        }
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clipSpectrum`.
pub unsafe fn clip_spectrum(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        let mut bkgd = 48_i32;
        let mut trunc = 0.02_f32;
        if (*opt).low != crate::imod::clip::clip::IP_DEFAULT as f32 {
            bkgd = if (*opt).low > 0. && (*opt).low < 1. {
                (255. * (*opt).low) as i32
            } else {
                (*opt).low as i32
            };
            if bkgd >= 192 {
                return 1;
            }
        }
        if (*opt).high != crate::imod::clip::clip::IP_DEFAULT as f32 {
            trunc = (*opt).high;
            if trunc < 0. || trunc >= 0.75 {
                return 1;
            }
        }
        if (*opt).ox != crate::imod::clip::clip::IP_DEFAULT
            || (*opt).oy != crate::imod::clip::clip::IP_DEFAULT
        {
            if (*opt).ox != crate::imod::clip::clip::IP_DEFAULT
                && (*opt).oy != crate::imod::clip::clip::IP_DEFAULT
                && (*opt).ox != (*opt).oy
            {
                crate::imod::clip::clip::show_error(
                    "clip spectrum: -ox and -oy cannot be entered with different sizes",
                );
                return 1;
            }
            if (*opt).ox != crate::imod::clip::clip::IP_DEFAULT {
                (*opt).oy = (*opt).ox
            } else {
                (*opt).ox = (*opt).oy
            }
        } else {
            (*opt).ox = 1024;
            (*opt).oy = 1024
        }
        if (*opt).oz != crate::imod::clip::clip::IP_DEFAULT {
            crate::imod::clip::clip::show_error("clip spectrum: -oz is not allowed");
            return 1;
        }
        let mode = if bkgd > 0 { 0 } else { 1 };
        if (*opt).mode == crate::imod::clip::clip::IP_DEFAULT {
            (*opt).mode = mode
        }
        if (*opt).add2file != 0 {
            crate::imod::clip::clip::show_error("clip spectrum: Cannot append to existing file");
            return 1;
        }
        crate::imod::clip::file_io::set_input_options(opt, hin);
        let pad = crate::imod::libcfshr::filtxcorr::nice_frame(
            (*opt).ix.max((*opt).iy),
            2,
            crate::imod::libfft::nice_fft_limit(),
        );
        if (pad as f32) < (*opt).ox as f32 * 1.02 {
            (*opt).ox = pad;
            (*opt).oy = pad
        }
        let mut zout = crate::imod::clip::file_io::set_output_options(opt, hout);
        if zout < 0 {
            return 1;
        }
        crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: scaled power spectrum");
        libc::printf(
            c"clip: Taking power spectrum of %d slices...\n".as_ptr(),
            (*opt).nofsecs,
        );
        for k in 0..(*opt).nofsecs {
            let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if s.is_null() {
                let mut message = [0_i8; 80];
                libc::snprintf(
                    message.as_mut_ptr(),
                    message.len(),
                    c"clip: Error reading slice %d.".as_ptr(),
                    *(*opt).secs.add(k as usize),
                );
                crate::imod::clip::clip::show_error(
                    core::ffi::CStr::from_ptr(message.as_ptr())
                        .to_str()
                        .unwrap(),
                );
                return -1;
            }
            let out = crate::imod::libcfshr::islice::slice_create((*opt).ox, (*opt).oy, mode);
            if out.is_null() {
                crate::imod::clip::clip::show_error(
                    "clip: Error getting memory for spectrum slice.",
                );
                return -1;
            }
            let err = crate::imod::libcfshr::spectrumscaled::spectrum_scaled(
                (*s).data.b.cast(),
                (*s).mode,
                (*s).xsize,
                (*s).ysize,
                (*out).data.b.cast(),
                pad,
                (*opt).ox,
                bkgd,
                trunc,
                3,
                crate::imod::libfft::todfft,
            );
            if err != 0 {
                let mut message = [0_i8; 80];
                libc::snprintf(
                    message.as_mut_ptr(),
                    message.len(),
                    c"clip: Error %d calling spectrumScaled".as_ptr(),
                    err,
                );
                crate::imod::clip::clip::show_error(
                    core::ffi::CStr::from_ptr(message.as_ptr())
                        .to_str()
                        .unwrap(),
                );
                return 1;
            }
            if crate::imod::clip::file_io::clip_write_slice(out, hout, opt, k, &mut zout, 1) != 0 {
                slice_free(out);
                return -1;
            }
        }
        if pad > (*opt).ox {
            let (mut x, mut y, mut z) = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hin);
            let f = pad as f32 / (*opt).ox as f32;
            x *= f;
            y *= f;
            z *= f;
            crate::imod::libiimod::mrcfiles::mrc_set_scale(
                &mut *hout, x as f64, y as f64, z as f64,
            );
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
            return -1;
        }
        0
    }
}
/// Matches C++ `clip_color`.
pub unsafe fn clip_color(hin: *mut MrcHeader, hout: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        const DEFAULT: f32 = crate::imod::clip::clip::IP_DEFAULT as f32;
        if (*opt).red == DEFAULT {
            (*opt).red = 1.;
        }
        if (*opt).green == DEFAULT {
            (*opt).green = 1.;
        }
        if (*opt).blue == DEFAULT {
            (*opt).blue = 1.;
        }
        if (*opt).dim == 2 {
            return clip2d_color(hin, hout, opt);
        }
        if [
            (*opt).ix,
            (*opt).iy,
            (*opt).iz,
            (*opt).ox,
            (*opt).oy,
            (*opt).oz,
        ]
        .iter()
        .any(|&v| v != crate::imod::clip::clip::IP_DEFAULT)
        {
            crate::imod::clip::clip::show_warning(
                "clip 3d color - input and output sizes ignored.",
            );
        }
        libc::printf(
            c"clip: color (red, green, blue) = ( %g, %g, %g).\n".as_ptr(),
            (*opt).red as f64,
            (*opt).green as f64,
            (*opt).blue as f64,
        );
        let data = crate::imod::libiimod::mrcfiles::mrc_read_byte(
            (*hin).fp.cast(),
            hin,
            core::ptr::null_mut(),
            None,
        );
        if data.is_null() {
            return -1;
        }
        (*hout).nx = (*hin).nx;
        (*hout).ny = (*hin).ny;
        (*hout).nz = (*hin).nz;
        (*hout).mode = 16;
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
            return -1;
        }
        for k in 0..(*hout).nz {
            for i in 0..(*hin).nx * (*hin).ny {
                let pixel = *(*data.add(k as usize)).add(i as usize) as f32;
                write_byte_pixel(pixel * (*opt).red, hout);
                write_byte_pixel(pixel * (*opt).green, hout);
                write_byte_pixel(pixel * (*opt).blue, hout);
            }
        }
        0
    }
}
/// Matches C++ `writeBytePixel`.
pub unsafe fn write_byte_pixel(mut pixel: f32, hout: *mut MrcHeader) {
    unsafe {
        if pixel > 255. {
            pixel = 255.;
        }
        let mut byte = (pixel + 0.5) as u8;
        if (*hout).bytes_signed != 0 {
            byte = ((byte as i32 - 128) & 255) as u8;
        }
        crate::imod::libcfshr::b3dutil::b3d_fwrite(
            (&byte as *const u8).cast(),
            1,
            1,
            (*hout).fp.cast(),
        );
    }
}
/// Matches C++ `clip2d_color`.
pub unsafe fn clip2d_color(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        if (*opt).mode != 16 && (*opt).mode != crate::imod::clip::clip::IP_DEFAULT {
            crate::imod::clip::clip::show_warning("clip - color output mode must be rgb.");
        }
        (*opt).mode = 16;
        let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
        if z < 0 {
            return z;
        }
        crate::imod::clip::clip::show_status("False Color...\n");
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"CLIP Color");
        let out = crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, 16);
        if out.is_null() {
            crate::imod::clip::clip::show_error("clip - creating slice");
            return -1;
        }
        for k in 0..(*opt).nofsecs {
            let source = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if source.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                slice_free(out);
                return -1;
            }
            for j in 0..(*opt).iy {
                for i in 0..(*opt).ix {
                    let mut value = [0.; 4];
                    slice_get_val(source, i, j, &mut value);
                    let pixel = value[0];
                    value[0] = (pixel * (*opt).red + 0.5).clamp(0., 255.);
                    value[1] = (pixel * (*opt).green + 0.5).clamp(0., 255.);
                    value[2] = (pixel * (*opt).blue + 0.5).clamp(0., 255.);
                    slice_put_val(out, i, j, value);
                }
            }
            if crate::imod::clip::file_io::clip_write_slice(out, hout, opt, k, &mut z, 0) != 0 {
                slice_free(source);
                slice_free(out);
                return -1;
            }
            slice_free(source);
        }
        slice_free(out);
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clip_joinrgb`.
pub unsafe fn clip_joinrgb(
    h1: *mut MrcHeader,
    h2: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_APPEND_OVERWRITE, IP_DEFAULT};
        if (*opt).red == IP_DEFAULT as f32 {
            (*opt).red = 1.;
        }
        if (*opt).green == IP_DEFAULT as f32 {
            (*opt).green = 1.;
        }
        if (*opt).blue == IP_DEFAULT as f32 {
            (*opt).blue = 1.;
        }
        if (*opt).infiles != 3 {
            libc::printf(c"ERROR: clip joinrgb - three input files must be specified\n".as_ptr());
            return -1;
        }
        let mut headers = [
            core::ptr::read(h1),
            core::ptr::read(h2),
            core::mem::zeroed::<MrcHeader>(),
        ];
        headers[2].fp =
            crate::imod::libiimod::iimage::ii_fopen(*(*opt).fnames.add(2), c"rb".as_ptr()).cast();
        if headers[2].fp.is_null() {
            libc::printf(
                c"ERROR: clip joinrgb - opening %s.\n".as_ptr(),
                *(*opt).fnames.add(2),
            );
            return -1;
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_read(headers[2].fp.cast(), &mut headers[2])
            != 0
        {
            libc::printf(
                c"ERROR: clip joinrgb - reading %s.\n".as_ptr(),
                *(*opt).fnames.add(2),
            );
            return -1;
        }
        if headers
            .iter()
            .any(|h| h.nx != (*h1).nx || h.ny != (*h1).ny || h.nz != (*h1).nz)
        {
            libc::printf(c"ERROR: clip joinrgb - all files must be same size.\n".as_ptr());
            crate::imod::libiimod::iimage::ii_fclose(headers[2].fp.cast());
            return -1;
        }
        if headers.iter().any(|h| h.mode != 0) {
            libc::printf(c"ERROR: clip joinrgb - all files must be bytes.\n".as_ptr());
            return -1;
        }
        let mut start = 0;
        if (*opt).add2file != 0 {
            if (*opt).add2file == IP_APPEND_OVERWRITE {
                libc::printf(
                    c"ERROR: clip joinrgb - Overwriting is not allowed, only appending\n".as_ptr(),
                );
                return -1;
            }
            if (*hout).mode != 16 {
                libc::printf(
                    c"ERROR: clip joinrgb - Mode of file being appended to must be 16\n".as_ptr(),
                );
                return -1;
            }
            if (*hout).nx != (*h1).nx || (*hout).ny != (*h1).ny {
                libc::printf(
                    c"ERROR: clip joinrgb - File being appended to is not same X/Y size as input files\n".as_ptr(),
                );
                return -1;
            }
            let (xs, ys, zs) = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hout);
            start = (*hout).nz;
            (*hout).nz += (*h1).nz;
            if (*hout).mz == start {
                (*hout).mz += (*h1).nz;
            }
            crate::imod::libiimod::mrcfiles::mrc_set_scale(
                &mut *hout, xs as f64, ys as f64, zs as f64,
            );
        } else {
            (*hout).mode = 16;
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut *hout,
                b"CLIP Join 3 files into RGB",
            );
        }
        (*hout).amin = 0.;
        (*hout).amax = 255.;
        (*hout).amean = 128.;
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
            return -1;
        }
        let rgb = crate::imod::libcfshr::islice::slice_create((*h1).nx, (*h1).ny, 16);
        let mut component = [core::ptr::null_mut(); 3];
        for p in &mut component {
            *p = crate::imod::libcfshr::islice::slice_create((*h1).nx, (*h1).ny, 0);
            if rgb.is_null() || (*p).is_null() {
                libc::printf(c"ERROR: CLIP - getting memory for slices\n".as_ptr());
                return -1;
            }
        }
        for k in 0..(*h1).nz {
            libc::printf(c"\rJoining section %d of %d".as_ptr(), k + 1, (*h1).nz);
            libc::fflush(core::ptr::null_mut());
            for n in 0..3 {
                if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                    (*component[n]).data.b.cast(),
                    headers[n].fp.cast(),
                    &mut headers[n],
                    k,
                    b'z' as i8,
                ) != 0
                {
                    return -1;
                }
            }
            for y in 0..(*h1).ny {
                for x in 0..(*h1).nx {
                    let mut value = [0.; 4];
                    for n in 0..3 {
                        let mut one = [0.; 4];
                        slice_get_val(component[n], x, y, &mut one);
                        value[n] = one[0] * [(*opt).red, (*opt).green, (*opt).blue][n];
                    }
                    slice_put_val(rgb, x, y, value);
                }
            }
            if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                (*rgb).data.b.cast(),
                (*hout).fp.cast(),
                hout,
                k + start,
                b'z' as i8,
            ) != 0
            {
                return -1;
            }
        }
        libc::printf(c"\n".as_ptr());
        for p in component {
            slice_free(p);
        }
        slice_free(rgb);
        crate::imod::libiimod::iimage::ii_fclose(headers[2].fp.cast());
        0
    }
}
/// Matches C++ `clip_splitrgb`.
pub unsafe fn clip_splitrgb(h1: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        if (*h1).mode != 16 {
            libc::printf(c"ERROR: clip splitrgb - mode is not RGB\n".as_ptr());
            return -1;
        }
        (*opt).ocanresize = 0;
        crate::imod::clip::file_io::set_multifile_input_options(opt, h1);
        let base =
            core::ffi::CStr::from_ptr(*(*opt).fnames.add((*opt).infiles as usize)).to_bytes();
        let mut headers: [MrcHeader; 3] = [
            core::mem::zeroed(),
            core::mem::zeroed(),
            core::mem::zeroed(),
        ];
        let rgb = crate::imod::libcfshr::islice::slice_create((*h1).nx, (*h1).ny, 16);
        let mut plane = [core::ptr::null_mut(); 3];
        if rgb.is_null() {
            libc::printf(c"ERROR: clip - getting memory for slices\n".as_ptr());
            return -1;
        }
        for n in 0..3 {
            let mut name = Vec::from(base);
            name.extend_from_slice([b".r", b".g", b".b"][n]);
            name.push(0);
            if libc::getenv(c"IMOD_NO_IMAGE_BACKUP".as_ptr()).is_null() {
                crate::imod::libcfshr::b3dutil::imod_backup_file(name.as_ptr().cast());
            }
            headers[n] = core::ptr::read(h1);
            headers[n].nz = (*opt).nofsecs * (*opt).infiles;
            headers[n].mz = headers[n].nz;
            crate::imod::libiimod::mrcfiles::mrc_coord_cp(&mut headers[n], &*h1);
            if headers[n].mz != 0 {
                headers[n].zorg -=
                    *(*opt).secs.add(0) as f32 * headers[n].zlen / headers[n].mz as f32;
            }
            headers[n].fp =
                crate::imod::libiimod::iimage::ii_fopen(name.as_ptr().cast(), c"wb+".as_ptr())
                    .cast();
            if headers[n].fp.is_null() {
                libc::printf(c"ERROR: clip - opening %s\n".as_ptr(), name.as_ptr());
                return -1;
            }
            headers[n].mode = 0;
            crate::imod::libiimod::mrcfiles::mrc_init_output_header(&mut headers[n]);
            headers[n].amin = 255.;
            headers[n].amax = 0.;
            headers[n].amean = 0.;
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut headers[n],
                b"CLIP Split RGB into 3 files",
            );
            if crate::imod::libiimod::mrcfiles::mrc_head_write(
                headers[n].fp.cast(),
                &mut headers[n],
            ) != 0
            {
                return -1;
            }
            plane[n] = crate::imod::libcfshr::islice::slice_create((*h1).nx, (*h1).ny, 0);
            if plane[n].is_null() {
                libc::printf(c"ERROR: clip - getting memory for slices\n".as_ptr());
                return -1;
            }
        }
        for file in 0..(*opt).infiles {
            let mut input = if file == 0 {
                core::ptr::read(h1)
            } else {
                core::mem::zeroed()
            };
            if file != 0 {
                input.fp = crate::imod::libiimod::iimage::ii_fopen(
                    *(*opt).fnames.add(file as usize),
                    c"rb".as_ptr(),
                )
                .cast();
                if input.fp.is_null()
                    || crate::imod::libiimod::mrcfiles::mrc_head_read(input.fp.cast(), &mut input)
                        != 0
                {
                    return -1;
                }
            }
            for k in 0..(*opt).nofsecs {
                libc::printf(
                    c"\rSplitting section %d of %d".as_ptr(),
                    k + 1,
                    (*opt).nofsecs,
                );
                if (*opt).infiles > 1 {
                    libc::printf(c", file %d".as_ptr(), file + 1);
                }
                libc::fflush(core::ptr::null_mut());
                if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                    (*rgb).data.b.cast(),
                    input.fp.cast(),
                    &mut input,
                    *(*opt).secs.add(k as usize),
                    b'z' as i8,
                ) != 0
                {
                    return -1;
                }
                for y in 0..(*h1).ny {
                    for x in 0..(*h1).nx {
                        let mut pixel = [0.; 4];
                        slice_get_val(rgb, x, y, &mut pixel);
                        for n in 0..3 {
                            slice_put_val(plane[n], x, y, [pixel[n], 0., 0., 0.]);
                        }
                    }
                }
                for n in 0..3 {
                    if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                        (*plane[n]).data.b.cast(),
                        headers[n].fp.cast(),
                        &mut headers[n],
                        file * (*opt).nofsecs + k,
                        b'z' as i8,
                    ) != 0
                    {
                        return -1;
                    }
                    crate::imod::libiimod::mrcslice::slice_mmm(plane[n]);
                    headers[n].amin = headers[n].amin.min((*plane[n]).min);
                    headers[n].amax = headers[n].amax.max((*plane[n]).max);
                    headers[n].amean += (*plane[n]).mean;
                }
            }
            if file != 0 {
                crate::imod::libiimod::iimage::ii_fclose(input.fp.cast());
            }
        }
        libc::printf(c"\n".as_ptr());
        for n in 0..3 {
            headers[n].amean /= (*opt).nofsecs as f32;
            if crate::imod::libiimod::mrcfiles::mrc_head_write(
                headers[n].fp.cast(),
                &mut headers[n],
            ) != 0
            {
                return -1;
            }
            slice_free(plane[n]);
            crate::imod::libiimod::iimage::ii_fclose(headers[n].fp.cast());
        }
        slice_free(rgb);
        0
    }
}
/// Matches C++ `clip_average`.
pub unsafe fn clip_average(
    h1: *mut MrcHeader,
    h2: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{
            IP_ADD, IP_AVERAGE, IP_DEFAULT, IP_STANDEV, IP_SUBTRACT, IP_VARIANCE,
        };
        if (*opt).infiles == 1 && matches!((*opt).process, IP_AVERAGE | IP_VARIANCE | IP_STANDEV) {
            return clip2d_average(h1, hout, opt);
        }
        if (*opt).add2file != 0 {
            crate::imod::clip::clip::show_error(
                "clip volume combining: you cannot add to an existing output file",
            );
            return -1;
        }
        if (*opt).infiles < 2 || ((*opt).process == IP_SUBTRACT && (*opt).infiles != 2) {
            crate::imod::clip::clip::show_error(if (*opt).process == IP_SUBTRACT {
                "clip subtract: needs exactly two input files."
            } else {
                "clip add: needs at least two input files."
            });
            return -1;
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, h1, hout);
        if z < 0 {
            return z;
        }
        let variance = match (*opt).process {
            IP_AVERAGE => 0,
            IP_ADD => 0,
            IP_VARIANCE => 1,
            IP_STANDEV => 2,
            IP_SUBTRACT => 0,
            _ => return -1,
        };
        let scale = if (*opt).low as i32 == IP_DEFAULT {
            1.
        } else {
            (*opt).low
        };
        let value_scale = if matches!((*opt).process, IP_AVERAGE | IP_VARIANCE | IP_STANDEV) {
            scale / (*opt).infiles as f32
        } else {
            scale
        };
        crate::imod::libiimod::mrcfiles::mrc_head_label(
            &mut *hout,
            match (*opt).process {
                IP_AVERAGE => b"clip: 3D Averaged",
                IP_ADD => b"clip: Summed",
                IP_VARIANCE => b"clip: 3D Variance",
                IP_STANDEV => b"clip: 3D Standard Deviation",
                _ => b"clip: Subtract",
            },
        );
        // h1/h2 are the dispatcher-opened first two files; subsequent input files follow
        // the exact C loop by being opened from fnames for each section.
        for k in 0..(*opt).nofsecs {
            libc::printf(
                c"\rclip: %s slice %d of %d\n".as_ptr(),
                match (*opt).process {
                    IP_ADD => c"Adding",
                    IP_SUBTRACT => c"Subtracting",
                    _ => c"Averaging",
                }
                .as_ptr(),
                k + 1,
                (*opt).nofsecs,
            );
            libc::fflush(core::ptr::null_mut());
            let slice_mode =
                if (*h1).mode == crate::imod::libiimod::mrcfiles::MRC_MODE_COMPLEX_FLOAT {
                    crate::imod::libiimod::mrcfiles::MRC_MODE_COMPLEX_FLOAT
                } else if (*h1).mode == crate::imod::libiimod::mrcfiles::MRC_MODE_RGB {
                    99
                } else {
                    crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT
                };
            let out = crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, slice_mode);
            let sq = if variance != 0 {
                crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, slice_mode)
            } else {
                core::ptr::null_mut()
            };
            if out.is_null() || variance != 0 && sq.is_null() {
                return -1;
            }
            let val = [0.; 4];
            for j in 0..(*opt).iy {
                for i in 0..(*opt).ix {
                    slice_put_val(out, i, j, val);
                    if variance != 0 {
                        slice_put_val(sq, i, j, val);
                    }
                }
            }
            for file in 0..(*opt).infiles {
                let (mut hdr, close) = if file == 0 {
                    (h1, false)
                } else if file == 1 {
                    (h2, false)
                } else {
                    let h = libc::calloc(1, core::mem::size_of::<MrcHeader>()).cast::<MrcHeader>();
                    if h.is_null() {
                        return -1;
                    }
                    (*h).fp = crate::imod::libiimod::iimage::ii_fopen(
                        *(*opt).fnames.add(file as usize),
                        c"rb".as_ptr(),
                    )
                    .cast();
                    if (*h).fp.is_null()
                        || crate::imod::libiimod::mrcfiles::mrc_head_read((*h).fp.cast(), h) != 0
                    {
                        return -1;
                    }
                    (h, true)
                };
                if (*hdr).nx != (*h1).nx
                    || (*hdr).ny != (*h1).ny
                    || (*hdr).nz != (*h1).nz
                    || (*hdr).mode != (*h1).mode
                {
                    crate::imod::clip::clip::show_error(
                        "clip volume combining: all files must be the same size and mode.",
                    );
                    return -1;
                }
                let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                    hdr,
                    *(*opt).secs.add(k as usize),
                    b'z' as i8,
                    (*opt).ix,
                    (*opt).iy,
                    (*opt).cx as i32,
                    (*opt).cy as i32,
                );
                if s.is_null() {
                    crate::imod::clip::clip::show_error("clip: Error reading slice.");
                    return -1;
                }
                let sign = if (*opt).process == IP_SUBTRACT && file > 0 {
                    -1.
                } else {
                    1.
                };
                for y in 0..(*opt).iy {
                    for x in 0..(*opt).ix {
                        let (mut a, mut v) = ([0.; 4], [0.; 4]);
                        slice_get_val(out, x, y, &mut a);
                        slice_get_val(s, x, y, &mut v);
                        for n in 0..3 {
                            a[n] += sign * v[n];
                        }
                        slice_put_val(out, x, y, a);
                        if variance != 0 {
                            slice_get_val(sq, x, y, &mut a);
                            a[0] += v[0] * v[0];
                            a[1] += v[1] * v[1];
                            a[1] += v[2] * v[2];
                            slice_put_val(sq, x, y, a);
                        }
                    }
                }
                slice_free(s);
                if close {
                    crate::imod::libiimod::iimage::ii_fclose((*hdr).fp.cast());
                    libc::free(hdr.cast());
                }
            }
            for y in 0..(*opt).iy {
                for x in 0..(*opt).ix {
                    let mut a = [0.; 4];
                    slice_get_val(out, x, y, &mut a);
                    for n in 0..3 {
                        a[n] *= value_scale;
                    }
                    if variance != 0 {
                        let mut q = [0.; 4];
                        slice_get_val(sq, x, y, &mut q);
                        for n in 0..3 {
                            a[n] = ((q[n] * scale * scale - (*opt).infiles as f32 * a[n] * a[n])
                                / ((*opt).infiles - 1) as f32)
                                .max(0.);
                            if variance == 2 {
                                a[n] = a[n].sqrt();
                            }
                        }
                    }
                    slice_put_val(out, x, y, a);
                }
            }
            if crate::imod::clip::file_io::clip_write_slice(out, hout, opt, k, &mut z, 1) != 0 {
                return -1;
            }
            if !sq.is_null() {
                slice_free(sq);
            }
        }
        libc::printf(c"\n".as_ptr());
        (*hout).amean /= (*opt).nofsecs as f32;
        crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout);
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clip2d_average`.
pub unsafe fn clip2d_average(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_AVERAGE, IP_DEFAULT, IP_STANDEV, IP_VARIANCE};
        if (*opt).ox != IP_DEFAULT || (*opt).oy != IP_DEFAULT || (*opt).oz != IP_DEFAULT {
            crate::imod::clip::clip::show_warning(
                "clip - ox, oy, oz have no effect for 2d average.",
            );
        }
        (*opt).ox = IP_DEFAULT;
        (*opt).oy = IP_DEFAULT;
        (*opt).oz = IP_DEFAULT;
        crate::imod::clip::file_io::set_input_options(opt, hin);
        (*opt).oz = 1;
        let z = crate::imod::clip::file_io::set_output_options(opt, hout);
        if z < 0 {
            return z;
        }
        crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
        let variance = match (*opt).process {
            IP_AVERAGE => 0,
            IP_VARIANCE => 1,
            IP_STANDEV => 2,
            _ => return -1,
        };
        crate::imod::libiimod::mrcfiles::mrc_head_label(
            &mut *hout,
            match variance {
                0 => b"clip: 2D Average",
                1 => b"clip: 2D Variance",
                _ => b"clip: 2D Standard Deviation",
            },
        );
        crate::imod::clip::clip::show_status("2D Averaging...\n");
        let mode = if (*hin).mode == 16 { 16 } else { 2 };
        let avgs = crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, mode);
        let mut counts = vec![0_f32; ((*opt).ix * (*opt).iy) as usize];
        let squares = if variance != 0 {
            crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, mode)
        } else {
            core::ptr::null_mut()
        };
        if avgs.is_null() || (variance != 0 && squares.is_null()) {
            crate::imod::clip::clip::show_error("CLIP - Memory error");
            return -1;
        }
        let aval = [0.; 4];
        for j in 0..(*avgs).ysize {
            for i in 0..(*avgs).xsize {
                slice_put_val(avgs, i, j, aval);
                if variance != 0 {
                    slice_put_val(squares, i, j, aval);
                }
            }
        }
        for k in 0..(*opt).nofsecs {
            let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if s.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            for y in 0..(*opt).iy {
                for x in 0..(*opt).ix {
                    if (*opt).val as i32 != IP_DEFAULT
                        && slice_get_pixel_magnitude(s, x, y) <= (*opt).val
                    {
                        continue;
                    }
                    let ind = (x + y * (*opt).ix) as usize;
                    let (mut a, mut v) = ([0.; 4], [0.; 4]);
                    slice_get_val(avgs, x, y, &mut a);
                    slice_get_val(s, x, y, &mut v);
                    for n in 0..3 {
                        a[n] += v[n];
                    }
                    slice_put_val(avgs, x, y, a);
                    if variance != 0 {
                        slice_get_val(squares, x, y, &mut a);
                        for n in 0..3 {
                            a[n] += v[n] * v[n];
                        }
                        slice_put_val(squares, x, y, a);
                    }
                    counts[ind] += 1.;
                }
            }
            slice_free(s);
        }
        let scale = if (*opt).low as i32 == IP_DEFAULT {
            1.
        } else {
            (*opt).low
        };
        for y in 0..(*opt).iy {
            for x in 0..(*opt).ix {
                let count = counts[(x + y * (*opt).ix) as usize];
                let mut a = [0.; 4];
                slice_get_val(avgs, x, y, &mut a);
                if count > 0. {
                    for n in 0..3 {
                        a[n] *= scale / count;
                    }
                    if variance != 0 {
                        if count > 1. {
                            let mut ss = [0.; 4];
                            slice_get_val(squares, x, y, &mut ss);
                            for n in 0..3 {
                                a[n] = ((ss[n] * scale * scale - count * a[n] * a[n])
                                    / (count - 1.))
                                    .max(0.);
                                if variance == 2 {
                                    a[n] = a[n].sqrt();
                                }
                            }
                        } else {
                            a = [0.; 4];
                        }
                    }
                } else {
                    a = [0.; 4];
                }
                slice_put_val(avgs, x, y, a);
            }
        }
        if (*avgs).mode != (*hout).mode
            && crate::imod::libiimod::mrcslice::slice_new_mode(avgs, (*hout).mode) < 0
        {
            crate::imod::clip::clip::show_error("CLIP - getting memory for slice array");
            return -1;
        }
        crate::imod::libcfshr::islice::slice_min_max(avgs);
        (*hout).amin = (*hout).amin.min((*avgs).min);
        (*hout).amax = (*hout).amax.max((*avgs).max);
        if (*opt).add2file != 1 {
            (*hout).amean += (*avgs).mean / (*hout).nz as f32;
        }
        if crate::imod::libiimod::mrcfiles::mrc_write_slice(
            (*avgs).data.b.cast(),
            (*hout).fp.cast(),
            hout,
            z,
            b'z' as i8,
        ) != 0
        {
            return -1;
        }
        let ret = crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout);
        slice_free(avgs);
        if !squares.is_null() {
            slice_free(squares);
        }
        ret
    }
}
/// Matches C++ `clip_multdiv`.
pub unsafe fn clip_multdiv(
    h1: *mut MrcHeader,
    h2: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_APPEND_FALSE, IP_DEFAULT, IP_DIVIDE, IP_MULTIPLY};
        use crate::imod::libiimod::mrcfiles::{
            MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, mrc_getdcsize,
        };
        if (*opt).infiles != 2 {
            crate::imod::clip::clip::show_error(
                "clip multiply/divide: Need exactly two input files.",
            );
            return -1;
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, h1, hout);
        if z < 0 {
            return z;
        }
        if (*opt).add2file != IP_APPEND_FALSE {
            crate::imod::clip::clip::show_error(
                "clip multiply/divide: you cannot add to an existing output file",
            );
            return -1;
        }
        let (mut dsize, mut csize1, mut csize2) = (0, 0, 0);
        mrc_getdcsize((*h1).mode, &mut dsize, &mut csize1);
        mrc_getdcsize((*h2).mode, &mut dsize, &mut csize2);
        if !(csize2 == 1
            || ((*h1).mode == MRC_MODE_COMPLEX_FLOAT && (*h2).mode == MRC_MODE_COMPLEX_FLOAT)
            || (*hout).mode == MRC_MODE_FLOAT)
        {
            crate::imod::clip::clip::show_error(
                "clip multiply/divide: second file must have single-channel data unless both are FFTs or output mode is float",
            );
            return -1;
        }
        if (*h1).nx != (*h2).nx || (*h1).ny != (*h2).ny {
            crate::imod::clip::clip::show_error(
                "clip  multiply/divide: X and Y sizes must be equal",
            );
            return -1;
        }
        if (*h1).nz != (*h2).nz && (*h2).nz > 1 {
            crate::imod::clip::clip::show_error(
                "clip  multiply/divide: Z sizes must be the same, or equal to 1 for second file",
            );
            return -1;
        }
        let scale = if (*opt).val as i32 == IP_DEFAULT {
            1.
        } else {
            (*opt).val
        };
        let (message, title_proc) = match (*opt).process {
            IP_MULTIPLY => (c"Multiplying", "Multiply"),
            IP_DIVIDE => (c"Dividing", "Divide"),
            _ => return -1,
        };
        let title = if (*opt).val as i32 != IP_DEFAULT {
            format!("clip: {title_proc}, scaled by {scale:.2}")
        } else {
            format!("clip: {title_proc}")
        };
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, title.as_bytes());
        let do_round = csize1 * csize2 == 1
            && (*hout).mode != MRC_MODE_FLOAT
            && ((*h1).mode == MRC_MODE_FLOAT
                || (*h2).mode == MRC_MODE_FLOAT
                || (*opt).process == IP_DIVIDE);
        let mut read_once = if (*h2).nz == 1 && (*h1).nz > 1 { 1 } else { 0 };
        let mut s: *mut Islice = core::ptr::null_mut();
        let mut div_by_zero = 0;
        for k in 0..(*opt).nofsecs {
            libc::printf(
                c"\rclip: %s slice %d of %d".as_ptr(),
                message.as_ptr(),
                k + 1,
                (*opt).nofsecs,
            );
            libc::fflush(core::ptr::null_mut());
            let out = crate::imod::libiimod::mrcslice::slice_read_subm(
                h1,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if out.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            if (*out).mode != (*hout).mode
                && crate::imod::libiimod::mrcslice::slice_new_mode(out, (*hout).mode) < 0
            {
                crate::imod::clip::clip::show_error("CLIP - getting memory for slice array");
                return -1;
            }
            if read_once >= 0 {
                s = crate::imod::libiimod::mrcslice::slice_read_subm(
                    h2,
                    if read_once != 0 {
                        0
                    } else {
                        *(*opt).secs.add(k as usize)
                    },
                    b'z' as i8,
                    (*opt).ix,
                    (*opt).iy,
                    (*opt).cx as i32,
                    (*opt).cy as i32,
                );
                if s.is_null() {
                    crate::imod::clip::clip::show_error("clip: Error reading slice.");
                    return -1;
                }
                read_once = -read_once;
                if csize2 > 1
                    && (*hout).mode == MRC_MODE_FLOAT
                    && crate::imod::libiimod::mrcslice::slice_float(s) != 0
                {
                    libc::printf(c"ERROR: CLIP - getting memory for slice array\n".as_ptr());
                    return -1;
                }
            }
            for y in 0..(*opt).iy {
                for x in 0..(*opt).ix {
                    let (mut a, mut b) = ([0.; 4], [0.; 4]);
                    slice_get_val(out, x, y, &mut a);
                    slice_get_val(s, x, y, &mut b);
                    if do_round {
                        if (*opt).process == IP_MULTIPLY {
                            a[0] = (a[0] * b[0] * scale).round();
                        } else if b[0] != 0. {
                            a[0] = (a[0] * (scale / b[0])).round();
                        } else {
                            div_by_zero += 1;
                            a[0] = 0.;
                        }
                    } else if csize2 == 1 || (*hout).mode == MRC_MODE_FLOAT {
                        if (*opt).process == IP_MULTIPLY {
                            let scaled_val = b[0] * scale;
                            a[0] *= scaled_val;
                            a[1] *= scaled_val;
                            a[2] *= scaled_val;
                        } else if b[0] != 0. {
                            let scaled_val = scale / b[0];
                            a[0] *= scaled_val;
                            a[1] *= scaled_val;
                            a[2] *= scaled_val;
                        } else {
                            div_by_zero += 1;
                            a[0] = 0.;
                            a[1] = 0.;
                            a[2] = 0.;
                        }
                    } else {
                        if (*opt).process == IP_MULTIPLY {
                            let re = (a[0] * b[0] - a[1] * b[1]) * scale;
                            a[1] = (a[0] * b[1] + b[0] * a[1]) * scale;
                            a[0] = re;
                        } else {
                            let den = b[0] * b[0] + b[1] * b[1];
                            if den != 0. {
                                let re = (a[0] * b[0] + a[1] * b[1]) * scale / den;
                                a[1] = (a[1] * b[0] - a[0] * b[1]) * scale / den;
                                a[0] = re;
                            } else {
                                a = [0.; 4];
                                div_by_zero += 1;
                            }
                        }
                    }
                    slice_put_val(out, x, y, a);
                }
            }
            if (*opt).read_defects != 0 && correct_defects(out, (*h2).nx, (*h2).ny, opt) != 0 {
                return -1;
            }
            if crate::imod::clip::file_io::clip_write_slice(out, hout, opt, k, &mut z, 1) != 0 {
                return -1;
            }
            if read_once == 0 || k == (*opt).nofsecs - 1 {
                slice_free(s);
            }
        }
        libc::printf(c"\n".as_ptr());
        if div_by_zero > 0 {
            libc::printf(
                c"WARNING: Division by zero occurred %d times\n".as_ptr(),
                div_by_zero,
            );
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
            return -1;
        }
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `clipPlanarFit`.
pub unsafe fn clip_planar_fit(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_DEFAULT, IP_FLATFIELD, IP_PLANARFIT};
        crate::imod::clip::file_io::set_input_options(opt, hin);
        let fitting = (*opt).process == IP_PLANARFIT || (*opt).val != IP_DEFAULT as f32;
        let mut order = 0_i32;
        if (*opt).process == IP_FLATFIELD {
            if fitting {
                order = ((*opt).val as i32).clamp(1, 4);
            }
            (*opt).oz = 1;
            let e = crate::imod::clip::file_io::set_output_options(opt, hout);
            if e < 0 {
                return e;
            }
            if fitting {
                let label = format!("clip: Flatfield based on order {order} fit to image sum");
                crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, label.as_bytes());
            } else {
                crate::imod::libiimod::mrcfiles::mrc_head_label(
                    &mut *hout,
                    b"clip: Flatfield based on image sum",
                );
            }
        }
        let n = ((*opt).ix * (*opt).iy) as usize;
        let mut sum = vec![0_f32; n];
        let base = if (*opt).low as i32 == IP_DEFAULT {
            0.
        } else {
            (*opt).low
        };
        let mut count = 0;
        let prefix = if (*opt).process == IP_PLANARFIT {
            "Doing plane fit:"
        } else {
            "Making flatfield:"
        };
        libc::printf(c"clip: summing slices...".as_ptr());
        libc::fflush(core::ptr::null_mut());
        // C `clipPlanarFit` opens each named input independently: the header
        // handed to sliceReadSubm must be that file's header, not `hin`.
        for f in 0..(*opt).infiles {
            let mut hdr: MrcHeader = core::mem::zeroed();
            let input_header = if f == 0 {
                hin
            } else {
                hdr.fp = crate::imod::libiimod::iimage::ii_fopen(
                    *(*opt).fnames.add(f as usize),
                    c"rb".as_ptr(),
                )
                .cast();
                if hdr.fp.is_null()
                    || crate::imod::libiimod::mrcfiles::mrc_head_read(hdr.fp.cast(), &mut hdr) != 0
                {
                    if !hdr.fp.is_null() {
                        crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
                    }
                    crate::imod::clip::clip::show_error(&format!(
                        "\n{prefix} error opening or reading header of {}.",
                        core::ffi::CStr::from_ptr(*(*opt).fnames.add(f as usize)).to_string_lossy()
                    ));
                    return -1;
                }
                if (*hin).nx != hdr.nx || (*hin).ny != hdr.ny {
                    crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
                    crate::imod::clip::clip::show_error(&format!(
                        "\n{prefix} files must be same size in X and Y; {} differs",
                        core::ffi::CStr::from_ptr(*(*opt).fnames.add(f as usize)).to_string_lossy()
                    ));
                    return -1;
                }
                &mut hdr as *mut MrcHeader
            };
            for k in 0..(*opt).nofsecs {
                if *(*opt).secs.add(k as usize) >= (*input_header).nz {
                    continue;
                }
                let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                    input_header,
                    *(*opt).secs.add(k as usize),
                    b'z' as i8,
                    (*opt).ix,
                    (*opt).iy,
                    (*opt).cx as i32,
                    (*opt).cy as i32,
                );
                if s.is_null() {
                    if f != 0 {
                        crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
                    }
                    crate::imod::clip::clip::show_error(&format!(
                        "\n{prefix} reading slice {} of {}",
                        *(*opt).secs.add(k as usize),
                        core::ffi::CStr::from_ptr(*(*opt).fnames.add(f as usize)).to_string_lossy()
                    ));
                    return -1;
                }
                for y in 0..(*opt).iy {
                    for x in 0..(*opt).ix {
                        sum[(x + y * (*opt).ix) as usize] +=
                            slice_get_pixel_magnitude(s, x, y) - base;
                    }
                }
                count += 1;
                slice_free(s);
            }
            if f != 0 {
                crate::imod::libiimod::iimage::ii_fclose(hdr.fp.cast());
            }
        }
        libc::printf(c"\n".as_ptr());
        if count == 0 {
            return -1;
        }
        let nx = (*opt).ix as f64;
        let ny = (*opt).iy as f64;
        let mut sx = 0.;
        let mut sy = 0.;
        let mut sxx = 0.;
        let mut syy = 0.;
        let mut sz = 0.;
        let mut sxz = 0.;
        let mut syz = 0.;
        for y in 0..(*opt).iy {
            for x in 0..(*opt).ix {
                let xx = x as f64 - (nx - 1.) / 2.;
                let yy = y as f64 - (ny - 1.) / 2.;
                let v = sum[(x + y * (*opt).ix) as usize] as f64;
                sx += xx;
                sy += yy;
                sxx += xx * xx;
                syy += yy * yy;
                sz += v;
                sxz += xx * v;
                syz += yy * v;
            }
        }
        // C normalizes the binned sum to its center before `lsFit2`; a plane
        // through the normalized, zero-centered bins has no constant term.
        let (aa, bb, plane_rms) = if fitting {
            let mut nx_trim = (0.005 * (*hin).nx as f32) as i32;
            let nx_in = ((*hin).nx - 2 * nx_trim).min((*opt).ix);
            nx_trim = 0.max(((*hin).nx - nx_in) / 2);
            let mut ny_trim = (0.005 * (*hin).ny as f32) as i32;
            let ny_in = ((*hin).ny - 2 * ny_trim).min((*opt).iy);
            ny_trim = 0.max(((*hin).ny - ny_in) / 2);
            let ind = if (*opt).process == IP_PLANARFIT {
                11
            } else {
                15
            };
            let x_binning = 1.max(nx_in / ind);
            let y_binning = 1.max(ny_in / ind);
            let nx_bin = nx_in / x_binning;
            let ny_bin = ny_in / y_binning;
            let mut bin_sum = vec![0_f64; (nx_bin * ny_bin) as usize];
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    for by in 0..y_binning {
                        for bx in 0..x_binning {
                            bin_sum[(ix + iy * nx_bin) as usize] +=
                                sum[((nx_trim + ix * x_binning + bx)
                                    + (*opt).ix * (ny_trim + iy * y_binning + by))
                                    as usize] as f64;
                        }
                    }
                }
            }
            let center = bin_sum[(nx_bin * (ny_bin / 2) + nx_bin / 2) as usize];
            if center == 0. {
                return -1;
            }
            if bin_sum.iter().any(|value| *value <= 0.) {
                crate::imod::clip::clip::show_warning(
                    "Some binned values are negative; you must set a base value to subtract with the -l option",
                );
            }
            let mut fit_sxx = 0.;
            let mut fit_syy = 0.;
            let mut fit_sxz = 0.;
            let mut fit_syz = 0.;
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    let k = (ix + iy * nx_bin) as usize;
                    let xx = x_binning as f64 * (ix as f64 + 0.5 - nx_bin as f64 / 2.);
                    let yy = y_binning as f64 * (iy as f64 + 0.5 - ny_bin as f64 / 2.);
                    let value = bin_sum[k] / center - 1.;
                    fit_sxx += xx * xx;
                    fit_syy += yy * yy;
                    fit_sxz += xx * value;
                    fit_syz += yy * value;
                }
            }
            let aa = fit_sxz / fit_sxx;
            let bb = fit_syz / fit_syy;
            let mut residual_sum = 0.;
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    let k = (ix + iy * nx_bin) as usize;
                    let xx = x_binning as f64 * (ix as f64 + 0.5 - nx_bin as f64 / 2.);
                    let yy = y_binning as f64 * (iy as f64 + 0.5 - ny_bin as f64 / 2.);
                    let residual = 100. * (bin_sum[k] / center - 1. - (aa * xx + bb * yy));
                    residual_sum += residual * residual;
                }
            }
            (aa, bb, (residual_sum / (nx_bin * ny_bin) as f64).sqrt())
        } else {
            (sxz / sxx, syz / syy, 0.)
        };
        if (*opt).process == IP_PLANARFIT {
            let text = format!("{aa:.8}  {bb:.8}\n");
            crate::imod::libcfshr::b3dutil::b3d_fwrite(
                text.as_ptr().cast(),
                1,
                text.len(),
                (*hout).fp.cast(),
            );
            libc::printf(
                c"Plane slopes imply a gradient over full extent in X and Y of %.3f and %.3f\n"
                    .as_ptr(),
                100. * aa * (*hin).nx as f64,
                100. * bb * (*hin).ny as f64,
            );
            libc::printf(c"Root-mean-squared residual = %.3f\n".as_ptr(), plane_rms);
            return 0;
        }
        if fitting && (*opt).process == IP_FLATFIELD {
            // C fills the same column-major `xMat[col][row]` used by
            // multRegress, then evaluates its polynomial inverse over the
            // full output.  Rebuild the binned normalized image here because
            // the plane-only `lsFit2` path above deliberately has no
            // constant column.
            let mut nx_trim = (0.005 * (*hin).nx as f32) as i32;
            let nx_in = ((*hin).nx - 2 * nx_trim).min((*opt).ix);
            nx_trim = 0.max(((*hin).nx - nx_in) / 2);
            let mut ny_trim = (0.005 * (*hin).ny as f32) as i32;
            let ny_in = ((*hin).ny - 2 * ny_trim).min((*opt).iy);
            ny_trim = 0.max(((*hin).ny - ny_in) / 2);
            let x_binning = 1.max(nx_in / 15);
            let y_binning = 1.max(ny_in / 15);
            let nx_bin = nx_in / x_binning;
            let ny_bin = ny_in / y_binning;
            let dim = 15 * 15 + 10;
            let col_dim = 18;
            let mut bin_sum = vec![0_f32; (nx_bin * ny_bin) as usize];
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    for by in 0..y_binning {
                        for bx in 0..x_binning {
                            bin_sum[(ix + iy * nx_bin) as usize] +=
                                sum[((nx_trim + ix * x_binning + bx)
                                    + (*opt).ix * (ny_trim + iy * y_binning + by))
                                    as usize];
                        }
                    }
                }
            }
            let center = bin_sum[(nx_bin * (ny_bin / 2) + nx_bin / 2) as usize];
            if center == 0. {
                return -1;
            }
            if bin_sum.iter().any(|value| *value <= 0.) {
                crate::imod::clip::clip::show_warning(
                    "Some binned values are negative; you must set a base value to subtract with the -l option",
                );
            }
            for value in &mut bin_sum {
                *value = *value / center - 1.;
            }
            let mut x_mat = vec![0_f32; (col_dim * dim) as usize];
            let mut sol = [0_f32; 18];
            let mut x_mean = [0_f32; 18];
            let mut x_sd = [0_f32; 18];
            let mut work = [0_f32; 18 * 18];
            let mut cons = 0_f32;
            let mut num_col = 0_i32;
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    let k = ix + iy * nx_bin;
                    let xx = x_binning as f32 * (ix as f32 + 0.5 - nx_bin as f32 / 2.);
                    let yy = y_binning as f32 * (iy as f32 + 0.5 - ny_bin as f32 / 2.);
                    let mut col = 0_i32;
                    for ind in 1..=order {
                        for py in 0..=ind {
                            let px = ind - py;
                            x_mat[(col * dim + k) as usize] = xx.powi(px) * yy.powi(py);
                            col += 1;
                        }
                    }
                    x_mat[(col * dim + k) as usize] = bin_sum[k as usize];
                    num_col = col;
                }
            }
            if crate::imod::libcfshr::regression::mult_regress(
                x_mat.as_mut_ptr(),
                dim,
                0,
                num_col,
                nx_bin * ny_bin,
                1,
                0,
                sol.as_mut_ptr(),
                col_dim,
                &mut cons,
                x_mean.as_mut_ptr(),
                x_sd.as_mut_ptr(),
                work.as_mut_ptr(),
            ) != 0
            {
                return -1;
            }
            let x_center = (*hin).nx as f32 / 2. - 0.5;
            let y_center = (*hin).ny as f32 / 2. - 0.5;
            libc::printf(
                c"Constant term %.6f\nX & Y order and coefficients (times half-size in X/Y to respective powers):\n"
                    .as_ptr(),
                cons as f64,
            );
            let mut col = 0_usize;
            for ind in 1..=order {
                for py in 0..=ind {
                    let px = ind - py;
                    libc::printf(
                        c"%d  %d  %.9f\n".as_ptr(),
                        px,
                        py,
                        (sol[col] * x_center.powi(px) * y_center.powi(py)) as f64,
                    );
                    col += 1;
                }
            }
            let mut residual_sum = 0_f32;
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    let k = (ix + iy * nx_bin) as usize;
                    let xx = x_binning as f32 * (ix as f32 + 0.5 - nx_bin as f32 / 2.);
                    let yy = y_binning as f32 * (iy as f32 + 0.5 - ny_bin as f32 / 2.);
                    let mut residual = bin_sum[k] - cons;
                    let mut col = 0_usize;
                    for ind in 1..=order {
                        for py in 0..=ind {
                            residual -= sol[col] * xx.powi(ind - py) * yy.powi(py);
                            col += 1;
                        }
                    }
                    residual_sum += 10_000. * residual * residual;
                }
            }
            libc::printf(
                c"Root-mean-squared residual = %.3f\n".as_ptr(),
                (residual_sum / (nx_bin * ny_bin) as f32).sqrt() as f64,
            );
            for iy in 0..(*hin).ny {
                for ix in 0..(*hin).nx {
                    let mut residual = 1. + cons;
                    let mut col = 0_usize;
                    for ind in 1..=order {
                        for py in 0..=ind {
                            residual += sol[col]
                                * (ix as f32 - x_center).powi(ind - py)
                                * (iy as f32 - y_center).powi(py);
                            col += 1;
                        }
                    }
                    sum[(ix + (*hin).nx * iy) as usize] = 1. / residual;
                }
            }
        } else if fitting {
            for y in 0..(*opt).iy {
                for x in 0..(*opt).ix {
                    let xx = x as f64 - (nx - 1.) / 2.;
                    let yy = y as f64 - (ny - 1.) / 2.;
                    sum[(x + y * (*opt).ix) as usize] = (1. / (1. + aa * xx + bb * yy)) as f32;
                }
            }
        } else {
            let mut minimum = f32::INFINITY;
            let mut maximum = f32::NEG_INFINITY;
            for value in &sum {
                minimum = minimum.min(*value);
                maximum = maximum.max(*value);
            }
            let mean = sz / n as f64;
            libc::printf(
                c"Averaged image min = %.5g, max = %.5g, mean = %.5g\n".as_ptr(),
                minimum as f64 / count as f64,
                maximum as f64 / count as f64,
                mean / count as f64,
            );
            if minimum < 0. {
                crate::imod::clip::clip::show_warning(
                    "Some summed values are negative; you must set a base value to subtract with the -l option",
                );
            }
            for value in &mut sum {
                *value = (mean / (0.05 * mean).max(*value as f64)) as f32;
            }
        }
        if (*opt).process == IP_FLATFIELD {
            let out = crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, 2);
            if out.is_null() {
                return -1;
            }
            core::ptr::copy_nonoverlapping(sum.as_ptr(), (*out).data.f, n);
            crate::imod::libcfshr::islice::slice_min_max(out);
            (*hout).amin = (*out).min;
            (*hout).amax = (*out).max;
            (*hout).amean = (*out).mean;
            if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0 {
                return -1;
            }
            let e = crate::imod::libiimod::mrcfiles::mrc_write_slice(
                (*out).data.b.cast(),
                (*hout).fp.cast(),
                hout,
                0,
                b'z' as i8,
            );
            slice_free(out);
            return e;
        }
        0
    }
}
/// Matches C++ `clipUnpack`.
pub unsafe fn clip_unpack(
    hin1: *mut MrcHeader,
    hin2: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        use crate::imod::clip::clip::{IP_APPEND_FALSE, IP_DEFAULT, IP_UNPACK};
        use crate::imod::libiimod::mrcfiles::{MRC_MODE_FLOAT, mrc_head_label};
        let mut do_ref = (*opt).infiles == 2;
        if (*opt).add2file != IP_APPEND_FALSE {
            crate::imod::clip::clip::show_error(
                "clip unpack/normalize - you cannot add to an existing output file",
            );
            return -1;
        }
        if do_ref && (*hin2).mode != MRC_MODE_FLOAT {
            crate::imod::clip::clip::show_error(
                "clip unpack/normalize - mode of second input file must be floats",
            );
            return -1;
        }
        if (*opt).infiles > 2 {
            crate::imod::clip::clip::show_error(
                "clip unpack/normalize - There can be only 2 input files",
            );
            return -1;
        }
        let mut z = crate::imod::clip::file_io::set_options(opt, hin1, hout);
        if z < 0 {
            return z;
        }
        let image_file = crate::imod::libiimod::iimage::ii_lookup_file_from_fp((*hin1).fp.cast());
        let is_eer = !image_file.is_null()
            && (*image_file).file == crate::imod::libiimod::iimage::IIFILE_TIFF
            && (*image_file).num_frames_in_eerfile > 0;
        let antialias_eer = is_eer && (*image_file).antialias_eerfilter > 0;
        let red_factor = if antialias_eer {
            1_i32 << (2 - (*image_file).read_eer_as_super_res).max(0)
        } else {
            1
        };
        let mut scale = if do_ref { 16. } else { 1. };
        if antialias_eer {
            scale = 100.;
        }
        if (*opt).val as i32 != IP_DEFAULT {
            scale = (*opt).val;
        }
        let mut reference = core::ptr::null_mut();
        if do_ref {
            reference = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin2,
                0,
                b'z' as i8,
                (*hin2).nx,
                (*hin2).ny,
                (*hin2).nx / 2,
                (*hin2).ny / 2,
            );
            if reference.is_null() {
                return -1;
            }
            let mut nx = (*hin2).nx;
            let mut ny = (*hin2).ny;
            if (*opt).rotation_flip < 0 {
                for label_index in 0..(*hin1).nlabl {
                    let label = core::ffi::CStr::from_ptr(
                        (*hin1).labels[label_index as usize].as_ptr().cast(),
                    );
                    let bytes = label.to_bytes();
                    if let Some(index) = bytes.windows(5).position(|text| text == b" r/f ") {
                        (*opt).rotation_flip = libc::atoi(bytes[index + 5..].as_ptr().cast());
                        break;
                    }
                }
                if (*opt).rotation_flip < 0 {
                    crate::imod::clip::clip::show_error(
                        "Cannot find r/f entry in header of first input file",
                    );
                    return -1;
                }
            }
            if (*opt).rotation_flip > 7 {
                crate::imod::clip::clip::show_error(&format!(
                    "Rotation/flip value of {} is out of range",
                    (*opt).rotation_flip
                ));
                return -1;
            }
            if (*opt).rotation_flip > 0
                && rotate_flip_gain_reference(
                    (*reference).data.f,
                    &mut nx,
                    &mut ny,
                    (*opt).rotation_flip,
                ) != 0
            {
                crate::imod::clip::clip::show_error(
                    "Error allocating memory for rotating gain reference",
                );
                return -1;
            }
            (*reference).xsize = nx;
            (*reference).ysize = ny;
            let super_fac = if nx > 0 { (*hin1).nx / nx } else { 0 };
            let use_fac = if antialias_eer { 4 } else { super_fac };
            if ((*hin1).ny / (*hin2).ny == super_fac
                && super_fac * (*hin2).nx == (*hin1).nx
                && super_fac * (*hin2).ny == (*hin1).ny
                && matches!(super_fac, 1 | 2 | 4))
                || antialias_eer
            {
                if super_fac > 1 || antialias_eer {
                    let expanded = crate::imod::libcfshr::islice::slice_create(
                        (*hin1).nx * red_factor,
                        (*hin1).ny * red_factor,
                        2,
                    );
                    if expanded.is_null() {
                        slice_free(reference);
                        crate::imod::clip::clip::show_error(
                            "Error allocating memory for expanded gain reference",
                        );
                        return -1;
                    }
                    crate::imod::clip::correct_defects::cor_def_expand_gain_reference(
                        (*reference).data.f,
                        nx,
                        ny,
                        use_fac,
                        (*expanded).data.f,
                    );
                    slice_free(reference);
                    reference = expanded;
                    nx = (*hin1).nx * red_factor;
                    ny = (*hin1).ny * red_factor;
                    if !(*opt).super_gain_name.is_null() {
                        let mut biases = Vec::new();
                        let (mut num_in_x, mut x_start, mut x_interval) = (0, 0, 0);
                        let (mut num_in_y, mut y_start, mut y_interval) = (0, 0, 0);
                        let name =
                            core::ffi::CStr::from_ptr((*opt).super_gain_name).to_string_lossy();
                        let error = crate::imod::clip::correct_defects::cor_def_read_super_gain(
                            &name,
                            use_fac,
                            &mut biases,
                            &mut num_in_x,
                            &mut x_start,
                            &mut x_interval,
                            &mut num_in_y,
                            &mut y_start,
                            &mut y_interval,
                        );
                        if error != 0 {
                            crate::imod::clip::clip::show_error(&format!(
                                "Reading file with super-resolution gain adjustments (error {error})"
                            ));
                            return -1;
                        }
                        crate::imod::clip::correct_defects::cor_def_refine_super_res_ref(
                            core::slice::from_raw_parts_mut(
                                (*reference).data.f,
                                (nx * ny) as usize,
                            ),
                            nx,
                            ny,
                            use_fac,
                            &biases,
                            num_in_x,
                            x_start,
                            x_interval,
                            num_in_y,
                            y_start,
                            y_interval,
                        );
                    }
                }
            }
            if nx != (*hin1).nx * red_factor || ny != (*hin1).ny * red_factor {
                crate::imod::clip::clip::show_error(
                    "clip unpack/normalize - reference size must match the input file size",
                );
                return -1;
            }
            if !antialias_eer {
                let llx = (*opt).cx as i32 - (*opt).ix / 2;
                let lly = (*opt).cy as i32 - (*opt).iy / 2;
                let urx = llx + (*opt).ix;
                let ury = lly + (*opt).iy;
                if llx < 0 || lly < 0 || urx > nx || ury > ny {
                    crate::imod::clip::clip::show_error(
                        "Selected area goes outside of actual data for gain reference",
                    );
                    return -1;
                }
                if (llx > 0 || lly > 0 || urx < nx || ury < ny)
                    && crate::imod::libiimod::mrcslice::slice_box_in(reference, llx, lly, urx, ury)
                        != 0
                {
                    crate::imod::clip::clip::show_error(
                        "Error allocating memory for taking subarea of gain reference",
                    );
                    return -1;
                }
                for ind in 0..(*opt).ix * (*opt).iy {
                    *(*reference).data.f.add(ind as usize) *= scale;
                }
            } else {
                crate::imod::libiimod::iitif::tiff_gain_reference_for_eer((*reference).data.f);
            }
        }
        let offset = if (*hout).mode == 2 { 0. } else { 0.5 };
        if antialias_eer {
            let kernel = (*image_file).eerkernel_scale as f32;
            if kernel == 0. {
                return -1;
            }
            scale /= kernel;
        }
        if antialias_eer {
            do_ref = false;
        }
        let threshold = if (*opt).high as i32 == IP_DEFAULT {
            f32::INFINITY
        } else {
            (*opt).high
                * scale
                * if antialias_eer {
                    (*image_file).eerkernel_scale as f32
                } else {
                    1.
                }
        };
        let out = crate::imod::libcfshr::islice::slice_create((*opt).ix, (*opt).iy, (*hout).mode);
        if out.is_null() {
            return -1;
        }
        mrc_head_label(
            &mut *hout,
            format!(
                "clip: {}, scaled by {:.2}",
                if (*opt).process == IP_UNPACK {
                    "Unpack 4-bit values"
                } else {
                    "Normalize"
                },
                scale
            )
            .as_bytes(),
        );
        for k in 0..(*opt).nofsecs {
            libc::printf(
                c"\rclip: processing slice %d of %d".as_ptr(),
                k + 1,
                (*opt).nofsecs,
            );
            libc::fflush(core::ptr::null_mut());
            let input = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin1,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if input.is_null() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            for y in 0..(*opt).iy {
                for x in 0..(*opt).ix {
                    let mut v = [0.; 4];
                    slice_get_val(input, x, y, &mut v);
                    let gain = if do_ref {
                        *(*reference).data.f.add((x + y * (*opt).ix) as usize)
                    } else {
                        scale
                    };
                    v[0] = v[0] * gain + offset;
                    if v[0] > threshold {
                        v[0] = if (*opt).low as i32 == IP_DEFAULT {
                            crate::imod::clip::correct_defects::cor_def_surrounding_mean(
                                (*input).data.b.cast(),
                                (*input).mode,
                                (*input).xsize,
                                (*input).ysize,
                                if scale == 0. {
                                    f32::INFINITY
                                } else {
                                    threshold / gain
                                },
                                x,
                                y,
                            ) * gain
                                + offset
                        } else {
                            (*opt).low * gain + offset
                        };
                    }
                    slice_put_val(out, x, y, v);
                }
            }
            slice_free(input);
            if (*opt).read_defects != 0 && correct_defects(out, (*hin1).nx, (*hin1).ny, opt) != 0 {
                return -1;
            }
            if crate::imod::clip::file_io::clip_write_slice(out, hout, opt, k, &mut z, 0) != 0 {
                return -1;
            }
        }
        slice_free(out);
        if !reference.is_null() {
            slice_free(reference);
        }
        libc::printf(c"\n".as_ptr());
        crate::imod::clip::file_io::set_mrc_coords(opt)
    }
}
/// Matches C++ `rotateFlipGainReference`.
pub unsafe fn rotate_flip_gain_reference(
    reference: *mut f32,
    nx_gain: *mut i32,
    ny_gain: *mut i32,
    rotation_flip: i32,
) -> i32 {
    unsafe {
        let nx_in = *nx_gain;
        let ny_in = *ny_gain;
        let summed =
            libc::malloc((nx_in * ny_in) as usize * core::mem::size_of::<f32>()).cast::<f32>();
        if summed.is_null() {
            return 1;
        }
        let error = crate::imod::libcfshr::rotateflip::rotate_flip_image(
            reference.cast(),
            2,
            nx_in,
            ny_in,
            rotation_flip,
            0,
            0,
            0,
            summed.cast(),
            nx_gain,
            ny_gain,
            0,
        );
        if error == 0 {
            core::ptr::copy_nonoverlapping(summed, reference, (*nx_gain * *ny_gain) as usize);
        }
        libc::free(summed.cast());
        error
    }
}
/// Matches C++ `clipDefectMap`.
pub unsafe fn clip_defect_map(
    hin: *mut MrcHeader,
    hout: *mut MrcHeader,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        let tolerance = 10;
        if (*opt).read_defects == 0 {
            crate::imod::clip::clip::show_error(
                "clip defectmap - you must provide a defect file with the -D option",
            );
            return -1;
        }
        if (*opt).defects.k2_type > 0 {
            if (*opt).defects.was_scaled > 0
                && ((*opt).cam_size_x / 2 - (*hin).nx).abs() < tolerance
                && ((*opt).cam_size_y / 2 - (*hin).ny).abs() < tolerance
            {
                crate::imod::clip::correct_defects::cor_def_scale_defects_for_k2(
                    &mut (*opt).defects,
                    true,
                );
                (*opt).cam_size_x /= 2;
                (*opt).cam_size_y /= 2;
            } else if (*opt).defects.was_scaled <= 0
                && ((*opt).cam_size_x * 2 - (*hin).nx).abs() < tolerance
                && ((*opt).cam_size_y * 2 - (*hin).ny).abs() < tolerance
            {
                crate::imod::clip::correct_defects::cor_def_scale_defects_for_k2(
                    &mut (*opt).defects,
                    false,
                );
                (*opt).cam_size_x *= 2;
                (*opt).cam_size_y *= 2;
            }
        }
        if (*opt).defects.falcon_type > 0 {
            let ind = ((*hin).nx as f32 / (*opt).cam_size_x as f32).round() as i32;
            if (ind == 2 || ind == 4)
                && ((*opt).cam_size_x * ind - (*hin).nx).abs() < tolerance
                && ((*opt).cam_size_y * ind - (*hin).ny).abs() < tolerance
            {
                crate::imod::clip::correct_defects::cor_def_scale_defects_for_falcon(
                    &mut (*opt).defects,
                    ind,
                );
                (*opt).cam_size_x *= ind;
                (*opt).cam_size_y *= ind;
            }
        }
        if ((*opt).cam_size_x - (*hin).nx).abs() >= tolerance
            || ((*opt).cam_size_y - (*hin).ny).abs() >= tolerance
        {
            crate::imod::clip::clip::show_error(
                "clip defectmap - Image size must be within 10 pixels of the camera size stored in the defect list, after possible scaling up or down by 2 for K2",
            );
            return -1;
        }
        (*hout).mode = crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
        if crate::imod::libcfshr::b3dutil::b3d_output_file_type() == 2 {
            (*hout).bytes_signed = 0;
        }
        let map = libc::malloc(((*hin).nx * (*hin).ny) as usize).cast::<u8>();
        if map.is_null() {
            crate::imod::clip::clip::show_error("CLIP - Memory error");
            return -1;
        }
        if crate::imod::clip::correct_defects::cor_def_fill_defect_array(
            &(*opt).defects,
            (*opt).cam_size_x,
            (*opt).cam_size_y,
            map,
            (*hin).nx,
            (*hin).ny,
            (*opt).sano != 0,
        ) != 0
        {
            libc::free(map.cast());
            crate::imod::clip::clip::show_error(
                "clip defectmap - Bad size parameters passed to routine",
            );
            return -1;
        }
        crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
        crate::imod::libiimod::mrcfiles::mrc_head_label(
            &mut *hout,
            b"clip defectmap: Map of defective pixels in image",
        );
        (*hout).amin = 0.;
        (*hout).amax = 0.;
        (*hout).amean = 0.;
        for ind in 0..(*hin).nx * (*hin).ny {
            let value = *map.add(ind as usize) as f32;
            (*hout).amean += value;
            (*hout).amax = (*hout).amax.max(value);
        }
        (*hout).amean /= ((*hin).nx * (*hin).ny) as f32;
        (*hout).nz = 1;
        let (sx, sy, sz) = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hin);
        crate::imod::libiimod::mrcfiles::mrc_set_scale(&mut *hout, sx as f64, sy as f64, sz as f64);
        if crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout) != 0
            || crate::imod::libiimod::mrcfiles::mrc_write_slice(
                map.cast(),
                (*hout).fp.cast(),
                hout,
                0,
                b'z' as i8,
            ) != 0
        {
            libc::free(map.cast());
            return -1;
        }
        libc::free(map.cast());
        0
    }
}
/// Matches C++ `clipSuperGain`.
pub unsafe fn clip_super_gain(
    h1: *mut MrcHeader,
    out_fp: *mut libc::FILE,
    opt: *mut ClipOptions,
) -> i32 {
    unsafe {
        if (*opt).val < 2. || (*opt).val > 64. {
            crate::imod::clip::clip::show_error(
                "clip supergain: the number of subdivisions must be between 2 and 64.",
            );
            return -1;
        }
        if out_fp.is_null() {
            return -1;
        }
        let subdivisions = (*opt).val as i32;
        let divisions = 2 * subdivisions;
        let n = ((*h1).nx * (*h1).ny) as usize;
        let mut image = vec![0_f64; n];
        let mut input = vec![0_u8; n];
        let mut extra = Vec::<MrcHeader>::with_capacity(((*opt).infiles - 1).max(0) as usize);
        for file_index in 1..(*opt).infiles {
            let mut header: MrcHeader = core::mem::zeroed();
            header.fp = crate::imod::libiimod::iimage::ii_fopen(
                *(*opt).fnames.add(file_index as usize),
                c"rb".as_ptr(),
            )
            .cast();
            if header.fp.is_null() {
                crate::imod::clip::clip::show_error(&format!(
                    "clip supergain: error opening {}.",
                    core::ffi::CStr::from_ptr(*(*opt).fnames.add(file_index as usize))
                        .to_string_lossy()
                ));
                for opened in &extra {
                    crate::imod::libiimod::iimage::ii_fclose(opened.fp.cast());
                }
                return -1;
            }
            if crate::imod::libiimod::mrcfiles::mrc_head_read(header.fp.cast(), &mut header) != 0 {
                crate::imod::clip::clip::show_error(&format!(
                    "clip supergain: error reading header of {}.",
                    core::ffi::CStr::from_ptr(*(*opt).fnames.add(file_index as usize))
                        .to_string_lossy()
                ));
                if !header.fp.is_null() {
                    crate::imod::libiimod::iimage::ii_fclose(header.fp.cast());
                }
                for opened in &extra {
                    crate::imod::libiimod::iimage::ii_fclose(opened.fp.cast());
                }
                return -1;
            }
            extra.push(header);
        }
        for file_index in 0..(*opt).infiles {
            let header = if file_index == 0 {
                h1
            } else {
                extra.as_mut_ptr().add((file_index - 1) as usize)
            };
            if (*header).nx != (*h1).nx
                || (*header).ny != (*h1).ny
                || (*header).mode != crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE
            {
                crate::imod::clip::clip::show_error(
                    "clip supergain: all input files must be the same X/Y size and mode 0.",
                );
                for opened in &extra {
                    crate::imod::libiimod::iimage::ii_fclose(opened.fp.cast());
                }
                return -1;
            }
            let image_file =
                crate::imod::libiimod::iimage::ii_lookup_file_from_fp((*header).fp.cast());
            if image_file.is_null()
                || (*image_file).file != crate::imod::libiimod::iimage::IIFILE_TIFF
                || (*image_file).num_frames_in_eerfile == 0
            {
                crate::imod::clip::clip::show_error(
                    "clip supergain: all input files must be EER files.",
                );
                for opened in &extra {
                    crate::imod::libiimod::iimage::ii_fclose(opened.fp.cast());
                }
                return -1;
            }
            for z in 0..(*header).nz {
                if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                    input.as_mut_ptr().cast(),
                    (*header).fp.cast(),
                    header,
                    z,
                    b'z' as i8,
                ) != 0
                {
                    for opened in &extra {
                        crate::imod::libiimod::iimage::ii_fclose(opened.fp.cast());
                    }
                    return -1;
                }
                for i in 0..n {
                    image[i] = input[i] as f64;
                }
            }
        }
        let phys_x = (*h1).nx / 4 - 4;
        let phys_y = (*h1).ny / 4 - 4;
        let x_spacing = 4 * (phys_x / divisions);
        let y_spacing = 4 * (phys_y / divisions);
        let x_offset = 4 * ((phys_x % divisions) / 2);
        let y_offset = 4 * ((phys_y % divisions) / 2);
        if x_spacing <= 0 || y_spacing <= 0 {
            return -1;
        }
        let mut bias = vec![[0_f64; 16]; (divisions * divisions) as usize];
        let mut total = [0_f64; 16];
        for yd in 0..divisions {
            for xd in 0..divisions {
                let a = &mut bias[(xd + yd * divisions) as usize];
                for y in y_offset + yd * y_spacing..y_offset + (yd + 1) * y_spacing {
                    for x in x_offset + xd * x_spacing..x_offset + (xd + 1) * x_spacing {
                        let ind = (x % 4 + 4 * (y % 4)) as usize;
                        a[ind] += image[(x + y * (*h1).nx) as usize];
                    }
                }
                for i in 0..16 {
                    total[i] += a[i];
                }
            }
        }
        let mut bsum = 0_f64;
        for value in total {
            bsum += value / 16.;
        }
        libc::printf(c"Overall gain factors for 4x4 super-resolution:\n".as_ptr());
        for i in (0..=12).rev().step_by(4) {
            for j in 0..4 {
                libc::printf(c"  %.6f".as_ptr(), bsum / total[i + j]);
            }
            libc::printf(c"\n".as_ptr());
        }
        let mut two = total;
        let b = combine_area_sums(two.as_mut_ptr()) as f64;
        libc::printf(c"Overall gain factors for 2x2 super-resolution:\n".as_ptr());
        for i in (0..=2).rev().step_by(2) {
            libc::printf(c"  %.6f  %.6f\n".as_ptr(), b / two[i], b / two[i + 1]);
        }
        libc::fprintf(out_fp, c"1  4\n".as_ptr());
        libc::fprintf(
            out_fp,
            c"%d %d %d %d %d %d\n".as_ptr(),
            divisions - 1,
            x_offset + x_spacing,
            x_spacing,
            divisions - 1,
            y_offset + y_spacing,
            y_spacing,
        );
        for yd in 0..divisions - 1 {
            for xd in 0..divisions - 1 {
                let ix = (xd + yd * divisions) as usize;
                let mut a = [0_f64; 16];
                for i in 0..16 {
                    a[i] = bias[ix][i]
                        + bias[ix + 1][i]
                        + bias[ix + divisions as usize][i]
                        + bias[ix + divisions as usize + 1][i];
                }
                bsum = 0.;
                for value in a {
                    bsum += value / 16.;
                }
                for value in a {
                    libc::fprintf(out_fp, c" %.5f".as_ptr(), bsum / value);
                }
                libc::fprintf(out_fp, c"\n".as_ptr());
                let q = combine_area_sums(a.as_mut_ptr()) as f64;
                for i in 0..4 {
                    libc::fprintf(out_fp, c" %.5f".as_ptr(), q / a[i]);
                }
                libc::fprintf(out_fp, c"\n".as_ptr());
            }
        }
        crate::imod::libiimod::iimage::ii_fclose(out_fp);
        for opened in &extra {
            crate::imod::libiimod::iimage::ii_fclose(opened.fp.cast());
        }
        0
    }
}
/// Matches C++ `combineAreaSums`.
pub unsafe fn combine_area_sums(area_sum: *mut f64) -> f32 {
    unsafe {
        *area_sum = *area_sum.add(0) + *area_sum.add(1) + *area_sum.add(4) + *area_sum.add(5);
        *area_sum.add(1) =
            *area_sum.add(2) + *area_sum.add(3) + *area_sum.add(6) + *area_sum.add(7);
        *area_sum.add(2) =
            *area_sum.add(8) + *area_sum.add(9) + *area_sum.add(12) + *area_sum.add(13);
        *area_sum.add(3) =
            *area_sum.add(10) + *area_sum.add(11) + *area_sum.add(14) + *area_sum.add(15);
        let mut bsum = 0.;
        for index in 0..4 {
            bsum += *area_sum.add(index) / 4.;
        }
        bsum as f32
    }
}
/// Matches C++ `clip_parxyz`.
pub unsafe fn clip_parxyz(
    v: *mut Istack,
    xmax: i32,
    ymax: i32,
    zmax: i32,
    rx: *mut f32,
    ry: *mut f32,
    rz: *mut f32,
) -> i32 {
    unsafe {
        let sl = *(*v).vol.add(zmax as usize);
        let first = *(*v).vol;
        let x1 = if xmax == 0 {
            (*first).xsize - 1
        } else {
            xmax - 1
        };
        let x3 = if xmax + 1 >= (*first).xsize {
            0
        } else {
            xmax + 1
        };
        let a = slice_get_pixel_magnitude(sl, x1, ymax) * -1.
            + slice_get_pixel_magnitude(sl, xmax, ymax) * 2.
            + slice_get_pixel_magnitude(sl, x3, ymax) * -1.;
        let b = ((xmax - 1) * (xmax - 1)) as f32
            * (slice_get_pixel_magnitude(sl, xmax, ymax) - slice_get_pixel_magnitude(sl, x3, ymax))
            + (xmax * xmax) as f32
                * (slice_get_pixel_magnitude(sl, x3, ymax)
                    - slice_get_pixel_magnitude(sl, x1, ymax))
            + ((xmax + 1) * (xmax + 1)) as f32
                * (slice_get_pixel_magnitude(sl, x1, ymax)
                    - slice_get_pixel_magnitude(sl, xmax, ymax));
        *rx = if a != 0. { -b / (2. * a) } else { xmax as f32 };
        let y1 = if ymax == 0 {
            (*first).ysize - 1
        } else {
            ymax - 1
        };
        let y3 = if ymax + 1 >= (*first).ysize {
            0
        } else {
            ymax + 1
        };
        let a = slice_get_pixel_magnitude(sl, xmax, y1) * -1.
            + slice_get_pixel_magnitude(sl, xmax, ymax) * 2.
            + slice_get_pixel_magnitude(sl, xmax, y3) * -1.;
        let b = ((ymax - 1) * (ymax - 1)) as f32
            * (slice_get_pixel_magnitude(sl, xmax, ymax) - slice_get_pixel_magnitude(sl, xmax, y3))
            + (ymax * ymax) as f32
                * (slice_get_pixel_magnitude(sl, xmax, y3)
                    - slice_get_pixel_magnitude(sl, xmax, y1))
            + ((ymax + 1) * (ymax + 1)) as f32
                * (slice_get_pixel_magnitude(sl, xmax, y1)
                    - slice_get_pixel_magnitude(sl, xmax, ymax));
        *ry = if a != 0. { -b / (2. * a) } else { ymax as f32 };
        let z1 = if zmax == 0 { (*v).zsize - 1 } else { zmax - 1 };
        let z3 = if zmax + 1 >= (*v).zsize { 0 } else { zmax + 1 };
        let a = slice_get_pixel_magnitude(*(*v).vol.add(z1 as usize), xmax, ymax) * -1.
            + slice_get_pixel_magnitude(sl, xmax, ymax) * 2.
            + slice_get_pixel_magnitude(*(*v).vol.add(z3 as usize), xmax, ymax) * -1.;
        let b = ((zmax - 1) * (zmax - 1)) as f32
            * (slice_get_pixel_magnitude(sl, xmax, ymax)
                - slice_get_pixel_magnitude(*(*v).vol.add(z3 as usize), xmax, ymax))
            + (zmax * zmax) as f32
                * (slice_get_pixel_magnitude(*(*v).vol.add(z3 as usize), xmax, ymax)
                    - slice_get_pixel_magnitude(*(*v).vol.add(z1 as usize), xmax, ymax))
            + ((zmax + 1) * (zmax + 1)) as f32
                * (slice_get_pixel_magnitude(*(*v).vol.add(z1 as usize), xmax, ymax)
                    - slice_get_pixel_magnitude(sl, xmax, ymax));
        *rz = if a != 0. { -b / (2. * a) } else { zmax as f32 };
        0
    }
}
/// Matches C++ `clip_stat3d`.
pub unsafe fn clip_stat3d(v: *mut Istack) -> i32 {
    unsafe {
        let (mut min, mut max, mut mean) = (0., 0., 0.);
        let (mut xmax, mut ymax, mut zmax) = (0, 0, 0);
        clip_get_stat3d(
            v, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
        );
        let (mut x, mut y, mut z) = (0., 0., 0.);
        clip_parxyz(v, xmax, ymax, zmax, &mut x, &mut y, &mut z);
        libc::printf(
            c"max = %g  min = %g  mean = %g\n".as_ptr(),
            max as f64,
            min as f64,
            mean as f64,
        );
        libc::printf(
            c"location of max pixel ( %d, %d, %d) is \n".as_ptr(),
            xmax,
            ymax,
            zmax,
        );
        libc::printf(
            c"( %.2f, %.2f, %.2f)\n".as_ptr(),
            x as f64,
            y as f64,
            z as f64,
        );
        0
    }
}
/// Matches C++ `clip_get_stat3d`.
pub unsafe fn clip_get_stat3d(
    v: *mut Istack,
    rmin: *mut f32,
    rmax: *mut f32,
    rmean: *mut f32,
    rx: *mut i32,
    ry: *mut i32,
    rz: *mut i32,
) -> i32 {
    unsafe {
        let first = *(*v).vol;
        let (nx, ny, nz) = ((*first).xsize, (*first).ysize, (*v).zsize);
        let mut min = 1e36_f32;
        let mut max = f32::MIN_POSITIVE;
        let mut sum = 0_f64;
        let (mut xmax, mut ymax, mut zmax) = (0, 0, 0);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let value = slice_get_pixel_magnitude(*(*v).vol.add(k as usize), i, j);
                    if value > max {
                        max = value;
                        xmax = i;
                        ymax = j;
                        zmax = k;
                    }
                    if value < min {
                        min = value;
                    }
                    sum += value as f64;
                }
            }
        }
        *rmin = min;
        *rmax = max;
        *rmean = (sum / (nx * ny * nz) as f64) as f32;
        *rx = xmax;
        *ry = ymax;
        *rz = zmax;
        0
    }
}
/// Matches C++ `clip_stat`.
pub unsafe fn clip_stat(hin: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        use crate::imod::clip::clip::IP_DEFAULT;
        let outliers = (*opt).val != IP_DEFAULT as f32 || (*opt).low != IP_DEFAULT as f32;
        let mut li: crate::imod::libiimod::mrcfiles::LoadInfo = core::mem::zeroed();
        crate::imod::libiimod::mrcfiles::mrc_init_li(Some(&mut li), None);
        let mut pcoords = Vec::new();
        if !(*opt).plname.is_null() {
            if crate::imod::libiimod::plist::mrc_plist_li(&mut li, hin, (*opt).plname) != 0 {
                crate::imod::clip::clip::show_error("stat: error reading piece list file");
                return -1;
            }
            if li.plist < (*hin).nz {
                crate::imod::clip::clip::show_error("stat: not enough piece coordinates in file");
                return -1;
            }
            pcoords = core::slice::from_raw_parts(li.pcoords, (3 * li.plist) as usize).to_vec();
            libc::free(li.pcoords.cast());
            let mut min_x = 0;
            let mut num_x = 0;
            let mut overlap_x = 0;
            let mut min_y = 0;
            let mut num_y = 0;
            let mut overlap_y = 0;
            if crate::imod::libcfshr::piecefuncs::check_piece_list(
                &pcoords,
                3,
                li.plist as usize,
                1,
                (*hin).nx,
                &mut min_x,
                &mut num_x,
                &mut overlap_x,
            ) != 0
                || crate::imod::libcfshr::piecefuncs::check_piece_list(
                    &pcoords[1..],
                    3,
                    li.plist as usize,
                    1,
                    (*hin).ny,
                    &mut min_y,
                    &mut num_y,
                    &mut overlap_y,
                ) != 0
            {
                crate::imod::clip::clip::show_error(
                    "stat: piece coordinates are not regularly spaced",
                );
                return -1;
            }
            if (*opt).new_xoverlap == IP_DEFAULT {
                (*opt).new_xoverlap = 0;
            }
            if (*opt).new_yoverlap == IP_DEFAULT {
                (*opt).new_yoverlap = 0;
            }
            if num_x > 1 {
                crate::imod::libcfshr::piecefuncs::adjust_piece_overlap(
                    &mut pcoords,
                    3,
                    li.plist as usize,
                    (*hin).nx,
                    min_x,
                    overlap_x,
                    (*opt).new_xoverlap,
                );
            }
            if num_y > 1 {
                crate::imod::libcfshr::piecefuncs::adjust_piece_overlap(
                    &mut pcoords[1..],
                    3,
                    li.plist as usize,
                    (*hin).ny,
                    min_y,
                    overlap_y,
                    (*opt).new_yoverlap,
                );
            }
        }
        if !pcoords.is_empty() {
            let axis = if (*opt).from_one != 0 {
                "y, view"
            } else {
                " y,   z"
            };
            libc::printf(c"piece|   min   |(   x,  %s)|    max  |(   x,  %s)|   mean\n-----|---------|----------------|---------|----------------|---------\n".as_ptr(),
                std::ffi::CString::new(axis).unwrap().as_ptr(), std::ffi::CString::new(axis).unwrap().as_ptr());
        } else if (*opt).from_one != 0 {
            libc::printf(
                c"view |   min   |(   x,   y)|    max  |(      x,      y)|   mean    |  std dev.\n-----|---------|-----------|---------|-----------------|-----------|----------\n"
                    .as_ptr(),
            );
        } else {
            libc::printf(
                c"slice|   min   |(   x,   y)|    max  |(      x,      y)|   mean    |  std dev.\n-----|---------|-----------|---------|-----------------|-----------|----------\n"
                    .as_ptr(),
            );
        }
        crate::imod::clip::file_io::set_input_options(opt, hin);
        let mut total_sum = 0_f64;
        let mut total_sq = 0_f64;
        let mut total_n = 0_i64;
        let mut all_min = f32::INFINITY;
        let mut all_max = f32::NEG_INFINITY;
        let mut zmin = 0_i32;
        let mut zmax = 0_i32;
        let add = if (*opt).from_one != 0 { 1 } else { 0 };
        let mut allmins = Vec::new();
        let mut allmaxes = Vec::new();
        let mut stat_rows = Vec::new();
        for k in 0..(*opt).nofsecs {
            let iz = *(*opt).secs.add(k as usize);
            if iz < 0 || iz >= (*hin).nz {
                crate::imod::clip::clip::show_error("stat: slice out of range.");
                return -1;
            }
            let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                iz,
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if s.is_null() {
                crate::imod::clip::clip::show_error("stat: error reading slice.");
                return -1;
            }
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut xmin = 0;
            let mut ymin = 0;
            let mut xmax = 0;
            let mut ymax = 0;
            let mut sum = 0_f64;
            let mut square = 0_f64;
            for y in 0..(*s).ysize {
                for x in 0..(*s).xsize {
                    let value = slice_get_pixel_magnitude(s, x, y);
                    if value < min {
                        min = value;
                        xmin = x;
                        ymin = y;
                    }
                    if value > max {
                        max = value;
                        xmax = x;
                        ymax = y;
                    }
                    sum += value as f64;
                    square += (value as f64) * (value as f64);
                }
            }
            let n = ((*s).xsize * (*s).ysize) as f64;
            let mean = sum / n;
            let sd = ((square - n * mean * mean) / 1_f64.max(n - 1.))
                .max(0.)
                .sqrt();
            if !outliers && !pcoords.is_empty() {
                let base = (3 * iz) as usize;
                libc::printf(
                    c"%4d  %9.4f (%4d,%4d,%4d) %9.4f (%4d,%4d,%4d) %9.4f\n".as_ptr(),
                    iz + add,
                    min as f64,
                    xmin + pcoords[base] + add,
                    ymin + pcoords[base + 1] + add,
                    pcoords[base + 2] + add,
                    max as f64,
                    xmax + pcoords[base],
                    ymax + pcoords[base + 1],
                    pcoords[base + 2] + add,
                    mean,
                );
            } else if !outliers {
                libc::printf(
                    c"%4d  %9.4f (%4d,%4d) %9.4f (%7.2f,%7.2f) %9.4f  %9.4f\n".as_ptr(),
                    iz + add,
                    min as f64,
                    xmin + add,
                    ymin + add,
                    max as f64,
                    xmax as f64,
                    ymax as f64,
                    mean,
                    sd,
                );
            }
            if min < all_min {
                all_min = min;
                zmin = iz;
            }
            if max > all_max {
                all_max = max;
                zmax = iz;
            }
            total_sum += sum;
            total_sq += square;
            total_n += n as i64;
            allmins.push(min);
            allmaxes.push(max);
            stat_rows.push((iz, xmin, ymin, xmax, ymax, mean, sd));
            slice_free(s);
        }
        if outliers {
            let mut length = (*opt).nofsecs;
            if (*opt).low != IP_DEFAULT as f32 {
                length = (*opt).low.round() as i32;
            }
            length = length.clamp(5, (*opt).nofsecs);
            let kcrit = if (*opt).val != IP_DEFAULT as f32 {
                (*opt).val
            } else {
                2.24
            };
            let mut flagged = vec![false; stat_rows.len()];
            for kk in 0..stat_rows.len() {
                let mut di = (kk as i32 - length / 2).max(0);
                let mut dj = (di + length).min((*opt).nofsecs);
                di = (dj - length).max(0);
                let mut min_drops = vec![0_f32; (dj - di) as usize];
                let mut mins = allmins[di as usize..dj as usize].to_vec();
                crate::imod::libcfshr::robuststat::rs_mad_median_outliers(
                    mins.as_mut_ptr(),
                    length,
                    kcrit,
                    min_drops.as_mut_ptr(),
                );
                let index = kk - di as usize;
                if min_drops[index] < 0. {
                    flagged[kk] = true;
                }
                let mut maxes = allmaxes[di as usize..dj as usize].to_vec();
                let mut max_drops = vec![0_f32; (dj - di) as usize];
                crate::imod::libcfshr::robuststat::rs_mad_median_outliers(
                    maxes.as_mut_ptr(),
                    length,
                    kcrit,
                    max_drops.as_mut_ptr(),
                );
                if max_drops[index] > 0. {
                    flagged[kk] = true;
                }
                let (iz, xmin, ymin, xmax, ymax, mean, sd) = stat_rows[kk];
                let starmin = if min_drops[index] < 0. { b'*' } else { b' ' };
                let starmax = if max_drops[index] > 0. { b'*' } else { b' ' };
                if !pcoords.is_empty() {
                    let base = (3 * iz) as usize;
                    libc::printf(
                        c"%4d  %9.4f%c(%4d,%4d,%4d) %9.4f%c(%4d,%4d,%4d) %9.4f  %9.4f\n".as_ptr(),
                        iz + add,
                        allmins[kk] as f64,
                        starmin as i32,
                        xmin + pcoords[base] + add,
                        ymin + pcoords[base + 1] + add,
                        pcoords[base + 2] + add,
                        allmaxes[kk] as f64,
                        starmax as i32,
                        xmax + pcoords[base],
                        ymax + pcoords[base + 1],
                        pcoords[base + 2] + add,
                        mean,
                        sd,
                    );
                } else {
                    libc::printf(
                        c"%4d  %9.4f%c(%4d,%4d) %9.4f%c(%7d,%7d) %9.4f  %9.4f\n".as_ptr(),
                        iz + add,
                        allmins[kk] as f64,
                        starmin as i32,
                        xmin,
                        ymin,
                        allmaxes[kk] as f64,
                        starmax as i32,
                        xmax,
                        ymax,
                        mean,
                        sd,
                    );
                }
            }
            libc::printf(
                c"\n%s with %sextreme values:".as_ptr(),
                if pcoords.is_empty() {
                    if (*opt).from_one != 0 {
                        c"Views".as_ptr()
                    } else {
                        c"Slices".as_ptr()
                    }
                } else {
                    c"Pieces".as_ptr()
                },
                if (*opt).low != IP_DEFAULT as f32 {
                    c"locally ".as_ptr()
                } else {
                    c"".as_ptr()
                },
            );
            let mut number = 0;
            for (index, row) in stat_rows.iter().enumerate() {
                if flagged[index] {
                    libc::printf(c" %3d".as_ptr(), row.0 + add);
                    number += 1;
                }
            }
            if number == 0 {
                libc::printf(c" None".as_ptr());
            }
            libc::printf(c"\n".as_ptr());
        }
        if total_n > 0 {
            let mean = total_sum / total_n as f64;
            let sd = ((total_sq - total_n as f64 * mean * mean) / 1_f64.max(total_n as f64 - 1.))
                .max(0.)
                .sqrt();
            if !pcoords.is_empty() {
                libc::printf(
                    c" all  %9.4f (@ piece =%5d) %9.4f (@ piece =%5d) %9.4f  %9.4f\n".as_ptr(),
                    all_min as f64,
                    zmin + 1,
                    all_max as f64,
                    zmax + 1,
                    mean,
                    sd,
                );
            } else {
                libc::printf(
                    c" all  %9.4f (@ z=%5d) %9.4f (@ z=%5d      ) %9.4f  %9.4f\n".as_ptr(),
                    all_min as f64,
                    zmin + add,
                    all_max as f64,
                    zmax + add,
                    mean,
                    sd,
                );
            }
        }
        0
    }
}
/// Matches C++ `clipHistogram`.
pub unsafe fn clip_histogram(hin: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        use crate::imod::clip::clip::IP_DEFAULT;
        crate::imod::clip::file_io::set_input_options(opt, hin);
        if (*opt).sano != 0 {
            return histogram_peaks_and_dip(hin, opt);
        }
        let floating = matches!((*hin).mode, 2 | 4);
        let (hist_min, hist_max, delta, bins_len, offset) = if floating {
            let lo = if (*opt).low as i32 == IP_DEFAULT {
                (*hin).amin
            } else {
                (*opt).low
            };
            let hi = if (*opt).high as i32 == IP_DEFAULT {
                (*hin).amax
            } else {
                (*opt).high
            };
            if lo >= hi {
                libc::printf(
                    c"ERROR: clip histogram - minimum (%f) must be less than maximum (%f)\n"
                        .as_ptr(),
                    lo as f64,
                    hi as f64,
                );
                return -1;
            }
            let d = if (*opt).val as i32 == IP_DEFAULT {
                (hi - lo) / 256.
            } else {
                (*opt).val
            };
            if d <= 0. {
                libc::printf(
                    c"ERROR: clip histogram - histogram bin size (%f) must be positive\n".as_ptr(),
                    d as f64,
                );
                return -1;
            }
            let mut number = ((hi - lo) / d).ceil() as i32;
            if (hi - lo) / d >= number as f32 - 0.01 {
                number += 1;
            }
            (lo, hi, d, number.clamp(0, 65535) as usize, 0)
        } else {
            match (*hin).mode {
                0 | 16 => (0., 255., 1., 256, 0),
                1 => (-32768., 32767., 1., 65536, 32768),
                6 => (0., 65535., 1., 65536, 0),
                _ => return -1,
            }
        };
        let mut bins = vec![0_i64; bins_len];
        let mut nx = 0;
        let mut ny = 0;
        for k in 0..(*opt).nofsecs {
            let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                *(*opt).secs.add(k as usize),
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if s.is_null() {
                crate::imod::clip::clip::show_error(&format!(
                    "clip histogram - error reading slice {}",
                    *(*opt).secs.add(k as usize)
                ));
                return -1;
            }
            nx = (*s).xsize;
            ny = (*s).ysize;
            for y in 0..ny {
                for x in 0..nx {
                    let v = slice_get_pixel_magnitude(s, x, y);
                    let ind = if floating {
                        ((v - hist_min) / delta) as isize
                    } else {
                        v.round() as isize + offset
                    };
                    if ind >= 0 && (ind as usize) < bins.len() {
                        bins[ind as usize] += 1;
                    }
                }
            }
            slice_free(s);
        }
        let first = bins.iter().position(|&n| n != 0);
        let last = bins.iter().rposition(|&n| n != 0);
        if let (Some(mut a), Some(mut b)) = (first, last) {
            if !floating {
                if (*opt).low as i32 != IP_DEFAULT
                    && (*opt).high as i32 != IP_DEFAULT
                    && (*opt).low > (*opt).high
                {
                    libc::printf(
                        c"ERROR: clip histogram - minimum (%f) must be less than maximum (%f)\n"
                            .as_ptr(),
                        (*opt).low as f64,
                        (*opt).high as f64,
                    );
                    return -1;
                }
                if (*opt).low as i32 != IP_DEFAULT {
                    a = a.max(((*opt).low + offset as f32).round() as usize);
                }
                if (*opt).high as i32 != IP_DEFAULT {
                    b = b.min(((*opt).high + offset as f32).round() as usize);
                }
                if a > b {
                    a = 1;
                }
            }
            let combine = if floating {
                1
            } else if (*opt).val as i32 == IP_DEFAULT {
                1.max((b - a) / 256)
            } else {
                (*opt).val.round().max(1.) as usize
            };
            let mut cumulative = 0_f64;
            let threshold = if (*opt).thresh as i32 == IP_DEFAULT {
                None
            } else {
                Some((*opt).thresh * (*opt).nofsecs as f32 * nx as f32 * ny as f32)
            };
            let mut threshold_value = 0_f32;
            let mut got_threshold = false;
            if floating {
                libc::printf(
                    c" Bin midpoint   counts    (bin interval is %f)\n".as_ptr(),
                    delta as f64,
                );
            } else if combine > 1 {
                libc::printf(
                    c"Bin midpoint   counts    (bin interval is %d)\n".as_ptr(),
                    combine as i32,
                );
            } else {
                libc::printf(
                    c" Value   counts    (bin interval is %d)\n".as_ptr(),
                    combine as i32,
                );
            }
            for i in (a..=b).step_by(combine) {
                let count: (i64) = (i..(i + combine).min(b + 1)).map(|n| bins[n]).sum();
                let middle = if floating {
                    hist_min + (i as f32 + 0.5) * delta
                } else if combine > 1 {
                    i as f32 - offset as f32 + combine as f32 * 0.5
                } else {
                    i as f32 - offset as f32
                };
                if floating {
                    libc::printf(c"%13.6g %8d\n".as_ptr(), middle as f64, count as i32);
                } else if combine > 1 {
                    libc::printf(c"%12.1f %8d\n".as_ptr(), middle as f64, count as i32);
                } else {
                    libc::printf(c"%6d %8d\n".as_ptr(), middle as i32, count as i32);
                }
                cumulative += count as f64;
                if let Some(t) = threshold {
                    if !got_threshold && cumulative >= t as f64 {
                        let frac = ((cumulative - t as f64) / count as f64) as f32;
                        threshold_value = if floating {
                            hist_min + (i as f32 - frac) * delta
                        } else {
                            i as f32 - frac - offset as f32
                        };
                        got_threshold = true;
                    }
                }
            }
            if got_threshold {
                libc::printf(
                    c"Threshold value for reaching %g of counts = %g\n".as_ptr(),
                    (*opt).thresh as f64,
                    threshold_value as f64,
                );
            }
            let mut combo_bins = Vec::new();
            for i in (a..=b).step_by(combine) {
                combo_bins.push(
                    (i..(i + combine).min(b + 1))
                        .map(|index| bins[index])
                        .sum::<i64>(),
                );
            }
            let num_bins = combo_bins.len() as i32;
            let combo_left = if floating {
                hist_min + a as f32 * delta
            } else {
                a as f32 - offset as f32
            };
            let bin_delta = if floating { delta } else { combine as f32 };
            let peak_ind = combo_bins
                .iter()
                .enumerate()
                .max_by_key(|(_, count)| *count)
                .map_or(0, |(index, _)| index as i32);
            if (*opt).falloff_frac as i32 != IP_DEFAULT {
                let (dir, mut ind) = if (*opt).falloff_frac > 0. {
                    (1_i32, 1_i32)
                } else {
                    (-1, num_bins - 2)
                };
                let mut diff_ind = -1_i32;
                let mut max_diff = 0_f32;
                let mut cumulative_counts = 0_f64;
                let threshold_counts = (*opt).falloff_frac.abs() as f64
                    * (*opt).nofsecs as f64
                    * nx as f64
                    * ny as f64;
                while ind > 0 && ind < num_bins - 1 {
                    cumulative_counts += combo_bins[(ind - dir) as usize] as f64;
                    let current = combo_bins[ind as usize];
                    let previous = combo_bins[(ind - dir) as usize];
                    let frac = if current > 3000 && previous > 3000 {
                        previous as f32 / current as f32
                    } else {
                        let num_fit = if current > 1000 && previous > 1000 {
                            3
                        } else if current > 300 && previous > 300 {
                            5
                        } else if current > 100 && previous > 100 {
                            7
                        } else if current > 30 && previous > 30 {
                            9
                        } else {
                            11
                        };
                        let mut start = 0.max(ind - num_fit / 2);
                        let end = (num_bins - 1).min(start + num_fit - 1);
                        start = 0.max(end + 1 - num_fit);
                        let count = end + 1 - start;
                        let mut xx = [0_f32; 11];
                        let mut yy = [0_f32; 11];
                        for j in 0..count {
                            xx[j as usize] = j as f32;
                            yy[j as usize] = combo_bins[(j + start) as usize] as f32;
                        }
                        let (mut slope, mut intercept, mut ro) = (0_f32, 0_f32, 0_f32);
                        crate::imod::libcfshr::simplestat::ls_fit(
                            xx.as_ptr(),
                            yy.as_ptr(),
                            count,
                            &mut slope,
                            &mut intercept,
                            &mut ro,
                        );
                        (slope * (ind - dir - start) as f32 + intercept)
                            / 1_f32.max(slope * (ind - start) as f32 + intercept)
                    };
                    if current > 100 && frac > max_diff && cumulative_counts > threshold_counts {
                        max_diff = frac;
                        diff_ind = ind;
                    }
                    ind += dir;
                }
                if diff_ind < 0 {
                    libc::printf(
                        c"ERROR: CLIP - No point of maximum falloff could be found past %.3f of total cumulative counts\n".as_ptr(),
                        (*opt).falloff_frac.abs() as f64,
                    );
                    return -1;
                }
                libc::printf(
                    c"Maximum falloff occurs at %g\n".as_ptr(),
                    (combo_left + diff_ind as f32 * bin_delta) as f64,
                );
            }
            if (*opt).pctl_frac as i32 == IP_DEFAULT {
                return 0;
            }
            if peak_ind as f32 > 0.8 * num_bins as f32 || (peak_ind as f32) < 0.2 * num_bins as f32
            {
                libc::printf(
                    c"ERROR: CLIP - Peak is at %f, too close to end of range to analyze for extra counts\n".as_ptr(),
                    (combo_left + (peak_ind as f32 + 0.5) * bin_delta) as f64,
                );
                return -1;
            }
            let frac = crate::imod::libcfshr::filtxcorr::parabolic_fit_position(
                combo_bins[(peak_ind - 1) as usize] as f32,
                combo_bins[peak_ind as usize] as f32,
                combo_bins[(peak_ind + 1) as usize] as f32,
            ) as f32;
            let (dir, mut ind) = if (*opt).pctl_frac > 0. {
                (1_i32, num_bins - 1)
            } else {
                (-1, 0)
            };
            libc::printf(
                c"Interpolated peak position %13.6g\n".as_ptr(),
                (combo_left + (peak_ind as f32 + frac + 0.5) * bin_delta) as f64,
            );
            libc::printf(
                c"Bins %s peak minus bins %s peak:\n".as_ptr(),
                if dir > 0 { c"above" } else { c"below" }.as_ptr(),
                if dir > 0 { c"below" } else { c"above" }.as_ptr(),
            );
            let mut diff_ind = -1_i32;
            while dir * (ind - (peak_ind + dir * 3)) > 0 {
                let r_ind = 2. * (peak_ind as f32 + frac) - ind as f32;
                let j = r_ind.floor() as i32;
                let ff = r_ind - j as f32;
                if j >= 0 && j < num_bins - 1 {
                    let value = (1. - ff) * combo_bins[j as usize] as f32
                        + ff * combo_bins[(j + 1) as usize] as f32;
                    let difference = combo_bins[ind as usize] - value.round() as i64;
                    libc::printf(
                        c"%13.6g %8d\n".as_ptr(),
                        (combo_left + (ind as f32 + 0.5) * bin_delta) as f64,
                        difference as i32,
                    );
                    combo_bins[ind as usize] = difference;
                    if diff_ind < 0 || difference > combo_bins[diff_ind as usize] {
                        diff_ind = ind;
                    }
                }
                ind -= dir;
            }
            if diff_ind < 0 || combo_bins[diff_ind as usize] <= 0 {
                libc::printf(
                    c"ERROR: CLIP - There are fewer counts %s the peak than %s it\n".as_ptr(),
                    if dir > 0 { c"above" } else { c"below" }.as_ptr(),
                    if dir > 0 { c"below" } else { c"above" }.as_ptr(),
                );
                return -1;
            }
            let mut cumul = 0_f64;
            ind = if dir > 0 { num_bins - 1 } else { 0 };
            while dir * (ind - (peak_ind + dir * 3)) > 0 {
                if dir * (ind - diff_ind) < 0 && combo_bins[ind as usize] < 0 {
                    break;
                }
                cumul += combo_bins[ind as usize] as f64;
                ind -= dir;
            }
            if cumul <= 0. {
                libc::printf(
                    c"ERROR: CLIP - There are fewer counts %s the peak than %s it\n".as_ptr(),
                    if dir > 0 { c"above" } else { c"below" }.as_ptr(),
                    if dir > 0 { c"below" } else { c"above" }.as_ptr(),
                );
                return -1;
            }
            let threshold_counts = (*opt).pctl_frac.abs() as f64 * cumul;
            cumul = 0.;
            ind = if dir > 0 { num_bins - 1 } else { 0 };
            while dir * (ind - (peak_ind + dir * 3)) > 0 {
                if dir * (ind - diff_ind) < 0 && combo_bins[ind as usize] < 0 {
                    break;
                }
                cumul += combo_bins[ind as usize] as f64;
                if cumul > threshold_counts {
                    let fraction =
                        ((cumul - threshold_counts) / combo_bins[ind as usize] as f64) as f32;
                    let value = combo_left + (ind as f32 + dir as f32 * fraction) * bin_delta;
                    libc::printf(
                        c"%.3f of the extra counts %s the peak occur %s %g\n".as_ptr(),
                        (*opt).pctl_frac.abs() as f64,
                        if dir > 0 { c"above" } else { c"below" }.as_ptr(),
                        if dir > 0 { c"above" } else { c"below" }.as_ptr(),
                        value as f64,
                    );
                    return 0;
                }
                ind -= dir;
            }
            libc::printf(
                c"ERROR: CLIP - Extra counts occurred %s the peak but percentile analysis failed\n"
                    .as_ptr(),
                if dir > 0 { c"above" } else { c"below" }.as_ptr(),
            );
            return -1;
        } else {
            libc::printf(c"There are no values within the specified range\n".as_ptr());
            0
        }
    }
}
/// Matches C++ `histogramPeaksAndDip`.
pub unsafe fn histogram_peaks_and_dip(hin: *mut MrcHeader, opt: *mut ClipOptions) -> i32 {
    unsafe {
        use crate::imod::clip::clip::IP_DEFAULT;
        if (*opt).low as i32 != IP_DEFAULT
            || (*opt).high as i32 != IP_DEFAULT
            || (*opt).val as i32 != IP_DEFAULT
        {
            libc::printf(
                c"ERROR: CLIP - The -n, -l, and -h options have no effect when doing a histogram with -s\n"
                    .as_ptr(),
            );
            return -1;
        }
        let volume = (*opt).dim == 3;
        let iz_add = if (*opt).from_one != 0 { 1 } else { 0 };
        let mut sample = Vec::<f32>::new();
        let mut bins = [0_f32; 1000];
        let (mut interval, mut num_sample, mut x, mut y, mut sample_index) = (0, 0, 0, 0, 0);
        let (mut first_val, mut last_val) = (0_f32, 0_f32);
        for k in 0..(*opt).nofsecs {
            let iz = *(*opt).secs.add(k as usize);
            let s = crate::imod::libiimod::mrcslice::slice_read_subm(
                hin,
                iz,
                b'z' as i8,
                (*opt).ix,
                (*opt).iy,
                (*opt).cx as i32,
                (*opt).cy as i32,
            );
            if s.is_null() {
                libc::printf(
                    c"ERROR: CLIP - reading %s %d".as_ptr(),
                    if (*opt).from_one != 0 {
                        c"view"
                    } else {
                        c"slice"
                    }
                    .as_ptr(),
                    iz + iz_add,
                );
                return -1;
            }
            if k == 0 {
                let full_size = ((*s).xsize as usize)
                    * ((*s).ysize as usize)
                    * if volume { (*opt).nofsecs as usize } else { 1 };
                let sample_size = full_size.min(1_000_000);
                sample.resize(sample_size, 0.);
                interval = 1.max((full_size + sample_size - 1) / sample_size) as i32;
                num_sample = (full_size / interval as usize) as i32;
            }
            if k == 0 || !volume {
                x = 0;
                y = 0;
                sample_index = 0;
                last_val = -1.0e37;
                first_val = 1.0e37;
            }
            let mut wrap = false;
            while sample_index < num_sample && !wrap {
                sample[sample_index as usize] = slice_get_pixel_magnitude(s, x, y);
                last_val = last_val.max(sample[sample_index as usize]);
                first_val = first_val.min(sample[sample_index as usize]);
                sample_index += 1;
                x += interval;
                while x >= (*s).xsize {
                    x -= (*s).xsize;
                    y += 1;
                    if y >= (*s).ysize {
                        y = 0;
                        wrap = true;
                    }
                }
            }
            if !volume || k == (*opt).nofsecs - 1 {
                if num_sample > 4000 {
                    let ind = (0.0005 * num_sample as f32) as i32;
                    first_val = crate::imod::libcfshr::percentile::percentile_float(
                        ind + 1,
                        sample.as_mut_ptr(),
                        num_sample,
                    );
                    last_val = crate::imod::libcfshr::percentile::percentile_float(
                        num_sample - ind,
                        sample.as_mut_ptr(),
                        num_sample,
                    );
                }
                let (mut dip, mut below_peak, mut above_peak) = (0., 0., 0.);
                let error = if last_val > first_val {
                    crate::imod::libcfshr::histogram::find_histogram_dip(
                        sample.as_mut_ptr(),
                        num_sample,
                        0,
                        bins.as_mut_ptr(),
                        1000,
                        first_val,
                        last_val,
                        &mut dip,
                        &mut below_peak,
                        &mut above_peak,
                        0,
                    )
                } else {
                    1
                };
                if volume {
                    libc::printf(
                        c"All %ss: ".as_ptr(),
                        if (*opt).from_one != 0 {
                            c"view"
                        } else {
                            c"slice"
                        }
                        .as_ptr(),
                    );
                } else {
                    libc::printf(
                        c"%s %d: ".as_ptr(),
                        if (*opt).from_one != 0 {
                            c"View"
                        } else {
                            c"Slice"
                        }
                        .as_ptr(),
                        iz + iz_add,
                    );
                }
                if error != 0 {
                    libc::printf(c"no histogram dip could be found\n".as_ptr());
                } else {
                    let mut num_below = 0;
                    for index in 0..num_sample {
                        if sample[index as usize] < dip {
                            num_below += 1;
                        }
                    }
                    libc::printf(
                        c"peaks at %.5g and %.5g  dip at %.5g  fraction below dip = %.4f\n"
                            .as_ptr(),
                        below_peak as f64,
                        above_peak as f64,
                        dip as f64,
                        num_below as f64 / num_sample as f64,
                    );
                }
            }
            slice_free(s);
        }
        0
    }
}
/// Matches C++ `correctDefects`.
pub unsafe fn correct_defects(
    slice: *mut Islice,
    nx_full: i32,
    ny_full: i32,
    opt: *mut ClipOptions,
) -> i32 {
    static mut FIRST_TIME: bool = true;
    unsafe {
        if crate::imod::libcfshr::islice::slice_mode_if_real((*slice).mode) < 0 {
            crate::imod::clip::clip::show_error(
                "clip with defect correction - The output slice mode must be byte, integer or floating point",
            );
            return -1;
        }
        let mut binning = 1;
        let first = FIRST_TIME;
        if crate::imod::clip::correct_defects::cor_def_setup_to_correct(
            nx_full,
            ny_full,
            &mut (*opt).defects,
            &mut (*opt).cam_size_x,
            &mut (*opt).cam_size_y,
            (*opt).scale_defects,
            (*opt).binning,
            &mut binning,
            if first { Some("-B") } else { None },
        ) != 0
        {
            crate::imod::clip::clip::show_error(
                "clip with defect correction - Image size is more than twice the size stored in the defect list",
            );
            return -1;
        }
        FIRST_TIME = false;
        let left = ((*opt).cam_size_x / binning - nx_full) / 2 + (*opt).cx as i32 - (*opt).ix / 2;
        let right = left + (*opt).ix;
        let top = ((*opt).cam_size_y / binning - ny_full) / 2 + (*opt).cy as i32 - (*opt).iy / 2;
        let bottom = top + (*opt).iy;
        if left < 0
            || top < 0
            || right > (*opt).cam_size_x / binning
            || bottom > (*opt).cam_size_y / binning
        {
            crate::imod::clip::clip::show_error(
                "clip with defect correction - The size and centering options select an area outside the camera field",
            );
            return -1;
        }
        crate::imod::clip::correct_defects::cor_def_correct_defects(
            &(*opt).defects,
            (*slice).data.b.cast(),
            (*slice).mode,
            binning,
            top,
            left,
            bottom,
            right,
        );
        0
    }
}
/// Matches C++ `write_vol`.
pub unsafe fn write_vol(vol: *mut *mut Islice, hout: *mut MrcHeader) -> i32 {
    unsafe {
        for k in 0..(*hout).nz {
            let slice = *vol.add(k as usize);
            if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                (*slice).data.b.cast(),
                (*hout).fp.cast(),
                hout,
                k,
                b'z' as i8,
            ) != 0
            {
                return -1;
            }
            slice_mmm(slice);
            if k == 0 {
                (*hout).amin = (*slice).min;
                (*hout).amax = (*slice).max;
                (*hout).amean = (*slice).mean;
            } else {
                if (*slice).min < (*hout).amin {
                    (*hout).amin = (*slice).min;
                }
                if (*slice).max > (*hout).amax {
                    (*hout).amax = (*slice).max;
                }
                (*hout).amean += (*slice).mean;
            }
        }
        (*hout).amean /= (*hout).nz as f32;
        crate::imod::libiimod::mrcfiles::mrc_head_write((*hout).fp.cast(), hout)
    }
}
/// Matches C++ `free_vol`.
pub unsafe fn free_vol(vol: *mut *mut Islice, z: i32) -> i32 {
    unsafe {
        for k in 0..z {
            slice_free(*vol.add(k as usize));
        }
        libc::free(vol.cast());
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::islice::{slice_create, slice_free, slice_put_val};
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write, mrc_read_slice,
    };

    #[test]
    fn combine_area_sums_uses_source_quadrant_grouping() {
        let mut sums = [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        assert_eq!(unsafe { combine_area_sums(sums.as_mut_ptr()) }, 34.);
        assert_eq!(&sums[..4], &[14., 22., 46., 54.]);
    }

    #[test]
    fn volume_statistics_follow_slice_magnitudes() {
        unsafe {
            let first = slice_create(2, 1, MRC_MODE_FLOAT);
            let second = slice_create(2, 1, MRC_MODE_FLOAT);
            slice_put_val(first, 0, 0, [1., 0., 0., 0.]);
            slice_put_val(first, 1, 0, [4., 0., 0., 0.]);
            slice_put_val(second, 0, 0, [2., 0., 0., 0.]);
            slice_put_val(second, 1, 0, [3., 0., 0., 0.]);
            let mut slices = [first, second];
            let mut volume = Istack {
                vol: slices.as_mut_ptr(),
                zsize: 2,
            };
            let (mut min, mut max, mut mean) = (0., 0., 0.);
            let (mut x, mut y, mut z) = (0, 0, 0);
            assert_eq!(
                clip_get_stat3d(
                    &mut volume,
                    &mut min,
                    &mut max,
                    &mut mean,
                    &mut x,
                    &mut y,
                    &mut z
                ),
                0
            );
            assert_eq!((min, max, mean), (1., 4., 2.5));
            assert_eq!((x, y, z), (1, 0, 0));
            slice_free(first);
            slice_free(second);
        }
    }

    #[test]
    fn write_vol_writes_real_mrc_slices_and_source_header_statistics() {
        unsafe {
            let first = slice_create(2, 1, MRC_MODE_FLOAT);
            let second = slice_create(2, 1, MRC_MODE_FLOAT);
            slice_put_val(first, 0, 0, [1., 0., 0., 0.]);
            slice_put_val(first, 1, 0, [5., 0., 0., 0.]);
            slice_put_val(second, 0, 0, [3., 0., 0., 0.]);
            slice_put_val(second, 1, 0, [7., 0., 0., 0.]);
            let vol = libc::malloc(2 * core::mem::size_of::<*mut Islice>()).cast::<*mut Islice>();
            *vol = first;
            *vol.add(1) = second;
            let fp = libc::tmpfile();
            let mut header: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_new(&mut header, 2, 1, 2, MRC_MODE_FLOAT), 0);
            header.fp = fp.cast();
            assert_eq!(mrc_head_write(fp, &mut header), 0);
            assert_eq!(write_vol(vol, &mut header), 0);
            assert_eq!((header.amin, header.amax, header.amean), (1., 7., 4.));
            libc::rewind(fp);
            let mut read: MrcHeader = core::mem::zeroed();
            assert_eq!(mrc_head_read(fp, &mut read), 0);
            let mut pixels = [0_f32; 2];
            assert_eq!(
                mrc_read_slice(pixels.as_mut_ptr().cast(), fp, &mut read, 1, b'z' as i8),
                0
            );
            assert_eq!(pixels, [3., 7.]);
            assert_eq!(free_vol(vol, 2), 0);
            libc::fclose(fp);
        }
    }
}
