//! Translation of `IMOD/clip/processing.cpp`.
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use std::io::Write as _;

use crate::imod::clip::clip::ClipOptions;
use crate::imod::libcfshr::islice::{
    Islice, Istack, MrcData, slice_get_pixel_magnitude, slice_get_val, slice_put_val,
};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_RGB, MrcHeader,
};
use crate::imod::libiimod::mrcslice::{slice_box, slice_mmm};

/// Matches C++ `clip_scaling`.
pub fn clip_scaling(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    // This deliberately follows the pixel loop in processing.cpp rather than using a
    // Rust image abstraction: its round-to-mode and complex-pixel behavior is part of
    // CLIP's observable output.
    use crate::imod::clip::clip::*;
    // This condition is intentionally evaluated before set_multifile_input_options,
    // which turns the default section list into the full input list.
    let copy_extra = opt.process == crate::imod::clip::clip::ClipOperation::Unwrap
        && opt.nofsecs == crate::imod::clip::clip::IP_DEFAULT
        && opt.oz == crate::imod::clip::clip::IP_DEFAULT;
    crate::imod::clip::file_io::set_multifile_input_options(opt, hin);
    let mut z = crate::imod::clip::file_io::set_output_options(opt, hout);
    if z < 0 {
        return z;
    }
    crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
    let mut base = if opt.val != crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.val
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
    let mut sd_arr: Vec<f32> = Vec::new();
    let mut sum_arr: Vec<f32> = Vec::new();
    let mut sqr_arr: Vec<f32> = Vec::new();
    let mut point_fp: Option<ImodFile> = None;
    match opt.process {
        crate::imod::clip::clip::ClipOperation::Brightness => {
            crate::imod::clip::clip::show_status("Brightness...\n");
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: brightness");
            min = hin.amin as f64;
        }
        crate::imod::clip::clip::ClipOperation::Shadow => {
            crate::imod::clip::clip::show_status("Shadow...\n");
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: shadow");
            min = hin.amax as f64;
        }
        crate::imod::clip::clip::ClipOperation::Contrast => {
            crate::imod::clip::clip::show_status("Contrast...\n");
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: contrast");
            min = hin.amean as f64;
        }
        crate::imod::clip::clip::ClipOperation::Resize => {
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: resized image");
        }
        crate::imod::clip::clip::ClipOperation::Threshold => {
            crate::imod::clip::clip::show_status("Threshold...\n");
            if opt.sano != 0 {
                threshold_low = hin.amin;
                threshold_high = hin.amax;
            }
            if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
                threshold_low = opt.low;
            }
            if opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 {
                threshold_high = opt.high;
            }
            if opt.thresh == crate::imod::clip::clip::IP_DEFAULT as f32 {
                crate::imod::clip::clip::show_error(
                    "clip threshold: You must enter a threshold value",
                );
                return -1;
            }
            if opt.min_size != crate::imod::clip::clip::IP_DEFAULT {
                return match crate::imod::clip::threshminsize::threshold_with_min_size(
                    hin,
                    hout,
                    opt,
                    threshold_low,
                    threshold_high,
                    z,
                ) {
                    Ok(()) => 0,
                    status => status.unwrap_err(),
                };
            };
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: thresholded");
            if let Some(name) = opt.point_out_name.clone() {
                crate::imod::libcfshr::b3dutil::imod_backup_file(&name);
                point_fp = ImodFile::open(&name, "w");
                if point_fp.is_none() {
                    crate::imod::clip::clip::show_error(&c_format(
                        "clip threshold: Error opening output file for points %s",
                        &[CArg::Str(&name)],
                    ));
                    return -1;
                }
            }
        }
        crate::imod::clip::clip::ClipOperation::Truncate => {
            crate::imod::clip::clip::show_status("Truncate...\n");
            trunc_low = opt.low != crate::imod::clip::clip::IP_DEFAULT as f32;
            trunc_high = opt.high != crate::imod::clip::clip::IP_DEFAULT as f32;
            trunc_mean = opt.sano != 0;
            if !trunc_low && !trunc_high {
                crate::imod::clip::clip::show_error(
                    "clip truncate: You must enter a low or a high limit",
                );
                return -1;
            }
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: truncated");
        }
        crate::imod::clip::clip::ClipOperation::Unwrap => {
            crate::imod::clip::clip::show_status("Unwrap...\n");
            if !matches!(hin.mode, 1 | 6) {
                crate::imod::clip::clip::show_error(
                    "clip truncate: Mode must be short or unsigned short integers",
                );
                return -1;
            }
            if hin.mode == 6 && opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
                crate::imod::clip::clip::show_error(
                    "clip truncate: You must enter a value to add with -n for mode 6 input",
                );
                return -1;
            }
            if opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
                opt.val = 32768.;
            }
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut *hout,
                b"clip: unwrapped integer values",
            );
            if copy_extra
                && crate::imod::libiimod::mrcfiles::mrc_copy_extra_header(&mut *hin, &mut *hout)
                    != 0
            {
                crate::imod::clip::clip::show_warning(
                    "clip warning: failed to copy extra header data",
                );
            }
        }
        crate::imod::clip::clip::ClipOperation::Logarithm => {
            crate::imod::clip::clip::show_status("Logarithm...\n");
            let title = c_format(
                "clip: logarithm after adding %g",
                &[CArg::Dbl((base as f64) as f64)],
            );
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, title.as_bytes());
        }
        crate::imod::clip::clip::ClipOperation::Sqroot => {
            crate::imod::clip::clip::show_status("Square root...\n");
            let title = c_format(
                "clip: square root after adding %g",
                &[CArg::Dbl((base as f64) as f64)],
            );
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, title.as_bytes());
        }
        crate::imod::clip::clip::ClipOperation::Integral => {
            polarity = if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
                1.
            } else {
                -1.
            };
            base = if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
                opt.high
            } else {
                opt.low
            };
            radius_center = opt.val;
            radius_inner = radius_center + 1.;
            radius_outer = radius_inner + 1_f32.max(0.5 * radius_center);
            if opt.new_xoverlap != crate::imod::clip::clip::IP_DEFAULT {
                radius_inner = opt.new_xoverlap as f32;
            }
            if opt.new_yoverlap != crate::imod::clip::clip::IP_DEFAULT {
                radius_outer = opt.new_yoverlap as f32;
            }
            crate::imod::clip::clip::show_status("Local integral...\n");
            border = radius_outer.ceil() as i32 + 1;
            let title = c_format(
                "clip: integral, threshold %g, radius %g",
                &[
                    CArg::Dbl((base as f64) as f64),
                    CArg::Dbl((radius_center as f64) as f64),
                ],
            );
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, title.as_bytes());
        }
        crate::imod::clip::clip::ClipOperation::BoxStandardDeviation => {
            sd_binning = opt.val.abs().round() as i32;
            let bin_x = opt.ix / sd_binning;
            let bin_y = opt.iy / sd_binning;
            let bin_size = (bin_x * bin_y) as usize;
            // `processing.cpp:224` B3DMALLOCs three float arrays and
            // exits if any is NULL; a `Vec` cannot fail here.
            sum_arr = vec![0_f32; bin_size];
            sqr_arr = vec![0_f32; bin_size];
            sd_arr = vec![0_f32; bin_size];
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
    if opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.val = 1.;
    }
    let alpha = opt.val as f64;
    // `processing.cpp:57`: 1.e-20 and 1.e-5 are double literals, so the
    // whole B3DMAX expression is evaluated in double before narrowing.
    let min_for_log = 1.0e-20_f64.max(1.0e-5 * (hin.amax - hin.amin) as f64) as f32;
    // Keep the source's file-major order.  `set_multifile_input_options`
    // has already checked every header, but each nonfirst input is opened
    // again here and its own header is passed to `sliceReadSubm`.
    let input_mode = (*hin).mode;
    let input_nx = (*hin).nx;
    let input_ny = (*hin).ny;
    let mut first_defect_setup = true;
    for f in 0..opt.infiles {
        let mut hdr = MrcHeader::default();
        let input_header: &mut MrcHeader = if f != 0 {
            hdr.fp =
                crate::imod::libiimod::iimage::ii_fopen(opt.fnames[(f) as usize].as_bytes(), "rb");
            if hdr.fp.is_none() {
                crate::imod::libcfshr::parse_params::exit_error(
                    c_format("Opening file %s.", &[CArg::Str(&opt.fnames[(f) as usize])])
                        .as_bytes(),
                );
            }
            if crate::imod::libiimod::mrcfiles::mrc_head_read(
                &mut hdr.fp.clone().unwrap(),
                &mut hdr,
            ) != 0
            {
                crate::imod::libcfshr::parse_params::exit_error(
                    c_format(
                        "Reading header of %s.",
                        &[CArg::Str(&opt.fnames[(f) as usize])],
                    )
                    .as_bytes(),
                );
            }
            &mut hdr
        } else {
            &mut *hin
        };
        for k in 0..opt.nofsecs {
            let Some(mut slice) = crate::imod::libiimod::mrcslice::slice_read_subm(
                input_header,
                opt.secs[(k) as usize],
                b'z',
                opt.ix,
                opt.iy,
                opt.cx as i32,
                opt.cy as i32,
                None,
            ) else {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            };
            if (opt.process == crate::imod::clip::clip::ClipOperation::Logarithm
                || opt.process == crate::imod::clip::clip::ClipOperation::Sqroot)
                && hout.mode == 2
                && crate::imod::libiimod::mrcslice::slice_float(&mut slice) < 0
            {
                crate::imod::clip::clip::show_error(
                    "clip: Error getting memory to convert slice to float.",
                );
                return -1;
            };
            if (opt.dim == 2 && opt.process != crate::imod::clip::clip::ClipOperation::Resize)
                || (opt.process == crate::imod::clip::clip::ClipOperation::Truncate && trunc_mean)
            {
                crate::imod::libiimod::mrcslice::slice_mmm(&mut slice);
                min = match opt.process {
                    crate::imod::clip::clip::ClipOperation::Brightness => slice.min as f64,
                    crate::imod::clip::clip::ClipOperation::Shadow => slice.max as f64,
                    _ => slice.mean as f64,
                };
            }
            // `processing.cpp:246-347` is an if/else chain: only these
            // operations walk the pixels at all.  Brightness, contrast and
            // shadow fall through to the final `mrc_slice_lie` arm and get
            // no per-pixel pass of their own.
            if opt.process == crate::imod::clip::clip::ClipOperation::Threshold {
                for j in 0..opt.iy {
                    for i in 0..opt.ix {
                        let mut val = [0_f32; 4];
                        slice_get_val(&slice, i, j, &mut val);
                        for (l, item) in val.iter_mut().take(slice.csize as usize).enumerate() {
                            // The <= makes the result the same as in 3dmod
                            *item = if *item <= opt.thresh {
                                threshold_low
                            } else {
                                if l == 0
                                    && let Some(fp) = point_fp.as_mut()
                                {
                                    let _ = fp.write_all(
                                        c_format(
                                            "%6d %6d %4d  %g\n",
                                            &[
                                                CArg::Int(i as i64),
                                                CArg::Int(j as i64),
                                                CArg::Int(opt.secs[k as usize] as i64),
                                                CArg::Dbl(*item as f64),
                                            ],
                                        )
                                        .as_bytes(),
                                    );
                                }
                                threshold_high
                            };
                        }
                        slice_put_val(&mut slice, i, j, val);
                    }
                }
            } else if opt.process == crate::imod::clip::clip::ClipOperation::Truncate {
                for j in 0..opt.iy {
                    for i in 0..opt.ix {
                        let mut val = [0_f32; 4];
                        slice_get_val(&slice, i, j, &mut val);
                        for item in val.iter_mut().take(slice.csize as usize) {
                            if trunc_low && *item < opt.low {
                                *item = if trunc_mean { min as f32 } else { opt.low };
                            }
                            if trunc_high && *item > opt.high {
                                *item = if trunc_mean { min as f32 } else { opt.high };
                            }
                        }
                        slice_put_val(&mut slice, i, j, val);
                    }
                }
            } else if opt.process == crate::imod::clip::clip::ClipOperation::Unwrap {
                // Unwrap: add a value and wrap values around
                let high = if input_mode == 6 { 65535. } else { 32767. };
                let low = high - 65535.;
                for j in 0..opt.iy {
                    for i in 0..opt.ix {
                        let mut val = [0_f32; 4];
                        slice_get_val(&slice, i, j, &mut val);
                        val[0] += opt.val;
                        if val[0] > high {
                            val[0] -= 65536.;
                        } else if val[0] < low {
                            val[0] += 65536.;
                        }
                        slice_put_val(&mut slice, i, j, val);
                    }
                }
            } else if opt.process == crate::imod::clip::clip::ClipOperation::Logarithm {
                for j in 0..opt.iy {
                    for i in 0..opt.ix {
                        let mut val = [0_f32; 4];
                        slice_get_val(&slice, i, j, &mut val);
                        for item in val.iter_mut().take(slice.csize as usize) {
                            // `processing.cpp:290` is `B3DMAX(minForLog, val[l] + base)`,
                            // i.e. `a > b ? a : b`, which yields the *second* operand when
                            // either is NaN.  `f32::max` returns the non-NaN one, so a NaN
                            // pixel would come out as `minForLog` here and as NaN natively.
                            let shifted = *item + base;
                            *item = if min_for_log > shifted {
                                min_for_log
                            } else {
                                shifted
                            }
                            .log10();
                        }
                        slice_put_val(&mut slice, i, j, val);
                    }
                }
            } else if opt.process == crate::imod::clip::clip::ClipOperation::Sqroot {
                for j in 0..opt.iy {
                    for i in 0..opt.ix {
                        let mut val = [0_f32; 4];
                        slice_get_val(&slice, i, j, &mut val);
                        for item in val.iter_mut().take(slice.csize as usize) {
                            // `processing.cpp:301`: `B3DMAX(0., val[l] + base)` -- same
                            // second-operand-on-NaN rule as the logarithm arm above.
                            let shifted = *item + base;
                            *item = if 0_f32 > shifted { 0_f32 } else { shifted }.sqrt();
                        }
                        slice_put_val(&mut slice, i, j, val);
                    }
                }
            } else if opt.process == crate::imod::clip::clip::ClipOperation::Integral {
                // `processing.cpp:317-318`: `sliceInit(&flSlice, ..., slice->data.f)`
                // then `sliceFloatEx(&flSlice, 0)`.  For a float slice that
                // conversion is a no-op (`mrcslice.c:259`), so `beadIntegral`
                // reads the *same* storage `slicePutVal` is writing below;
                // every other mode gets a converted copy.
                let mut integral_data = Vec::new();
                if slice.mode != crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT {
                    let length = usize::try_from(slice.xsize).ok().and_then(|xsize| {
                        usize::try_from(slice.ysize)
                            .ok()
                            .and_then(|ysize| xsize.checked_mul(ysize))
                    });
                    if length.is_none_or(|length| integral_data.try_reserve_exact(length).is_err())
                    {
                        crate::imod::clip::clip::show_error(
                            "clip: Error getting memory to convert slice to float.",
                        );
                        return -1;
                    }
                    for j in 0..slice.ysize {
                        for i in 0..slice.xsize {
                            let mut value = [0.; 4];
                            slice_get_val(&slice, i, j, &mut value);
                            if matches!(slice.mode, MRC_MODE_COMPLEX_SHORT | MRC_MODE_COMPLEX_FLOAT)
                            {
                                value[0] = (value[0] * value[0] + value[1] * value[1]).sqrt();
                            } else if slice.mode == MRC_MODE_RGB {
                                value[0] = value[0] * 0.3 + value[1] * 0.59 + value[2] * 0.11;
                            }
                            integral_data.push(value[0]);
                        }
                    }
                }
                for j in 0..opt.iy {
                    for i in 0..opt.ix {
                        let mut val = [0_f32; 4];
                        slice_get_val(&slice, i, j, &mut val);
                        if i < border
                            || i >= opt.ix - border
                            || j < border
                            // processing.cpp deliberately uses ix for this lower
                            // Y boundary too.  Preserve that non-square behavior.
                            || j >= opt.ix - border
                            || (opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 && val[0] > opt.low)
                            || (opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 && val[0] > opt.high)
                        {
                            val[0] = 0.;
                        } else {
                            let mut center_mean = 0.;
                            let mut annulus_mean = 0.;
                            // `processing.cpp:324`: `val[0] = B3DMAX(0., polarity * val[0])` --
                            // `a > b ? a : b`, so a NaN integral stays NaN; `f32::max` would return 0.
                            let integral = (polarity
                                * crate::imod::libcfshr::beadutil::bead_integral(
                                    if slice.mode
                                        == crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT
                                    {
                                        slice.data.f()
                                    } else {
                                        &integral_data
                                    },
                                    slice.xsize,
                                    slice.xsize,
                                    slice.ysize,
                                    radius_center,
                                    radius_inner,
                                    radius_outer,
                                    i as f32 + 0.5,
                                    j as f32 + 0.5,
                                    &mut center_mean,
                                    &mut annulus_mean,
                                    None,
                                    0.,
                                    Some(&mut base),
                                ) as f32);
                            val[0] = if 0. > integral { 0. } else { integral };
                        }
                        slice_put_val(&mut slice, i, j, val);
                    }
                }
            } else if opt.process == crate::imod::clip::clip::ClipOperation::BoxStandardDeviation {
                if crate::imod::libiimod::mrcslice::slice_float(&mut slice) < 0 {
                    crate::imod::clip::clip::show_error(
                        "clip: Error getting memory to convert slice to float.",
                    );
                    return -1;
                }
                let nx_bin = opt.ix / sd_binning;
                let ny_bin = opt.iy / sd_binning;
                if nx_bin <= 0 || ny_bin <= 0 {
                    return -1;
                }
                let size = (nx_bin * ny_bin) as usize;
                let (mut x_offset, mut y_offset) = (0_i32, 0_i32);
                crate::imod::libcfshr::multibinstat::make_standard_dev_map(
                    slice.data.f(),
                    opt.ix,
                    0,
                    (opt.ix - 1) * if opt.sano != 0 { -1 } else { 1 },
                    0,
                    opt.iy - 1,
                    sd_binning * if opt.val < 0. { 1 } else { -1 },
                    (opt.low.round() as i32) / sd_binning,
                    &mut sd_arr[..size],
                    &mut sum_arr[..size],
                    &mut sqr_arr[..size],
                    &mut x_offset,
                    &mut y_offset,
                );
                slice.xsize = nx_bin;
                slice.ysize = ny_bin;
                // `processing.cpp:344`: `memcpy(slice->data.f, sdArr, binSize * sizeof(float))`.
                slice.data.f_mut()[..size].copy_from_slice(&sd_arr[..size]);
            } else if opt.process != crate::imod::clip::clip::ClipOperation::Resize {
                crate::imod::libiimod::mrcslice::mrc_slice_lie(&mut slice, min, alpha);
            }
            if opt.read_defects != 0
                && correct_defects(&mut slice, input_nx, input_ny, opt, &mut first_defect_setup)
                    .is_err()
            {
                return -1;
            }
            if crate::imod::clip::file_io::clip_write_slice(&mut slice, hout, opt, k, &mut z, 1)
                .is_err()
            {
                if f != 0 {
                    crate::imod::libiimod::iimage::ii_fclose(&mut hdr.fp.clone().unwrap());
                }
                return -1;
            }
        }
        if f != 0 {
            crate::imod::libiimod::iimage::ii_fclose(&mut hdr.fp.clone().unwrap());
        }
    }
    if point_fp.is_some() {
        // `processing.cpp:346` fcloses the file; dropping the handle closes
        // it the same way.
        point_fp = None;
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clipEdge`.
pub fn clip_edge(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    use crate::imod::clip::clip::ClipOperation;
    if opt.mode == crate::imod::clip::clip::IP_DEFAULT {
        opt.mode = if opt.process == crate::imod::clip::clip::ClipOperation::Gradient {
            hin.mode
        } else {
            0
        };
    }
    if !matches!(hin.mode, 0 | 1 | 6 | 2)
        && !(hin.mode == 4 && opt.process != crate::imod::clip::clip::ClipOperation::Gradient)
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
    let (message, title): (&str, &[u8]) = match opt.process {
        crate::imod::clip::clip::ClipOperation::Gradient => {
            ("Taking gradient of", b"clip: gradient")
        }
        crate::imod::clip::clip::ClipOperation::Prewitt => {
            ("Applying Prewitt filter to", b"clip: Prewitt filter")
        }
        crate::imod::clip::clip::ClipOperation::Graham => {
            ("Applying Graham filter to", b"clip: Graham filter")
        }
        crate::imod::clip::clip::ClipOperation::Sobel => {
            ("Applying Sobel filter to", b"clip: Sobel filter")
        }
        _ => return -1,
    };
    crate::imod::libiimod::mrcfiles::mrc_head_label(hout, title);
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "clip: %s %d slices...\n",
            &[CArg::Str(message), CArg::Int((opt.nofsecs) as i64)],
        )
        .as_bytes(),
    );
    // `processing.cpp:416-458` reads a slice and, for the gradient, creates
    // an output and frees the input, every section.  As in `clip_convolve`,
    // freed slices are kept in `spares` and offered back by mode
    // (`slice_recreate`) to the next read and the next gradient output; both
    // overwrite every pixel before reading it.  At most two are kept.
    let mut spares: Vec<Islice> = Vec::new();
    let take_spare = |spares: &mut Vec<Islice>, mode: i32| {
        let index = spares.iter().position(|spare| spare.mode == mode)?;
        Some(spares.swap_remove(index))
    };
    for k in 0..opt.nofsecs {
        let Some(mut source) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            take_spare(&mut spares, hin.mode),
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        let mut out = if opt.process == crate::imod::clip::clip::ClipOperation::Gradient {
            let reuse = take_spare(&mut spares, source.mode);
            let Some(result) = crate::imod::libiimod::mrcslice::slice_gradient(&mut source, reuse)
            else {
                return -1;
            };
            // `sliceFree(s)` — kept for reuse.
            spares.push(source);
            result
        } else {
            if source.mode != 0 {
                crate::imod::libcfshr::islice::slice_min_max(&mut source);
                let mut scale = 1.;
                if source.max > source.min {
                    scale = 255. / (source.max - source.min) as f64;
                }
                if (0.95..=1.05).contains(&scale) {
                    scale = 0.95;
                }
                let fixed = source.min as f64 * scale / (scale - 1.);
                crate::imod::libiimod::mrcslice::mrc_slice_lie(&mut source, fixed, scale);
                crate::imod::libiimod::mrcslice::slice_new_mode(&mut source, 0);
            }
            crate::imod::libcfshr::islice::slice_min_max(&mut source);
            if opt.process == crate::imod::clip::clip::ClipOperation::Graham {
                crate::imod::libiimod::sliceproc::slice_byte_graham(&mut source);
            } else if opt.process == crate::imod::clip::clip::ClipOperation::Sobel {
                crate::imod::libiimod::sliceproc::slice_byte_edge_sobel(&mut source);
            } else {
                crate::imod::libiimod::sliceproc::slice_byte_edge_prewitt(&mut source);
            }
            source
        };
        if crate::imod::clip::file_io::clip_write_slice(out.as_mut(), hout, opt, k, &mut z, 1)
            .is_err()
        {
            return -1;
        };
        // A slice `clipWriteSlice` converted to another mode can serve
        // neither the read nor the gradient; it is freed here, as in the C.
        if out.mode == hin.mode {
            spares.push(out);
        }
        while spares.len() > 2 {
            spares.remove(0);
        }
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clip_convolve`.
pub fn clip_convolve(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
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
    // `processing.cpp:466` declares `char title[100]`; `snprintf` into it
    // keeps at most 99 bytes.
    let mut title = String::new();
    let message: &str;
    let blur: &mut [f32];
    if opt.mode == crate::imod::clip::clip::IP_DEFAULT {
        opt.mode = if opt.process == crate::imod::clip::clip::ClipOperation::Smooth {
            hin.mode
        } else {
            crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT
        };
    }
    let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
    if z < 0 {
        return z;
    }
    match opt.process {
        crate::imod::clip::clip::ClipOperation::Smooth => {
            if opt.val < 0. {
                smooth_3d = true;
                if opt.dim == 2 {
                    crate::imod::clip::clip::show_error(
                        "clip smooth: the -2d option cannot be entered with smoothing in 3D",
                    );
                    return -1;
                }
                if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
                    opt.low = 0.85;
                }
            }
            if opt.val > 1. {
                niter = opt.val.round() as i32;
            }
            if opt.low > 0. {
                crate::imod::libcfshr::filtxcorr::scaled_gaussian_kernel(
                    &mut gaussian_kernel,
                    &mut dim,
                    7,
                    opt.low,
                );
                if smooth_3d {
                    if hin.nz < dim {
                        title = c_format(
                            "clip smooth: 3D smoothing with sigma %.2f requires %d slices; input has only %d",
                            &[
                                CArg::Dbl(opt.low as f64),
                                CArg::Int(dim as i64),
                                CArg::Int(hin.nz as i64),
                            ],
                        );
                        title.truncate(99.min(title.len()));
                        crate::imod::clip::clip::show_error(&title);
                        return -1;
                    }
                    message = "Gaussian kernel 3D smoothing";
                    title = c_format(
                        "clip: Gaussian 3D smoothing, sigma %.2f",
                        &[CArg::Dbl(opt.low as f64)],
                    );
                } else {
                    message = "Gaussian kernel smoothing";
                    title = c_format(
                        "clip: Gaussian smoothing, sigma %.2f, %d iterations",
                        &[CArg::Dbl(opt.low as f64), CArg::Int(niter as i64)],
                    );
                }
                blur = &mut gaussian_kernel;
            } else {
                message = "Smoothing";
                title = c_format(
                    "clip: Standard smoothing, %d iterations",
                    &[CArg::Int(niter as i64)],
                );
                blur = &mut smooth_kernel;
            }
        }
        crate::imod::clip::clip::ClipOperation::Sharpen => {
            message = "Sharpening";
            crate::imod::libiimod::mrcfiles::mrc_head_label(hout, b"clip: sharpen");
            blur = &mut sharpen_kernel;
        }
        crate::imod::clip::clip::ClipOperation::Laplacian => {
            message = "Applying Laplacian to";
            crate::imod::libiimod::mrcfiles::mrc_head_label(hout, b"clip: Laplacian");
            blur = &mut laplacian_kernel;
        }
        _ => return -1,
    }
    if opt.process == crate::imod::clip::clip::ClipOperation::Smooth {
        title.truncate(99.min(title.len()));
        crate::imod::libiimod::mrcfiles::mrc_head_label(hout, title.as_bytes());
    }
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "clip: %s %d slices...\n",
            &[CArg::Str(message), CArg::Int(opt.nofsecs as i64)],
        )
        .as_bytes(),
    );
    if smooth_3d {
        return clip_median(hin, hout, opt, blur, dim, z);
    }
    // `processing.cpp:548-570` creates a slice for the read and one per
    // filter iteration and frees each when done; here the freed slices are
    // kept in `spares` and offered back by mode (`slice_recreate`) to the
    // next read (input mode) and the next filter output (float), so a
    // running loop allocates nothing when the input is float.  Each reused
    // slice is wholly overwritten before it is read.  At most two are kept
    // between sections, the next read's and the next filter output's.
    let mut spares: Vec<Islice> = Vec::new();
    let take_spare = |spares: &mut Vec<Islice>, mode: i32| {
        let index = spares.iter().position(|spare| spare.mode == mode)?;
        Some(spares.swap_remove(index))
    };
    for k in 0..opt.nofsecs {
        let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            take_spare(&mut spares, hin.mode),
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        for _ in 0..niter {
            s.mean = hin.amean;
            if opt.process == crate::imod::clip::clip::ClipOperation::Smooth {
                crate::imod::libiimod::mrcslice::slice_mmm(s.as_mut());
            }
            let Some(slice) = crate::imod::libcfshr::islice::slice_mat_filter(
                s.as_mut(),
                blur,
                dim,
                take_spare(&mut spares, crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT),
            ) else {
                crate::imod::clip::clip::show_error("clip: Error getting new slice for filtering.");
                return -1;
            };
            // `sliceFree(s); s = slice;` — the freed slice is kept for reuse.
            spares.push(std::mem::replace(&mut s, slice));
        }
        // When `clipWriteSlice` converts back to a non-float mode, that
        // conversion (`sliceNewMode`) allocates the new-mode slice itself;
        // spares that are not float are freed before it so they do not add
        // to the peak.  The converted slice is then the next read's spare.
        if s.mode != opt.mode {
            spares.retain(|spare| spare.mode == crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT);
        }
        if crate::imod::clip::file_io::clip_write_slice(s.as_mut(), hout, opt, k, &mut z, 1)
            .is_err()
        {
            return -1;
        }
        spares.push(s);
        if spares.len() > 2 {
            spares.remove(0);
        }
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clipMedian`.
pub fn clip_median(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
    kernel: &[f32],
    size_in: i32,
    mut z: i32,
) -> i32 {
    let mut size = size_in;
    let mut z_kernel = [0_f32; 7];
    let mut zsum = 0_f32;
    if kernel.is_empty() {
        if !matches!(hin.mode, 0 | 1 | 6 | 2) {
            crate::imod::clip::clip::show_error(
                "clip median: only byte, integer and float modes can be used",
            );
            return -1;
        }
        z = crate::imod::clip::file_io::set_options(opt, hin, hout);
        if z < 0 {
            return z;
        }
        if opt.val as i32 == crate::imod::clip::clip::IP_DEFAULT {
            opt.val = 3.;
        }
        if opt.mode != 0 && opt.mode != 1 && hin.mode != 6 && opt.mode != 2 {
            opt.mode = hin.mode;
        }
        size = 2.max(opt.val as i32);
        let title = c_format(
            "clip: %dD median filter, size %d",
            &[CArg::Int((opt.dim) as i64), CArg::Int((size) as i64)],
        );
        crate::imod::libiimod::mrcfiles::mrc_head_label(hout, title.as_bytes());
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "clip: median filtering %d slices...\n",
                &[CArg::Int((opt.nofsecs) as i64)],
            )
            .as_bytes(),
        );
    } else {
        for k in 0..size {
            z_kernel[k as usize] = kernel[(k + size * (size / 2)) as usize];
            zsum += z_kernel[k as usize];
        }
        for k in 0..size {
            z_kernel[k as usize] /= zsum;
        }
    }
    let depth = if opt.dim == 2 { 1 } else { size };
    let mut stack = Istack {
        slices: Vec::with_capacity(depth as usize),
    };
    let mut first = 0_i32;
    let mut last = -1_i32;
    let mut output: Option<Islice> = None;
    for k in 0..opt.nofsecs {
        if k == 0 || !kernel.is_empty() {
            // `clipWriteSlice` freed the previous slice at the end of the
            // last iteration (`file_io.cpp:358`), so the new one is the only
            // output slice alive.
            output = None;
            output = crate::imod::libcfshr::islice::slice_create(
                opt.ix,
                opt.iy,
                if kernel.is_empty() { opt.mode } else { 2 },
            );
            if output.is_none() {
                return -1;
            }
        }
        // `processing.cpp:640-650` keeps three separate cases.  Only the
        // kernel/smoothing case re-derives firstNeed from lastNeed so the
        // window always holds `size` slices; the median case leaves
        // firstNeed at secs[k] - size / 2, which makes the window narrower
        // at the ends of the volume.
        let (mut needed_first, mut needed_last) = if opt.dim == 2 {
            let sec = opt.secs[(k) as usize];
            (sec, sec)
        } else if !kernel.is_empty() {
            let sec = opt.secs[(k) as usize];
            let first_need = 0.max(sec - size / 2);
            let last_need = (hin.nz - 1).min(first_need + size - 1);
            (0.max(last_need + 1 - size), last_need)
        } else {
            let sec = opt.secs[(k) as usize];
            let first_need = 0.max(sec - size / 2);
            let last_need = (hin.nz - 1).min(sec + (size - 1) / 2);
            (first_need, last_need)
        };
        // `processing.cpp:652-662` compacts `v.vol` in place: a slice outside
        // the needed range is freed and the kept ones are copied down over it
        // (`v.vol[j++] = v.vol[i]`).  No second array is allocated there, so
        // this retains in place -- the closure sees the elements in their
        // original order and drops the discarded ones as the C frees them.
        let mut in_vol = first;
        stack.slices.retain(|_| {
            let sec = in_vol;
            in_vol += 1;
            sec >= needed_first && sec <= needed_last
        });
        if !stack.slices.is_empty() {
            first = last + 1 - stack.slices.len() as i32;
        }
        for sec in needed_first..=needed_last {
            if stack.slices.is_empty() || sec > last {
                let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
                    hin,
                    sec,
                    b'z',
                    opt.ix,
                    opt.iy,
                    opt.cx as i32,
                    opt.cy as i32,
                    None,
                ) else {
                    return -1;
                };
                let s = if kernel.is_empty() {
                    s
                } else {
                    let Some(f) = crate::imod::libcfshr::islice::slice_mat_filter(
                        s.as_mut(),
                        &kernel,
                        size,
                        None,
                    ) else {
                        crate::imod::clip::clip::show_error("clip: Error getting filtered slice.");
                        return -1;
                    };
                    f
                };
                if stack.slices.is_empty() {
                    first = sec;
                }
                stack.slices.push(s);
                last = sec;
            }
        }
        if !kernel.is_empty() {
            let output = output.as_mut().unwrap();
            let pixel_count = (opt.ix * opt.iy) as usize;
            // `processing.cpp:695`: `memset(slice->data.f, 0, opt->ix * opt->iy * 4)`.
            output.data.f_mut()[..pixel_count].fill(0.);
            for n in 0..size {
                let ind = (n + k - size / 2 - first).clamp(0, size - 1);
                // `processing.cpp:699-701`: `dataPtr = v.vol[j]->data.f;
                // slice->data.f[j] += zKernel[i] * dataPtr[j]`.
                let data_ptr = stack.slices[ind as usize].data.f();
                let out = output.data.f_mut();
                // Same elements in the same order with the same `a += w * d`
                // per element; only the per-element bounds checks go (a short
                // slice now panics before the loop rather than inside it,
                // with nothing written to any file in between).
                let weight = z_kernel[n as usize];
                for (o, &d) in out[..pixel_count].iter_mut().zip(&data_ptr[..pixel_count]) {
                    *o += weight * d;
                }
            }
        } else if crate::imod::libiimod::sliceproc::slice_median_filter(
            output.as_mut().unwrap(),
            &stack.slices,
            size,
        ) != 0
        {
            return -1;
        }
        if crate::imod::clip::file_io::clip_write_slice(
            output.as_mut().unwrap(),
            hout,
            opt,
            k,
            &mut z,
            if kernel.is_empty() { 0 } else { 1 },
        )
        .is_err()
        {
            return -1;
        }
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clipBlankFile`.
pub fn clip_blank_file(hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    let setup_z = crate::imod::clip::file_io::set_output_options(opt, hout);
    if setup_z < 0 {
        return setup_z;
    }
    let title = c_format(
        "clip blankfile: constant value %g",
        &[CArg::Dbl((opt.pad as f64) as f64)],
    );
    crate::imod::libiimod::mrcfiles::mrc_head_label(hout, title.as_bytes());
    let Some(mut slice) = crate::imod::libcfshr::islice::slice_create(opt.ox, opt.oy, opt.mode)
    else {
        crate::imod::libcfshr::parse_params::exit_error(
            b"Creating slice structure with data array",
        );
    };
    let val = [opt.pad, opt.pad, opt.pad, 0.];
    for iy in 0..opt.oy {
        for ix in 0..opt.ox {
            slice_put_val(&mut slice, ix, iy, val);
        }
    }
    let mut z = 0;
    opt.nofsecs = opt.oz;
    for iz in 0..opt.oz {
        if crate::imod::clip::file_io::clip_write_slice(&mut slice, hout, opt, iz, &mut z, 0)
            .is_err()
        {
            return -1;
        }
    }
    hout.xorg = 0.;
    hout.yorg = 0.;
    hout.zorg = 0.;
    hout.amin = opt.pad;
    hout.amax = opt.pad;
    hout.amean = opt.pad;
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return -1;
    }
    0
}
/// Matches C++ `clipDiffusion`.
pub fn clip_diffusion(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    if !matches!(hin.mode, 0 | 1 | 6 | 2) {
        crate::imod::clip::clip::show_error(
            "clip diffusion: only byte, integer and float modes can be used",
        );
        return -1;
    }
    let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
    if z < 0 {
        return z;
    }
    if opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.val = 5.;
    }
    let iterations = 1.max(opt.val as i32);
    if opt.thresh == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.thresh = 2.;
    }
    let cc = 1.max(3.min(opt.thresh as i32));
    if opt.weight == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.weight = 2.;
    }
    let kk = 0_f64.max(opt.weight as f64);
    if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.low = 0.2;
    }
    let lambda = 0.001_f64.max(opt.low as f64);
    crate::imod::libiimod::mrcfiles::mrc_head_label(hout, b"clip: diffusion");
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "clip: anistropic diffusion %d slices...\n",
            &[CArg::Int((opt.nofsecs) as i64)],
        )
        .as_bytes(),
    );
    for k in 0..opt.nofsecs {
        let Some(mut slice) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        if crate::imod::libiimod::sliceproc::slice_aniso_diff(
            &mut slice,
            opt.mode,
            cc,
            kk,
            lambda,
            iterations,
            crate::imod::libiimod::sliceproc::ANISO_CLEAR_AT_END,
        ) != 0
        {
            return -1;
        }
        if crate::imod::clip::file_io::clip_write_slice(&mut slice, hout, opt, k, &mut z, 1)
            .is_err()
        {
            return -1;
        }
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clip_flip`.
pub fn clip_flip(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> Result<(), i32> {
    // `processing.cpp:844` uses `opt->command` directly; `main` always sets
    // it from `argv[1]`, and the field is a `String` now.
    let command = opt.command.clone().into_bytes();
    hout.mode = if opt.mode == crate::imod::clip::clip::IP_DEFAULT {
        hin.mode
    } else {
        opt.mode
    };
    let changed_mode = hout.mode != hin.mode;
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
        hout.nx = hin.nx;
        hout.ny = hin.ny;
        hout.nz = hin.nz;
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, kind);
        if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0
        {
            return Err(-1);
        }
        // `processing.cpp:834-835`: "For most variations, it gets and frees a
        // slice inside the loop when the mode is changing, outside the loop
        // otherwise".  `:1157-1158` and `:1188-1189` create the slice once
        // ahead of the loop for `!newMode`; since `newMode` is
        // `hout->mode != hin->mode`, the mode argument is the same either way.
        let mut held: Option<Islice> = None;
        if !changed_mode {
            held = crate::imod::libcfshr::islice::slice_create(hin.nx, hin.ny, hin.mode);
            if held.is_none() {
                return Err(-1);
            }
        }
        for k in 0..hin.nz {
            if changed_mode {
                held = crate::imod::libcfshr::islice::slice_create(hin.nx, hin.ny, hin.mode);
            }
            let Some(sl) = held.as_mut() else {
                return Err(-1);
            };
            let input = if axis == b'z' { hin.nz - k - 1 } else { k };
            if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                sl.data.bytes_mut(),
                &mut hin.fp.clone().unwrap(),
                hin,
                input,
                b'z',
            ) != 0
            {
                return Err(-1);
            }
            if changed_mode && crate::imod::libiimod::mrcslice::slice_new_mode(sl, hout.mode) < 0 {
                return Err(-1);
            }
            if axis != b'z' {
                crate::imod::libiimod::mrcslice::slice_mirror(sl, axis);
            }
            if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                sl.data.bytes(),
                &mut hout.fp.clone().unwrap(),
                hout,
                k,
                b'z',
            ) != 0
            {
                return Err(-1);
            }
        }
        let _ = ImodFile::Stdout.write_all(" Done!\n".as_bytes());
        return Ok(());
    }
    if command.starts_with(b"flipxy") || command.starts_with(b"flipyx") {
        hout.nx = hin.ny;
        hout.ny = hin.nx;
        hout.nz = hin.nz;
        hout.mx = hin.my;
        hout.my = hin.mx;
        hout.mz = hin.mz;
        hout.xlen = hin.ylen;
        hout.ylen = hin.xlen;
        hout.zlen = hin.zlen;
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: flipxy");
        let limits = [0i32; 3];
        let mut num_tiles = [0i32; 3];
        let mut tile_sizes = [0i32; 3];
        if crate::imod::clip::file_io::set_chunk_output(
            opt,
            hout,
            &limits,
            &mut num_tiles,
            &mut tile_sizes,
        )
        .is_err()
        {
            return Err(-1);
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0
        {
            return Err(-1);
        }
        // `processing.cpp:860-861` creates the slice once ahead of the loop
        // when `newMode` is 0, and only inside it (`:863-864`) when the mode
        // is changing; `hout->mode == hin->mode` in the hoisted case.
        let mut held: Option<Islice> = None;
        if !changed_mode {
            held = crate::imod::libcfshr::islice::slice_create(hout.nx, hout.nz, hin.mode);
            if held.is_none() {
                return Err(-1);
            }
        }
        for x in 0..hin.nx {
            if changed_mode {
                held = crate::imod::libcfshr::islice::slice_create(hout.nx, hout.nz, hin.mode);
            }
            let Some(sl) = held.as_mut() else {
                return Err(-1);
            };
            if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                sl.data.bytes_mut(),
                &mut hin.fp.clone().unwrap(),
                hin,
                x,
                b'x',
            ) != 0
            {
                return Err(-1);
            }
            if changed_mode && crate::imod::libiimod::mrcslice::slice_new_mode(sl, hout.mode) < 0 {
                return Err(-1);
            }
            if opt.sano != 0 {
                crate::imod::libiimod::mrcslice::slice_mirror(sl, b'y');
            }
            if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                sl.data.bytes(),
                &mut hout.fp.clone().unwrap(),
                hout,
                x,
                b'y',
            ) != 0
            {
                return Err(-1);
            }
        }
        let _ = ImodFile::Stdout.write_all(" Done!\n".as_bytes());
        return Ok(());
    }
    if command.starts_with(b"flipxz") || command.starts_with(b"flipzx") {
        hout.nx = hin.nz;
        hout.ny = hin.ny;
        hout.nz = hin.nx;
        hout.mx = hin.mz;
        hout.mz = hin.mx;
        hout.xlen = hin.zlen;
        hout.zlen = hin.xlen;
        crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: flipxz");
        let limits = [0i32; 3];
        let mut num_tiles = [0i32; 3];
        let mut tile_sizes = [0i32; 3];
        if crate::imod::clip::file_io::set_chunk_output(
            opt,
            hout,
            &limits,
            &mut num_tiles,
            &mut tile_sizes,
        )
        .is_err()
        {
            return Err(-1);
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0
        {
            return Err(-1);
        }
        let Some(mut source) =
            crate::imod::libcfshr::islice::slice_create(hin.ny, hin.nz, hin.mode)
        else {
            return Err(-1);
        };
        let Some(mut transposed) =
            crate::imod::libcfshr::islice::slice_create(hout.nx, hout.ny, hout.mode)
        else {
            return Err(-1);
        };
        for x in 0..hin.nx {
            if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                source.data.bytes_mut(),
                &mut hin.fp.clone().unwrap(),
                hin,
                x,
                b'x',
            ) != 0
            {
                return Err(-1);
            }
            for y in 0..hin.ny {
                for z in 0..hin.nz {
                    let mut v = [0.; 4];
                    slice_get_val(&source, y, z, &mut v);
                    slice_put_val(transposed.as_mut(), z, y, v);
                }
            }
            if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                transposed.data.bytes(),
                &mut hout.fp.clone().unwrap(),
                hout,
                x,
                b'z',
            ) != 0
            {
                return Err(-1);
            }
        }
        let _ = ImodFile::Stdout.write_all(" Done!\n".as_bytes());
        return Ok(());
    }
    // `processing.cpp:925-1139`, YZ and ROTX.
    if command.starts_with(b"flipyz")
        || command.starts_with(b"flipzy")
        || command.starts_with(b"rotx")
    {
        let rotx = command.starts_with(b"rotx");
        crate::imod::libiimod::mrcfiles::mrc_head_label(
            &mut *hout,
            if rotx {
                b"clip: rotx - rotation by -90 around X"
            } else {
                b"clip: flipyz"
            },
        );

        // This one is not allowed to change mode
        hout.mode = hin.mode;
        hout.nx = hin.nx;
        hout.mx = hin.mx;
        hout.xlen = hin.xlen;
        hout.ny = hin.nz;
        hout.my = hin.mz;
        hout.ylen = hin.zlen;
        hout.nz = hin.ny;
        hout.mz = hin.my;
        hout.zlen = hin.ylen;

        // For rotation try to adjust the header as in rotatevol.  `:947-950`
        // assign to `float ycen, zcen` from *double* expressions (`/ 2.`), and
        // `yorg`/`zorg` are a subtract, multiply and divide all in double,
        // rounded to float once at the store; only `yorg * my / ylen` is a
        // float product.
        if rotx && hin.my != 0 && hin.ylen != 0. && hin.mz != 0 && hin.zlen != 0. {
            for i in 0..3 {
                hout.tiltangles[i] = hout.tiltangles[i + 3];
            }
            hout.tiltangles[3] -= 90.;
            let ycen = (hin.ny as f64 / 2. - (hin.yorg * hin.my as f32 / hin.ylen) as f64) as f32;
            let zcen = (hin.nz as f64 / 2. - (hin.zorg * hin.mz as f32 / hin.zlen) as f64) as f32;
            hout.yorg =
                ((hout.ny as f64 / 2. - zcen as f64) * hout.ylen as f64 / hout.my as f64) as f32;
            hout.zorg =
                ((hout.nz as f64 / 2. + ycen as f64) * hout.zlen as f64 / hout.mz as f64) as f32;
        }

        // Get the memory limit
        let memmin: f32 = 512000000.;
        let mut memmax: f32 = 2900000000.;
        let chunk_xy_crit: f32 = 1.0e6;
        if core::mem::size_of::<usize>() > 4 {
            memmax = (0.4 * crate::imod::libcfshr::b3dutil::b3d_physical_memory()) as f32;
        }
        let mut dsize = 0;
        let mut csize = 0;
        crate::imod::libiimod::mrcfiles::mrc_getdcsize(hout.mode, &mut dsize, &mut csize);
        let mut memlim = ((dsize as f32 * hout.nx as f32) * hout.ny as f32) * hout.nz as f32;
        // `B3DMAX(memmin, B3DMIN(memmax, memlim))`, spelled as the macros are.
        let inner = if memmax < memlim { memmax } else { memlim };
        memlim = if memmin > inner { memmin } else { inner };

        // Get the maximum number of full output slices to load.  `memlim` is
        // clamped to at most `dsize*nx*ny*nz` (or `memmin`), so the quotient
        // is at most about `nz` and the float-to-int conversion is in range.
        let dsize_nx = dsize.wrapping_mul(hout.nx);
        let mut max_slices = ((memlim / dsize_nx as f32) / hout.ny as f32) as i32;
        max_slices = max_slices.min(hout.nz).max(1);
        let mut k = (hout.nz + max_slices - 1) / max_slices;
        max_slices = (hout.nz + k - 1) / k;
        let mut num_load_slices = hin.nz;
        let mut ny_load_slice = max_slices;

        // If output is HDF, and this is not full size of input in Y, see if
        // chunks can be done and full slices loaded for that
        let ii_file =
            crate::imod::libiimod::iimage::ii_lookup_file_from_fp(&hout.fp.clone().unwrap());
        let limits = [0i32; 3];
        let mut num_tiles = [0i32; 3];
        let mut tile_sizes = [0i32; 3];
        tile_sizes[1] = 0;
        // `b3dutil.h:60` OUTPUT_TYPE_HDF is 5, the same value as IIFILE_HDF.
        if crate::imod::libcfshr::b3dutil::b3d_output_file_type()
            == crate::imod::libiimod::iimage::IIFILE_HDF
            && ii_file.is_some()
            && max_slices < hout.nz
        {
            let mut max_load = ((memlim / dsize_nx as f32) / hout.nz as f32) as i32;
            max_load = max_load.min(hout.ny).max(1);
            k = (hout.ny + max_load - 1) / max_load;
            max_load = (hout.ny + k - 1) / k;
            if max_load < hout.ny
                && max_load as f32 * (dsize as f32 * hout.nx as f32) > chunk_xy_crit
            {
                // Get a tile size that is no more than this
                if opt.chunk_y == crate::imod::clip::clip::IP_DEFAULT {
                    opt.chunk_y = max_load;
                }
                opt.chunk_y = opt.chunk_y.min(max_load);
                let chunk_lims = [0, max_load, 0];
                if crate::imod::clip::file_io::set_chunk_output(
                    opt,
                    hout,
                    &chunk_lims,
                    &mut num_tiles,
                    &mut tile_sizes,
                )
                .is_err()
                {
                    return Err(-1);
                }
                num_load_slices = tile_sizes[1];
                ny_load_slice = hin.ny;
            }
        }
        if tile_sizes[1] == 0
            && crate::imod::clip::file_io::set_chunk_output(
                opt,
                hout,
                &limits,
                &mut num_tiles,
                &mut tile_sizes,
            )
            .is_err()
        {
            return Err(-1);
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0
        {
            return Err(-1);
        }

        // Get the slice storage
        let Some(mut out_slice) =
            crate::imod::libcfshr::islice::slice_create(hout.nx, num_load_slices, hout.mode)
        else {
            let _ = ImodFile::Stdout
                .write_all(b"ERROR: CLIP - getting memory for slice array or output slice\n");
            return Err(-1);
        };
        let mut yslice: Vec<Islice> = Vec::with_capacity(num_load_slices.max(0) as usize);
        for _ in 0..num_load_slices {
            let Some(sl) =
                crate::imod::libcfshr::islice::slice_create(hout.nx, ny_load_slice, hout.mode)
            else {
                let _ = ImodFile::Stdout.write_all(b"ERROR: CLIP - getting memory for slices\n");
                return Err(-1);
            };
            yslice.push(sl);
        }

        let mut li = crate::imod::libiimod::mrcfiles::LoadInfo::default();
        crate::imod::libiimod::mrcfiles::mrc_init_li(Some(&mut li), None);
        crate::imod::libiimod::mrcfiles::mrc_init_li(Some(&mut li), Some(&*hin));
        let mut num_done = 0;
        // `:1056`/`:1118` copy `dsize * hout->nx` bytes per row and step both
        // offsets by the same amount, with `dsize` from `mrc_getdcsize` and
        // `csize` never used.  For the multi-channel modes (complex, RGB) that
        // is a fraction of a row, so native writes a scrambled volume; this is
        // translated as written (an upstream defect, not ours to repair).
        // Every access stays inside both slices, so no bound is exceeded.
        // The part of `outSlice` never copied into is `malloc` residue in the
        // C and zero here (`NATIVE.md`'s zero-filled `Vec` vs `malloc`).
        let row = dsize_nx as usize;
        if tile_sizes[1] != 0 {
            // `tileSizes[1]` is non-zero only after `setChunkOutput` tiled the
            // output, which fails when `iiLookupFileFromFP` finds no file
            // (`file_io.cpp`), so `iiFile` is non-NULL here as in the C.
            let Some(ii_file) = ii_file else {
                return Err(-1);
            };
            // SAFETY: `ii_file` is the live entry in the image-file list for
            // `hout->fp`, which stays open for the whole of this routine and
            // is not otherwise borrowed while this reference is used.
            let ii_file = unsafe { &mut *ii_file };
            ii_file.llx = 0;
            ii_file.urx = ii_file.nx - 1;
            ii_file.pad_left = 0;
            ii_file.pad_right = 0;
            for _chunk in 0..num_tiles[1] {
                let num_todo = tile_sizes[1].min(hin.nz - num_done);
                // Where the fallback `setChunkOutput` produced the tiling,
                // `nyLoadSlice` is `maxSlices` and can be less than `hin->ny`;
                // the C's full-section `mrcReadZ` then overruns `yslice[k]`.
                // `mrc_read_z` checks the buffer and returns an error instead.
                for k in 0..num_todo {
                    let err = crate::imod::libiimod::mrcsec::mrc_read_z(
                        hin,
                        &mut li,
                        yslice[k as usize].data.bytes_mut(),
                        k + num_done,
                    );
                    if err != 0 {
                        let _ = ImodFile::Stdout.write_all(
                            c_format(
                                "ERROR: CLIP - Reading full section %d (error # %d)\n",
                                &[CArg::Int(k as i64), CArg::Int(err as i64)],
                            )
                            .as_bytes(),
                        );
                        return Err(-1);
                    }
                }

                // Set limits for output loop
                let (ydir, yst, ynd) = if rotx {
                    (-1, hin.ny - 1, 0)
                } else {
                    (1, 0, hin.ny - 1)
                };

                // Write current portion of Z slices in order after copying
                // into output slice
                let mut j = yst;
                while j * ydir <= ynd * ydir {
                    let line_ofs = row * j as usize;
                    let mut slice_ofs = 0usize;
                    let out = out_slice.data.bytes_mut();
                    for k in 0..num_todo as usize {
                        out[slice_ofs..slice_ofs + row]
                            .copy_from_slice(&yslice[k].data.bytes()[line_ofs..line_ofs + row]);
                        slice_ofs += row;
                    }

                    ii_file.lly = num_done;
                    ii_file.ury = num_done + num_todo - 1;
                    let k = ydir * (j - yst);
                    if crate::imod::libiimod::iimage::ii_write_section(
                        ii_file,
                        out_slice.data.bytes_mut(),
                        k,
                    ) != 0
                    {
                        let _ = ImodFile::Stdout.write_all(
                            c_format(
                                "ERROR: CLIP - Writing y %d to %d of section %d\n",
                                &[
                                    CArg::Int(ii_file.lly as i64),
                                    CArg::Int(ii_file.ury as i64),
                                    CArg::Int(k as i64),
                                ],
                            )
                            .as_bytes(),
                        );
                        return Err(-1);
                    }
                    j += ydir;
                }
                num_done += num_todo;
            }
        } else {
            // Loop on chunks in Z of output
            while num_done < hout.nz {
                let num_todo = max_slices.min(hout.nz - num_done);

                // Set up loading limits and limits for output loop
                let ydir;
                let yst;
                let ynd;
                if rotx {
                    li.ymax = hout.nz - 1 - num_done;
                    li.ymin = li.ymax - (num_todo - 1);
                    ydir = -1;
                    yst = num_todo - 1;
                    ynd = 0;
                } else {
                    li.ymin = num_done;
                    li.ymax = num_done + num_todo - 1;
                    ydir = 1;
                    ynd = num_todo - 1;
                    yst = 0;
                }

                // Load the slices within the Y range
                for k in 0..hin.nz {
                    let err = crate::imod::libiimod::mrcsec::mrc_read_z(
                        hin,
                        &mut li,
                        yslice[k as usize].data.bytes_mut(),
                        k,
                    );
                    if err != 0 {
                        let _ = ImodFile::Stdout.write_all(
                            c_format(
                                "ERROR: CLIP - Reading section %d, y %d to %d (error # %d)\n",
                                &[
                                    CArg::Int(k as i64),
                                    CArg::Int(li.ymin as i64),
                                    CArg::Int(li.ymax as i64),
                                    CArg::Int(err as i64),
                                ],
                            )
                            .as_bytes(),
                        );
                        return Err(-1);
                    }
                }

                // Write Z slices in order after copying into output slice
                let mut j = yst;
                while j * ydir <= ynd * ydir {
                    let line_ofs = row * j as usize;
                    let mut slice_ofs = 0usize;
                    let out = out_slice.data.bytes_mut();
                    for sl in &yslice[..hin.nz as usize] {
                        out[slice_ofs..slice_ofs + row]
                            .copy_from_slice(&sl.data.bytes()[line_ofs..line_ofs + row]);
                        slice_ofs += row;
                    }

                    if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                        out_slice.data.bytes(),
                        &mut hout.fp.clone().unwrap(),
                        hout,
                        num_done,
                        b'z',
                    ) != 0
                    {
                        let _ = ImodFile::Stdout.write_all(
                            c_format(
                                "ERROR: CLIP - Writing section %d\n",
                                &[CArg::Int(num_done as i64)],
                            )
                            .as_bytes(),
                        );
                        return Err(-1);
                    }
                    num_done += 1;
                    j += ydir;
                }
            }
        }

        // Clean up
        drop(yslice);
        drop(out_slice);
        let _ = ImodFile::Stdout.write_all(" Done!\n".as_bytes());
        return Ok(());
    }
    crate::imod::clip::clip::show_warning("clip flip - no flipping was done.");
    Err(-1)
}
/// Matches C++ `clip_quadrant`.
pub fn clip_quadrant(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> Result<(), i32> {
    let (nx, ny, nz) = (hin.nx, hin.ny, hin.nz);
    let mut width = 20;
    let mut group_size = 1;
    let mut user_base = 0_f64;
    let mut num_todo = nz;
    if opt.ix != crate::imod::clip::clip::IP_DEFAULT
        || opt.iy != crate::imod::clip::clip::IP_DEFAULT
        || opt.ox != crate::imod::clip::clip::IP_DEFAULT
        || opt.oy != crate::imod::clip::clip::IP_DEFAULT
        || opt.oz != crate::imod::clip::clip::IP_DEFAULT
    {
        crate::imod::clip::clip::show_warning("clip quadrant - input and output sizes ignored.");
    }
    if opt.nofsecs != crate::imod::clip::clip::IP_DEFAULT {
        num_todo = opt.nofsecs;
        for iz in 0..num_todo - 1 {
            for jz in iz + 1..num_todo {
                if opt.secs[(jz) as usize] < opt.secs[(iz) as usize] {
                    opt.secs.swap(iz as usize, jz as usize);
                }
            }
        }
        let mut jz = 0;
        for iz in 0..num_todo {
            if jz == 0 || opt.secs[(jz - 1) as usize] != opt.secs[(iz) as usize] {
                opt.secs[(jz) as usize] = opt.secs[(iz) as usize];
                jz += 1;
            }
        }
        num_todo = jz;
    } else {
        crate::imod::clip::file_io::set_input_options(opt, hin);
    }
    opt.nofsecs = nz;
    if opt.val != crate::imod::clip::clip::IP_DEFAULT as f32 {
        group_size = opt.val.round() as i32;
    }
    if opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 {
        width = opt.high.round() as i32;
    }
    if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
        user_base = opt.low as f64;
    }
    if width < 2 || width > nx / 4 || width > ny / 4 {
        crate::imod::clip::clip::show_error("clip: width entry too small or too large.");
        return Err(-1);
    }
    group_size = group_size.clamp(1, nz);
    let num_groups = 1.max(num_todo / group_size);
    hout.mode = hin.mode;
    let mut new_mode = -1;
    if opt.mode != crate::imod::clip::clip::IP_DEFAULT && opt.mode != hin.mode {
        new_mode = crate::imod::libcfshr::islice::slice_mode_if_real(opt.mode);
        if new_mode < 0 {
            crate::imod::clip::clip::show_error("clip: Inappropriate new mode entry.");
            return Err(-1);
        }
        hout.mode = opt.mode;
    }
    hout.amean = 0.;
    hout.amin = 1.0e37;
    hout.amax = -1.0e37;
    if crate::imod::libiimod::mrcfiles::mrc_copy_extra_header(&mut *hin, &mut *hout) != 0 {
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
            iz = opt.secs[(ind) as usize];
            let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_mrc(hin, iz, b'z') else {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return Err(-1);
            };
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
                    s.as_mut(),
                    coords[n].0,
                    coords[n].1,
                    coords[n].2,
                    coords[n].3,
                    &mut mean,
                )
                .is_err()
                {
                    return Err(-1);
                }
                d[n] += mean;
            }
        }
        // `processing.cpp:1344-1350`: the seeds are 1.e30 / -1.e30 and the
        // divide happens inside the same loop as the min/max.
        let mut qmin = 1.0e30_f64;
        let mut qmax = -qmin;
        for v in &mut d {
            *v /= nin_group as f64;
            if qmin > *v {
                qmin = *v;
            }
            if qmax < *v {
                qmax = *v;
            }
        }
        let base = (if user_base != 0. { 0.01 } else { 0.05 }) * (qmax - qmin) - qmin - user_base;
        if base > 0. {
            crate::imod::clip::clip::show_warning(
                "clip - intensities being adjusted to avoid taking log of small or negative values",
            );
        }
        // `processing.cpp:1357`: `B3DMAX(0., base) + userBase` -- NaN-preserving.
        let base = if 0. > base { 0. } else { base } + user_base;
        let t = d.map(|v| (v + base).log10());
        let c2 = t[0] - t[6] + 2. * t[1] - 2. * t[3] + t[4] - t[2];
        let c3 = t[0] - t[6] + t[1] - t[3] + t[2] - t[4] + t[7] - t[5];
        let c4 = 2. * t[0] - 2. * t[6] + t[1] - t[3] + t[5] - t[7];
        // `processing.cpp:1370-1373` with `determ3` from `b3dutil.h:83`:
        // a1*b2*c3 - a1*b3*c2 + a2*b3*c1 - a2*b1*c3 + a3*b1*c2 - a3*b2*c1.
        let denom =
            6. * 4. * 6. - 6. * 2. * 2. + 2. * 2. * 4. - 2. * 2. * 6. + 4. * 2. * 2. - 4. * 4. * 4.;
        let g2 = (c2 * 4. * 6. - c2 * 2. * 2. + 2. * 2. * c4 - 2. * c3 * 6. + 4. * c3 * 2.
            - 4. * 4. * c4)
            / denom;
        let g3 = (6. * c3 * 6. - 6. * 2. * c4 + c2 * 2. * 4. - c2 * 2. * 6. + 4. * 2. * c4
            - 4. * c3 * 4.)
            / denom;
        let g4 = (6. * 4. * c4 - 6. * c3 * 2. + 2. * c3 * 4. - 2. * 2. * c4 + c2 * 2. * 2.
            - c2 * 4. * 4.)
            / denom;
        let gain = [
            10_f64.powf(-(g2 + g3 + g4)),
            10_f64.powf(g2),
            10_f64.powf(g3),
            10_f64.powf(g4),
        ];
        if nin_group > 1 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "Group from %d to %d:",
                    &[
                        CArg::Int((opt.secs[(start) as usize]) as i64),
                        CArg::Int((iz) as i64),
                    ],
                )
                .as_bytes(),
            );
        } else {
            let _ = ImodFile::Stdout
                .write_all(c_format("Section %d:", &[CArg::Int((iz) as i64)]).as_bytes());
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                " scale factors %.4f %.4f %.4f %.4f\n",
                &[
                    CArg::Dbl((gain[0]) as f64),
                    CArg::Dbl((gain[1]) as f64),
                    CArg::Dbl((gain[2]) as f64),
                    CArg::Dbl((gain[3]) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Boundary diffs before: %6.1f %6.1f %6.1f %6.1f\n",
                &[
                    CArg::Dbl((d[0] - d[6]) as f64),
                    CArg::Dbl((d[3] - d[1]) as f64),
                    CArg::Dbl((d[4] - d[2]) as f64),
                    CArg::Dbl((d[7] - d[5]) as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Boundary diffs after: %6.1f %6.1f %6.1f %6.1f\n",
                &[
                    CArg::Dbl((gain[0] * (d[0] + base) - gain[3] * (d[6] + base)) as f64),
                    CArg::Dbl((gain[1] * (d[3] + base) - gain[0] * (d[1] + base)) as f64),
                    CArg::Dbl((gain[2] * (d[4] + base) - gain[1] * (d[2] + base)) as f64),
                    CArg::Dbl((gain[3] * (d[7] + base) - gain[2] * (d[5] + base)) as f64),
                ],
            )
            .as_bytes(),
        );
        let mut end_out = iz;
        if group == num_groups - 1 {
            end_out = nz - 1;
        }
        for iz in last_out + 1..=end_out {
            let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_mrc(hin, iz, b'z') else {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return Err(-1);
            };
            if new_mode >= 0
                && crate::imod::libiimod::mrcslice::slice_new_mode(s.as_mut(), new_mode) < 0
            {
                crate::imod::clip::clip::show_error("clip: Error converting slice to new mode.");
                return Err(-1);
            }
            if (start..end).any(|ind| iz == opt.secs[(ind) as usize]) {
                correct_quadrant(s.as_mut(), nx / 2, ny / 2, nx, ny, gain[0], base);
                correct_quadrant(s.as_mut(), 0, ny / 2, nx / 2, ny, gain[1], base);
                correct_quadrant(s.as_mut(), 0, 0, nx / 2, ny / 2, gain[2], base);
                correct_quadrant(s.as_mut(), nx / 2, 0, nx, ny / 2, gain[3], base);
            }
            crate::imod::libiimod::mrcslice::slice_mmm(s.as_mut());
            hout.amin = hout.amin.min(s.min);
            hout.amax = hout.amax.max(s.max);
            hout.amean += s.mean / nz as f32;
            if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                s.data.bytes(),
                &mut hout.fp.clone().unwrap(),
                hout,
                iz,
                b'z',
            ) != 0
            {
                crate::imod::clip::clip::show_error("clip: Error writing slice.");
                return Err(-1);
            }
            last_out = iz;
        }
        start += nin_group;
    }
    crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: quadrant correction");
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return Err(-1);
    }
    Ok(())
}
/// Matches C++ `quadrantSample`.
fn quadrant_sample(
    slice: &mut Islice,
    llx: i32,
    lly: i32,
    urx: i32,
    ury: i32,
    mean: &mut f64,
) -> Result<(), i32> {
    let Some(mut box_slice) = slice_box(slice, llx, lly, urx, ury) else {
        crate::imod::clip::clip::show_error("clip: Error extracting subslice.");
        return Err(-1);
    };
    slice_mmm(box_slice.as_mut());
    *mean = box_slice.mean as f64;
    Ok(())
}
/// Matches C++ `correctQuadrant`.
fn correct_quadrant(
    slice: &mut Islice,
    llx: i32,
    lly: i32,
    urx: i32,
    ury: i32,
    gain: f64,
    base: f64,
) {
    // `processing.cpp:1458-1471`: the mode test sits at row level, with a
    // separate column loop for each arm.
    for iy in lly..ury {
        if slice.mode == crate::imod::libiimod::mrcslice::SLICE_MODE_FLOAT {
            for ix in llx..urx {
                let mut val = [0.; 4];
                slice_get_val(slice, ix, iy, &mut val);
                val[0] = (gain * (val[0] as f64 + base) - base) as f32;
                slice_put_val(slice, ix, iy, val);
            }
        } else {
            for ix in llx..urx {
                let mut val = [0.; 4];
                slice_get_val(slice, ix, iy, &mut val);
                val[0] = (gain * (val[0] as f64 + base) - base + 0.5).floor() as f32;
                slice_put_val(slice, ix, iy, val);
            }
        }
    }
}
/// Matches C++ `fillDriftCorrectedEdges`.
pub fn fill_drift_corrected_edges(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> i32 {
    let mut width = if hin.nx < 3000 { 30 } else { 60 };
    let mut length = 1024.max(hin.nx / 4);
    let mut crit = 9.;
    if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
        length = opt.low.round() as i32;
    }
    if opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 {
        crit = opt.high;
    }
    if opt.val != crate::imod::clip::clip::IP_DEFAULT as f32 {
        width = opt.val.round() as i32;
    }
    if !matches!(hin.mode, 1 | 6) {
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
    for k in 0..opt.nofsecs {
        let Some(mut slice) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "\nSLICE %d\n",
                &[CArg::Int((opt.secs[(k) as usize]) as i64)],
            )
            .as_bytes(),
        );
        if crate::imod::clip::correct_defects::cor_def_find_drift_corr_edges(
            slice.data.bytes(),
            hin.mode,
            opt.ix,
            opt.iy,
            length,
            width,
            crit,
            &mut defects.usable_left,
            &mut defects.usable_right,
            &mut defects.usable_top,
            &mut defects.usable_bottom,
        ) != 0
        {
            let message = c_format(
                "clip: error from CorDefFindDriftCorrEdges for slice %d.",
                &[CArg::Int((k) as i64)],
            );
            crate::imod::clip::clip::show_error(&message);
            return -1;
        }
        crate::imod::clip::correct_defects::cor_def_correct_defects(
            &defects,
            slice.data.bytes_mut(),
            hin.mode,
            1,
            0,
            0,
            hin.ny,
            hin.nx,
        );
        if crate::imod::clip::file_io::clip_write_slice(&mut slice, hout, opt, k, &mut z, 1)
            .is_err()
        {
            return -1;
        }
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clipSpectrum`.
pub fn clip_spectrum(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    let mut bkgd = 48_i32;
    let mut trunc = 0.02_f32;
    if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
        bkgd = if opt.low > 0. && opt.low < 1. {
            (255. * opt.low) as i32
        } else {
            opt.low as i32
        };
        if bkgd >= 192 {
            return 1;
        }
    }
    if opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 {
        trunc = opt.high;
        if trunc < 0. || trunc >= 0.75 {
            return 1;
        }
    }
    if opt.ox != crate::imod::clip::clip::IP_DEFAULT
        || opt.oy != crate::imod::clip::clip::IP_DEFAULT
    {
        if opt.ox != crate::imod::clip::clip::IP_DEFAULT
            && opt.oy != crate::imod::clip::clip::IP_DEFAULT
            && opt.ox != opt.oy
        {
            crate::imod::clip::clip::show_error(
                "clip spectrum: -ox and -oy cannot be entered with different sizes",
            );
            return 1;
        }
        if opt.ox != crate::imod::clip::clip::IP_DEFAULT {
            opt.oy = opt.ox
        } else {
            opt.ox = opt.oy
        }
    } else {
        opt.ox = 1024;
        opt.oy = 1024
    }
    if opt.oz != crate::imod::clip::clip::IP_DEFAULT {
        crate::imod::clip::clip::show_error("clip spectrum: -oz is not allowed");
        return 1;
    }
    let mode = if bkgd > 0 { 0 } else { 1 };
    if opt.mode == crate::imod::clip::clip::IP_DEFAULT {
        opt.mode = mode
    }
    if opt.add2file != 0 {
        crate::imod::clip::clip::show_error("clip spectrum: Cannot append to existing file");
        return 1;
    }
    crate::imod::clip::file_io::set_input_options(opt, hin);
    let pad = crate::imod::libcfshr::filtxcorr::nice_frame(
        opt.ix.max(opt.iy),
        2,
        crate::imod::libfft::nice_fft_limit(),
    );
    if (pad as f32) < opt.ox as f32 * 1.02 {
        opt.ox = pad;
        opt.oy = pad
    }
    let mut zout = crate::imod::clip::file_io::set_output_options(opt, hout);
    if zout < 0 {
        return 1;
    }
    crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
    crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: scaled power spectrum");
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "clip: Taking power spectrum of %d slices...\n",
            &[CArg::Int((opt.nofsecs) as i64)],
        )
        .as_bytes(),
    );
    // `processing.cpp:1611-1636` reads a slice and creates the output slice
    // every section (the read one is never freed; `clipWriteSlice` frees the
    // output).  Here both are handed back to the next section as storage
    // (`slice_recreate`), and the float copy of the input reuses its vector:
    // the read fills every input pixel, and `spectrumScaled` writes every
    // output pixel on each path it takes for clip's filter type 3 (the byte
    // scaling loop over `finalSize * finalSize`, `zoomWithFilter` into the
    // output when reducing, and the mirrored fill of every column otherwise).
    let mut spare_read: Option<Islice> = None;
    let mut spare_out: Option<Islice> = None;
    let mut input: Vec<f32> = Vec::new();
    for k in 0..opt.nofsecs {
        let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            spare_read.take(),
        ) else {
            let message = c_format(
                "clip: Error reading slice %d.",
                &[CArg::Int((opt.secs[(k) as usize]) as i64)],
            );
            crate::imod::clip::clip::show_error(&message);
            return -1;
        };
        let Some(mut out) =
            crate::imod::libcfshr::islice::slice_recreate(spare_out.take(), opt.ox, opt.oy, mode)
        else {
            crate::imod::clip::clip::show_error("clip: Error getting memory for spectrum slice.");
            return -1;
        };
        // `processing.cpp:1625` passes `s->data.b` with `s->mode` straight
        // through, and `sliceTaperInPad` (`taperpad.c:236-266`) converts each
        // mode to float itself.  Only a mode that switch has no case for
        // (complex) still goes through the per-pixel float conversion.
        use crate::imod::libcfshr::spectrumscaled::SpectrumInput;
        use crate::imod::libiimod::mrcslice::{
            SLICE_MODE_BYTE, SLICE_MODE_FLOAT, SLICE_MODE_RGB, SLICE_MODE_SHORT, SLICE_MODE_USHORT,
        };
        input.clear();
        let image = match s.mode {
            SLICE_MODE_BYTE => SpectrumInput::Byte(s.data.b()),
            SLICE_MODE_SHORT => SpectrumInput::Short(s.data.s()),
            SLICE_MODE_USHORT => SpectrumInput::UShort(s.data.us()),
            SLICE_MODE_FLOAT => SpectrumInput::Float(s.data.f()),
            SLICE_MODE_RGB => SpectrumInput::Rgb(s.data.b()),
            _ => {
                for y in 0..s.ysize {
                    for x in 0..s.xsize {
                        let mut value = [0.; 4];
                        crate::imod::libcfshr::islice::slice_get_val(&s, x, y, &mut value);
                        input.push(value[0]);
                    }
                }
                SpectrumInput::Float(&input)
            }
        };
        // `processing.cpp:1625` hands `spectrumScaled` the output slice's own
        // storage (`slice->data.b`); it fills the buffer in place and nothing
        // is copied afterwards.  `mode` here is byte when `bkgd > 0` and short
        // otherwise, which is exactly the member each arm writes.
        let err = if bkgd > 0 {
            crate::imod::libcfshr::spectrumscaled::spectrum_scaled(
                image,
                s.xsize,
                s.ysize,
                crate::imod::libcfshr::spectrumscaled::SpectrumOutput::Byte(out.data.b_mut()),
                pad,
                opt.ox,
                bkgd,
                trunc,
                3,
                crate::imod::libfft::todfft,
            )
        } else {
            crate::imod::libcfshr::spectrumscaled::spectrum_scaled(
                image,
                s.xsize,
                s.ysize,
                crate::imod::libcfshr::spectrumscaled::SpectrumOutput::Short(out.data.s_mut()),
                pad,
                opt.ox,
                bkgd,
                trunc,
                3,
                crate::imod::libfft::todfft,
            )
        };
        if err != 0 {
            let message = c_format(
                "clip: Error %d calling spectrumScaled",
                &[CArg::Int((err) as i64)],
            );
            crate::imod::clip::clip::show_error(&message);
            return 1;
        }
        if crate::imod::clip::file_io::clip_write_slice(out.as_mut(), hout, opt, k, &mut zout, 1)
            .is_err()
        {
            return -1;
        }
        spare_read = Some(s);
        spare_out = Some(out);
    }
    if pad > opt.ox {
        let (mut x, mut y, mut z) = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hin);
        let f = pad as f32 / opt.ox as f32;
        x *= f;
        y *= f;
        z *= f;
        crate::imod::libiimod::mrcfiles::mrc_set_scale(&mut *hout, x as f64, y as f64, z as f64);
    }
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return -1;
    }
    0
}
/// Matches C++ `clip_color`.
pub fn clip_color(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> Result<(), i32> {
    const DEFAULT: f32 = crate::imod::clip::clip::IP_DEFAULT as f32;
    if opt.red == DEFAULT {
        opt.red = 1.;
    }
    if opt.green == DEFAULT {
        opt.green = 1.;
    }
    if opt.blue == DEFAULT {
        opt.blue = 1.;
    }
    if opt.dim == 2 {
        return match clip2d_color(hin, hout, opt) {
            0 => Ok(()),
            status => Err(status),
        };
    }
    if [opt.ix, opt.iy, opt.iz, opt.ox, opt.oy, opt.oz]
        .iter()
        .any(|&v| v != crate::imod::clip::clip::IP_DEFAULT)
    {
        crate::imod::clip::clip::show_warning("clip 3d color - input and output sizes ignored.");
    }
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "clip: color (red, green, blue) = ( %g, %g, %g).\n",
            &[
                CArg::Dbl((opt.red as f64) as f64),
                CArg::Dbl((opt.green as f64) as f64),
                CArg::Dbl((opt.blue as f64) as f64),
            ],
        )
        .as_bytes(),
    );
    let data = crate::imod::libiimod::mrcfiles::mrc_read_byte(
        &mut hin.fp.clone().unwrap(),
        hin,
        None,
        None,
    );
    let Some(data) = data else {
        return Err(-1);
    };
    hout.nx = hin.nx;
    hout.ny = hin.ny;
    hout.nz = hin.nz;
    hout.mode = 16;
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return Err(-1);
    }
    for k in 0..hout.nz {
        for i in 0..hin.nx * hin.ny {
            let pixel = data[k as usize][i as usize] as f32;
            write_byte_pixel(pixel * opt.red, hout);
            write_byte_pixel(pixel * opt.green, hout);
            write_byte_pixel(pixel * opt.blue, hout);
        }
    }
    Ok(())
}
/// Matches C++ `writeBytePixel`.
fn write_byte_pixel(mut pixel: f32, hout: &mut MrcHeader) {
    if pixel > 255. {
        pixel = 255.;
    }
    // `processing.cpp:1702`: a float-to-unsigned-char conversion, which on
    // this target truncates toward zero into an int and keeps the low byte.
    let mut byte = (pixel + 0.5) as i32 as u8;
    if hout.bytes_signed != 0 {
        byte = ((byte as i32 - 128) & 255) as u8;
    }
    // Borrow the stream rather than cloning it: dropping an `ImodFile`
    // clone flushes the shared buffer (`b3dutil.rs`, `Drop for ImodFile`), so
    // a clone per call turned each byte into its own `write` syscall — `clip
    // color` ran ~40x slower than native.  The C writes through its buffered
    // `FILE *`.
    crate::imod::libcfshr::b3dutil::b3d_fwrite(
        core::slice::from_ref(&byte),
        1,
        1,
        hout.fp.as_mut().unwrap(),
    );
}
/// Matches C++ `clip2d_color`.
pub fn clip2d_color(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    if opt.mode != 16 && opt.mode != crate::imod::clip::clip::IP_DEFAULT {
        crate::imod::clip::clip::show_warning("clip - color output mode must be rgb.");
    }
    opt.mode = 16;
    let mut z = crate::imod::clip::file_io::set_options(opt, hin, hout);
    if z < 0 {
        return z;
    }
    crate::imod::clip::clip::show_status("False Color...\n");
    crate::imod::libiimod::mrcfiles::mrc_head_label(hout, b"CLIP Color");
    let Some(mut out) = crate::imod::libcfshr::islice::slice_create(opt.ix, opt.iy, 16) else {
        crate::imod::clip::clip::show_error("clip - creating slice");
        return -1;
    };
    for k in 0..opt.nofsecs {
        let Some(mut source) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        for j in 0..opt.iy {
            for i in 0..opt.ix {
                let mut value = [0.; 4];
                slice_get_val(&source, i, j, &mut value);
                let pixel = value[0];
                value[0] = (pixel * opt.red + 0.5).clamp(0., 255.);
                value[1] = (pixel * opt.green + 0.5).clamp(0., 255.);
                value[2] = (pixel * opt.blue + 0.5).clamp(0., 255.);
                slice_put_val(out.as_mut(), i, j, value);
            }
        }
        if crate::imod::clip::file_io::clip_write_slice(out.as_mut(), hout, opt, k, &mut z, 0)
            .is_err()
        {
            return -1;
        }
    }
    crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
}
/// Matches C++ `clip_joinrgb`.
pub fn clip_joinrgb(
    h1: &mut MrcHeader,
    h2: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> Result<(), i32> {
    use crate::imod::clip::clip::IP_APPEND_OVERWRITE;
    if opt.red == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.red = 1.;
    }
    if opt.green == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.green = 1.;
    }
    if opt.blue == crate::imod::clip::clip::IP_DEFAULT as f32 {
        opt.blue = 1.;
    }
    if opt.infiles != 3 {
        let _ = ImodFile::Stdout
            .write_all(b"ERROR: clip joinrgb - three input files must be specified\n");
        return Err(-1);
    }
    let mut headers = [h1.clone(), h2.clone(), MrcHeader::default()];
    headers[2].fp = crate::imod::libiimod::iimage::ii_fopen(opt.fnames[2].as_bytes(), "rb");
    if headers[2].fp.is_none() {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "ERROR: clip joinrgb - opening %s.\n",
                &[CArg::Str(&opt.fnames[2])],
            )
            .as_bytes(),
        );
        return Err(-1);
    }
    if crate::imod::libiimod::mrcfiles::mrc_head_read(
        &mut headers[2].fp.clone().unwrap(),
        &mut headers[2],
    ) != 0
    {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "ERROR: clip joinrgb - reading %s.\n",
                &[CArg::Str(&opt.fnames[2])],
            )
            .as_bytes(),
        );
        return Err(-1);
    }
    if headers
        .iter()
        .any(|h| h.nx != h1.nx || h.ny != h1.ny || h.nz != h1.nz)
    {
        let _ = ImodFile::Stdout.write_all(b"ERROR: clip joinrgb - all files must be same size.\n");
        crate::imod::libiimod::iimage::ii_fclose(&mut headers[2].fp.clone().unwrap());
        return Err(-1);
    }
    if headers.iter().any(|h| h.mode != 0) {
        let _ = ImodFile::Stdout.write_all(b"ERROR: clip joinrgb - all files must be bytes.\n");
        return Err(-1);
    }
    let mut start = 0;
    if opt.add2file != 0 {
        if opt.add2file == IP_APPEND_OVERWRITE {
            let _ = ImodFile::Stdout
                .write_all(b"ERROR: clip joinrgb - Overwriting is not allowed, only appending\n");
            return Err(-1);
        }
        if hout.mode != 16 {
            let _ = ImodFile::Stdout
                .write_all(b"ERROR: clip joinrgb - Mode of file being appended to must be 16\n");
            return Err(-1);
        }
        if hout.nx != h1.nx || hout.ny != h1.ny {
            let _ = ImodFile::Stdout.write_all(b"ERROR: clip joinrgb - File being appended to is not same X/Y size as input files\n");
            return Err(-1);
        }
        let (xs, ys, zs) = crate::imod::libiimod::mrcfiles::mrc_get_scale(hout);
        start = hout.nz;
        hout.nz += h1.nz;
        if hout.mz == start {
            hout.mz += h1.nz;
        }
        crate::imod::libiimod::mrcfiles::mrc_set_scale(hout, xs as f64, ys as f64, zs as f64);
    } else {
        hout.mode = 16;
        crate::imod::libiimod::mrcfiles::mrc_head_label(hout, b"CLIP Join 3 files into RGB");
    }
    hout.amin = 0.;
    hout.amax = 255.;
    hout.amean = 128.;
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return Err(-1);
    }
    let Some(mut rgb) = crate::imod::libcfshr::islice::slice_create(h1.nx, h1.ny, 16) else {
        let _ = ImodFile::Stdout.write_all(b"ERROR: CLIP - getting memory for slices\n");
        return Err(-1);
    };
    let mut component = Vec::with_capacity(3);
    for _ in 0..3 {
        let Some(p) = crate::imod::libcfshr::islice::slice_create(h1.nx, h1.ny, 0) else {
            let _ = ImodFile::Stdout.write_all(b"ERROR: CLIP - getting memory for slices\n");
            return Err(-1);
        };
        component.push(p);
    }
    for k in 0..h1.nz {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "\rJoining section %d of %d",
                &[CArg::Int((k + 1) as i64), CArg::Int((h1.nz) as i64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.flush();
        for n in 0..3 {
            if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                component[n].data.bytes_mut(),
                &mut headers[n].fp.clone().unwrap(),
                &mut headers[n],
                k,
                b'z',
            ) != 0
            {
                return Err(-1);
            }
        }
        for y in 0..h1.ny {
            for x in 0..h1.nx {
                let mut value = [0.; 4];
                for n in 0..3 {
                    let mut one = [0.; 4];
                    slice_get_val(component[n].as_mut(), x, y, &mut one);
                    value[n] = one[0] * [opt.red, opt.green, opt.blue][n];
                }
                slice_put_val(rgb.as_mut(), x, y, value);
            }
        }
        if crate::imod::libiimod::mrcfiles::mrc_write_slice(
            rgb.data.bytes(),
            &mut hout.fp.clone().unwrap(),
            hout,
            k + start,
            b'z',
        ) != 0
        {
            return Err(-1);
        }
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
    crate::imod::libiimod::iimage::ii_fclose(&mut headers[2].fp.clone().unwrap());
    Ok(())
}
/// Matches C++ `clip_splitrgb`.
pub fn clip_splitrgb(h1: &mut MrcHeader, opt: &mut ClipOptions) -> Result<(), i32> {
    if h1.mode != 16 {
        let _ = ImodFile::Stdout.write_all(b"ERROR: clip splitrgb - mode is not RGB\n");
        return Err(-1);
    }
    opt.ocanresize = 0;
    crate::imod::clip::file_io::set_multifile_input_options(opt, h1);
    // `processing.cpp:1901` sizes `fname` from `opt->fnames[1]` but writes
    // `opt->fnames[opt->infiles]` into it; a `String` has no such limit.
    let base = opt.fnames[opt.infiles as usize].clone();
    let mut headers: [MrcHeader; 3] = [
        MrcHeader::default(),
        MrcHeader::default(),
        MrcHeader::default(),
    ];
    let Some(mut rgb) = crate::imod::libcfshr::islice::slice_create(h1.nx, h1.ny, 16) else {
        let _ = ImodFile::Stdout.write_all(b"ERROR: clip - getting memory for slices\n");
        return Err(-1);
    };
    let mut planes = Vec::with_capacity(3);
    for n in 0..3 {
        let name = c_format(
            "%s%s",
            &[CArg::Str(&base), CArg::Str([".r", ".g", ".b"][n])],
        );
        if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none() {
            crate::imod::libcfshr::b3dutil::imod_backup_file(&name);
        }
        headers[n] = (*h1).clone();
        headers[n].nz = opt.nofsecs * opt.infiles;
        headers[n].mz = headers[n].nz;
        crate::imod::libiimod::mrcfiles::mrc_coord_cp(&mut headers[n], &*h1);
        if headers[n].mz != 0 {
            headers[n].zorg -=
                opt.secs[(0) as usize] as f32 * headers[n].zlen / headers[n].mz as f32;
        }
        headers[n].fp = crate::imod::libiimod::iimage::ii_fopen(name.as_bytes(), "wb+");
        if headers[n].fp.is_none() {
            let _ = ImodFile::Stdout
                .write_all(c_format("ERROR: clip - opening %s\n", &[CArg::Str(&name)]).as_bytes());
            return Err(-1);
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
            &mut headers[n].fp.clone().unwrap(),
            &mut headers[n],
        ) != 0
        {
            return Err(-1);
        }
        let Some(plane) = crate::imod::libcfshr::islice::slice_create(h1.nx, h1.ny, 0) else {
            let _ = ImodFile::Stdout.write_all(b"ERROR: clip - getting memory for slices\n");
            return Err(-1);
        };
        planes.push(plane);
    }
    for file in 0..opt.infiles {
        let mut input = if file == 0 {
            (*h1).clone()
        } else {
            MrcHeader::default()
        };
        if file != 0 {
            input.fp = crate::imod::libiimod::iimage::ii_fopen(
                opt.fnames[(file) as usize].as_bytes(),
                "rb",
            );
            if input.fp.is_none()
                || crate::imod::libiimod::mrcfiles::mrc_head_read(
                    &mut input.fp.clone().unwrap(),
                    &mut input,
                ) != 0
            {
                return Err(-1);
            }
        }
        for k in 0..opt.nofsecs {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "\rSplitting section %d of %d",
                    &[CArg::Int((k + 1) as i64), CArg::Int((opt.nofsecs) as i64)],
                )
                .as_bytes(),
            );
            if opt.infiles > 1 {
                let _ = ImodFile::Stdout
                    .write_all(c_format(", file %d", &[CArg::Int((file + 1) as i64)]).as_bytes());
            }
            let _ = ImodFile::Stdout.flush();
            if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                rgb.data.bytes_mut(),
                &mut input.fp.clone().unwrap(),
                &mut input,
                opt.secs[(k) as usize],
                b'z',
            ) != 0
            {
                return Err(-1);
            }
            for y in 0..h1.ny {
                for x in 0..h1.nx {
                    let mut pixel = [0.; 4];
                    slice_get_val(rgb.as_mut(), x, y, &mut pixel);
                    for n in 0..3 {
                        slice_put_val(planes[n].as_mut(), x, y, [pixel[n], 0., 0., 0.]);
                    }
                }
            }
            for n in 0..3 {
                if crate::imod::libiimod::mrcfiles::mrc_write_slice(
                    planes[n].data.bytes(),
                    &mut headers[n].fp.clone().unwrap(),
                    &mut headers[n],
                    file * opt.nofsecs + k,
                    b'z',
                ) != 0
                {
                    return Err(-1);
                }
                crate::imod::libiimod::mrcslice::slice_mmm(planes[n].as_mut());
                headers[n].amin = headers[n].amin.min(planes[n].min);
                headers[n].amax = headers[n].amax.max(planes[n].max);
                headers[n].amean += planes[n].mean;
            }
        }
        if file != 0 {
            crate::imod::libiimod::iimage::ii_fclose(&mut input.fp.clone().unwrap());
        }
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
    for n in 0..3 {
        headers[n].amean /= opt.nofsecs as f32;
        if crate::imod::libiimod::mrcfiles::mrc_head_write(
            &mut headers[n].fp.clone().unwrap(),
            &mut headers[n],
        ) != 0
        {
            return Err(-1);
        }
        crate::imod::libiimod::iimage::ii_fclose(&mut headers[n].fp.clone().unwrap());
    }
    Ok(())
}
/// Matches C++ `clip_average`.
pub fn clip_average(
    h1: &mut MrcHeader,
    h2: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> i32 {
    if opt.infiles == 1
        && matches!(
            opt.process,
            crate::imod::clip::clip::ClipOperation::Average
                | crate::imod::clip::clip::ClipOperation::Variance
                | crate::imod::clip::clip::ClipOperation::StandardDeviation
        )
    {
        return clip2d_average(h1, hout, opt);
    }
    if opt.add2file != 0 {
        crate::imod::clip::clip::show_error(
            "clip volume combining: you cannot add to an existing output file",
        );
        return -1;
    }
    if opt.infiles < 2
        || (opt.process == crate::imod::clip::clip::ClipOperation::Subtract && opt.infiles != 2)
    {
        crate::imod::clip::clip::show_error(
            if opt.process == crate::imod::clip::clip::ClipOperation::Subtract {
                "clip subtract: needs exactly two input files."
            } else {
                "clip add: needs at least two input files."
            },
        );
        return -1;
    }
    let mut z = crate::imod::clip::file_io::set_options(opt, h1, hout);
    if z < 0 {
        return z;
    }
    // `processing.cpp:2017`: both scales are double and start at 1.
    let mut valscale = 1.0_f64;
    let mut varscale = 1.0_f64;
    if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
        valscale = opt.low as f64;
        varscale = (opt.low * opt.low) as f64;
    }
    let mut variance = 0;
    // `processing.cpp:2043-2073`: the process switch also sets valscale, and
    // IP_SUBTRACT resets it to 1, discarding any -l entry.
    match opt.process {
        crate::imod::clip::clip::ClipOperation::Average => {
            valscale /= opt.infiles as f64;
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: 3D Averaged");
        }
        crate::imod::clip::clip::ClipOperation::Add => {
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: Summed");
        }
        crate::imod::clip::clip::ClipOperation::Variance => {
            variance = 1;
            valscale /= opt.infiles as f64;
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: 3D Variance");
        }
        crate::imod::clip::clip::ClipOperation::StandardDeviation => {
            variance = 2;
            valscale /= opt.infiles as f64;
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                &mut *hout,
                b"clip: 3D Standard Deviation",
            );
        }
        crate::imod::clip::clip::ClipOperation::Subtract => {
            valscale = 1.;
            crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, b"clip: Subtract");
        }
        _ => return -1,
    }
    let slice_mode = if h1.mode == crate::imod::libiimod::mrcfiles::MRC_MODE_COMPLEX_FLOAT {
        crate::imod::libiimod::mrcfiles::MRC_MODE_COMPLEX_FLOAT
    } else if h1.mode == crate::imod::libiimod::mrcfiles::MRC_MODE_RGB {
        99
    } else {
        crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT
    };
    // `processing.cpp:2075-2083`: the sum-of-squares slice is made once.
    let mut sq = if variance != 0 {
        let Some(slice) = crate::imod::libcfshr::islice::slice_create(opt.ix, opt.iy, slice_mode)
        else {
            return -1;
        };
        Some(slice)
    } else {
        None
    };
    // `processing.cpp:2084-2103`: every input header is opened and checked
    // before the section loop starts.
    // `processing.cpp:2084` mallocs an array of header pointers and one
    // header per input; a `Vec<MrcHeader>` owns both.
    let mut headers: Vec<MrcHeader> = (0..opt.infiles).map(|_| MrcHeader::default()).collect();
    for f in 0..opt.infiles {
        let h = &mut headers[f as usize];
        h.fp = crate::imod::libiimod::iimage::ii_fopen(opt.fnames[(f) as usize].as_bytes(), "rb");
        if h.fp.is_none() {
            crate::imod::clip::clip::show_error(&c_format(
                "clip volume combining: error opening %s.",
                &[CArg::Str(&opt.fnames[(f) as usize])],
            ));
            return -1;
        };
        if crate::imod::libiimod::mrcfiles::mrc_head_read(&mut h.fp.clone().unwrap(), h) != 0 {
            crate::imod::clip::clip::show_error(&c_format(
                "clip volume combining: error reading header of %s.",
                &[CArg::Str(&opt.fnames[(f) as usize])],
            ));
            return -1;
        }
        if h1.nx != h.nx || h1.ny != h.ny || h1.nz != h.nz || h1.mode != h.mode {
            crate::imod::clip::clip::show_error(
                "clip volume combining: all files must be the same size and mode.",
            );
            return -1;
        };
    }
    // h1/h2 are the dispatcher-opened first two files; subsequent input files follow
    // the exact C loop by being opened from fnames for each section.
    // `processing.cpp:2112-2152` creates the output slice every section (and
    // never frees it) and reads and frees one slice per input file.  Here
    // each is handed on as the next one's storage (`slice_recreate`): the
    // output is zeroed pixel by pixel before it is read, and the read writes
    // every pixel of the area.
    let mut spare_out: Option<Islice> = None;
    let mut spare_in: Option<Islice> = None;
    for k in 0..opt.nofsecs {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "\rclip: %s slice %d of %d\n",
                &[
                    CArg::Str(match opt.process {
                        crate::imod::clip::clip::ClipOperation::Add => "Adding",
                        crate::imod::clip::clip::ClipOperation::Subtract => "Subtracting",
                        _ => "Averaging",
                    }),
                    CArg::Int((k + 1) as i64),
                    CArg::Int((opt.nofsecs) as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.flush();
        let Some(mut out) = crate::imod::libcfshr::islice::slice_recreate(
            spare_out.take(),
            opt.ix,
            opt.iy,
            slice_mode,
        ) else {
            return -1;
        };
        let mut factor = 1.0_f32;
        let val = [0.; 4];
        for j in 0..opt.iy {
            for i in 0..opt.ix {
                slice_put_val(out.as_mut(), i, j, val);
                if variance != 0 {
                    slice_put_val(sq.as_mut().unwrap(), i, j, val);
                }
            }
        }
        for file in 0..opt.infiles {
            let input_header = &mut headers[file as usize];
            let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
                input_header,
                opt.secs[(k) as usize],
                b'z',
                opt.ix,
                opt.iy,
                opt.cx as i32,
                opt.cy as i32,
                spare_in.take(),
            ) else {
                return -1;
            };
            for y in 0..opt.iy {
                for x in 0..opt.ix {
                    let (mut a, mut v) = ([0.; 4], [0.; 4]);
                    slice_get_val(s.as_mut(), x, y, &mut v);
                    slice_get_val(out.as_mut(), x, y, &mut a);
                    a[0] += factor * v[0];
                    a[1] += factor * v[1];
                    a[2] += factor * v[2];
                    slice_put_val(out.as_mut(), x, y, a);
                    if variance != 0 {
                        slice_get_val(sq.as_mut().unwrap(), x, y, &mut a);
                        a[0] += v[0] * v[0];
                        a[1] += v[1] * v[1];
                        // BUGS.md, fixed in translation: `processing.cpp:2150` is
                        // `oval[1] += val[2] * val[2];` -- index 1, not 2, so native folds
                        // channel 2's sum of squares into channel 1 and never accumulates
                        // channel 2's own.  Each channel accumulates its own square here.
                        a[2] += v[2] * v[2];
                        slice_put_val(sq.as_mut().unwrap(), x, y, a);
                    }
                }
            }
            // `sliceFree(s)` — kept for the next read.
            spare_in = Some(s);
            if opt.process == crate::imod::clip::clip::ClipOperation::Subtract {
                factor = -1.;
            }
        }
        // `processing.cpp:2160-2161`.
        if valscale != 1. {
            crate::imod::libiimod::mrcslice::mrc_slice_valscale(out.as_mut(), valscale);
        }
        if variance != 0 {
            let f = opt.infiles;
            for y in 0..opt.iy {
                for x in 0..opt.ix {
                    let (mut a, mut v) = ([0.; 4], [0.; 4]);
                    slice_get_val(sq.as_mut().unwrap(), x, y, &mut v);
                    slice_get_val(out.as_mut(), x, y, &mut a);
                    for n in 0..3 {
                        // `processing.cpp:2168`: val[l] * varscale is double,
                        // f * oval[l] * oval[l] is float, the subtraction and
                        // the division by (f - 1.) are double.
                        let mut t = (v[n] as f64 * varscale - (f as f32 * a[n] * a[n]) as f64)
                            / (f as f64 - 1.);
                        if 0. > t {
                            t = 0.;
                        }
                        a[n] = t as f32;
                        if variance > 1 {
                            a[n] = (a[n] as f64).sqrt() as f32;
                        }
                    }
                    slice_put_val(out.as_mut(), x, y, a);
                }
            }
        }
        // When `clipWriteSlice` converts the output, no slice is kept: the
        // spare read slice is freed first so the conversion does not add to
        // the peak, and the converted slice is freed as in the C.  (Keeping
        // the converted slice for the next read measured worse on byte input.)
        if out.mode != opt.mode {
            spare_in = None;
        }
        if crate::imod::clip::file_io::clip_write_slice(out.as_mut(), hout, opt, k, &mut z, 1)
            .is_err()
        {
            return -1;
        }
        if out.mode == slice_mode {
            spare_out = Some(out);
        }
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
    hout.amean /= opt.nofsecs as f32;
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return -1;
    }
    crate::imod::clip::file_io::set_mrc_coords(h1, hout, opt)
}
/// Matches C++ `clip2d_average`.
pub fn clip2d_average(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    if opt.ox != crate::imod::clip::clip::IP_DEFAULT
        || opt.oy != crate::imod::clip::clip::IP_DEFAULT
        || opt.oz != crate::imod::clip::clip::IP_DEFAULT
    {
        crate::imod::clip::clip::show_warning("clip - ox, oy, oz have no effect for 2d average.");
    }
    opt.ox = crate::imod::clip::clip::IP_DEFAULT;
    opt.oy = crate::imod::clip::clip::IP_DEFAULT;
    opt.oz = crate::imod::clip::clip::IP_DEFAULT;
    crate::imod::clip::file_io::set_input_options(opt, hin);
    opt.oz = 1;
    let z = crate::imod::clip::file_io::set_output_options(opt, hout);
    if z < 0 {
        return z;
    }
    crate::imod::libiimod::mrcfiles::mrc_head_label_cp(&*hin, &mut *hout);
    let variance = match opt.process {
        crate::imod::clip::clip::ClipOperation::Average => 0,
        crate::imod::clip::clip::ClipOperation::Variance => 1,
        crate::imod::clip::clip::ClipOperation::StandardDeviation => 2,
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
    // `processing.cpp:2236`: SLICE_MODE_MAX is 99 (`mrcslice.h:28`), not
    // the RGB byte mode.
    let mode = if hin.mode == crate::imod::libiimod::mrcfiles::MRC_MODE_RGB {
        99
    } else {
        2
    };
    let Some(mut avgs) = crate::imod::libcfshr::islice::slice_create(opt.ix, opt.iy, mode) else {
        return -1;
    };
    let mut counts = vec![0_f32; (opt.ix * opt.iy) as usize];
    let mut squares = if variance != 0 {
        let Some(slice) = crate::imod::libcfshr::islice::slice_create(opt.ix, opt.iy, mode) else {
            return -1;
        };
        Some(slice)
    } else {
        None
    };
    let aval = [0.; 4];
    for j in 0..avgs.ysize {
        for i in 0..avgs.xsize {
            slice_put_val(avgs.as_mut(), i, j, aval);
            if variance != 0 {
                slice_put_val(squares.as_mut().unwrap(), i, j, aval);
            }
        }
    }
    for k in 0..opt.nofsecs {
        let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        // `processing.cpp:2232-2233`: `thresh` is decided once, not per pixel.
        let thresh = opt.val != crate::imod::clip::clip::IP_DEFAULT as f32;
        for y in 0..opt.iy {
            for x in 0..opt.ix {
                if thresh && slice_get_pixel_magnitude(s.as_ref(), x, y) <= opt.val {
                    continue;
                }
                let ind = (x + y * opt.ix) as usize;
                let (mut a, mut v) = ([0.; 4], [0.; 4]);
                slice_get_val(avgs.as_mut(), x, y, &mut a);
                slice_get_val(s.as_mut(), x, y, &mut v);
                for n in 0..3 {
                    a[n] += v[n];
                }
                slice_put_val(avgs.as_mut(), x, y, a);
                if variance != 0 {
                    slice_get_val(squares.as_mut().unwrap(), x, y, &mut a);
                    for n in 0..3 {
                        a[n] += v[n] * v[n];
                    }
                    slice_put_val(squares.as_mut().unwrap(), x, y, a);
                }
                counts[ind] += 1.;
            }
        }
    }
    let scale = if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
        1.
    } else {
        opt.low
    };
    for y in 0..opt.iy {
        for x in 0..opt.ix {
            let count = counts[(x + y * opt.ix) as usize];
            let mut a = [0.; 4];
            slice_get_val(avgs.as_mut(), x, y, &mut a);
            if count > 0. {
                for n in 0..3 {
                    a[n] *= scale / count;
                }
                if variance != 0 {
                    if count > 1. {
                        let mut ss = [0.; 4];
                        slice_get_val(squares.as_mut().unwrap(), x, y, &mut ss);
                        for n in 0..3 {
                            // `processing.cpp:2168-2169`: the variance, then
                            // `B3DMAX(0., oval[l])` -- NaN-preserving, not `f32::max`.
                            let variance_n =
                                (ss[n] * scale * scale - count * a[n] * a[n]) / (count - 1.);
                            a[n] = if 0. > variance_n { 0. } else { variance_n };
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
            slice_put_val(avgs.as_mut(), x, y, a);
        }
    }
    // `processing.cpp:2328-2329` returns without a diagnostic here.
    if avgs.mode != hout.mode
        && crate::imod::libiimod::mrcslice::slice_new_mode(avgs.as_mut(), hout.mode) < 0
    {
        return -1;
    }
    // `processing.cpp:2330` is sliceMMM, which sets mean as well as min and
    // max.  sliceMinMax leaves `mean` at zero, so the output header's amean
    // was written as 0 instead of the averaged mean.
    crate::imod::libiimod::mrcslice::slice_mmm(avgs.as_mut());
    // `processing.cpp:2332-2333`: `B3DMIN(avgs->min, hout->amin)` and the
    // matching `B3DMAX` -- `a < b ? a : b` with the SECTION value first, so a NaN
    // in either leaves `hout->amin` untouched.  `f32::min`/`max` would instead
    // return whichever operand is not NaN.
    hout.amin = if avgs.min < hout.amin {
        avgs.min
    } else {
        hout.amin
    };
    hout.amax = if avgs.max > hout.amax {
        avgs.max
    } else {
        hout.amax
    };
    if opt.add2file != 1 {
        hout.amean += avgs.mean / hout.nz as f32;
    }
    // `processing.cpp:2336-2337`: carry the input pixel spacing to the output.
    let (sx, sy, sz) = crate::imod::libiimod::mrcfiles::mrc_get_scale(&*hin);
    crate::imod::libiimod::mrcfiles::mrc_set_scale(&mut *hout, sx as f64, sy as f64, sz as f64);
    if crate::imod::libiimod::mrcfiles::mrc_write_slice(
        avgs.data.bytes(),
        &mut hout.fp.clone().unwrap(),
        hout,
        z,
        b'z',
    ) != 0
    {
        return -1;
    }
    let ret = crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout);
    ret
}
/// Matches C++ `clip_multdiv`.
pub fn clip_multdiv(
    h1: &mut MrcHeader,
    h2: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> i32 {
    use crate::imod::clip::clip::IP_APPEND_FALSE;
    use crate::imod::libiimod::mrcfiles::{MRC_MODE_COMPLEX_FLOAT, MRC_MODE_FLOAT, mrc_getdcsize};
    if opt.infiles != 2 {
        crate::imod::clip::clip::show_error("clip multiply/divide: Need exactly two input files.");
        return -1;
    }
    let mut z = crate::imod::clip::file_io::set_options(opt, h1, hout);
    if z < 0 {
        return z;
    }
    if opt.add2file != IP_APPEND_FALSE {
        crate::imod::clip::clip::show_error(
            "clip multiply/divide: you cannot add to an existing output file",
        );
        return -1;
    }
    let (mut dsize, mut csize1, mut csize2) = (0, 0, 0);
    mrc_getdcsize(h1.mode, &mut dsize, &mut csize1);
    mrc_getdcsize(h2.mode, &mut dsize, &mut csize2);
    if !(csize2 == 1
        || (h1.mode == MRC_MODE_COMPLEX_FLOAT && h2.mode == MRC_MODE_COMPLEX_FLOAT)
        || hout.mode == MRC_MODE_FLOAT)
    {
        crate::imod::clip::clip::show_error(
            "clip multiply/divide: second file must have single-channel data unless both are FFTs or output mode is float",
        );
        return -1;
    }
    if h1.nx != h2.nx || h1.ny != h2.ny {
        crate::imod::clip::clip::show_error("clip  multiply/divide: X and Y sizes must be equal");
        return -1;
    }
    if h1.nz != h2.nz && h2.nz > 1 {
        crate::imod::clip::clip::show_error(
            "clip  multiply/divide: Z sizes must be the same, or equal to 1 for second file",
        );
        return -1;
    }
    let scale = if opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
        1.
    } else {
        opt.val
    };
    let (message, title_proc) = match opt.process {
        crate::imod::clip::clip::ClipOperation::Multiply => ("Multiplying", "Multiply"),
        crate::imod::clip::clip::ClipOperation::Divide => ("Dividing", "Divide"),
        _ => return -1,
    };
    let title = if opt.val != crate::imod::clip::clip::IP_DEFAULT as f32 {
        c_format(
            "clip: %s, scaled by %.2f",
            &[CArg::Str(title_proc), CArg::Dbl(scale as f64)],
        )
    } else {
        c_format("clip: %s", &[CArg::Str(title_proc)])
    };
    crate::imod::libiimod::mrcfiles::mrc_head_label(&mut *hout, title.as_bytes());
    let do_round = csize1 * csize2 == 1
        && hout.mode != MRC_MODE_FLOAT
        && (h1.mode == MRC_MODE_FLOAT
            || h2.mode == MRC_MODE_FLOAT
            || opt.process == crate::imod::clip::clip::ClipOperation::Divide);
    let mut read_once = if h2.nz == 1 && h1.nz > 1 { 1 } else { 0 };
    let mut s: Option<Islice> = None;
    let mut div_by_zero = 0;
    let mut first_defect_setup = true;
    // `processing.cpp:2393-2480` reads (and frees) both slices every section;
    // each freed one is handed to the next read of its file as storage
    // (`slice_recreate`), which the read overwrites completely.
    let mut spare_out: Option<Islice> = None;
    let mut spare_s: Option<Islice> = None;
    for k in 0..opt.nofsecs {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "\rclip: %s slice %d of %d",
                &[
                    CArg::Str(message),
                    CArg::Int((k + 1) as i64),
                    CArg::Int((opt.nofsecs) as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.flush();
        let Some(mut out) = crate::imod::libiimod::mrcslice::slice_read_subm(
            h1,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            spare_out.take(),
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        if out.mode != hout.mode
            && crate::imod::libiimod::mrcslice::slice_new_mode(out.as_mut(), hout.mode) < 0
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
                    opt.secs[(k) as usize]
                },
                b'z',
                opt.ix,
                opt.iy,
                opt.cx as i32,
                opt.cy as i32,
                spare_s.take(),
            );
            if s.is_none() {
                crate::imod::clip::clip::show_error("clip: Error reading slice.");
                return -1;
            }
            read_once = -read_once;
            if csize2 > 1
                && hout.mode == MRC_MODE_FLOAT
                && crate::imod::libiimod::mrcslice::slice_float(s.as_mut().unwrap()) != 0
            {
                let _ =
                    ImodFile::Stdout.write_all(b"ERROR: CLIP - getting memory for slice array\n");
                return -1;
            }
        }
        for y in 0..opt.iy {
            for x in 0..opt.ix {
                let (mut a, mut b) = ([0.; 4], [0.; 4]);
                slice_get_val(out.as_mut(), x, y, &mut a);
                slice_get_val(s.as_mut().unwrap(), x, y, &mut b);
                if do_round {
                    // `processing.cpp:2454`: B3DNINT is (int)floor(x + 0.5).
                    if opt.process == crate::imod::clip::clip::ClipOperation::Multiply {
                        a[0] = ((a[0] * b[0] * scale) as f64 + 0.5).floor() as i32 as f32;
                    } else if b[0] != 0. {
                        a[0] = ((a[0] * (scale / b[0])) as f64 + 0.5).floor() as i32 as f32;
                    } else {
                        div_by_zero += 1;
                        a[0] = 0.;
                    }
                } else if csize2 == 1 || hout.mode == MRC_MODE_FLOAT {
                    if opt.process == crate::imod::clip::clip::ClipOperation::Multiply {
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
                    if opt.process == crate::imod::clip::clip::ClipOperation::Multiply {
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
                slice_put_val(out.as_mut(), x, y, a);
            }
        }
        if opt.read_defects != 0
            && correct_defects(out.as_mut(), h2.nx, h2.ny, opt, &mut first_defect_setup).is_err()
        {
            return -1;
        }
        if crate::imod::clip::file_io::clip_write_slice(out.as_mut(), hout, opt, k, &mut z, 1)
            .is_err()
        {
            return -1;
        }
        // Each is kept only while still in its file's mode; a converted
        // slice is freed here, as in the C.
        if out.mode == h1.mode {
            spare_out = Some(out);
        }
        if read_once == 0 || k == opt.nofsecs - 1 {
            // `sliceFree(s)` — kept for the next read, unless the next
            // section converts its first slice: holding this one through
            // that conversion would only raise the peak.
            spare_s = s
                .take()
                .filter(|s| s.mode == h2.mode && h1.mode == hout.mode);
        }
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
    if div_by_zero > 0 {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "WARNING: Division by zero occurred %d times\n",
                &[CArg::Int((div_by_zero) as i64)],
            )
            .as_bytes(),
        );
    }
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0 {
        return -1;
    }
    crate::imod::clip::file_io::set_mrc_coords(h1, hout, opt)
}
/// Matches C++ `clipPlanarFit`.
pub fn clip_planar_fit(hin: &mut MrcHeader, hout: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    crate::imod::clip::file_io::set_input_options(opt, hin);
    let fitting = opt.process == crate::imod::clip::clip::ClipOperation::Planarfit
        || opt.val != crate::imod::clip::clip::IP_DEFAULT as f32;
    let mut order = 0_i32;
    if opt.process == crate::imod::clip::clip::ClipOperation::Flatfield {
        if fitting {
            order = (opt.val as i32).clamp(1, 4);
        }
        opt.oz = 1;
        let e = crate::imod::clip::file_io::set_output_options(opt, hout);
        if e < 0 {
            return e;
        }
        if fitting {
            let label = c_format(
                "clip: Flatfield based on order %d fit to image sum",
                &[CArg::Int(order as i64)],
            );
            crate::imod::libiimod::mrcfiles::mrc_head_label(hout, label.as_bytes());
        } else {
            crate::imod::libiimod::mrcfiles::mrc_head_label(
                hout,
                b"clip: Flatfield based on image sum",
            );
        }
    }
    let n = (opt.ix * opt.iy) as usize;
    // `processing.cpp:2575`: `sumBuf = B3DMALLOC(float, numPix)`, which
    // `fullArrayMinMaxMean` and `mrc_write_slice` later take as a slice's
    // float storage.
    let mut sum = MrcData::F(vec![0_f32; n]);
    let base = if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
        0.
    } else {
        opt.low
    };
    let mut count = 0;
    // `processing.cpp:2532` declares `int indProc = 0` and never assigns
    // it, so `prefix[indProc]` is always the plane-fit string.
    let prefix = "Doing plane fit:";
    let _ = ImodFile::Stdout.write_all(b"clip: summing slices...");
    let _ = ImodFile::Stdout.flush();
    // C `clipPlanarFit` opens each named input independently: the header
    // handed to sliceReadSubm must be that file's header, not `hin`.
    for f in 0..opt.infiles {
        let mut hdr = MrcHeader::default();
        let input_header: &mut MrcHeader = if f == 0 {
            hin
        } else {
            hdr.fp =
                crate::imod::libiimod::iimage::ii_fopen(opt.fnames[(f) as usize].as_bytes(), "rb");
            if hdr.fp.is_none() {
                crate::imod::libcfshr::parse_params::exit_error(
                    c_format(
                        "\n%s error opening %s.",
                        &[CArg::Str(prefix), CArg::Str(&opt.fnames[(f) as usize])],
                    )
                    .as_bytes(),
                );
            }
            if crate::imod::libiimod::mrcfiles::mrc_head_read(
                &mut hdr.fp.clone().unwrap(),
                &mut hdr,
            ) != 0
            {
                crate::imod::libcfshr::parse_params::exit_error(
                    c_format(
                        "\n%s error reading header of %s.",
                        &[CArg::Str(prefix), CArg::Str(&opt.fnames[(f) as usize])],
                    )
                    .as_bytes(),
                );
            }
            if hin.nx != hdr.nx || hin.ny != hdr.ny {
                crate::imod::libcfshr::parse_params::exit_error(
                    c_format(
                        "\n%s files must be same size in X and Y; %s differs",
                        &[CArg::Str(prefix), CArg::Str(&opt.fnames[(f) as usize])],
                    )
                    .as_bytes(),
                );
            }
            &mut hdr
        };
        for k in 0..opt.nofsecs {
            if opt.secs[(k) as usize] >= input_header.nz {
                continue;
            }
            let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
                input_header,
                opt.secs[(k) as usize],
                b'z',
                opt.ix,
                opt.iy,
                opt.cx as i32,
                opt.cy as i32,
                None,
            ) else {
                crate::imod::libcfshr::parse_params::exit_error(
                    c_format(
                        "\n%s reading slice %d of %s",
                        &[
                            CArg::Str(prefix),
                            CArg::Int(opt.secs[k as usize] as i64),
                            CArg::Str(&opt.fnames[f as usize]),
                        ],
                    )
                    .as_bytes(),
                );
            };
            // `processing.cpp:2608-2632`: any mode other than float, short,
            // ushort or byte is floated first, then each mode is summed
            // straight from its own union member.
            let mut mode = input_header.mode;
            if mode != crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT
                && mode != crate::imod::libiimod::mrcfiles::MRC_MODE_SHORT
                && mode != crate::imod::libiimod::mrcfiles::MRC_MODE_USHORT
                && mode != crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE
            {
                if crate::imod::libiimod::mrcslice::slice_float(&mut s) != 0 {
                    crate::imod::libcfshr::parse_params::exit_error(
                        c_format("\n%s converting slice to float", &[CArg::Str(prefix)]).as_bytes(),
                    );
                }
                mode = crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT;
            }
            let sum_buf = sum.f_mut();
            match mode {
                crate::imod::libiimod::mrcfiles::MRC_MODE_FLOAT => {
                    let data = s.data.f();
                    for (dst, &src) in sum_buf[..n].iter_mut().zip(&data[..n]) {
                        *dst += src - base;
                    }
                }
                crate::imod::libiimod::mrcfiles::MRC_MODE_SHORT => {
                    let data = s.data.s();
                    for (dst, &src) in sum_buf[..n].iter_mut().zip(&data[..n]) {
                        *dst += src as f32 - base;
                    }
                }
                crate::imod::libiimod::mrcfiles::MRC_MODE_USHORT => {
                    let data = s.data.us();
                    for (dst, &src) in sum_buf[..n].iter_mut().zip(&data[..n]) {
                        *dst += src as f32 - base;
                    }
                }
                crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE => {
                    let data = s.data.b();
                    for (dst, &src) in sum_buf[..n].iter_mut().zip(&data[..n]) {
                        *dst += src as f32 - base;
                    }
                }
                _ => {}
            }
            count += 1;
        }
        if f != 0 {
            crate::imod::libiimod::iimage::ii_fclose(&mut hdr.fp.clone().unwrap());
        }
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
    if count == 0 {
        return -1;
    }
    // `processing.cpp:2643-2688`: bin the sum, normalize it to the centre
    // bin, build the xx/yy coordinate arrays and fit a plane with no
    // constant term.  aa and bb are floats in the source and lsFit2 is its
    // own routine, not an inline normal-equation solve.
    if opt.process == crate::imod::clip::clip::ClipOperation::Planarfit {
        let mut nx_trim = (0.005 * hin.nx as f32) as i32;
        let nx_in = (hin.nx - 2 * nx_trim).min(opt.ix);
        nx_trim = 0.max((hin.nx - nx_in) / 2);
        let mut ny_trim = (0.005 * hin.ny as f32) as i32;
        let ny_in = (hin.ny - 2 * ny_trim).min(opt.iy);
        ny_trim = 0.max((hin.ny - ny_in) / 2);
        let x_binning = 1.max(nx_in / 11);
        let y_binning = 1.max(ny_in / 11);
        let nx_bin = nx_in / x_binning;
        let ny_bin = ny_in / y_binning;
        let num_bin = (nx_bin * ny_bin) as usize;
        let mut bin_sum = vec![0_f32; num_bin];
        crate::imod::libcfshr::reduce_by_binning::bin_into_slice(
            &sum.f()[(ny_trim * opt.ix + nx_trim) as usize..],
            opt.ix,
            &mut bin_sum,
            nx_bin,
            ny_bin,
            x_binning,
            y_binning,
            1.,
        );
        let cen_val = bin_sum[(nx_bin * (ny_bin / 2) + nx_bin / 2) as usize];
        let mut warned = 0;
        for value in &mut bin_sum {
            if *value <= 0. && warned == 0 {
                warned += 1;
                crate::imod::clip::clip::show_warning(
                    "Some binned values are negative; you must set a base value to subtract with the -l option",
                );
            }
            *value = *value / cen_val - 1.;
        }
        let xcen = (nx_bin as f64 / 2.) as f32;
        let ycen = (ny_bin as f64 / 2.) as f32;
        let mut xx = vec![0_f32; num_bin];
        let mut yy = vec![0_f32; num_bin];
        let mut k = 0_usize;
        for iy in 0..ny_bin {
            for ix in 0..nx_bin {
                xx[k] = (x_binning as f64 * (ix as f64 + 0.5 - xcen as f64)) as f32;
                yy[k] = (y_binning as f64 * (iy as f64 + 0.5 - ycen as f64)) as f32;
                k += 1;
            }
        }
        let mut aa = 0_f32;
        let mut bb = 0_f32;
        crate::imod::libcfshr::simplestat::ls_fit2(
            &xx,
            &yy,
            &bin_sum,
            nx_bin * ny_bin,
            &mut aa,
            &mut bb,
            None,
        );
        let text = c_format(
            "%.8f  %.8f\n",
            &[CArg::Dbl(aa as f64), CArg::Dbl(bb as f64)],
        );
        crate::imod::libcfshr::b3dutil::b3d_fwrite(
            text.as_bytes(),
            1,
            text.len(),
            &mut hout.fp.clone().unwrap(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Plane slopes imply a gradient over full extent in X and Y of %.3f and %.3f\n",
                &[
                    CArg::Dbl((100. * aa as f64 * hin.nx as f64) as f64),
                    CArg::Dbl((100. * bb as f64 * hin.ny as f64) as f64),
                ],
            )
            .as_bytes(),
        );
        let mut dmean = 0_f32;
        for iy in 0..ny_bin {
            for ix in 0..nx_bin {
                let k = (ix + iy * nx_bin) as usize;
                let resid = (100. * (bin_sum[k] - (aa * xx[k] + bb * yy[k])) as f64) as f32;
                dmean += resid * resid;
            }
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Root-mean-squared residual = %.3f\n",
                &[CArg::Dbl(
                    (((dmean / (nx_bin * ny_bin) as f32) as f64).sqrt()) as f64,
                )],
            )
            .as_bytes(),
        );
        return 0;
    }
    if fitting && opt.process == crate::imod::clip::clip::ClipOperation::Flatfield {
        // C fills the same column-major `xMat[col][row]` used by
        // multRegress, then evaluates its polynomial inverse over the
        // full output.  Rebuild the binned normalized image here because
        // the plane-only `lsFit2` path above deliberately has no
        // constant column.
        let mut nx_trim = (0.005 * hin.nx as f32) as i32;
        let nx_in = (hin.nx - 2 * nx_trim).min(opt.ix);
        nx_trim = 0.max((hin.nx - nx_in) / 2);
        let mut ny_trim = (0.005 * hin.ny as f32) as i32;
        let ny_in = (hin.ny - 2 * ny_trim).min(opt.iy);
        ny_trim = 0.max((hin.ny - ny_in) / 2);
        let x_binning = 1.max(nx_in / 15);
        let y_binning = 1.max(ny_in / 15);
        let nx_bin = nx_in / x_binning;
        let ny_bin = ny_in / y_binning;
        let dim = 15 * 15 + 10;
        let col_dim = 18;
        // `processing.cpp:2645` calls binIntoSlice, whose per-output-row
        // partial sums are each scaled by 1/(binFacX*binFacY) before being
        // accumulated; a plain unscaled sum rounds differently.
        let mut bin_sum = vec![0_f32; (nx_bin * ny_bin) as usize];
        crate::imod::libcfshr::reduce_by_binning::bin_into_slice(
            &sum.f()[(ny_trim * opt.ix + nx_trim) as usize..],
            opt.ix,
            &mut bin_sum,
            nx_bin,
            ny_bin,
            x_binning,
            y_binning,
            1.,
        );
        let center = bin_sum[(nx_bin * (ny_bin / 2) + nx_bin / 2) as usize];
        // `processing.cpp:2650-2657`: the warning fires only for the first
        // non-positive bin, and the normalization runs on every bin.
        let mut warned = 0;
        for value in &mut bin_sum {
            if *value <= 0. && warned == 0 {
                warned += 1;
                crate::imod::clip::clip::show_warning(
                    "Some binned values are negative; you must set a base value to subtract with the -l option",
                );
            }
            *value = *value / center - 1.;
        }
        let mut x_mat = vec![0_f32; (col_dim * dim) as usize];
        let mut sol = [0_f32; 18];
        let mut x_mean = [0_f32; 18];
        let mut x_sd = [0_f32; 18];
        let mut work = [0_f32; 18 * 18];
        let mut cons = 0_f32;
        let mut num_col = 0_i32;
        // `processing.cpp:2659-2665`: xcen/ycen are floats from a double
        // quotient, and each xx/yy is a double expression stored as float.
        let bin_xcen = (nx_bin as f64 / 2.) as f32;
        let bin_ycen = (ny_bin as f64 / 2.) as f32;
        for iy in 0..ny_bin {
            for ix in 0..nx_bin {
                let k = ix + iy * nx_bin;
                let xx = (x_binning as f64 * (ix as f64 + 0.5 - bin_xcen as f64)) as f32;
                let yy = (y_binning as f64 * (iy as f64 + 0.5 - bin_ycen as f64)) as f32;
                let mut col = 0_i32;
                for ind in 1..=order {
                    for py in 0..=ind {
                        let px = ind - py;
                        // `processing.cpp:2698`: pow() is the double version.
                        x_mat[(col * dim + k) as usize] =
                            ((xx as f64).powf(px as f64) * (yy as f64).powf(py as f64)) as f32;
                        col += 1;
                    }
                }
                x_mat[(col * dim + k) as usize] = bin_sum[k as usize];
                num_col = col;
            }
        }
        let mut cons_arr = [cons];
        let regress_err = crate::imod::libcfshr::regression::mult_regress(
            &x_mat,
            dim,
            0,
            num_col,
            nx_bin * ny_bin,
            1,
            0,
            &mut sol,
            col_dim,
            Some(&mut cons_arr),
            &mut x_mean,
            &mut x_sd,
            &mut work,
        );
        cons = cons_arr[0];
        if regress_err != 0 {
            return -1;
        }
        let x_center = hin.nx as f32 / 2. - 0.5;
        let y_center = hin.ny as f32 / 2. - 0.5;
        let _ = ImodFile::Stdout.write_all(c_format("Constant term %.6f\nX & Y order and coefficients (times half-size in X/Y to respective powers):\n", &[CArg::Dbl((cons as f64) as f64)]).as_bytes());
        let mut col = 0_usize;
        for ind in 1..=order {
            for py in 0..=ind {
                let px = ind - py;
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%d  %d  %.9f\n",
                        &[
                            CArg::Int((px) as i64),
                            CArg::Int((py) as i64),
                            CArg::Dbl(
                                (sol[col] as f64
                                    * (x_center as f64).powf(px as f64)
                                    * (y_center as f64).powf(py as f64))
                                    as f64,
                            ),
                        ],
                    )
                    .as_bytes(),
                );
                col += 1;
            }
        }
        let mut residual_sum = 0_f32;
        for iy in 0..ny_bin {
            for ix in 0..nx_bin {
                let k = (ix + iy * nx_bin) as usize;
                let xx = (x_binning as f64 * (ix as f64 + 0.5 - bin_xcen as f64)) as f32;
                let yy = (y_binning as f64 * (iy as f64 + 0.5 - bin_ycen as f64)) as f32;
                let mut residual = bin_sum[k] - cons;
                let mut col = 0_usize;
                for ind in 1..=order {
                    for py in 0..=ind {
                        residual = (residual as f64
                            - sol[col] as f64
                                * (xx as f64).powf((ind - py) as f64)
                                * (yy as f64).powf(py as f64))
                            as f32;
                        col += 1;
                    }
                }
                residual_sum += 10_000. * residual * residual;
            }
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Root-mean-squared residual = %.3f\n",
                &[CArg::Dbl(
                    (((residual_sum / (nx_bin * ny_bin) as f32) as f64).sqrt()) as f64,
                )],
            )
            .as_bytes(),
        );
        // `processing.cpp:2746-2763`: `numOMPthreads(4)` and an OpenMP
        // `parallel for` over `iy`.  (a) Row `iy` writes only
        // `sumBuf[ix + nx * iy]`, so groups of whole rows own disjoint output;
        // (b) each pixel is evaluated from `cons`, `sol` and its own
        // coordinates alone, with `resid`, `col`, `ind`, `py` private and no
        // accumulation across pixels.  The result is therefore the same for
        // any thread count or partition.
        let nx = hin.nx as usize;
        let sum_buf = &mut sum.f_mut()[..nx * hin.ny as usize];
        let num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(4);
        let rows_per_group = if num_threads > 1 {
            (hin.ny as usize).div_ceil(num_threads as usize).max(1)
        } else {
            (hin.ny as usize).max(1)
        };
        let run_group = |(g, rows): (usize, &mut [f32])| {
            for (r, row) in rows.chunks_mut(nx).enumerate() {
                let iy = (g * rows_per_group + r) as i32;
                for ix in 0..hin.nx {
                    let mut residual = 1. + cons;
                    let mut col = 0_usize;
                    for ind in 1..=order {
                        for py in 0..=ind {
                            residual = (residual as f64
                                + sol[col] as f64
                                    * ((ix as f32 - x_center) as f64).powf((ind - py) as f64)
                                    * ((iy as f32 - y_center) as f64).powf(py as f64))
                                as f32;
                            col += 1;
                        }
                    }
                    row[ix as usize] = 1. / residual;
                }
            }
        };
        if nx > 0 {
            if num_threads > 1 {
                // Same pool sizing as `reduce_by_binning.rs`.
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(crate::imod::libcfshr::b3dutil::num_omp_threads(i32::MAX) as usize)
                    .build_global();
                use rayon::iter::{IndexedParallelIterator, ParallelIterator};
                use rayon::slice::ParallelSliceMut;
                sum_buf
                    .par_chunks_mut(rows_per_group * nx)
                    .enumerate()
                    .for_each(run_group);
            } else {
                sum_buf
                    .chunks_mut(rows_per_group * nx)
                    .enumerate()
                    .for_each(run_group);
            }
        }
    } else {
        // C `processing.cpp:2769-2777` takes min/max/mean through
        // `fullArrayMinMaxMean`, so `dmean` is rounded to float before it is
        // used; `0.05 * dmean` then promotes the comparison and the divide
        // to double, and the quotient is rounded to float exactly once.
        let (dmin, dmax, dmean) = crate::imod::libiimod::mrcslice::full_array_min_max_mean(
            &mut sum,
            crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT,
            opt.ix,
            opt.iy,
        )
        .expect("sum buffer dimensions match the flatfield image");
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Averaged image min = %.5g, max = %.5g, mean = %.5g\n",
                &[
                    CArg::Dbl(((dmin / count as f32) as f64) as f64),
                    CArg::Dbl(((dmax / count as f32) as f64) as f64),
                    CArg::Dbl(((dmean / count as f32) as f64) as f64),
                ],
            )
            .as_bytes(),
        );
        if dmin < 0. {
            crate::imod::clip::clip::show_warning(
                "Some summed values are negative; you must set a base value to subtract with the -l option",
            );
        }
        for value in sum.f_mut() {
            *value = (dmean as f64 / (0.05 * dmean as f64).max(*value as f64)) as f32;
        }
    }
    if opt.process == crate::imod::clip::clip::ClipOperation::Flatfield {
        // `processing.cpp:2779-2785` takes min/max/mean straight from
        // sumBuf with `fullArrayMinMaxMean` and writes sumBuf itself.  The
        // intermediate slice that used to stand in here was filled by
        // `sliceMinMax`, which never sets `mean`, so the output header
        // carried a stale value rather than the flatfield mean.
        (hout.amin, hout.amax, hout.amean) =
            crate::imod::libiimod::mrcslice::full_array_min_max_mean(
                &mut sum,
                crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT,
                opt.ix,
                opt.iy,
            )
            .expect("sum buffer dimensions match the flatfield image");
        if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0
        {
            let message = c_format("%s writing header", &[CArg::Str(prefix)]);
            crate::imod::libcfshr::parse_params::exit_error(message.as_bytes());
        }
        if crate::imod::libiimod::mrcfiles::mrc_write_slice(
            sum.bytes(),
            &mut hout.fp.clone().unwrap(),
            hout,
            0,
            b'z',
        ) != 0
        {
            let message = c_format("%s writing image", &[CArg::Str(prefix)]);
            crate::imod::libcfshr::parse_params::exit_error(message.as_bytes());
        }
    }
    0
}
/// Matches C++ `clipUnpack`.
pub fn clip_unpack(
    hin1: &mut MrcHeader,
    hin2: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> i32 {
    use crate::imod::clip::clip::IP_APPEND_FALSE;
    use crate::imod::libiimod::mrcfiles::{MRC_MODE_FLOAT, mrc_head_label};
    let mut do_ref = opt.infiles == 2;
    if opt.add2file != IP_APPEND_FALSE {
        crate::imod::clip::clip::show_error(
            "clip unpack/normalize - you cannot add to an existing output file",
        );
        return -1;
    }
    if do_ref && hin2.mode != MRC_MODE_FLOAT {
        crate::imod::clip::clip::show_error(
            "clip unpack/normalize - mode of second input file must be floats",
        );
        return -1;
    }
    if opt.infiles > 2 {
        crate::imod::clip::clip::show_error(
            "clip unpack/normalize - There can be only 2 input files",
        );
        return -1;
    }
    let mut z = crate::imod::clip::file_io::set_options(opt, hin1, hout);
    if z < 0 {
        return z;
    }
    let eer_info = hin1
        .fp
        .as_ref()
        .and_then(crate::imod::libiimod::iimage::ii_eer_info_from_fp);
    let is_eer = eer_info.is_some_and(|info| info.is_eer);
    let antialias_eer = eer_info.is_some_and(|info| info.is_eer && info.antialias_filter > 0);
    let read_eer_as_super_res = eer_info.map_or(0, |info| info.read_as_super_res);
    let eerkernel_scale = eer_info.map_or(1., |info| info.kernel_scale);
    let red_factor = if antialias_eer {
        1_i32 << (2 - read_eer_as_super_res).max(0)
    } else {
        1
    };
    let mut scale = if do_ref { 16. } else { 1. };
    if antialias_eer {
        scale = 100.;
    }
    if opt.val != crate::imod::clip::clip::IP_DEFAULT as f32 {
        scale = opt.val;
    }
    let mut reference: Option<Islice> = None;
    if do_ref {
        reference = crate::imod::libiimod::mrcslice::slice_read_mrc(hin2, 0, b'z');
        if reference.is_none() {
            return -1;
        }
        let mut nx = hin2.nx;
        let mut ny = hin2.ny;
        if opt.rotation_flip < 0 {
            for label_index in 0..hin1.nlabl {
                // `strstr(hin1->labels[i], " r/f ")` over the NUL-terminated
                // label, then `atoi(extraName + 4)` -- four past the start of
                // the match, so the scan begins on the trailing blank, which
                // `atoi` skips.
                let label = &hin1.labels[label_index as usize];
                let end = label.iter().position(|b| *b == 0).unwrap_or(label.len());
                let bytes = &label[..end];
                if let Some(index) = bytes.windows(5).position(|text| text == b" r/f ") {
                    opt.rotation_flip = crate::imod::clip::clip::atoi_bytes(&bytes[index + 4..]);
                    break;
                }
            }
            if opt.rotation_flip < 0 {
                crate::imod::clip::clip::show_error(
                    "Cannot find r/f entry in header of first input file",
                );
                return -1;
            }
        }
        if opt.rotation_flip > 7 {
            crate::imod::clip::clip::show_error(&c_format(
                "Rotation/flip value of %d is out of range",
                &[CArg::Int(opt.rotation_flip as i64)],
            ));
            return -1;
        }
        if opt.rotation_flip > 0
            && rotate_flip_gain_reference(
                reference.as_mut().unwrap().data.f_mut(),
                &mut nx,
                &mut ny,
                opt.rotation_flip,
            ) != 0
        {
            crate::imod::clip::clip::show_error(
                "Error allocating memory for rotating gain reference",
            );
            return -1;
        }
        reference.as_mut().unwrap().xsize = nx;
        reference.as_mut().unwrap().ysize = ny;
        // `processing.cpp:2881`: superFac comes from the reference file's
        // own width, not from the possibly rotated slice width.
        let super_fac = hin1.nx / hin2.nx;
        let use_fac = if antialias_eer { 4 } else { super_fac };
        if (hin1.ny / hin2.ny == super_fac
            && super_fac * hin2.nx == hin1.nx
            && super_fac * hin2.ny == hin1.ny
            && matches!(super_fac, 1 | 2 | 4))
            || antialias_eer
        {
            if super_fac > 1 || antialias_eer {
                // `processing.cpp:2885-2889`.
                let is_tiff_gain = hin2.fp.as_ref().is_some_and(|fp| {
                    crate::imod::libiimod::iimage::ii_file_type_from_fp(fp)
                        == Some(crate::imod::libiimod::iimage::IIFILE_TIFF)
                });
                if !is_eer && !is_tiff_gain {
                    let _ = ImodFile::Stdout.write_all(c_format("WARNING: clip - Expanding gain reference because image is exactly %d times as big as reference\n", &[CArg::Int((super_fac) as i64)]).as_bytes());
                }
                let Some(mut expanded) = crate::imod::libcfshr::islice::slice_create(
                    hin1.nx * red_factor,
                    hin1.ny * red_factor,
                    2,
                ) else {
                    crate::imod::clip::clip::show_error(
                        "Error allocating memory for expanded gain reference",
                    );
                    return -1;
                };
                // `processing.cpp:2896-2898`: the reference file's own
                // dimensions are expanded, and nxGain/nyGain are multiplied
                // by useFac rather than reset from the input file size.
                // `processing.cpp:2897-2901`: `CorDefExpandGainReference(slRef->data.f,
                // ..., refTemp)` and then `slRef->data.f = refTemp`.
                crate::imod::clip::correct_defects::cor_def_expand_gain_reference(
                    reference.as_ref().unwrap().data.f(),
                    hin2.nx,
                    hin2.ny,
                    use_fac,
                    expanded.data.f_mut(),
                );
                reference = Some(expanded);
                nx *= use_fac;
                ny *= use_fac;
                reference.as_mut().unwrap().xsize = nx;
                reference.as_mut().unwrap().ysize = ny;
                if let Some(super_gain_name) = opt.super_gain_name.clone() {
                    let mut biases = Vec::new();
                    let (mut num_in_x, mut x_start, mut x_interval) = (0, 0, 0);
                    let (mut num_in_y, mut y_start, mut y_interval) = (0, 0, 0);
                    let error = crate::imod::clip::correct_defects::cor_def_read_super_gain(
                        &super_gain_name,
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
                        crate::imod::libcfshr::parse_params::exit_error(
                            c_format(
                                "Reading file with super-resolution gain adjustments (error %d)",
                                &[CArg::Int(error as i64)],
                            )
                            .as_bytes(),
                        );
                    }
                    // `processing.cpp:2911`: `CorDefRefineSuperResRef(refTemp, ...)`
                    // works on the reference in place.
                    crate::imod::clip::correct_defects::cor_def_refine_super_res_ref(
                        reference.as_mut().unwrap().data.f_mut(),
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
        if nx != hin1.nx * red_factor || ny != hin1.ny * red_factor {
            crate::imod::clip::clip::show_error(
                "clip unpack/normalize - reference size must match the input file size",
            );
            return -1;
        }
        if !antialias_eer {
            let llx = opt.cx as i32 - opt.ix / 2;
            let lly = opt.cy as i32 - opt.iy / 2;
            let urx = llx + opt.ix;
            let ury = lly + opt.iy;
            if llx < 0 || lly < 0 || urx > nx || ury > ny {
                crate::imod::clip::clip::show_error(
                    "Selected area goes outside of actual data for gain reference",
                );
                return -1;
            }
            if (llx > 0 || lly > 0 || urx < nx || ury < ny)
                && crate::imod::libiimod::mrcslice::slice_box_in(
                    reference.as_mut().unwrap(),
                    llx,
                    lly,
                    urx,
                    ury,
                ) != 0
            {
                crate::imod::clip::clip::show_error(
                    "Error allocating memory for taking subarea of gain reference",
                );
                return -1;
            }
            // `processing.cpp:2944-2945`: `slRef->data.f[i] *= scale`.
            let data = reference.as_mut().unwrap().data.f_mut();
            for ind in 0..opt.ix * opt.iy {
                data[ind as usize] *= scale;
            }
        }
    }
    let offset = if hout.mode == 2 { 0. } else { 0.5 };
    let Some(mut out) = crate::imod::libcfshr::islice::slice_create(opt.ix, opt.iy, hout.mode)
    else {
        return -1;
    };
    mrc_head_label(
        &mut *hout,
        c_format(
            if opt.process == crate::imod::clip::clip::ClipOperation::Unpack {
                "clip: Unpack 4-bit values, scaled by %.2f"
            } else {
                "clip: Normalize, scaled by %.2f"
            },
            &[CArg::Dbl(scale as f64)],
        )
        .as_bytes(),
    );
    // `processing.cpp:2810`: `float truncThresh, unscaledThresh = 1.e30;`
    // -- the *unscaled* threshold is what `CorDefSurroundingMean` is given,
    // and `truncThresh` is it times the scale.
    let mut unscaled_thresh = 1.0e30_f32;
    if opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 {
        unscaled_thresh = opt.high;
    }
    if antialias_eer {
        if do_ref {
            if !crate::imod::libiimod::iitif::tiff_gain_reference_for_eer_bytes(
                reference.as_mut().unwrap().data.bytes_mut(),
            ) {
                return -1;
            }
        }
        do_ref = false;
        scale /= eerkernel_scale;
        unscaled_thresh *= eerkernel_scale;
    }
    let trunc_thresh = unscaled_thresh * scale;
    let mut first_defect_setup = true;
    for k in 0..opt.nofsecs {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "\rclip: processing slice %d of %d",
                &[CArg::Int((k + 1) as i64), CArg::Int((opt.nofsecs) as i64)],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.flush();
        let Some(input) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin1,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error("clip: Error reading slice.");
            return -1;
        };
        // `processing.cpp:2991-2992`: the thread count is recomputed for every
        // section, from the float `scale` promoted to double; `B3DNINT` is
        // `(int)floor(x + 0.5)`.
        let num_threads = crate::imod::libcfshr::b3dutil::num_omp_threads(
            (4. * scale as f64 * ((opt.ix as f64 * opt.iy as f64).sqrt() / 944.).ln() / 2f64.ln()
                + 0.5)
                .floor() as i32,
        );
        // `processing.cpp:2994-3039`: an OpenMP `parallel for` over `j`.
        // (a) Row `j` writes only output pixels `(i, j)` through
        // `slicePutVal`, so groups of whole rows own disjoint output; (b)
        // every pixel is computed from the unmodified input section, the
        // reference and scalars fixed for the section — `CorDefSurroundingMean`
        // only reads the input frame — with `val`, `ival`, `scaleUse` private
        // and nothing accumulated across pixels.  The result is therefore the
        // same for any thread count or partition.
        //
        // `pixel` is the loop body up to the `slicePutVal`; the store itself
        // is written per output member below, as `slicePutVal`
        // (`islice.c:270-285`, `islice.rs::slice_put_val`) does it: the first
        // `csize` components of `val`, cast through `int` to the member type
        // for the integer modes.  Every `(i, j)` is inside the slice, so its
        // bounds check never fails.
        let input: &Islice = &input;
        let ref_data: &[f32] = if do_ref {
            reference.as_ref().unwrap().data.f()
        } else {
            &[]
        };
        let opt_low = opt.low;
        let in_mode_is_byte = hin1.mode == crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
        let pixel = |x: i32, y: i32, base: usize| -> [f32; 4] {
            // `processing.cpp:3002-3021` has a byte fast path that reads
            // `slIn->data.b[i + base]` straight out of the row, and only
            // falls back to `sliceGetVal` for "other modes, a bit slower"
            // (`processing.cpp:3022`).  The guard is on the *file* mode, as
            // in the source, and `sliceReadSubm` always builds the slice
            // with `hin->mode`, so `data.b()` matches it.
            let mut v = [0.; 4];
            if in_mode_is_byte {
                // `processing.cpp:3004`: `bval = slIn->data.b[i + base]`
                // into an `int`, which then converts exactly to float for
                // the multiply below.
                let bval = input.data.b()[base + x as usize] as i32;
                // `processing.cpp:3005`: `slRef->data.f[i + base]`.
                let gain = if do_ref {
                    ref_data[base + x as usize]
                } else {
                    scale
                };
                v[0] = bval as f32 * gain + offset;
                if v[0] > trunc_thresh {
                    v[0] = if opt_low == crate::imod::clip::clip::IP_DEFAULT as f32 {
                        // `processing.cpp:3011`: the byte arm passes the
                        // literal `MRC_MODE_BYTE`, not `slIn->mode`.
                        crate::imod::clip::correct_defects::cor_def_surrounding_mean(
                            input.data.bytes(),
                            crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE,
                            input.xsize,
                            input.ysize,
                            unscaled_thresh,
                            x,
                            y,
                        ) * gain
                            + offset
                    } else {
                        opt_low * gain + offset
                    };
                }
            } else {
                slice_get_val(input, x, y, &mut v);
                // `processing.cpp:3024`: `slRef->data.f[i + base]`.
                let gain = if do_ref {
                    ref_data[base + x as usize]
                } else {
                    scale
                };
                v[0] = v[0] * gain + offset;
                if v[0] > trunc_thresh {
                    v[0] = if opt_low == crate::imod::clip::clip::IP_DEFAULT as f32 {
                        crate::imod::clip::correct_defects::cor_def_surrounding_mean(
                            input.data.bytes(),
                            input.mode,
                            input.xsize,
                            input.ysize,
                            unscaled_thresh,
                            x,
                            y,
                        ) * gain
                            + offset
                    } else {
                        opt_low * gain + offset
                    };
                }
            }
            v
        };
        let ix = opt.ix as usize;
        let iy = opt.iy as usize;
        let csize = match out.mode {
            0 | 1 | 2 | 6 => 1,
            3 | 4 => 2,
            16 | 99 => 3,
            // `slicePutVal` stores nothing for any other mode, and `pixel`
            // has no side effect, so there is nothing to run.
            _ => 0,
        };
        let rows_per_group = if num_threads > 1 {
            iy.div_ceil(num_threads as usize).max(1)
        } else {
            iy.max(1)
        };
        let group_len = rows_per_group * ix * csize;
        let run_b = |(g, rows): (usize, &mut [u8])| {
            for (r, row) in rows.chunks_mut(ix * csize).enumerate() {
                let y = g * rows_per_group + r;
                for x in 0..ix {
                    let v = pixel(x as i32, y as i32, y * ix);
                    for c in 0..csize {
                        row[x * csize + c] = v[c] as i32 as u8;
                    }
                }
            }
        };
        let run_s = |(g, rows): (usize, &mut [i16])| {
            for (r, row) in rows.chunks_mut(ix * csize).enumerate() {
                let y = g * rows_per_group + r;
                for x in 0..ix {
                    let v = pixel(x as i32, y as i32, y * ix);
                    for c in 0..csize {
                        row[x * csize + c] = v[c] as i32 as i16;
                    }
                }
            }
        };
        let run_us = |(g, rows): (usize, &mut [u16])| {
            for (r, row) in rows.chunks_mut(ix * csize).enumerate() {
                let y = g * rows_per_group + r;
                for x in 0..ix {
                    let v = pixel(x as i32, y as i32, y * ix);
                    for c in 0..csize {
                        row[x * csize + c] = v[c] as i32 as u16;
                    }
                }
            }
        };
        let run_f = |(g, rows): (usize, &mut [f32])| {
            for (r, row) in rows.chunks_mut(ix * csize).enumerate() {
                let y = g * rows_per_group + r;
                for x in 0..ix {
                    let v = pixel(x as i32, y as i32, y * ix);
                    for c in 0..csize {
                        row[x * csize + c] = v[c];
                    }
                }
            }
        };
        if group_len > 0 {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::slice::ParallelSliceMut;
            let len = ix * iy * csize;
            if num_threads > 1 {
                // Same pool sizing as `reduce_by_binning.rs`.
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(crate::imod::libcfshr::b3dutil::num_omp_threads(i32::MAX) as usize)
                    .build_global();
                match &mut out.data {
                    MrcData::B(d) => d[..len]
                        .par_chunks_mut(group_len)
                        .enumerate()
                        .for_each(run_b),
                    MrcData::S(d) => d[..len]
                        .par_chunks_mut(group_len)
                        .enumerate()
                        .for_each(run_s),
                    MrcData::Us(d) => d[..len]
                        .par_chunks_mut(group_len)
                        .enumerate()
                        .for_each(run_us),
                    MrcData::F(d) => d[..len]
                        .par_chunks_mut(group_len)
                        .enumerate()
                        .for_each(run_f),
                }
            } else {
                match &mut out.data {
                    MrcData::B(d) => d[..len].chunks_mut(group_len).enumerate().for_each(run_b),
                    MrcData::S(d) => d[..len].chunks_mut(group_len).enumerate().for_each(run_s),
                    MrcData::Us(d) => d[..len].chunks_mut(group_len).enumerate().for_each(run_us),
                    MrcData::F(d) => d[..len].chunks_mut(group_len).enumerate().for_each(run_f),
                }
            }
        }
        if opt.read_defects != 0
            && correct_defects(out.as_mut(), hin1.nx, hin1.ny, opt, &mut first_defect_setup)
                .is_err()
        {
            return -1;
        }
        if crate::imod::clip::file_io::clip_write_slice(out.as_mut(), hout, opt, k, &mut z, 0)
            .is_err()
        {
            return -1;
        }
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
    crate::imod::clip::file_io::set_mrc_coords(hin1, hout, opt)
}
/// Matches C++ `rotateFlipGainReference`.
fn rotate_flip_gain_reference(
    reference: &mut [f32],
    nx_gain: &mut i32,
    ny_gain: &mut i32,
    rotation_flip: i32,
) -> i32 {
    let nx_in = *nx_gain;
    let ny_in = *ny_gain;
    let Some(value_count) = nx_in
        .checked_mul(ny_in)
        .and_then(|count| usize::try_from(count).ok())
    else {
        return 1;
    };
    // `processing.cpp:3064` B3DMALLOCs the rotated copy and returns 1 on
    // failure; a `Vec` cannot fail here.
    let mut summed = vec![0_f32; value_count];
    let error = crate::imod::libcfshr::rotateflip::rotate_flip_image(
        crate::imod::libcfshr::rotateflip::RotateFlipData::Float {
            array: reference,
            brray: &mut summed,
        },
        nx_in,
        ny_in,
        rotation_flip,
        0,
        0,
        0,
        nx_gain,
        ny_gain,
        0,
    );
    if error == 0 {
        // `processing.cpp:3071`: `memcpy(ref, summed, nxGain * nyGain * sizeof(float))`.
        reference[..summed.len()].copy_from_slice(&summed);
    }
    error
}
/// Matches C++ `clipDefectMap`.
pub fn clip_defect_map(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
) -> Result<(), i32> {
    let tolerance = 10;
    if opt.read_defects == 0 {
        crate::imod::clip::clip::show_error(
            "clip defectmap - you must provide a defect file with the -D option",
        );
        return Err(-1);
    }
    if opt.defects.k2_type > 0 {
        if opt.defects.was_scaled > 0
            && (opt.cam_size_x / 2 - hin.nx).abs() < tolerance
            && (opt.cam_size_y / 2 - hin.ny).abs() < tolerance
        {
            crate::imod::clip::correct_defects::cor_def_scale_defects_for_k2(
                &mut opt.defects,
                true,
            );
            opt.cam_size_x /= 2;
            opt.cam_size_y /= 2;
        } else if opt.defects.was_scaled <= 0
            && (opt.cam_size_x * 2 - hin.nx).abs() < tolerance
            && (opt.cam_size_y * 2 - hin.ny).abs() < tolerance
        {
            crate::imod::clip::correct_defects::cor_def_scale_defects_for_k2(
                &mut opt.defects,
                false,
            );
            opt.cam_size_x *= 2;
            opt.cam_size_y *= 2;
        }
    }
    if opt.defects.falcon_type > 0 {
        let ind = (hin.nx as f32 / opt.cam_size_x as f32).round() as i32;
        if (ind == 2 || ind == 4)
            && (opt.cam_size_x * ind - hin.nx).abs() < tolerance
            && (opt.cam_size_y * ind - hin.ny).abs() < tolerance
        {
            crate::imod::clip::correct_defects::cor_def_scale_defects_for_falcon(
                &mut opt.defects,
                ind,
            );
            opt.cam_size_x *= ind;
            opt.cam_size_y *= ind;
        }
    }
    if (opt.cam_size_x - hin.nx).abs() >= tolerance || (opt.cam_size_y - hin.ny).abs() >= tolerance
    {
        crate::imod::clip::clip::show_error(
            "clip defectmap - Image size must be within 10 pixels of the camera size stored in the defect list, after possible scaling up or down by 2 for K2",
        );
        return Err(-1);
    }
    hout.mode = crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE;
    if crate::imod::libcfshr::b3dutil::b3d_output_file_type() == 2 {
        hout.bytes_signed = 0;
    }
    // `processing.cpp:2996` B3DMALLOCs the map and reports a memory error;
    // a `Vec` cannot fail here.
    let mut map = vec![0_u8; (hin.nx * hin.ny) as usize];
    if crate::imod::clip::correct_defects::cor_def_fill_defect_array(
        &opt.defects,
        opt.cam_size_x,
        opt.cam_size_y,
        &mut map,
        hin.nx,
        hin.ny,
        opt.sano != 0,
    ) != 0
    {
        crate::imod::clip::clip::show_error(
            "clip defectmap - Bad size parameters passed to routine",
        );
        return Err(-1);
    }
    crate::imod::libiimod::mrcfiles::mrc_head_label_cp(hin, hout);
    crate::imod::libiimod::mrcfiles::mrc_head_label(
        hout,
        b"clip defectmap: Map of defective pixels in image",
    );
    hout.amin = 0.;
    hout.amax = 0.;
    hout.amean = 0.;
    for ind in 0..hin.nx * hin.ny {
        let value = map[ind as usize] as f32;
        hout.amean += value;
        hout.amax = hout.amax.max(value);
    }
    hout.amean /= (hin.nx * hin.ny) as f32;
    hout.nz = 1;
    let (sx, sy, sz) = crate::imod::libiimod::mrcfiles::mrc_get_scale(hin);
    crate::imod::libiimod::mrcfiles::mrc_set_scale(hout, sx as f64, sy as f64, sz as f64);
    if crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) != 0
        || crate::imod::libiimod::mrcfiles::mrc_write_slice(
            &map,
            &mut hout.fp.clone().unwrap(),
            hout,
            0,
            b'z',
        ) != 0
    {
        return Err(-1);
    }
    Ok(())
}
/// Matches C++ `clipSuperGain`.
pub fn clip_super_gain(
    h1: &mut MrcHeader,
    out_fp: &mut ImodFile,
    opt: &mut ClipOptions,
) -> Result<(), i32> {
    if opt.val < 2. || opt.val > 64. {
        crate::imod::clip::clip::show_error(
            "clip supergain: the number of subdivisions must be between 2 and 64.",
        );
        return Err(-1);
    }
    // `processing.cpp:3179`: numDiv = 2 * opt->val truncates the float
    // product, so -n 2.7 gives 5, not 4.
    let divisions = (2. * opt.val) as i32;
    let (input_nx, input_ny) = (h1.nx, h1.ny);
    let n = (input_nx * input_ny) as usize;
    let mut image = vec![0_f64; n];
    let mut input = vec![0_u8; n];
    let mut extra = Vec::<MrcHeader>::with_capacity((opt.infiles - 1).max(0) as usize);
    for file_index in 1..opt.infiles {
        let mut header = MrcHeader::default();
        header.fp = crate::imod::libiimod::iimage::ii_fopen(
            opt.fnames[(file_index) as usize].as_bytes(),
            "rb",
        );
        if header.fp.is_none() {
            crate::imod::clip::clip::show_error(&c_format(
                "clip supergain: error opening %s.",
                &[CArg::Str(&opt.fnames[(file_index) as usize])],
            ));
            for opened in &extra {
                crate::imod::libiimod::iimage::ii_fclose(&mut opened.fp.clone().unwrap());
            }
            return Err(-1);
        }
        if crate::imod::libiimod::mrcfiles::mrc_head_read(
            &mut header.fp.clone().unwrap(),
            &mut header,
        ) != 0
        {
            crate::imod::clip::clip::show_error(&c_format(
                "clip supergain: error reading header of %s.",
                &[CArg::Str(&opt.fnames[(file_index) as usize])],
            ));
            if !header.fp.is_none() {
                crate::imod::libiimod::iimage::ii_fclose(&mut header.fp.clone().unwrap());
            }
            for opened in &extra {
                crate::imod::libiimod::iimage::ii_fclose(&mut opened.fp.clone().unwrap());
            }
            return Err(-1);
        }
        extra.push(header);
    }
    for file_index in 0..opt.infiles {
        let header: &mut MrcHeader = if file_index == 0 {
            h1
        } else {
            &mut extra[(file_index - 1) as usize]
        };
        if header.nx != input_nx
            || header.ny != input_ny
            || header.mode != crate::imod::libiimod::mrcfiles::MRC_MODE_BYTE
        {
            crate::imod::clip::clip::show_error(
                "clip supergain: all input files must be the same X/Y size and mode 0.",
            );
            for opened in &extra {
                crate::imod::libiimod::iimage::ii_fclose(&mut opened.fp.clone().unwrap());
            }
            return Err(-1);
        }
        let is_eer = header.fp.as_ref().is_some_and(|fp| {
            crate::imod::libiimod::iimage::ii_eer_info_from_fp(fp).is_some_and(|info| info.is_eer)
        });
        if !is_eer {
            crate::imod::clip::clip::show_error(
                "clip supergain: all input files must be EER files.",
            );
            for opened in &extra {
                crate::imod::libiimod::iimage::ii_fclose(&mut opened.fp.clone().unwrap());
            }
            return Err(-1);
        }
        for z in 0..header.nz {
            if crate::imod::libiimod::mrcfiles::mrc_read_slice(
                &mut input,
                &mut header.fp.clone().unwrap(),
                header,
                z,
                b'z',
            ) != 0
            {
                for opened in &extra {
                    crate::imod::libiimod::iimage::ii_fclose(&mut opened.fp.clone().unwrap());
                }
                return Err(-1);
            }
            for i in 0..n {
                image[i] = input[i] as f64;
            }
        }
    }
    let phys_x = h1.nx / 4 - 4;
    let phys_y = h1.ny / 4 - 4;
    let x_spacing = 4 * (phys_x / divisions);
    let y_spacing = 4 * (phys_y / divisions);
    let x_offset = 4 * ((phys_x % divisions) / 2);
    let y_offset = 4 * ((phys_y % divisions) / 2);
    if x_spacing <= 0 || y_spacing <= 0 {
        return Err(-1);
    }
    let mut bias = vec![[0_f64; 16]; (divisions * divisions) as usize];
    let mut total = [0_f64; 16];
    for yd in 0..divisions {
        for xd in 0..divisions {
            let a = &mut bias[(xd + yd * divisions) as usize];
            for y in y_offset + yd * y_spacing..y_offset + (yd + 1) * y_spacing {
                for x in x_offset + xd * x_spacing..x_offset + (xd + 1) * x_spacing {
                    let ind = (x % 4 + 4 * (y % 4)) as usize;
                    a[ind] += image[(x + y * h1.nx) as usize];
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
    let _ = ImodFile::Stdout.write_all(b"Overall gain factors for 4x4 super-resolution:\n");
    for i in (0..=12).rev().step_by(4) {
        for j in 0..4 {
            let _ = ImodFile::Stdout.write_all(
                c_format("  %.6f", &[CArg::Dbl((bsum / total[i + j]) as f64)]).as_bytes(),
            );
        }
        let _ = ImodFile::Stdout.write_all(b"\n");
    }
    let mut two = total;
    let b = combine_area_sums(&mut two) as f64;
    let _ = ImodFile::Stdout.write_all(b"Overall gain factors for 2x2 super-resolution:\n");
    for i in (0..=2).rev().step_by(2) {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "  %.6f  %.6f\n",
                &[
                    CArg::Dbl((b / two[i]) as f64),
                    CArg::Dbl((b / two[i + 1]) as f64),
                ],
            )
            .as_bytes(),
        );
    }
    let _ = out_fp.write_all(b"1  4\n");
    let _ = out_fp.write_all(
        c_format(
            "%d %d %d %d %d %d\n",
            &[
                CArg::Int((divisions - 1) as i64),
                CArg::Int((x_offset + x_spacing) as i64),
                CArg::Int((x_spacing) as i64),
                CArg::Int((divisions - 1) as i64),
                CArg::Int((y_offset + y_spacing) as i64),
                CArg::Int((y_spacing) as i64),
            ],
        )
        .as_bytes(),
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
                let _ = out_fp
                    .write_all(c_format(" %.5f", &[CArg::Dbl((bsum / value) as f64)]).as_bytes());
            }
            let _ = out_fp.write_all(b"\n");
            let q = combine_area_sums(&mut a) as f64;
            for i in 0..4 {
                let _ =
                    out_fp.write_all(c_format(" %.5f", &[CArg::Dbl((q / a[i]) as f64)]).as_bytes());
            }
            let _ = out_fp.write_all(b"\n");
        }
    }
    crate::imod::libiimod::iimage::ii_fclose(out_fp);
    for opened in &extra {
        crate::imod::libiimod::iimage::ii_fclose(&mut opened.fp.clone().unwrap());
    }
    Ok(())
}
/// Matches C++ `combineAreaSums`.
fn combine_area_sums(area_sum: &mut [f64]) -> f32 {
    {
        area_sum[0] = area_sum[0] + area_sum[1] + area_sum[4] + area_sum[5];
        area_sum[1] = area_sum[2] + area_sum[3] + area_sum[6] + area_sum[7];
        area_sum[2] = area_sum[8] + area_sum[9] + area_sum[12] + area_sum[13];
        area_sum[3] = area_sum[10] + area_sum[11] + area_sum[14] + area_sum[15];
        // `processing.cpp:3317`: bsum is a float, so each term is rounded to
        // single precision as it is added.
        let mut bsum = 0_f32;
        for index in 0..4 {
            bsum += (area_sum[index] / 4.) as f32;
        }
        bsum
    }
}
/// Matches C++ `clip_parxyz`.
pub fn clip_parxyz(
    v: &mut Istack,
    xmax: i32,
    ymax: i32,
    zmax: i32,
    rx: &mut f32,
    ry: &mut f32,
    rz: &mut f32,
) -> i32 {
    let sl = &v.slices[zmax as usize];
    let (xsize, ysize) = (v.slices[0].xsize, v.slices[0].ysize);
    let x1 = if xmax == 0 { xsize - 1 } else { xmax - 1 };
    let x3 = if xmax + 1 >= xsize { 0 } else { xmax + 1 };
    let a = slice_get_pixel_magnitude(sl, x1, ymax) * -1.
        + slice_get_pixel_magnitude(sl, xmax, ymax) * 2.
        + slice_get_pixel_magnitude(sl, x3, ymax) * -1.;
    let b = ((xmax - 1) * (xmax - 1)) as f32
        * (slice_get_pixel_magnitude(sl, xmax, ymax) - slice_get_pixel_magnitude(sl, x3, ymax))
        + (xmax * xmax) as f32
            * (slice_get_pixel_magnitude(sl, x3, ymax) - slice_get_pixel_magnitude(sl, x1, ymax))
        + ((xmax + 1) * (xmax + 1)) as f32
            * (slice_get_pixel_magnitude(sl, x1, ymax) - slice_get_pixel_magnitude(sl, xmax, ymax));
    *rx = if a != 0. { -b / (2. * a) } else { xmax as f32 };
    let y1 = if ymax == 0 { ysize - 1 } else { ymax - 1 };
    let y3 = if ymax + 1 >= ysize { 0 } else { ymax + 1 };
    let a = slice_get_pixel_magnitude(sl, xmax, y1) * -1.
        + slice_get_pixel_magnitude(sl, xmax, ymax) * 2.
        + slice_get_pixel_magnitude(sl, xmax, y3) * -1.;
    let b = ((ymax - 1) * (ymax - 1)) as f32
        * (slice_get_pixel_magnitude(sl, xmax, ymax) - slice_get_pixel_magnitude(sl, xmax, y3))
        + (ymax * ymax) as f32
            * (slice_get_pixel_magnitude(sl, xmax, y3) - slice_get_pixel_magnitude(sl, xmax, y1))
        + ((ymax + 1) * (ymax + 1)) as f32
            * (slice_get_pixel_magnitude(sl, xmax, y1) - slice_get_pixel_magnitude(sl, xmax, ymax));
    *ry = if a != 0. { -b / (2. * a) } else { ymax as f32 };
    let z1 = if zmax == 0 {
        v.slices.len() as i32 - 1
    } else {
        zmax - 1
    };
    let z3 = if zmax + 1 >= v.slices.len() as i32 {
        0
    } else {
        zmax + 1
    };
    let a = slice_get_pixel_magnitude(&v.slices[z1 as usize], xmax, ymax) * -1.
        + slice_get_pixel_magnitude(sl, xmax, ymax) * 2.
        + slice_get_pixel_magnitude(&v.slices[z3 as usize], xmax, ymax) * -1.;
    let b = ((zmax - 1) * (zmax - 1)) as f32
        * (slice_get_pixel_magnitude(sl, xmax, ymax)
            - slice_get_pixel_magnitude(&v.slices[z3 as usize], xmax, ymax))
        + (zmax * zmax) as f32
            * (slice_get_pixel_magnitude(&v.slices[z3 as usize], xmax, ymax)
                - slice_get_pixel_magnitude(&v.slices[z1 as usize], xmax, ymax))
        + ((zmax + 1) * (zmax + 1)) as f32
            * (slice_get_pixel_magnitude(&v.slices[z1 as usize], xmax, ymax)
                - slice_get_pixel_magnitude(sl, xmax, ymax));
    *rz = if a != 0. { -b / (2. * a) } else { zmax as f32 };
    0
}
/// Matches C++ `clip_stat3d`.
pub fn clip_stat3d(v: &mut Istack) -> i32 {
    let (mut min, mut max, mut mean) = (0., 0., 0.);
    let (mut xmax, mut ymax, mut zmax) = (0, 0, 0);
    clip_get_stat3d(
        v, &mut min, &mut max, &mut mean, &mut xmax, &mut ymax, &mut zmax,
    );
    let (mut x, mut y, mut z) = (0., 0., 0.);
    clip_parxyz(v, xmax, ymax, zmax, &mut x, &mut y, &mut z);
    let _ = ImodFile::Stdout.write_all(
        c_format(
            "max = %g  min = %g  mean = %g\n",
            &[
                CArg::Dbl((max as f64) as f64),
                CArg::Dbl((min as f64) as f64),
                CArg::Dbl((mean as f64) as f64),
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
                CArg::Dbl((x as f64) as f64),
                CArg::Dbl((y as f64) as f64),
                CArg::Dbl((z as f64) as f64),
            ],
        )
        .as_bytes(),
    );
    0
}
/// Matches C++ `clip_get_stat3d`.
pub fn clip_get_stat3d(
    v: &mut Istack,
    rmin: &mut f32,
    rmax: &mut f32,
    rmean: &mut f32,
    rx: &mut i32,
    ry: &mut i32,
    rz: &mut i32,
) -> i32 {
    let (nx, ny, nz) = (v.slices[0].xsize, v.slices[0].ysize, v.slices.len() as i32);
    let mut min = 1e36_f32;
    // `processing.cpp:22-23` redefines FLT_MIN as INT_MIN for this file and
    // <float.h> is not in its include set, so `max = FLT_MIN` seeds -2^31.
    let mut max = i32::MIN as f32;
    let mut sum = 0_f64;
    let (mut xmax, mut ymax, mut zmax) = (0, 0, 0);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let value = slice_get_pixel_magnitude(&mut v.slices[k as usize], i, j);
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
thread_local! {
    /// Where [`clip_stat`] records the mean and SD of each section row it
    /// prints, when a direct caller set it through [`clip_recording_stat`].
    static STAT_SINK: std::cell::RefCell<Option<std::sync::Arc<std::sync::Mutex<Vec<(f64, f64)>>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Rust-only: runs program `clip` (its `argv` from the in-process runner)
/// with the mean and SD of every section row `clip stat` prints (the
/// `%9.4f  %9.4f` columns of a plain, non-montage, non-outlier report)
/// recorded into `sink`, for a direct caller that used to parse those
/// columns (`copytomocoms`; see `CLAUDE.md`, "Wherever we control both
/// sides, use a direct function call now").  The program's output is
/// unchanged.  Run it under `commands::call_in_process`.
pub fn clip_recording_stat(sink: std::sync::Arc<std::sync::Mutex<Vec<(f64, f64)>>>) {
    STAT_SINK.with_borrow_mut(|slot| *slot = Some(sink));
    crate::imod::clip::clip::clip();
}

/// Matches C++ `clip_stat`.
pub fn clip_stat(hin: &mut MrcHeader, opt: &mut ClipOptions) -> Result<(), i32> {
    let outliers = opt.val != crate::imod::clip::clip::IP_DEFAULT as f32
        || opt.low != crate::imod::clip::clip::IP_DEFAULT as f32;
    let mut li = crate::imod::libiimod::mrcfiles::LoadInfo::default();
    crate::imod::libiimod::mrcfiles::mrc_init_li(Some(&mut li), None);
    let mut pcoords = Vec::new();
    if let Some(plname) = opt.plname.clone() {
        if crate::imod::libiimod::plist::mrc_plist_li(&mut li, hin, &plname) != 0 {
            crate::imod::clip::clip::show_error("stat: error reading piece list file");
            return Err(-1);
        }
        if li.plist < hin.nz {
            crate::imod::clip::clip::show_error("stat: not enough piece coordinates in file");
            return Err(-1);
        }
        pcoords = li.pcoords.take().unwrap_or_default();
        pcoords.truncate((3 * li.plist) as usize);
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
            hin.nx,
            &mut min_x,
            &mut num_x,
            &mut overlap_x,
        ) != 0
            || crate::imod::libcfshr::piecefuncs::check_piece_list(
                &pcoords[1..],
                3,
                li.plist as usize,
                1,
                hin.ny,
                &mut min_y,
                &mut num_y,
                &mut overlap_y,
            ) != 0
        {
            crate::imod::clip::clip::show_error("stat: piece coordinates are not regularly spaced");
            return Err(-1);
        }
        if opt.new_xoverlap == crate::imod::clip::clip::IP_DEFAULT {
            opt.new_xoverlap = 0;
        }
        if opt.new_yoverlap == crate::imod::clip::clip::IP_DEFAULT {
            opt.new_yoverlap = 0;
        }
        if num_x > 1 {
            crate::imod::libcfshr::piecefuncs::adjust_piece_overlap(
                &mut pcoords,
                3,
                li.plist as usize,
                hin.nx,
                min_x,
                overlap_x,
                opt.new_xoverlap,
            );
        }
        if num_y > 1 {
            crate::imod::libcfshr::piecefuncs::adjust_piece_overlap(
                &mut pcoords[1..],
                3,
                li.plist as usize,
                hin.ny,
                min_y,
                overlap_y,
                opt.new_yoverlap,
            );
        }
    }
    if !pcoords.is_empty() {
        let axis = if opt.from_one != 0 {
            "y, view"
        } else {
            " y,   z"
        };
        let _ = ImodFile::Stdout.write_all(c_format("piece|   min   |(   x,  %s)|    max  |(   x,  %s)|   mean\n-----|---------|----------------|---------|----------------|---------\n", &[CArg::Str(axis), CArg::Str(axis)]).as_bytes());
    } else if opt.from_one != 0 {
        let _ = ImodFile::Stdout.write_all(b"view |   min   |(   x,   y)|    max  |(      x,      y)|   mean    |  std dev.\n-----|---------|-----------|---------|-----------------|-----------|----------\n");
    } else {
        let _ = ImodFile::Stdout.write_all(b"slice|   min   |(   x,   y)|    max  |(      x,      y)|   mean    |  std dev.\n-----|---------|-----------|---------|-----------------|-----------|----------\n");
    }
    crate::imod::clip::file_io::set_input_options(opt, hin);
    // `processing.cpp:3480` float ptnum, `:3481` double vmean/vsumsq.
    let mut ptnum = 0_f32;
    let mut vmean = 0_f64;
    let mut all_min = f32::INFINITY;
    let mut all_max = f32::NEG_INFINITY;
    let mut zmin = 0_i32;
    let mut zmax = 0_i32;
    let add = if opt.from_one != 0 { 1 } else { 0 };
    // `processing.cpp:3534-3537` mallocs `stats`, `allmins`, `allmaxes` and
    // `ifdrop` at `opt->nofsecs` entries each, once, before the loop.
    let mut allmins = Vec::with_capacity(opt.nofsecs.max(0) as usize);
    let mut allmaxes = Vec::with_capacity(opt.nofsecs.max(0) as usize);
    let mut stat_rows = Vec::with_capacity(opt.nofsecs.max(0) as usize);
    for k in 0..opt.nofsecs {
        let iz = opt.secs[(k) as usize];
        if iz < 0 || iz >= hin.nz {
            crate::imod::clip::clip::show_error("stat: slice out of range.");
            return Err(-1);
        }
        let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            iz,
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error("stat: error reading slice.");
            return Err(-1);
        };
        // `processing.cpp:3541-3551` deliberately seeds both extrema from
        // the center pixel while their coordinates start at zero.  In
        // particular, a center maximum is not selected again by the
        // strict `>` comparison below; its fitted position therefore uses
        // the source's `(0, 0)` coordinate.
        let center = slice_get_pixel_magnitude(s.as_ref(), s.xsize / 2, s.ysize / 2);
        let mut min = center;
        let mut max = center;
        let mut xmin = 0;
        let mut ymin = 0;
        let mut xmax = 0;
        let mut ymax = 0;
        let mut sum = 0_f64;
        let mut square = 0_f64;
        // `processing.cpp:3575-3586`: a preliminary mean from 64 widely
        // spaced samples, which every pixel is then measured against.  The
        // source added it on 5/19/17 precisely because summing the raw
        // values loses the SD when it is small beside the mean, so dropping
        // it is not a simplification -- `clip stats` on a short volume with
        // mean 2009 and SD 5779 already prints a different last digit.
        let ptnum_slice = (s.xsize * s.ysize) as f32;
        let mut prelim_mean = center as f64;
        if s.xsize > 10 && s.ysize > 10 {
            let mut tsum = 0_f64;
            for j in 1..9 {
                for i in 1..9 {
                    tsum += slice_get_pixel_magnitude(
                        s.as_ref(),
                        (i * s.xsize) / 10,
                        (j * s.ysize) / 10,
                    ) as f64;
                }
            }
            prelim_mean = tsum / 64.;
        }
        // `double tsum, tsumsq` are function-scope in the source and reset
        // at the head of every row.
        let mut tsum;
        let mut tsumsq;
        // `processing.cpp:3459-3472`'s `PROCESS_PIXEL` macro, which each of
        // the four loops below expands.  `float m` means the subtraction
        // rounds back to single precision and `m * m` is a single-precision
        // product that only then widens for the accumulation.
        macro_rules! process_pixel {
            ($value:expr, $i:expr, $j:expr) => {{
                let mut m: f32 = $value;
                if m > max {
                    max = m;
                    xmax = $i;
                    ymax = $j;
                }
                if m < min {
                    min = m;
                    xmin = $i;
                    ymin = $j;
                }
                m = (m as f64 - prelim_mean) as f32;
                tsum += m as f64;
                tsumsq += (m * m) as f64;
            }};
        }
        // Performance: the same pixel sequence as `process_pixel!`, in two
        // passes over a row.  The extremes and the sums never read each
        // other, so they can be taken separately.  The extremes pass first
        // asks, over the whole row, whether any pixel is strictly beyond the
        // current extremes (a vectorisable test); only then does it rescan
        // the row in order with the macro's own `>`/`<` updates, which are
        // the only ones that could fire (max only grows, min only shrinks,
        // NaN never compares).  The sums then run in the source's order.
        // LLVM otherwise turned the two tests into a chain of four
        // conditional moves per pixel: 25% slower than native on a
        // 4096 x 4096 x 35 float stack.
        macro_rules! process_row {
            ($row:expr, $y:expr, $to_f32:expr) => {{
                let row = $row;
                let to_f32 = $to_f32;
                let mut beyond = false;
                for &v in row {
                    let m: f32 = to_f32(v);
                    beyond |= (m > max) | (m < min);
                }
                if beyond {
                    for (x, &v) in row.iter().enumerate() {
                        let m: f32 = to_f32(v);
                        if m > max {
                            max = m;
                            xmax = x as i32;
                            ymax = $y;
                        }
                        if m < min {
                            min = m;
                            xmin = x as i32;
                            ymin = $y;
                        }
                    }
                }
                for &v in row {
                    let m = (to_f32(v) as f64 - prelim_mean) as f32;
                    tsum += m as f64;
                    tsumsq += (m * m) as f64;
                }
            }};
        }
        for y in 0..s.ysize {
            // `processing.cpp:3587-3624` accumulates a row at a time, so
            // the summation order is per row, not over the whole slice.
            tsum = 0_f64;
            tsumsq = 0_f64;
            match s.mode {
                // `processing.cpp:3591`: "5/19/17: It now is about twice as
                // fast to do it directly, added USHORT/FLOAT".  All three of
                // these modes have `csize == 1`, so the datum read straight
                // out of the row is exactly what `sliceGetPixelMagnitude`
                // returns for them.
                crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_SHORT => {
                    // `processing.cpp:3594`: `sdata = &slice->data.s[slice->xsize * j]`.
                    let row = &s.data.s()[(s.xsize * y) as usize..][..s.xsize as usize];
                    process_row!(row, y, |v: i16| v as f32);
                }
                crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_USHORT => {
                    // `processing.cpp:3602`: `usdata = &slice->data.us[slice->xsize * j]`.
                    let row = &s.data.us()[(s.xsize * y) as usize..][..s.xsize as usize];
                    process_row!(row, y, |v: u16| v as f32);
                }
                crate::imod::libcfshr::reduce_by_binning::SLICE_MODE_FLOAT => {
                    // `processing.cpp:3610`: `fdata = &slice->data.f[slice->xsize * j]`.
                    let row = &s.data.f()[(s.xsize * y) as usize..][..s.xsize as usize];
                    process_row!(row, y, |v: f32| v);
                }
                // Byte takes the source's `default:` arm, whose
                // `sliceGetPixelMagnitude` returns the byte widened to
                // `float` for a one-component slice.  Reading the row directly
                // hands `PROCESS_PIXEL` the identical value in the same order.
                crate::imod::libiimod::mrcslice::SLICE_MODE_BYTE if s.csize == 1 => {
                    let row = &s.data.b()[(s.xsize * y) as usize..][..s.xsize as usize];
                    process_row!(row, y, |v: u8| v as f32);
                }
                _ => {
                    for x in 0..s.xsize {
                        process_pixel!(slice_get_pixel_magnitude(s.as_ref(), x, y), x, y);
                    }
                }
            }
            sum += tsum;
            square += tsumsq;
        }
        let mut mean = sum / ptnum_slice as f64;
        // `processing.cpp:3632-3633`: `sqrt(B3DMAX(0., std))`.  `B3DMAX` is
        // `a > b ? a : b`, so a NaN variance stays NaN and prints `nan`;
        // `f64::max` would return 0 and print `0.0000`.  (The denominator's
        // `B3DMAX(1., ptnum - 1.)` is left as `max`: `ptnum` is a count, never NaN.)
        let var = (square - ptnum_slice as f64 * mean * mean) / 1_f64.max(ptnum_slice as f64 - 1.);
        let sd = if 0. > var { 0. } else { var }.sqrt();
        mean += prelim_mean;
        // `processing.cpp:3636-3649`: refine the maximum with the same
        // source-mapped 3x3 parabolic fit before applying output coordinates.
        let mut data = [[0_f64; 3]; 3];
        for dj in -1..=1 {
            for di in -1..=1 {
                data[(dj + 1) as usize][(di + 1) as usize] =
                    slice_get_pixel_magnitude(s.as_ref(), xmax + di, ymax + dj) as f64;
            }
        }
        let (mut cx, mut cy) = (0_f64, 0_f64);
        crate::imod::clip::correlation::parabolic_fit(&mut cx, &mut cy, &data);
        // `processing.cpp:3481` declares `float x, y`: the fitted peak is
        // rounded to float here, and `:3653-3658` adjust it in float.
        let (mut peak_x, mut peak_y) = ((cx + xmax as f64) as f32, (cy + ymax as f64) as f32);
        if opt.sano != 0 {
            peak_x -= s.xsize as f32 / 2.;
            peak_y -= s.ysize as f32 / 2.;
        } else {
            let adjust_x = opt.cx as i32 - opt.ix / 2;
            let adjust_y = opt.cy as i32 - opt.iy / 2;
            peak_x += adjust_x as f32;
            peak_y += adjust_y as f32;
            xmin += adjust_x;
            ymin += adjust_y;
        }
        if !outliers && !pcoords.is_empty() {
            let base = (3 * iz) as usize;
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "%4d  %9.4f (%4d,%4d,%4d) %9.4f (%4d,%4d,%4d) %9.4f\n",
                    &[
                        CArg::Int((iz + add) as i64),
                        CArg::Dbl((min as f64) as f64),
                        CArg::Int((xmin + pcoords[base] + add) as i64),
                        CArg::Int((ymin + pcoords[base + 1] + add) as i64),
                        CArg::Int((pcoords[base + 2] + add) as i64),
                        CArg::Dbl((max as f64) as f64),
                        CArg::Int(
                            ((peak_x as f64 + 0.5).floor() as i32 + pcoords[base] + add) as i64,
                        ),
                        CArg::Int(
                            ((peak_y as f64 + 0.5).floor() as i32 + pcoords[base + 1] + add) as i64,
                        ),
                        CArg::Int((pcoords[base + 2] + add) as i64),
                        CArg::Dbl((mean) as f64),
                    ],
                )
                .as_bytes(),
            );
        } else if !outliers {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "%4d  %9.4f (%4d,%4d) %9.4f (%7.2f,%7.2f) %9.4f  %9.4f\n",
                    &[
                        CArg::Int((iz + add) as i64),
                        CArg::Dbl((min as f64) as f64),
                        CArg::Int((xmin + add) as i64),
                        CArg::Int((ymin + add) as i64),
                        CArg::Dbl((max as f64) as f64),
                        CArg::Dbl((peak_x) as f64),
                        CArg::Dbl((peak_y) as f64),
                        CArg::Dbl((mean) as f64),
                        CArg::Dbl((sd) as f64),
                    ],
                )
                .as_bytes(),
            );
            STAT_SINK.with_borrow(|slot| {
                if let Some(sink) = slot {
                    sink.lock()
                        .expect("clip stat sink")
                        .push((mean as f64, sd as f64));
                }
            });
        }
        // `processing.cpp:3660-3676`: the first selected section seeds the
        // extrema, and its zmax is set to 0 rather than to iz.
        if k == 0 {
            all_min = min;
            all_max = max;
            vmean = 0.;
            zmin = iz;
            zmax = 0;
        } else {
            if min < all_min {
                all_min = min;
                zmin = iz;
            }
            if max > all_max {
                all_max = max;
                zmax = iz;
            }
        }
        vmean += mean;
        ptnum = (s.xsize * s.ysize) as f32;
        allmins.push(min);
        allmaxes.push(max);
        stat_rows.push((iz, xmin, ymin, peak_x, peak_y, mean, sd));
    }
    let mut flagged: Vec<bool> = Vec::new();
    if outliers {
        let mut length = opt.nofsecs;
        if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
            length = opt.low.round() as i32;
        }
        // `processing.cpp:3548`: B3DMIN(nofsecs, B3DMAX(5, length)).
        length = opt.nofsecs.min(5.max(length));
        let kcrit = if opt.val != crate::imod::clip::clip::IP_DEFAULT as f32 {
            opt.val
        } else {
            2.24
        };
        flagged = vec![false; stat_rows.len()];
        // `processing.cpp:3537` mallocs `ifdrop` once at `nofsecs` entries,
        // and `:3711`/`:3716` pass `rsMadMedianOutliers` the sub-ranges
        // `&allmins[di]` / `&ifdrop[di]` in place: neither the input window
        // nor the output window is copied per section.
        let mut ifdrop = vec![0_f32; opt.nofsecs.max(0) as usize];
        for kk in 0..stat_rows.len() {
            let mut di = (kk as i32 - length / 2).max(0);
            let dj = (di + length).min(opt.nofsecs);
            di = (dj - length).max(0);
            crate::imod::libcfshr::robuststat::rs_mad_median_outliers(
                &allmins[di as usize..dj as usize],
                length,
                kcrit,
                &mut ifdrop[di as usize..dj as usize],
            );
            // `processing.cpp:3712-3715` reads `ifdrop[kk]` for the minima
            // before the maxima call overwrites the same array.
            let starmin = if ifdrop[kk] < 0. { b'*' } else { b' ' };
            if ifdrop[kk] < 0. {
                flagged[kk] = true;
            }
            crate::imod::libcfshr::robuststat::rs_mad_median_outliers(
                &allmaxes[di as usize..dj as usize],
                length,
                kcrit,
                &mut ifdrop[di as usize..dj as usize],
            );
            let starmax = if ifdrop[kk] > 0. { b'*' } else { b' ' };
            if ifdrop[kk] > 0. {
                flagged[kk] = true;
            }
            let (iz, xmin, ymin, peak_x, peak_y, mean, sd) = stat_rows[kk];
            if !pcoords.is_empty() {
                // `processing.cpp:3723-3726` indexes the piece list with the
                // loop counter kk for X and Y, but with secs[kk] for Z.
                let base = 3 * kk;
                let zbase = (3 * iz + 2) as usize;
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%4d  %9.4f%c(%4d,%4d,%4d) %9.4f%c(%4d,%4d,%4d) %9.4f  %9.4f\n",
                        &[
                            CArg::Int((iz + add) as i64),
                            CArg::Dbl((allmins[kk] as f64) as f64),
                            CArg::Chr((starmin as i32) as u8),
                            CArg::Int((xmin + pcoords[base] + add) as i64),
                            CArg::Int((ymin + pcoords[base + 1] + add) as i64),
                            CArg::Int((pcoords[zbase] + add) as i64),
                            CArg::Dbl((allmaxes[kk] as f64) as f64),
                            CArg::Chr((starmax as i32) as u8),
                            CArg::Int(
                                ((peak_x as f64 + 0.5).floor() as i32 + pcoords[base] + add) as i64,
                            ),
                            CArg::Int(
                                ((peak_y as f64 + 0.5).floor() as i32 + pcoords[base + 1] + add)
                                    as i64,
                            ),
                            CArg::Int((pcoords[zbase] + add) as i64),
                            CArg::Dbl((mean) as f64),
                            CArg::Dbl((sd) as f64),
                        ],
                    )
                    .as_bytes(),
                );
            } else {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%4d  %9.4f%c(%4d,%4d) %9.4f%c(%7d,%7d) %9.4f  %9.4f\n",
                        &[
                            CArg::Int((iz + add) as i64),
                            CArg::Dbl((allmins[kk] as f64) as f64),
                            CArg::Chr((starmin as i32) as u8),
                            CArg::Int((xmin) as i64),
                            CArg::Int((ymin) as i64),
                            CArg::Dbl((allmaxes[kk] as f64) as f64),
                            CArg::Chr((starmax as i32) as u8),
                            CArg::Int(((peak_x as f64 + 0.5).floor() as i32) as i64),
                            CArg::Int(((peak_y as f64 + 0.5).floor() as i32) as i64),
                            CArg::Dbl((mean) as f64),
                            CArg::Dbl((sd) as f64),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
    }
    // `processing.cpp:3742-3758`: the overall line is computed and printed
    // before the extreme-value list, pooling the per-section statistics.
    vmean /= opt.nofsecs as f64;
    let mut vsumsq = 0_f64;
    for row in &stat_rows {
        vsumsq +=
            ptnum as f64 * (row.5 * row.5 - vmean * vmean) + (ptnum as f64 - 1.) * row.6 * row.6;
    }
    ptnum *= opt.nofsecs as f32;
    let mut std = vsumsq / 1_f64.max(ptnum as f64 - 1.);
    // `processing.cpp:3752`: `sqrt(B3DMAX(0., std))` -- NaN-preserving, as above.
    std = if 0. > std { 0. } else { std }.sqrt();
    if !pcoords.is_empty() {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                " all  %9.4f (@ piece =%5d) %9.4f (@ piece =%5d) %9.4f  %9.4f\n",
                &[
                    CArg::Dbl((all_min as f64) as f64),
                    CArg::Int((zmin + 1) as i64),
                    CArg::Dbl((all_max as f64) as f64),
                    CArg::Int((zmax + 1) as i64),
                    CArg::Dbl((vmean) as f64),
                    CArg::Dbl((std) as f64),
                ],
            )
            .as_bytes(),
        );
    } else {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                " all  %9.4f (@ z=%5d) %9.4f (@ z=%5d      ) %9.4f  %9.4f\n",
                &[
                    CArg::Dbl((all_min as f64) as f64),
                    CArg::Int((zmin + add) as i64),
                    CArg::Dbl((all_max as f64) as f64),
                    CArg::Int((zmax + add) as i64),
                    CArg::Dbl((vmean) as f64),
                    CArg::Dbl((std) as f64),
                ],
            )
            .as_bytes(),
        );
    }
    if outliers {
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "\n%s with %sextreme values:",
                &[
                    CArg::Str(if pcoords.is_empty() {
                        if opt.from_one != 0 { "Views" } else { "Slices" }
                    } else {
                        "Pieces"
                    }),
                    CArg::Str(if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
                        "locally "
                    } else {
                        ""
                    }),
                ],
            )
            .as_bytes(),
        );
        let mut number = 0;
        // `processing.cpp:3766-3775`.
        let mut line_length = 28
            + if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
                8
            } else {
                0
            };
        for (index, row) in stat_rows.iter().enumerate() {
            if flagged[index] {
                let _ = ImodFile::Stdout
                    .write_all(c_format(" %3d", &[CArg::Int((row.0 + add) as i64)]).as_bytes());
                line_length += 4;
                if line_length > 74 {
                    let _ = ImodFile::Stdout.write_all(b"\n");
                    line_length = 0;
                }
                number += 1;
            }
        }
        if number == 0 {
            let _ = ImodFile::Stdout.write_all(b" None");
        }
        let _ = ImodFile::Stdout.write_all(b"\n");
    }
    Ok(())
}
/// Matches C++ `clipHistogram`.
pub fn clip_histogram(hin: &mut MrcHeader, opt: &mut ClipOptions) -> i32 {
    crate::imod::clip::file_io::set_input_options(opt, hin);
    if opt.sano != 0 {
        return match histogram_peaks_and_dip(hin, opt) {
            Ok(()) => 0,
            status => status.unwrap_err(),
        };
    }
    let floating = matches!(hin.mode, 2 | 4);
    let (hist_min, hist_max, delta, bins_len, offset) = if floating {
        let lo = if opt.low == crate::imod::clip::clip::IP_DEFAULT as f32 {
            hin.amin
        } else {
            opt.low
        };
        let hi = if opt.high == crate::imod::clip::clip::IP_DEFAULT as f32 {
            hin.amax
        } else {
            opt.high
        };
        if lo >= hi {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "ERROR: clip histogram - minimum (%f) must be less than maximum (%f)\n",
                    &[CArg::Dbl((lo as f64) as f64), CArg::Dbl((hi as f64) as f64)],
                )
                .as_bytes(),
            );
            return -1;
        }
        let d = if opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
            (hi - lo) / 256.
        } else {
            opt.val
        };
        if d <= 0. {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "ERROR: clip histogram - histogram bin size (%f) must be positive\n",
                    &[CArg::Dbl((d as f64) as f64)],
                )
                .as_bytes(),
            );
            return -1;
        }
        let mut number = ((hi - lo) / d).ceil() as i32;
        // `processing.cpp:3844`: a float quotient compared against
        // `numBins - 0.01`, which is a double.
        if ((hi - lo) / d) as f64 >= number as f64 - 0.01 {
            number += 1;
        }
        (lo, hi, d, number.clamp(0, 65535) as usize, 0)
    } else {
        match hin.mode {
            0 | 16 => (0., 255., 1., 256, 0),
            1 => (-32768., 32767., 1., 65536, 32768),
            6 => (0., 65535., 1., 65536, 0),
            _ => return -1,
        }
    };
    let mut bins = vec![0_i64; bins_len];
    let mut nx = 0;
    let mut ny = 0;
    for k in 0..opt.nofsecs {
        let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            opt.secs[(k) as usize],
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            crate::imod::clip::clip::show_error(&c_format(
                "clip histogram - error reading slice %d",
                &[CArg::Int(opt.secs[k as usize] as i64)],
            ));
            return -1;
        };
        nx = s.xsize;
        ny = s.ysize;
        // `processing.cpp:3862-3876`: `if (floatVals)` at row level, with a
        // separate column loop for each arm.
        for y in 0..ny {
            // `sliceGetPixelMagnitude` is called per pixel in the source.
            // For a one-component slice it returns the element widened to
            // `float` and nothing else (`slice_get_val` + `csize == 1`), so
            // for those modes the row is read directly, choosing the element
            // type once per row: the same `float` reaches the same binning
            // expression, in the same order.  Other modes keep the call.
            let row = y as usize * nx as usize..(y as usize + 1) * nx as usize;
            if floating {
                let mut add = |v: f32| {
                    // `ind = (val - histMin) / delta` converts a float to
                    // `int`.  x86's `cvttss2si` gives INT_MIN for NaN (and for
                    // out-of-range values), which the `ind >= 0` test then
                    // skips; Rust's `as` maps NaN to 0, which would count a
                    // NaN pixel in bin 0.  Out-of-range values saturate and
                    // fail the range test either way.
                    let q = (v - hist_min) / delta;
                    let ind = if q.is_nan() { isize::MIN } else { q as isize };
                    if ind >= 0 && (ind as usize) < bins.len() {
                        bins[ind as usize] += 1;
                    }
                };
                match (s.mode, s.csize) {
                    (crate::imod::libiimod::mrcslice::SLICE_MODE_FLOAT, 1) => {
                        s.data.f()[row].iter().for_each(|&v| add(v))
                    }
                    _ => {
                        for x in 0..nx {
                            add(slice_get_pixel_magnitude(s.as_ref(), x, y));
                        }
                    }
                }
            } else {
                let mut add = |v: f32| {
                    let ind = v.round() as isize + offset;
                    // `:3873` has no range test; see the deliberate check
                    // recorded in `TO_OPT.md` (the C's is a heap overflow).
                    if ind >= 0 && (ind as usize) < bins.len() {
                        bins[ind as usize] += 1;
                    }
                };
                match (s.mode, s.csize) {
                    (crate::imod::libiimod::mrcslice::SLICE_MODE_BYTE, 1) => {
                        s.data.b()[row].iter().for_each(|&v| add(v as f32))
                    }
                    (crate::imod::libiimod::mrcslice::SLICE_MODE_SHORT, 1) => {
                        s.data.s()[row].iter().for_each(|&v| add(v as f32))
                    }
                    (crate::imod::libiimod::mrcslice::SLICE_MODE_USHORT, 1) => {
                        s.data.us()[row].iter().for_each(|&v| add(v as f32))
                    }
                    _ => {
                        for x in 0..nx {
                            add(slice_get_pixel_magnitude(s.as_ref(), x, y));
                        }
                    }
                }
            }
        }
    }
    // `processing.cpp:3880-3894` computes the percentile threshold in its
    // own pass over the raw, uncombined bin array, before minBin/maxBin are
    // found and before any bin combining.  Folding it into the combined
    // print loop uses combined counts and a stepped index instead, which
    // only agrees when the combining factor is 1.
    let mut threshold_value = 0_f32;
    let mut got_threshold = false;
    if opt.thresh != crate::imod::clip::clip::IP_DEFAULT as f32 {
        let thresh_counts = ((opt.thresh * opt.nofsecs as f32) as f64 * nx as f64) * ny as f64;
        let mut cumul_counts = 0_f64;
        for ind in 0..bins.len() {
            cumul_counts += bins[ind] as f64;
            if cumul_counts >= thresh_counts {
                let frac = ((cumul_counts - thresh_counts) / bins[ind] as f64) as f32;
                threshold_value = if floating {
                    hist_min + (ind as f32 - frac) * delta
                } else {
                    (ind as f32 - frac) - offset as f32
                };
                got_threshold = true;
                break;
            }
        }
    }
    let first = bins.iter().position(|&n| n != 0);
    let last = bins.iter().rposition(|&n| n != 0);
    if let (Some(mut a), Some(mut b)) = (first, last) {
        if !floating {
            if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32
                && opt.high != crate::imod::clip::clip::IP_DEFAULT as f32
                && opt.low > opt.high
            {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "ERROR: clip histogram - minimum (%f) must be less than maximum (%f)\n",
                        &[
                            CArg::Dbl((opt.low as f64) as f64),
                            CArg::Dbl((opt.high as f64) as f64),
                        ],
                    )
                    .as_bytes(),
                );
                return -1;
            }
            if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32 {
                a = a.max((opt.low + offset as f32).round() as usize);
            }
            if opt.high != crate::imod::clip::clip::IP_DEFAULT as f32 {
                b = b.min((opt.high + offset as f32).round() as usize);
            }
            if a > b {
                a = 1;
            }
        }
        let combine = if floating {
            1
        } else if opt.val == crate::imod::clip::clip::IP_DEFAULT as f32 {
            // `processing.cpp:3949` is B3DNINT((maxBin - minBin) / 256.),
            // i.e. floor(x + 0.5) on a double quotient.  Integer division
            // truncates instead and picks the bin width one too small
            // whenever the quotient's fraction is at least a half.
            1.max((((b - a) as f64) / 256. + 0.5).floor() as usize)
        } else {
            // `processing.cpp:3942-3947`.
            let entered = (opt.val as f64 + 0.5).floor() as i32;
            if entered < 1 {
                crate::imod::clip::clip::show_error(&c_format(
                    "clip histogram - Entered bin size (%f) must be > 0.5",
                    &[CArg::Dbl(opt.val as f64)],
                ));
                return -1;
            }
            entered as usize
        };
        if floating {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    " Bin midpoint   counts    (bin interval is %f)\n",
                    &[CArg::Dbl((delta as f64) as f64)],
                )
                .as_bytes(),
            );
        } else if combine > 1 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "Bin midpoint   counts    (bin interval is %d)\n",
                    &[CArg::Int((combine as i32) as i64)],
                )
                .as_bytes(),
            );
        } else {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    " Value   counts    (bin interval is %d)\n",
                    &[CArg::Int((combine as i32) as i64)],
                )
                .as_bytes(),
            );
        }
        for i in (a..=b).step_by(combine) {
            // `processing.cpp:3966-3968` clamps each contributing index to
            // maxBin rather than stopping at it, so a final partial group
            // re-adds bins[maxBin] once per missing slot.  Truncating the
            // range instead undercounts that last group.
            let count: (i64) = (0..combine).map(|n| bins[(i + n).min(b)]).sum();
            // `processing.cpp:3932` and `:3963` evaluate the midpoint in
            // double: the integer bin index promotes against the `0.5`
            // literal, and the float `delta`/`histMin` widen into the same
            // expression.  Computing it in f32 and widening afterwards
            // changes the sixth significant digit that `%13.6g` prints.
            let middle = if floating {
                hist_min as f64 + (i as f64 + 0.5) * delta as f64
            } else if combine > 1 {
                i as f64 - offset as f64 + combine as f64 * 0.5
            } else {
                i as f64 - offset as f64
            };
            if floating {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%13.6g %8d\n",
                        &[CArg::Dbl((middle) as f64), CArg::Int((count as i32) as i64)],
                    )
                    .as_bytes(),
                );
            } else if combine > 1 {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%12.1f %8d\n",
                        &[CArg::Dbl((middle) as f64), CArg::Int((count as i32) as i64)],
                    )
                    .as_bytes(),
                );
            } else {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%6d %8d\n",
                        &[
                            CArg::Int((middle as i32) as i64),
                            CArg::Int((count as i32) as i64),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
        // `processing.cpp:3981-3982` prints unconditionally; when the
        // percentile is never reached, C's `thresh` is still uninitialised
        // stack storage.  Zero is used here for that indeterminate value.
        let _ = got_threshold;
        if opt.thresh != crate::imod::clip::clip::IP_DEFAULT as f32 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "Threshold value for reaching %g of counts = %g\n",
                    &[
                        CArg::Dbl((opt.thresh as f64) as f64),
                        CArg::Dbl((threshold_value as f64) as f64),
                    ],
                )
                .as_bytes(),
            );
        }
        let mut combo_bins = Vec::new();
        for i in (a..=b).step_by(combine) {
            combo_bins.push(
                (0..combine)
                    .map(|index| bins[(i + index).min(b)])
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
        // `processing.cpp:3934`/`:3974` is `if (peakInd < 0 || bins[ind] >
        // bins[peakInd]) peakInd = ind;` -- strictly greater, so the *first*
        // of two equal maxima wins.  `Iterator::max_by_key` returns the last
        // one, which picks a different peak whenever the histogram ties.
        let mut peak_ind = -1_i32;
        for (index, count) in combo_bins.iter().enumerate() {
            if peak_ind < 0 || *count > combo_bins[peak_ind as usize] {
                peak_ind = index as i32;
            }
        }
        let peak_ind = peak_ind.max(0);
        if opt.falloff_frac != crate::imod::clip::clip::IP_DEFAULT as f32 {
            let (dir, mut ind) = if opt.falloff_frac > 0. {
                (1_i32, 1_i32)
            } else {
                (-1, num_bins - 2)
            };
            let mut diff_ind = -1_i32;
            let mut max_diff = 0_f32;
            // `processing.cpp:3797`: numFit is a function-scope int seeded 7.
            let mut num_fit_state = 7_i32;
            let mut cumulative_counts = 0_f64;
            let threshold_counts =
                opt.falloff_frac.abs() as f64 * opt.nofsecs as f64 * nx as f64 * ny as f64;
            while ind > 0 && ind < num_bins - 1 {
                cumulative_counts += combo_bins[(ind - dir) as usize] as f64;
                let current = combo_bins[ind as usize];
                let previous = combo_bins[(ind - dir) as usize];
                let frac = if current > 3000 && previous > 3000 {
                    previous as f32 / current as f32
                } else {
                    // `processing.cpp:4006-4015` is four separate `if`
                    // statements with a single `else` on the last, so the
                    // first three assignments are always overwritten.
                    let mut num_fit = num_fit_state;
                    if current > 1000 && previous > 1000 {
                        num_fit = 3;
                    }
                    if current > 300 && previous > 300 {
                        num_fit = 5;
                    }
                    if current > 100 && previous > 100 {
                        num_fit = 7;
                    }
                    if current > 30 && previous > 30 {
                        num_fit = 9;
                    } else {
                        num_fit = 11;
                    }
                    num_fit_state = num_fit;
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
                        &xx,
                        &yy,
                        count,
                        &mut slope,
                        &mut intercept,
                        &mut ro,
                    );
                    // `processing.cpp:4020`: B3DMAX(1., ...) makes the
                    // denominator a double, so the division is in double.
                    (((slope * (ind - dir - start) as f32 + intercept) as f64)
                        / 1.0_f64.max((slope * (ind - start) as f32 + intercept) as f64))
                        as f32
                };
                if current > 100 && frac > max_diff && cumulative_counts > threshold_counts {
                    max_diff = frac;
                    diff_ind = ind;
                }
                ind += dir;
            }
            if diff_ind < 0 {
                let _ = ImodFile::Stdout.write_all(c_format("ERROR: CLIP - No point of maximum falloff could be found past %.3f of total cumulative counts\n", &[CArg::Dbl((opt.falloff_frac.abs() as f64) as f64)]).as_bytes());
                return -1;
            }
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "Maximum falloff occurs at %g\n",
                    &[CArg::Dbl(
                        ((combo_left + diff_ind as f32 * bin_delta) as f64) as f64,
                    )],
                )
                .as_bytes(),
            );
        }
        if opt.pctl_frac == crate::imod::clip::clip::IP_DEFAULT as f32 {
            return 0;
        }
        if peak_ind as f32 > 0.8 * num_bins as f32 || (peak_ind as f32) < 0.2 * num_bins as f32 {
            let _ = ImodFile::Stdout.write_all(c_format("ERROR: CLIP - Peak is at %f, too close to end of range to analyze for extra counts\n", &[CArg::Dbl((combo_left as f64 + (peak_ind as f64 + 0.5) * bin_delta as f64) as f64)]).as_bytes());
            return -1;
        }
        // `processing.cpp:4052` indexes `bins`, not `comboBins`.  For the
        // float path and for combine == 1, `comboBins` is `&bins[minBin]`,
        // so these three reads land minBin elements too low; only when
        // combine > 1 (where C sets `comboBins = bins` and writes the sums
        // back into bins[0..numBins]) do they coincide.
        let fit_at = |index: i32| -> f32 {
            if combine > 1 {
                combo_bins[index as usize] as f32
            } else {
                bins[index as usize] as f32
            }
        };
        let frac = crate::imod::libcfshr::filtxcorr::parabolic_fit_position(
            fit_at(peak_ind - 1),
            fit_at(peak_ind),
            fit_at(peak_ind + 1),
        ) as f32;
        let (dir, mut ind) = if opt.pctl_frac > 0. {
            (1_i32, num_bins - 1)
        } else {
            (-1, 0)
        };
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Interpolated peak position %13.6g\n",
                &[CArg::Dbl(
                    (combo_left as f64 + (peak_ind as f64 + frac as f64 + 0.5) * bin_delta as f64)
                        as f64,
                )],
            )
            .as_bytes(),
        );
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "Bins %s peak minus bins %s peak:\n",
                &[
                    CArg::Str(if dir > 0 { "above" } else { "below" }),
                    CArg::Str(if dir > 0 { "below" } else { "above" }),
                ],
            )
            .as_bytes(),
        );
        let mut diff_ind = -1_i32;
        while dir * (ind - (peak_ind + dir * 3)) > 0 {
            let r_ind = 2. * (peak_ind as f32 + frac) - ind as f32;
            let j = r_ind.floor() as i32;
            let ff = r_ind - j as f32;
            if j >= 0 && j < num_bins - 1 {
                // `processing.cpp:4076`: (1. - ff) makes the first product
                // a double; the sum is narrowed back into the float `val`.
                let value = ((1. - ff as f64) * combo_bins[j as usize] as f64
                    + (ff * combo_bins[(j + 1) as usize] as f32) as f64)
                    as f32;
                let difference = combo_bins[ind as usize] - (value as f64 + 0.5).floor() as i64;
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%13.6g %8d\n",
                        &[
                            CArg::Dbl(
                                (combo_left as f64 + (ind as f64 + 0.5) * bin_delta as f64) as f64,
                            ),
                            CArg::Int((difference as i32) as i64),
                        ],
                    )
                    .as_bytes(),
                );
                combo_bins[ind as usize] = difference;
                if diff_ind < 0 || difference > combo_bins[diff_ind as usize] {
                    diff_ind = ind;
                }
            }
            ind -= dir;
        }
        if diff_ind < 0 || combo_bins[diff_ind as usize] <= 0 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "ERROR: CLIP - There are fewer counts %s the peak than %s it\n",
                    &[
                        CArg::Str(if dir > 0 { "above" } else { "below" }),
                        CArg::Str(if dir > 0 { "below" } else { "above" }),
                    ],
                )
                .as_bytes(),
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
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "ERROR: CLIP - There are fewer counts %s the peak than %s it\n",
                    &[
                        CArg::Str(if dir > 0 { "above" } else { "below" }),
                        CArg::Str(if dir > 0 { "below" } else { "above" }),
                    ],
                )
                .as_bytes(),
            );
            return -1;
        }
        let threshold_counts = opt.pctl_frac.abs() as f64 * cumul;
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
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%.3f of the extra counts %s the peak occur %s %g\n",
                        &[
                            CArg::Dbl((opt.pctl_frac.abs() as f64) as f64),
                            CArg::Str(if dir > 0 { "above" } else { "below" }),
                            CArg::Str(if dir > 0 { "above" } else { "below" }),
                            CArg::Dbl((value as f64) as f64),
                        ],
                    )
                    .as_bytes(),
                );
                return 0;
            }
            ind -= dir;
        }
        let _ = ImodFile::Stdout.write_all(
            c_format(
                "ERROR: CLIP - Extra counts occurred %s the peak but percentile analysis failed\n",
                &[CArg::Str(if dir > 0 { "above" } else { "below" })],
            )
            .as_bytes(),
        );
        return -1;
    } else {
        let _ = ImodFile::Stdout.write_all(b"There are no values within the specified range\n");
        0
    }
}
/// Matches C++ `histogramPeaksAndDip`.
pub fn histogram_peaks_and_dip(hin: &mut MrcHeader, opt: &mut ClipOptions) -> Result<(), i32> {
    if opt.low != crate::imod::clip::clip::IP_DEFAULT as f32
        || opt.high != crate::imod::clip::clip::IP_DEFAULT as f32
        || opt.val != crate::imod::clip::clip::IP_DEFAULT as f32
    {
        let _ = ImodFile::Stdout.write_all(b"ERROR: CLIP - The -n, -l, and -h options have no effect when doing a histogram with -s\n");
        return Err(-1);
    }
    let volume = opt.dim == 3;
    let iz_add = if opt.from_one != 0 { 1 } else { 0 };
    let mut sample = Vec::<f32>::new();
    let mut bins = [0_f32; 1000];
    let (mut interval, mut num_sample, mut x, mut y, mut sample_index) = (0, 0, 0, 0, 0);
    let (mut first_val, mut last_val) = (0_f32, 0_f32);
    for k in 0..opt.nofsecs {
        let iz = opt.secs[(k) as usize];
        let Some(mut s) = crate::imod::libiimod::mrcslice::slice_read_subm(
            hin,
            iz,
            b'z',
            opt.ix,
            opt.iy,
            opt.cx as i32,
            opt.cy as i32,
            None,
        ) else {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "ERROR: CLIP - reading %s %d",
                    &[
                        CArg::Str(if opt.from_one != 0 { "view" } else { "slice" }),
                        CArg::Int((iz + iz_add) as i64),
                    ],
                )
                .as_bytes(),
            );
            return Err(-1);
        };
        if k == 0 {
            let full_size = (s.xsize as usize)
                * (s.ysize as usize)
                * if volume { opt.nofsecs as usize } else { 1 };
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
            sample[sample_index as usize] = slice_get_pixel_magnitude(s.as_ref(), x, y);
            last_val = last_val.max(sample[sample_index as usize]);
            first_val = first_val.min(sample[sample_index as usize]);
            sample_index += 1;
            x += interval;
            while x >= s.xsize {
                x -= s.xsize;
                y += 1;
                if y >= s.ysize {
                    y = 0;
                    wrap = true;
                }
            }
        }
        if !volume || k == opt.nofsecs - 1 {
            if num_sample > 4000 {
                let ind = (0.0005 * num_sample as f32) as i32;
                first_val = crate::imod::libcfshr::percentile::percentile_float(
                    ind + 1,
                    &mut sample,
                    num_sample,
                );
                last_val = crate::imod::libcfshr::percentile::percentile_float(
                    num_sample - ind,
                    &mut sample,
                    num_sample,
                );
            }
            let result = (last_val > first_val)
                .then(|| {
                    crate::imod::libcfshr::histogram::find_histogram_dip(
                        &sample[..num_sample as usize],
                        0,
                        &mut bins,
                        first_val,
                        last_val,
                        0,
                    )
                })
                .flatten();
            let error = i32::from(result.is_none());
            let (dip, below_peak, above_peak) = result
                .map(|result| (result.dip, result.peak_below, result.peak_above))
                .unwrap_or_default();
            if volume {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "All %ss: ",
                        &[CArg::Str(if opt.from_one != 0 { "view" } else { "slice" })],
                    )
                    .as_bytes(),
                );
            } else {
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "%s %d: ",
                        &[
                            CArg::Str(if opt.from_one != 0 { "View" } else { "Slice" }),
                            CArg::Int((iz + iz_add) as i64),
                        ],
                    )
                    .as_bytes(),
                );
            }
            if error != 0 {
                let _ = ImodFile::Stdout.write_all(b"no histogram dip could be found\n");
            } else {
                let mut num_below = 0;
                for index in 0..num_sample {
                    if sample[index as usize] < dip {
                        num_below += 1;
                    }
                }
                let _ = ImodFile::Stdout.write_all(
                    c_format(
                        "peaks at %.5g and %.5g  dip at %.5g  fraction below dip = %.4f\n",
                        &[
                            CArg::Dbl((below_peak as f64) as f64),
                            CArg::Dbl((above_peak as f64) as f64),
                            CArg::Dbl((dip as f64) as f64),
                            CArg::Dbl((num_below as f64 / num_sample as f64) as f64),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
    }
    Ok(())
}
/// Matches C++ `correctDefects`.
pub fn correct_defects(
    slice: &mut Islice,
    nx_full: i32,
    ny_full: i32,
    opt: &mut ClipOptions,
    first_time: &mut bool,
) -> Result<(), i32> {
    if crate::imod::libcfshr::islice::slice_mode_if_real(slice.mode) < 0 {
        crate::imod::clip::clip::show_error(
            "clip with defect correction - The output slice mode must be byte, integer or floating point",
        );
        return Err(-1);
    }
    let mut binning = 1;
    let first = *first_time;
    if crate::imod::clip::correct_defects::cor_def_setup_to_correct(
        nx_full,
        ny_full,
        &mut opt.defects,
        &mut opt.cam_size_x,
        &mut opt.cam_size_y,
        opt.scale_defects,
        opt.binning,
        &mut binning,
        if first { Some("-B") } else { None },
    ) != 0
    {
        crate::imod::clip::clip::show_error(
            "clip with defect correction - Image size is more than twice the size stored in the defect list",
        );
        return Err(-1);
    }
    *first_time = false;
    let left = (opt.cam_size_x / binning - nx_full) / 2 + opt.cx as i32 - opt.ix / 2;
    let right = left + opt.ix;
    let top = (opt.cam_size_y / binning - ny_full) / 2 + opt.cy as i32 - opt.iy / 2;
    let bottom = top + opt.iy;
    if left < 0 || top < 0 || right > opt.cam_size_x / binning || bottom > opt.cam_size_y / binning
    {
        crate::imod::clip::clip::show_error(
            "clip with defect correction - The size and centering options select an area outside the camera field",
        );
        return Err(-1);
    }
    crate::imod::clip::correct_defects::cor_def_correct_defects(
        &opt.defects,
        slice.data.bytes_mut(),
        slice.mode,
        binning,
        top,
        left,
        bottom,
        right,
    );
    Ok(())
}
/// Matches C++ `write_vol`.
pub fn write_vol(vol: &mut [Islice], hout: &mut MrcHeader) -> Result<(), i32> {
    for k in 0..hout.nz {
        let slice = vol[k as usize].as_mut();
        if crate::imod::libiimod::mrcfiles::mrc_write_slice(
            slice.data.bytes(),
            &mut hout.fp.clone().unwrap(),
            hout,
            k,
            b'z',
        ) != 0
        {
            return Err(-1);
        }
        slice_mmm(slice);
        if k == 0 {
            hout.amin = slice.min;
            hout.amax = slice.max;
            hout.amean = slice.mean;
        } else {
            if slice.min < hout.amin {
                hout.amin = slice.min;
            }
            if slice.max > hout.amax {
                hout.amax = slice.max;
            }
            hout.amean += slice.mean;
        }
    }
    hout.amean /= hout.nz as f32;
    match crate::imod::libiimod::mrcfiles::mrc_head_write(&mut hout.fp.clone().unwrap(), hout) {
        0 => Ok(()),
        status => Err(status),
    }
}
/// Matches C++ `free_vol`.
pub fn free_vol(vol: &mut Vec<Islice>, z: i32) -> i32 {
    vol.truncate(z.max(0) as usize);
    // `processing.cpp:4309` frees the pointer array itself; the `Vec` owns
    // it here, so clearing it is the same release.
    vol.clear();
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libcfshr::islice::{slice_create, slice_put_val};
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write, mrc_read_slice,
    };

    #[test]
    fn combine_area_sums_uses_source_quadrant_grouping() {
        let mut sums = [
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ];
        assert_eq!(combine_area_sums(&mut sums), 34.);
        assert_eq!(&sums[..4], &[14., 22., 46., 54.]);
    }

    #[test]
    fn rotate_flip_gain_reference_rotates_owned_floats() {
        let mut reference = vec![1_f32, 2., 3., 4., 5., 6.];
        let (mut nx, mut ny) = (3, 2);

        assert_eq!(
            rotate_flip_gain_reference(&mut reference, &mut nx, &mut ny, 1),
            0
        );
        assert_eq!((nx, ny), (2, 3));
        assert_eq!(reference, [4., 1., 5., 2., 6., 3.]);
    }

    #[test]
    fn write_byte_pixel_matches_source_rounding_clamping_and_signed_offset() {
        use std::io::Seek as _;

        let mut file = ImodFile::tmpfile().unwrap();
        let mut header = MrcHeader::default();
        header.fp = Some(file.clone());
        write_byte_pixel(12.5, &mut header);
        write_byte_pixel(300., &mut header);
        header.bytes_signed = 1;
        write_byte_pixel(128., &mut header);

        file.rewind().unwrap();
        let mut bytes = [0_u8; 3];
        assert_eq!(
            crate::imod::libcfshr::b3dutil::b3d_fread(&mut bytes, 1, 3, &mut file),
            3
        );
        assert_eq!(bytes, [13, 255, 0]);
    }

    #[test]
    fn volume_statistics_follow_slice_magnitudes() {
        let mut first = slice_create(2, 1, MRC_MODE_FLOAT).unwrap();
        let mut second = slice_create(2, 1, MRC_MODE_FLOAT).unwrap();
        slice_put_val(first.as_mut(), 0, 0, [1., 0., 0., 0.]);
        slice_put_val(first.as_mut(), 1, 0, [4., 0., 0., 0.]);
        slice_put_val(second.as_mut(), 0, 0, [2., 0., 0., 0.]);
        slice_put_val(second.as_mut(), 1, 0, [3., 0., 0., 0.]);
        let mut volume = Istack {
            slices: vec![first, second],
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
    }

    #[test]
    fn write_vol_writes_real_mrc_slices_and_source_header_statistics() {
        let mut first = slice_create(2, 1, MRC_MODE_FLOAT).unwrap();
        let mut second = slice_create(2, 1, MRC_MODE_FLOAT).unwrap();
        slice_put_val(first.as_mut(), 0, 0, [1., 0., 0., 0.]);
        slice_put_val(first.as_mut(), 1, 0, [5., 0., 0., 0.]);
        slice_put_val(second.as_mut(), 0, 0, [3., 0., 0., 0.]);
        slice_put_val(second.as_mut(), 1, 0, [7., 0., 0., 0.]);
        let mut vol = vec![first, second];
        let mut fp = ImodFile::tmpfile().unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 2, MRC_MODE_FLOAT), 0);
        header.fp = Some(fp.clone());
        assert_eq!(mrc_head_write(&mut fp, &mut header), 0);
        assert_eq!(write_vol(&mut vol, &mut header), Ok(()));
        assert_eq!((header.amin, header.amax, header.amean), (1., 7., 4.));
        use std::io::Seek as _;
        let _ = fp.rewind();
        let mut read = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut fp, &mut read), 0);
        let mut bytes = [0_u8; core::mem::size_of::<f32>() * 2];
        assert_eq!(mrc_read_slice(&mut bytes, &mut fp, &mut read, 1, b'z',), 0);
        let pixels = bytes
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(pixels, [3., 7.]);
        assert_eq!(free_vol(&mut vol, 2), 0);
    }
}
