//! Translation of `IMOD/libcfshr/multibinstat.c`.
#![allow(dead_code)]

pub const MAX_MBS_SCALES: usize = 20;

/// C `multiBinSetup`.
pub unsafe fn multi_bin_setup(
    binning: *mut [i32; 3],
    box_size: *mut [i32; 3],
    box_spacing: *mut [i32; 3],
    num_scales: i32,
    start_coord: *mut i32,
    end_coord: *mut i32,
    box_start: *mut [i32; 3],
    num_boxes: *mut [i32; 3],
    buffer_start_inds: *mut i32,
    stat_start_inds: *mut i32,
) -> i32 {
    unsafe {
        let mut pixels = [0; 3];
        for dim in 0..3 {
            pixels[dim] = *end_coord.add(dim) + 1 - *start_coord.add(dim);
            if pixels[dim] < 4 {
                return 1;
            }
        }
        if num_scales as usize > MAX_MBS_SCALES {
            return 2;
        }
        *buffer_start_inds = pixels[0] * pixels[1] + pixels[2] / 4 + 1;
        *stat_start_inds = 0;
        for ind in 0..num_scales as usize {
            let mut binned = [0; 3];
            for dim in 0..3 {
                binned[dim] = pixels[dim] / (*binning.add(ind))[dim];
                if binned[dim] < (*box_size.add(ind))[dim] {
                    return 3;
                }
                if (*box_spacing.add(ind))[dim] < 1 {
                    return 4;
                }
                (*num_boxes.add(ind))[dim] =
                    (binned[dim] - (*box_size.add(ind))[dim]) / (*box_spacing.add(ind))[dim] + 1;
                (*box_start.add(ind))[dim] = (binned[dim]
                    - (*box_size.add(ind))[dim]
                    - ((*num_boxes.add(ind))[dim] - 1) * (*box_spacing.add(ind))[dim])
                    / 2;
            }
            *buffer_start_inds.add(ind + 1) = *buffer_start_inds.add(ind) + binned[0] * binned[1];
            *stat_start_inds.add(ind + 1) = *stat_start_inds.add(ind)
                + (*num_boxes.add(ind))[0] * (*num_boxes.add(ind))[1] * (*num_boxes.add(ind))[2];
        }
        0
    }
}

/// Fortran wrapper C `multibinsetup`.
pub unsafe fn multibinsetup(
    binning: *mut [i32; 3],
    box_size: *mut [i32; 3],
    box_spacing: *mut [i32; 3],
    num_scales: *mut i32,
    start_coord: *mut i32,
    end_coord: *mut i32,
    box_start: *mut [i32; 3],
    num_boxes: *mut [i32; 3],
    buffer_start_inds: *mut i32,
    stat_start_inds: *mut i32,
) -> i32 {
    unsafe {
        multi_bin_setup(
            binning,
            box_size,
            box_spacing,
            *num_scales,
            start_coord,
            end_coord,
            box_start,
            num_boxes,
            buffer_start_inds,
            stat_start_inds,
        )
    }
}

/// C `multiBinStats`.
pub unsafe fn multi_bin_stats(
    binning: *mut [i32; 3],
    box_size: *mut [i32; 3],
    box_spacing: *mut [i32; 3],
    num_scales: i32,
    start_coord: *mut i32,
    end_coord: *mut i32,
    box_start: *mut [i32; 3],
    num_boxes: *mut [i32; 3],
    buffer_start_inds: *mut i32,
    stat_start_inds: *mut i32,
    buffer: *mut f32,
    means: *mut f32,
    sds: *mut f32,
    func_data: *mut i32,
    get_slice_func: unsafe extern "C" fn(*mut i32, *mut i32, *mut f32) -> i32,
) -> i32 {
    unsafe {
        let mut pixels = [0; 3];
        for dim in 0..3 {
            pixels[dim] = *end_coord.add(dim) + 1 - *start_coord.add(dim);
        }
        for ind in 0..*stat_start_inds.add(num_scales as usize) as usize {
            *means.add(ind) = 0.;
            *sds.add(ind) = 0.;
        }
        let mut nx_bin = vec![0; num_scales as usize];
        let mut ny_bin = vec![0; num_scales as usize];
        let mut num_box_pix = vec![0; num_scales as usize];
        let mut slices_added = vec![0; num_scales as usize];
        for scl in 0..num_scales as usize {
            nx_bin[scl] = pixels[0] / (*binning.add(scl))[0];
            ny_bin[scl] = pixels[1] / (*binning.add(scl))[1];
            num_box_pix[scl] =
                (*box_size.add(scl))[0] * (*box_size.add(scl))[1] * (*box_size.add(scl))[2];
        }
        let need_slice = core::slice::from_raw_parts_mut(buffer.cast::<u8>(), pixels[2] as usize);
        need_slice.fill(0);
        for scl in 0..num_scales as usize {
            for iz_box in 0..(*num_boxes.add(scl))[2] {
                let z_start = ((*box_start.add(scl))[2] + iz_box * (*box_spacing.add(scl))[2])
                    * (*binning.add(scl))[2];
                for iz in z_start..z_start + (*box_size.add(scl))[2] * (*binning.add(scl))[2] {
                    need_slice[iz as usize] = 1;
                }
            }
        }
        let read_buf = buffer.add((pixels[2] / 4 + 1) as usize);
        let mut last_z_used = -999;
        for mut iz in *start_coord.add(2)..=*end_coord.add(2) {
            if need_slice[(iz - *start_coord.add(2)) as usize] == 0 {
                continue;
            }
            if iz != last_z_used + 1 {
                for scl in 0..num_scales as usize {
                    slices_added[scl] = 0;
                    core::slice::from_raw_parts_mut(
                        buffer.add(*buffer_start_inds.add(scl) as usize),
                        (nx_bin[scl] * ny_bin[scl]) as usize,
                    )
                    .fill(0.);
                }
            }
            last_z_used = iz;
            let err = get_slice_func(&mut iz, func_data, read_buf);
            if err != 0 {
                return err;
            }
            for scl in 0..num_scales as usize {
                let fac_x = (*binning.add(scl))[0];
                let fac_y = (*binning.add(scl))[1];
                let factor = 1. / (*binning.add(scl))[2] as f32;
                for iy in 0..ny_bin[scl] {
                    for ix in 0..nx_bin[scl] {
                        let mut sum = 0.;
                        for by in 0..fac_y {
                            for bx in 0..fac_x {
                                sum += *read_buf.add(
                                    (ix * fac_x + bx + pixels[0] * (iy * fac_y + by)) as usize,
                                );
                            }
                        }
                        *buffer
                            .add((*buffer_start_inds.add(scl) + ix + nx_bin[scl] * iy) as usize) +=
                            sum * factor / (fac_x * fac_y) as f32;
                    }
                }
                slices_added[scl] += 1;
                if slices_added[scl] >= (*binning.add(scl))[2] {
                    let z_start = (*box_start.add(scl))[2];
                    let z_spacing = (*box_spacing.add(scl))[2];
                    let z_cur = (iz - *start_coord.add(2)) / (*binning.add(scl))[2];
                    let mut iz_box =
                        ((z_cur - z_start) / z_spacing).min((*num_boxes.add(scl))[2] - 1);
                    while iz_box >= 0
                        && z_cur >= z_start
                        && z_start + z_spacing * iz_box + (*box_size.add(scl))[2] - 1 >= z_cur
                    {
                        for iy_box in 0..(*num_boxes.add(scl))[1] {
                            for ix_box in 0..(*num_boxes.add(scl))[0] {
                                let iy_start =
                                    (*box_start.add(scl))[1] + (*box_spacing.add(scl))[1] * iy_box;
                                let ix_start =
                                    (*box_start.add(scl))[0] + (*box_spacing.add(scl))[0] * ix_box;
                                let box_ind = *stat_start_inds.add(scl)
                                    + (iz_box * (*num_boxes.add(scl))[1] + iy_box)
                                        * (*num_boxes.add(scl))[0]
                                    + ix_box;
                                for iy in iy_start..iy_start + (*box_size.add(scl))[1] {
                                    for ix in ix_start..ix_start + (*box_size.add(scl))[0] {
                                        let value = *buffer.add(
                                            (*buffer_start_inds.add(scl) + nx_bin[scl] * iy + ix)
                                                as usize,
                                        );
                                        *means.add(box_ind as usize) += value;
                                        *sds.add(box_ind as usize) += value * value;
                                    }
                                }
                            }
                        }
                        iz_box -= 1;
                    }
                    slices_added[scl] = 0;
                    core::slice::from_raw_parts_mut(
                        buffer.add(*buffer_start_inds.add(scl) as usize),
                        (nx_bin[scl] * ny_bin[scl]) as usize,
                    )
                    .fill(0.);
                }
            }
        }
        for scl in 0..num_scales as usize {
            for iz in 0..(*num_boxes.add(scl))[2] {
                for iy in 0..(*num_boxes.add(scl))[1] {
                    for ix in 0..(*num_boxes.add(scl))[0] {
                        let index = *stat_start_inds.add(scl)
                            + (iz * (*num_boxes.add(scl))[1] + iy) * (*num_boxes.add(scl))[0]
                            + ix;
                        let avg = *means.add(index as usize) / num_box_pix[scl] as f32;
                        let variance = (*sds.add(index as usize)
                            - num_box_pix[scl] as f32 * avg * avg)
                            / (num_box_pix[scl] - 1) as f32;
                        *means.add(index as usize) = avg;
                        *sds.add(index as usize) = if variance > 0. { variance.sqrt() } else { 0. };
                    }
                }
            }
        }
        0
    }
}

/// Fortran wrapper C `multibinstats`.
pub unsafe fn multibinstats(
    binning: *mut [i32; 3],
    box_size: *mut [i32; 3],
    box_spacing: *mut [i32; 3],
    num_scales: *mut i32,
    start_coord: *mut i32,
    end_coord: *mut i32,
    box_start: *mut [i32; 3],
    num_boxes: *mut [i32; 3],
    buffer_start_inds: *mut i32,
    stat_start_inds: *mut i32,
    buffer: *mut f32,
    means: *mut f32,
    sds: *mut f32,
    func_data: *mut i32,
    get_slice_func: unsafe extern "C" fn(*mut i32, *mut i32, *mut f32) -> i32,
) -> i32 {
    unsafe {
        multi_bin_stats(
            binning,
            box_size,
            box_spacing,
            *num_scales,
            start_coord,
            end_coord,
            box_start,
            num_boxes,
            buffer_start_inds,
            stat_start_inds,
            buffer,
            means,
            sds,
            func_data,
            get_slice_func,
        )
    }
}

/// C `makeStandardDevMap`.
pub unsafe fn make_standard_dev_map(
    array: *mut f32,
    nx_dim: i32,
    ix_start: i32,
    mut ix_end: i32,
    iy_start: i32,
    iy_end: i32,
    mut binning: i32,
    box_size: i32,
    sd_arr: *mut f32,
    sum_arr: *mut f32,
    sqr_arr: *mut f32,
    x_offset: *mut i32,
    y_offset: *mut i32,
) {
    unsafe {
        let mut need_bin = binning > 0;
        let mut variation = false;
        if ix_end < 0 {
            variation = true;
            ix_end = -ix_end;
        }
        if binning == -1 {
            binning = 1;
            need_bin = true;
        }
        let nx_bin = (ix_end + 1 - ix_start) / binning.abs();
        let ny_bin = (iy_end + 1 - iy_start) / binning.abs();
        if !need_bin {
            binning = -binning;
            let mut width = 0;
            let error = crate::imod::libcfshr::zoomdown::select_zoom_filter(
                4,
                1. / binning as f64,
                &mut width,
            );
            if error == 0 {
                let lines = crate::imod::libcfshr::b3dutil::make_line_pointers(
                    array.cast(),
                    nx_dim,
                    iy_end + 1,
                    4,
                );
                if !lines.is_null() {
                    let err = crate::imod::libcfshr::zoomdown::zoom_with_filter(
                        lines,
                        nx_dim,
                        iy_end + 1,
                        ix_start as f32,
                        iy_start as f32,
                        nx_bin,
                        ny_bin,
                        nx_bin,
                        0,
                        2,
                        sd_arr.cast(),
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                    );
                    libc::free(lines.cast());
                    if err != 0 {
                        println!(
                            "WARNING: makeStandardDevMap - zoomWithFilter failed with error {}; using binning instead of reduction",
                            err
                        );
                        need_bin = true;
                    }
                } else {
                    println!(
                        "WARNING: makeStandardDevMap - error allocating line pointers; using binning instead of reduction"
                    );
                    need_bin = true;
                }
            } else {
                println!(
                    "WARNING: makeStandardDevMap - selectZoomFilter failed with error {}; using binning instead of reduction",
                    error
                );
                need_bin = true;
            }
        }
        if need_bin {
            for iy in 0..ny_bin {
                for ix in 0..nx_bin {
                    let mut sum = 0.;
                    for by in 0..binning {
                        for bx in 0..binning {
                            sum += *array.add(
                                (ix_start
                                    + ix * binning
                                    + bx
                                    + nx_dim * (iy_start + iy * binning + by))
                                    as usize,
                            );
                        }
                    }
                    *sd_arr.add((ix + iy * nx_bin) as usize) = sum / (binning * binning) as f32;
                }
            }
        }
        core::slice::from_raw_parts_mut(sum_arr, (nx_bin * ny_bin) as usize).fill(0.);
        core::slice::from_raw_parts_mut(sqr_arr, (nx_bin * ny_bin) as usize).fill(0.);
        for iy in 0..ny_bin {
            for by in (0.max(iy + 1 - box_size))..((ny_bin + 1 - box_size).min(iy + 1)) {
                for ix in 0..nx_bin {
                    let val = *sd_arr.add((ix + iy * nx_bin) as usize);
                    for bx in (0.max(ix + 1 - box_size))..((nx_bin + 1 - box_size).min(ix + 1)) {
                        *sum_arr.add((bx + by * nx_bin) as usize) += val;
                        *sqr_arr.add((bx + by * nx_bin) as usize) += val * val;
                    }
                }
            }
        }
        let box_left = box_size / 2;
        let mut min_sum = 1.0e30_f32;
        let mut max_sum = -1.0e30_f32;
        if variation {
            for iy in 0..ny_bin {
                let by = (iy - box_left).clamp(0, ny_bin - box_size - 1);
                for ix in 0..nx_bin {
                    let bx = (ix - box_left).clamp(0, nx_bin - box_size - 1);
                    let mean =
                        *sum_arr.add((bx + by * nx_bin) as usize) / (box_size * box_size) as f32;
                    min_sum = min_sum.min(mean);
                    max_sum = max_sum.max(mean);
                }
            }
            if min_sum < 0. || min_sum < 0.01 * (max_sum - min_sum) {
                println!(
                    "WARNING: makeStandardDevMap - minimum box mean ({:.4}) is less than 0.01 times range of means ({:.4}); returning SDs instead of SD/means",
                    min_sum,
                    max_sum - min_sum
                );
                variation = false;
            }
        }
        for iy in 0..ny_bin {
            let by = (iy - box_left).clamp(0, ny_bin - box_size - 1);
            for ix in 0..nx_bin {
                let bx = (ix - box_left).clamp(0, nx_bin - box_size - 1);
                let sum = *sum_arr.add((bx + by * nx_bin) as usize);
                let avg = sum / (box_size * box_size) as f32;
                let den = (*sqr_arr.add((bx + by * nx_bin) as usize)
                    - (box_size * box_size) as f32 * avg * avg)
                    / ((box_size * box_size - 1) as f32);
                let mut sd = if den > 0. { den.sqrt() } else { 0. };
                if variation {
                    sd /= avg;
                }
                *sd_arr.add((ix + iy * nx_bin) as usize) = sd;
            }
        }
        *x_offset = -ix_start / binning;
        *y_offset = -iy_start / binning;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn slice_by_z(iz: *mut i32, _data: *mut i32, output: *mut f32) -> i32 {
        unsafe {
            for index in 0..16 {
                *output.add(index) = *iz as f32;
            }
        }
        0
    }
    #[test]
    fn setup_matches_source_layout() {
        let mut bin = [[1, 1, 1]];
        let mut size = [[2, 2, 2]];
        let mut spacing = [[1, 1, 1]];
        let mut start = [0, 0, 0];
        let mut end = [3, 3, 3];
        let mut bs = [[0; 3]];
        let mut nb = [[0; 3]];
        let mut buf = [0; 2];
        let mut stat = [0; 2];
        assert_eq!(
            unsafe {
                multi_bin_setup(
                    bin.as_mut_ptr(),
                    size.as_mut_ptr(),
                    spacing.as_mut_ptr(),
                    1,
                    start.as_mut_ptr(),
                    end.as_mut_ptr(),
                    bs.as_mut_ptr(),
                    nb.as_mut_ptr(),
                    buf.as_mut_ptr(),
                    stat.as_mut_ptr(),
                )
            },
            0
        );
        assert_eq!(nb[0], [3, 3, 3]);
        assert_eq!(stat[1], 27);
    }

    #[test]
    fn standard_deviation_map_matches_uniform_and_linear_boxes() {
        let mut array = vec![4.; 16];
        let mut sd = vec![0.; 16];
        let mut sums = vec![0.; 16];
        let mut squares = vec![0.; 16];
        let mut xo = 99;
        let mut yo = 99;
        unsafe {
            make_standard_dev_map(
                array.as_mut_ptr(),
                4,
                0,
                3,
                0,
                3,
                1,
                2,
                sd.as_mut_ptr(),
                sums.as_mut_ptr(),
                squares.as_mut_ptr(),
                &mut xo,
                &mut yo,
            );
        }
        assert_eq!(sd, vec![0.; 16]);
        assert_eq!((xo, yo), (0, 0));
        for (index, value) in array.iter_mut().enumerate() {
            *value = index as f32;
        }
        unsafe {
            make_standard_dev_map(
                array.as_mut_ptr(),
                4,
                0,
                3,
                0,
                3,
                1,
                2,
                sd.as_mut_ptr(),
                sums.as_mut_ptr(),
                squares.as_mut_ptr(),
                &mut xo,
                &mut yo,
            );
        }
        assert!((sd[0] - 2.380476).abs() < 1.0e-5);
        assert!((sd[15] - 2.380476).abs() < 1.0e-5);
    }

    #[test]
    fn multiscale_stats_accumulates_callback_slices_in_source_box_order() {
        let mut binning = [[1, 1, 1]];
        let mut size = [[2, 2, 2]];
        let mut spacing = [[1, 1, 1]];
        let mut start = [0, 0, 0];
        let mut end = [3, 3, 3];
        let mut starts = [[0; 3]];
        let mut counts = [[0; 3]];
        let mut buffer_starts = [0; 2];
        let mut stat_starts = [0; 2];
        unsafe {
            assert_eq!(
                multi_bin_setup(
                    binning.as_mut_ptr(),
                    size.as_mut_ptr(),
                    spacing.as_mut_ptr(),
                    1,
                    start.as_mut_ptr(),
                    end.as_mut_ptr(),
                    starts.as_mut_ptr(),
                    counts.as_mut_ptr(),
                    buffer_starts.as_mut_ptr(),
                    stat_starts.as_mut_ptr()
                ),
                0
            );
        }
        let mut buffer = vec![0.; buffer_starts[1] as usize];
        let mut means = vec![0.; stat_starts[1] as usize];
        let mut sds = vec![0.; stat_starts[1] as usize];
        unsafe {
            assert_eq!(
                multi_bin_stats(
                    binning.as_mut_ptr(),
                    size.as_mut_ptr(),
                    spacing.as_mut_ptr(),
                    1,
                    start.as_mut_ptr(),
                    end.as_mut_ptr(),
                    starts.as_mut_ptr(),
                    counts.as_mut_ptr(),
                    buffer_starts.as_mut_ptr(),
                    stat_starts.as_mut_ptr(),
                    buffer.as_mut_ptr(),
                    means.as_mut_ptr(),
                    sds.as_mut_ptr(),
                    core::ptr::null_mut(),
                    slice_by_z
                ),
                0
            );
        }
        assert_eq!(means[0], 0.5);
        assert!((sds[0] - (2_f32 / 7.).sqrt()).abs() < 1.0e-6);
        assert_eq!(means[18], 2.5);
    }
}
