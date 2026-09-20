//! Translation of `IMOD/libcfshr/multibinstat.c`: measure local mean/SD in an
//! array of boxes at multiple binnings.

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libcfshr::reduce_by_binning::bin_into_slice;
use crate::imod::libcfshr::simplestat::sums_to_avg_sd;
use crate::imod::libcfshr::zoomdown::{
    SLICE_MODE_FLOAT, ZoomLines, ZoomOut, select_zoom_filter, zoom_with_filter,
};
use std::io::Write;

pub const MAX_MBS_SCALES: usize = 20;

/// Complete function inventory for `multibinstat.c`.
pub const MULTI_BIN_STAT_SOURCE_FUNCTIONS: &[&str] = &[
    "multiBinSetup",
    "multibinsetup",
    "multiBinStats",
    "multibinstats",
    "makeStandardDevMap",
];

/// Original `multiBinSetup` (`multibinstat.c:50`).
#[allow(clippy::too_many_arguments)]
pub fn multi_bin_setup(
    binning: &[[i32; 3]],
    box_size: &[[i32; 3]],
    box_spacing: &[[i32; 3]],
    num_scales: i32,
    start_coord: &[i32],
    end_coord: &[i32],
    box_start: &mut [[i32; 3]],
    num_boxes: &mut [[i32; 3]],
    buffer_start_inds: &mut [i32],
    stat_start_inds: &mut [i32],
) -> i32 {
    if num_scales < 0 || num_scales as usize > MAX_MBS_SCALES {
        return 2;
    }
    let scale_count = num_scales as usize;
    if start_coord.len() < 3
        || end_coord.len() < 3
        || binning.len() < scale_count
        || box_size.len() < scale_count
        || box_spacing.len() < scale_count
        || box_start.len() < scale_count
        || num_boxes.len() < scale_count
        || buffer_start_inds.len() < scale_count + 1
        || stat_start_inds.len() < scale_count + 1
    {
        return 1;
    }
    let mut num_ub_pixels = [0i32; 3];
    let mut num_binned_pix = [0i32; 3];

    /* Get the number of unbinned pixels and check ranges */
    for dim in 0..3 {
        let Some(span) = end_coord[dim]
            .checked_sub(start_coord[dim])
            .and_then(|value| value.checked_add(1))
        else {
            return 1;
        };
        num_ub_pixels[dim] = span;
        if num_ub_pixels[dim] < 4 {
            return 1;
        }
    }
    /* Go through the scales, figure out how many binned pixels, and the number and
     * starting coordinates of the boxes in each dimension.
     * Allow space in bottom of buffer for an array of flags for each slice */
    buffer_start_inds[0] = num_ub_pixels[0] * num_ub_pixels[1] + num_ub_pixels[2] / 4 + 1;
    stat_start_inds[0] = 0;
    for ind in 0..scale_count {
        for dim in 0..3 {
            if binning[ind][dim] <= 0 {
                return 3;
            }
            num_binned_pix[dim] = num_ub_pixels[dim] / binning[ind][dim];
            if num_binned_pix[dim] < box_size[ind][dim] {
                return 3;
            }
            if box_spacing[ind][dim] < 1 {
                return 4;
            }
            num_boxes[ind][dim] =
                (num_binned_pix[dim] - box_size[ind][dim]) / box_spacing[ind][dim] + 1;
            box_start[ind][dim] = (num_binned_pix[dim]
                - box_size[ind][dim]
                - (num_boxes[ind][dim] - 1) * box_spacing[ind][dim])
                / 2;
        }

        /* Get the starting indexes for binned buffers and the means and SDs */
        buffer_start_inds[ind + 1] = buffer_start_inds[ind] + num_binned_pix[0] * num_binned_pix[1];
        stat_start_inds[ind + 1] =
            stat_start_inds[ind] + num_boxes[ind][0] * num_boxes[ind][1] * num_boxes[ind][2];
    }
    0
}

/// Original `multibinsetup` (`multibinstat.c:96`).
#[allow(clippy::too_many_arguments)]
pub fn multibinsetup(
    binning: &[[i32; 3]],
    box_size: &[[i32; 3]],
    box_spacing: &[[i32; 3]],
    num_scales: &i32,
    start_coord: &[i32],
    end_coord: &[i32],
    box_start: &mut [[i32; 3]],
    num_boxes: &mut [[i32; 3]],
    buffer_start_inds: &mut [i32],
    stat_start_inds: &mut [i32],
) -> i32 {
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

/// Original `multiBinStats` (`multibinstat.c:119`).
///
/// The source aliases the head of `buffer` as `unsigned char *needSlice`
/// (`multibinstat.c:145`), a byte flag per input slice living in the
/// `numUBpixels[b3dZ] / 4 + 1` floats that `multiBinSetup` reserved below
/// `readBuf`.  Nothing else ever reads that region, so the flags are a `Vec`
/// here and the reserved floats are simply left alone.
#[allow(clippy::too_many_arguments)]
pub fn multi_bin_stats(
    binning: &[[i32; 3]],
    box_size: &[[i32; 3]],
    box_spacing: &[[i32; 3]],
    num_scales: i32,
    start_coord: &[i32],
    end_coord: &[i32],
    box_start: &[[i32; 3]],
    num_boxes: &[[i32; 3]],
    buffer_start_inds: &[i32],
    stat_start_inds: &[i32],
    buffer: &mut [f32],
    means: &mut [f32],
    sds: &mut [f32],
    func_data: &mut [i32],
    get_slice_func: fn(&mut i32, &mut [i32], &mut [f32]) -> i32,
) -> i32 {
    if num_scales < 0 || num_scales as usize > MAX_MBS_SCALES {
        return 1;
    }
    let scale_count = num_scales as usize;
    if start_coord.len() < 3
        || end_coord.len() < 3
        || binning.len() < scale_count
        || box_size.len() < scale_count
        || box_spacing.len() < scale_count
        || box_start.len() < scale_count
        || num_boxes.len() < scale_count
        || buffer_start_inds.len() < scale_count + 1
        || stat_start_inds.len() < scale_count + 1
        || buffer_start_inds[0] < 0
        || buffer_start_inds[scale_count] < buffer_start_inds[0]
        || stat_start_inds[scale_count] < 0
        || buffer.len() < buffer_start_inds[scale_count] as usize
        || means.len() < stat_start_inds[scale_count] as usize
        || sds.len() < stat_start_inds[scale_count] as usize
    {
        return 1;
    }
    let mut num_ub_pixels = [0i32; 3];
    let mut num_box_pix = [0i32; MAX_MBS_SCALES];
    let mut slices_added = [0i32; MAX_MBS_SCALES];
    let mut nx_bin = [0i32; MAX_MBS_SCALES];
    let mut ny_bin = [0i32; MAX_MBS_SCALES];

    /* Initialize: set up the binned sizes etc. */
    for dim in 0..3 {
        let Some(span) = end_coord[dim]
            .checked_sub(start_coord[dim])
            .and_then(|value| value.checked_add(1))
        else {
            return 1;
        };
        if span < 4 {
            return 1;
        }
        num_ub_pixels[dim] = span;
    }
    let flag_floats = num_ub_pixels[2] / 4 + 1;
    if buffer_start_inds[0] < flag_floats {
        return 1;
    }
    for scale in 0..scale_count {
        if buffer_start_inds[scale] < buffer_start_inds[0]
            || buffer_start_inds[scale + 1] < buffer_start_inds[scale]
            || stat_start_inds[scale] < 0
            || stat_start_inds[scale + 1] < stat_start_inds[scale]
        {
            return 1;
        }
    }
    for ind in 0..stat_start_inds[num_scales as usize] as usize {
        sds[ind] = 0.;
        means[ind] = 0.;
    }
    for scl in 0..scale_count {
        if binning[scl].iter().any(|&value| value <= 0)
            || box_size[scl].iter().any(|&value| value <= 0)
            || box_spacing[scl].iter().any(|&value| value <= 0)
            || num_boxes[scl].iter().any(|&value| value <= 0)
        {
            return 1;
        }
        nx_bin[scl] = num_ub_pixels[0] / binning[scl][0];
        ny_bin[scl] = num_ub_pixels[1] / binning[scl][1];
        num_box_pix[scl] = box_size[scl][0] * box_size[scl][1] * box_size[scl][2];
    }

    /* Determine which slices are needed */
    let mut need_slice = vec![0u8; num_ub_pixels[2] as usize];
    for scl in 0..num_scales as usize {
        for iz_box in 0..num_boxes[scl][2] {
            let z_start = (box_start[scl][2] + iz_box * box_spacing[scl][2]) * binning[scl][2];
            for iz in z_start..z_start + box_size[scl][2] * binning[scl][2] {
                need_slice[iz as usize] = 1;
            }
        }
    }
    let mut last_z_used = -999;

    // `readBuf = buffer + numUBpixels[b3dZ] / 4 + 1` (`multibinstat.c:155`)
    // and the binned buffers start at `bufferStartInds[0]`, which
    // `multiBinSetup` put one whole unbinned slice above that; the two halves
    // of the source's single allocation are split here.
    let (read_part, bin_part) = buffer.split_at_mut(buffer_start_inds[0] as usize);
    let read_buf = &mut read_part[(num_ub_pixels[2] / 4 + 1) as usize..];

    /* Loop on slices */
    for iz in start_coord[2]..=end_coord[2] {
        /* Skip if slice is unneeded */
        if need_slice[(iz - start_coord[2]) as usize] == 0 {
            continue;
        }

        /* If starting a batch of slices, clear out the binned buffers */
        if iz != last_z_used + 1 {
            for scl in 0..num_scales as usize {
                slices_added[scl] = 0;
                let base = (buffer_start_inds[scl] - buffer_start_inds[0]) as usize;
                for ixy in 0..(nx_bin[scl] * ny_bin[scl]) as usize {
                    bin_part[base + ixy] = 0.;
                }
            }
        }
        last_z_used = iz;

        /* Get the next slice */
        let mut iz_arg = iz;
        let err = get_slice_func(&mut iz_arg, func_data, read_buf);
        if err != 0 {
            return err;
        }

        /* Add it into each binned buffer */
        for scl in 0..num_scales as usize {
            let base = (buffer_start_inds[scl] - buffer_start_inds[0]) as usize;
            bin_into_slice(
                read_buf,
                num_ub_pixels[0],
                &mut bin_part[base..],
                nx_bin[scl],
                ny_bin[scl],
                binning[scl][0],
                binning[scl][1],
                // `1. / binning[scl][b3dZ]` is a double quotient converted to
                // the `float zWeight` parameter.
                (1.0 / binning[scl][2] as f64) as f32,
            );
            slices_added[scl] += 1;
            if slices_added[scl] >= binning[scl][2] {
                /* When binned slice is done, add it into the needed boxes */
                let z_start = box_start[scl][2];
                let z_spacing = box_spacing[scl][2];
                let z_cur = (iz - start_coord[2]) / binning[scl][2];
                let mut iz_box = (z_cur - z_start) / z_spacing;
                iz_box = if iz_box < num_boxes[scl][2] - 1 {
                    iz_box
                } else {
                    num_boxes[scl][2] - 1
                };

                /* Loop backwards in Z until the slice is past the box */
                while iz_box >= 0
                    && z_cur >= z_start
                    && z_start + z_spacing * iz_box + box_size[scl][2] - 1 >= z_cur
                {
                    /* In each box, add in all the pixels in the box */
                    /* TODO: find out how effective this is! */
                    // The `numOMPthreads(8)` `parallel for` over `iyBox` is
                    // not reproduced; each box accumulates independently.
                    for iy_box in 0..num_boxes[scl][1] {
                        let iy_start = box_start[scl][1] + box_spacing[scl][1] * iy_box;
                        for ix_box in 0..num_boxes[scl][0] {
                            let ix_start = box_start[scl][0] + box_spacing[scl][0] * ix_box;
                            let box_ind = stat_start_inds[scl]
                                + (iz_box * num_boxes[scl][1] + iy_box) * num_boxes[scl][0]
                                + ix_box;
                            for iy in iy_start..iy_start + box_size[scl][1] {
                                let buf_base = buffer_start_inds[scl] + nx_bin[scl] * iy;
                                for ix in ix_start..ix_start + box_size[scl][0] {
                                    let pixel =
                                        bin_part[(buf_base + ix - buffer_start_inds[0]) as usize];
                                    means[box_ind as usize] += pixel;
                                    sds[box_ind as usize] += pixel * pixel;
                                }
                            }
                        }
                    }

                    iz_box -= 1;
                }

                /* Reset for adding more slices, and increment Z */
                slices_added[scl] = 0;
                for ixy in 0..(nx_bin[scl] * ny_bin[scl]) as usize {
                    bin_part[base + ixy] = 0.;
                }
            }
        }
    }

    /* Compute the means and SDs */
    for scl in 0..num_scales as usize {
        for iz_box in 0..num_boxes[scl][2] {
            for iy_box in 0..num_boxes[scl][1] {
                for ix_box in 0..num_boxes[scl][0] {
                    let box_ind = stat_start_inds[scl]
                        + (iz_box * num_boxes[scl][1] + iy_box) * num_boxes[scl][0]
                        + ix_box;
                    let (mut one_mean, mut one_sd) = (0., 0.);
                    sums_to_avg_sd(
                        means[box_ind as usize],
                        sds[box_ind as usize],
                        num_box_pix[scl],
                        &mut one_mean,
                        &mut one_sd,
                    );
                    means[box_ind as usize] = one_mean;
                    sds[box_ind as usize] = one_sd;
                }
            }
        }
    }
    0
}

/// Original `multibinstats` (`multibinstat.c:250`).
#[allow(clippy::too_many_arguments)]
pub fn multibinstats(
    binning: &[[i32; 3]],
    box_size: &[[i32; 3]],
    box_spacing: &[[i32; 3]],
    num_scales: &i32,
    start_coord: &[i32],
    end_coord: &[i32],
    box_start: &[[i32; 3]],
    num_boxes: &[[i32; 3]],
    buffer_start_inds: &[i32],
    stat_start_inds: &[i32],
    buffer: &mut [f32],
    means: &mut [f32],
    sds: &mut [f32],
    func_data: &mut [i32],
    get_slice_func: fn(&mut i32, &mut [i32], &mut [f32]) -> i32,
) -> i32 {
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

/// Original `makeStandardDevMap` (`multibinstat.c:274`).
#[allow(clippy::too_many_arguments)]
pub fn make_standard_dev_map(
    array: &[f32],
    nx_dim: i32,
    ix_start: i32,
    mut ix_end: i32,
    iy_start: i32,
    iy_end: i32,
    mut binning: i32,
    box_size: i32,
    sd_arr: &mut [f32],
    sum_arr: &mut [f32],
    sqr_arr: &mut [f32],
    x_offset: &mut i32,
    y_offset: &mut i32,
) {
    // The source leaves `nxBin`/`nyBin` uninitialised; every path through the
    // two branches below assigns them before use.
    let mut nx_bin: i32 = 0;
    let mut ny_bin: i32 = 0;
    let mut need_bin = i32::from(binning > 0);
    let mut do_variation = 0;
    let mut min_sum: f32 = 1.0e30;
    let mut max_sum: f32 = -1.0e30;
    let fnum_pix: f32 = (box_size * box_size) as f32;

    if ix_end < 0 {
        do_variation = 1;
        ix_end = -ix_end;
    }
    if binning == -1 {
        binning = 1;
        need_bin = 1;
    }

    if need_bin == 0 {
        binning = -binning;
        let mut iy = 0;
        let ix = select_zoom_filter(4, 1. / binning as f64, &mut iy);
        if ix != 0 {
            let _ = ImodFile::Stdout.write_all(
                format!(
                    "WARNING: makeStandardDevMap - selectZoomFilter failed with error {ix}; \
                     using binning instead of reduction\n",
                )
                .as_bytes(),
            );
            need_bin = 1;
        } else {
            // `makeLinePointers` cannot fail here, so the source's
            // "error allocating line pointers" warning is unreachable.
            let line_ptrs: Vec<&[f32]> = (0..(iy_end + 1) as usize)
                .map(|i| &array[i * nx_dim as usize..])
                .collect();
            nx_bin = (ix_end + 1 - ix_start) / binning;
            ny_bin = (iy_end + 1 - iy_start) / binning;
            let ix = zoom_with_filter(
                ZoomLines::Float(&line_ptrs),
                nx_dim,
                iy_end + 1,
                ix_start as f32,
                iy_start as f32,
                nx_bin,
                ny_bin,
                nx_bin,
                0,
                SLICE_MODE_FLOAT,
                &mut ZoomOut::Float(&mut sd_arr[..]),
                None,
                None,
            );
            if ix != 0 {
                let _ = ImodFile::Stdout.write_all(
                    format!(
                        "WARNING: makeStandardDevMap - zoomWithFilter failed with error {ix}; \
                         using binning instead of reduction\n",
                    )
                    .as_bytes(),
                );
                need_bin = 1;
            }
        }
    }
    if need_bin > 0 {
        // The source calls
        // `extractWithBinning(array, MRC_MODE_FLOAT, nxDim, ixStart, ixEnd,
        //  iyStart, iyEnd, binning, sdArr, 0, &nxBin, &nyBin)`.  That routine's
        // Rust translation takes the `void *` arrays as `&[u8]`
        // (`reduce_by_binning.rs:655`), which cannot be produced from an
        // `&[f32]` without `unsafe`, so its `SLICE_MODE_FLOAT` arm is written
        // out here: `reduce_by_binning.c:61-63` centres the extracted area
        // with `ixofs = (nxin % nbin) / 2` / `iyofs = (nyin % nbin) / 2`,
        // `:103` advances by `xStart + yStart * nxDim`, and `:400-412` sums
        // the block in row order and divides by `(b3dFloat)nbinsq`.
        let nxin = ix_end + 1 - ix_start;
        let nyin = iy_end + 1 - iy_start;
        let ixofs = (nxin % binning) / 2;
        let iyofs = (nyin % binning) / 2;
        nx_bin = nxin / binning;
        ny_bin = nyin / binning;
        for iy in 0..ny_bin {
            for ix in 0..nx_bin {
                let mut fsum = 0.0f32;
                for j in 0..binning {
                    for i in 0..binning {
                        fsum += array[(ix_start
                            + ixofs
                            + binning * ix
                            + i
                            + nx_dim * (iy_start + iyofs + binning * iy + j))
                            as usize];
                    }
                }
                sd_arr[(ix + iy * nx_bin) as usize] = fsum / (binning * binning) as f32;
            }
        }
    }
    for ind in 0..(nx_bin * ny_bin) as usize {
        sum_arr[ind] = 0.;
        sqr_arr[ind] = 0.;
    }
    let box_left = box_size / 2;
    for iy in 0..ny_bin {
        let sum_ystart = if 0 > iy + 1 - box_size {
            0
        } else {
            iy + 1 - box_size
        };
        let sum_yend = if ny_bin + 1 - box_size < iy + 1 {
            ny_bin + 1 - box_size
        } else {
            iy + 1
        };
        for by in sum_ystart..sum_yend {
            for ix in 0..nx_bin {
                let sum_xstart = if 0 > ix + 1 - box_size {
                    0
                } else {
                    ix + 1 - box_size
                };
                let sum_xend = if nx_bin + 1 - box_size < ix + 1 {
                    nx_bin + 1 - box_size
                } else {
                    ix + 1
                };
                let val = sd_arr[(ix + iy * nx_bin) as usize];
                let val_sq = val * val;
                for bx in sum_xstart..sum_xend {
                    sum_arr[(bx + by * nx_bin) as usize] += val;
                    sqr_arr[(bx + by * nx_bin) as usize] += val_sq;
                }
            }
        }
    }

    if do_variation != 0 {
        for iy in 0..ny_bin {
            // `B3DCLAMP(by, 0, nyBin - boxSize - 1)` is
            // `MAX(0, MIN(nyBin - boxSize - 1, by))`, in that nesting.
            let mut by = iy - box_left;
            by = if ny_bin - box_size - 1 < by {
                ny_bin - box_size - 1
            } else {
                by
            };
            by = if 0 > by { 0 } else { by };
            for ix in 0..nx_bin {
                let mut bx = ix - box_left;
                bx = if nx_bin - box_size - 1 < bx {
                    nx_bin - box_size - 1
                } else {
                    bx
                };
                bx = if 0 > bx { 0 } else { bx };
                // `ACCUM_MIN`/`ACCUM_MAX` are `a < b ? a : b` / `a > b ? a : b`,
                // which keep the second operand on a NaN comparison where
                // `f32::min`/`max` would keep the first.
                let v = sum_arr[(bx + by * nx_bin) as usize] / fnum_pix;
                min_sum = if min_sum < v { min_sum } else { v };
                max_sum = if max_sum > v { max_sum } else { v };
            }
        }

        // `0.01` is a double literal, so the product and the comparison are
        // evaluated in double.
        if min_sum < 0. || (min_sum as f64) < 0.01 * (max_sum - min_sum) as f64 {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "WARNING: makeStandardDevMap - minimum box mean (%.4g) is less than 0.01 \
                     times range of means (%.4g); returning SDs instead of SD/means\n",
                    &[
                        CArg::Dbl(min_sum as f64),
                        CArg::Dbl((max_sum - min_sum) as f64),
                    ],
                )
                .as_bytes(),
            );
            do_variation = 0;
        }
    }

    for iy in 0..ny_bin {
        let mut by = iy - box_left;
        by = if ny_bin - box_size - 1 < by {
            ny_bin - box_size - 1
        } else {
            by
        };
        by = if 0 > by { 0 } else { by };
        for ix in 0..nx_bin {
            let mut bx = ix - box_left;
            bx = if nx_bin - box_size - 1 < bx {
                nx_bin - box_size - 1
            } else {
                bx
            };
            bx = if 0 > bx { 0 } else { bx };
            let mut val = 0.0f32;
            let mut one_sd = 0.0f32;
            sums_to_avg_sd(
                sum_arr[(bx + by * nx_bin) as usize],
                sqr_arr[(bx + by * nx_bin) as usize],
                box_size * box_size,
                &mut val,
                &mut one_sd,
            );
            sd_arr[(ix + iy * nx_bin) as usize] = one_sd;
            if do_variation != 0 {
                sd_arr[(ix + iy * nx_bin) as usize] /= val;
            }
        }
    }

    *x_offset = -ix_start / binning;
    *y_offset = -iy_start / binning;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice_by_z(iz: &mut i32, _data: &mut [i32], output: &mut [f32]) -> i32 {
        for index in 0..16 {
            output[index] = *iz as f32;
        }
        0
    }
    #[test]
    fn setup_matches_source_layout() {
        let bin = [[1, 1, 1]];
        let size = [[2, 2, 2]];
        let spacing = [[1, 1, 1]];
        let start = [0, 0, 0];
        let end = [3, 3, 3];
        let mut bs = [[0; 3]];
        let mut nb = [[0; 3]];
        let mut buf = [0; 2];
        let mut stat = [0; 2];
        assert_eq!(
            multi_bin_setup(
                &bin, &size, &spacing, 1, &start, &end, &mut bs, &mut nb, &mut buf, &mut stat,
            ),
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
        make_standard_dev_map(
            &array,
            4,
            0,
            3,
            0,
            3,
            1,
            2,
            &mut sd,
            &mut sums,
            &mut squares,
            &mut xo,
            &mut yo,
        );
        assert_eq!(sd, vec![0.; 16]);
        assert_eq!((xo, yo), (0, 0));
        for (index, value) in array.iter_mut().enumerate() {
            *value = index as f32;
        }
        make_standard_dev_map(
            &array,
            4,
            0,
            3,
            0,
            3,
            1,
            2,
            &mut sd,
            &mut sums,
            &mut squares,
            &mut xo,
            &mut yo,
        );
        assert!((sd[0] - 2.380476).abs() < 1.0e-5);
        assert!((sd[15] - 2.380476).abs() < 1.0e-5);
    }

    #[test]
    fn multiscale_stats_accumulates_callback_slices_in_source_box_order() {
        let binning = [[1, 1, 1]];
        let size = [[2, 2, 2]];
        let spacing = [[1, 1, 1]];
        let start = [0, 0, 0];
        let end = [3, 3, 3];
        let mut starts = [[0; 3]];
        let mut counts = [[0; 3]];
        let mut buffer_starts = [0; 2];
        let mut stat_starts = [0; 2];
        assert_eq!(
            multi_bin_setup(
                &binning,
                &size,
                &spacing,
                1,
                &start,
                &end,
                &mut starts,
                &mut counts,
                &mut buffer_starts,
                &mut stat_starts
            ),
            0
        );
        let mut buffer = vec![0.; buffer_starts[1] as usize];
        let mut means = vec![0.; stat_starts[1] as usize];
        let mut sds = vec![0.; stat_starts[1] as usize];
        assert_eq!(
            multi_bin_stats(
                &binning,
                &size,
                &spacing,
                1,
                &start,
                &end,
                &starts,
                &counts,
                &buffer_starts,
                &stat_starts,
                &mut buffer,
                &mut means,
                &mut sds,
                &mut [],
                slice_by_z
            ),
            0
        );
        assert_eq!(means[0], 0.5);
        assert!((sds[0] - (2_f32 / 7.).sqrt()).abs() < 1.0e-6);
        assert_eq!(means[18], 2.5);
    }

    #[test]
    fn setup_rejects_invalid_owned_dimensions_before_division() {
        let mut starts = [[0; 3]];
        let mut counts = [[0; 3]];
        let mut buffer_starts = [0; 2];
        let mut stat_starts = [0; 2];
        assert_eq!(
            multi_bin_setup(
                &[[0, 1, 1]],
                &[[2, 2, 2]],
                &[[1, 1, 1]],
                1,
                &[0, 0, 0],
                &[3, 3, 3],
                &mut starts,
                &mut counts,
                &mut buffer_starts,
                &mut stat_starts,
            ),
            3
        );
        assert_eq!(MULTI_BIN_STAT_SOURCE_FUNCTIONS.len(), 5);
    }
}
