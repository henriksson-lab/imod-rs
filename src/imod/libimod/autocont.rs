//! Low-level flood-mask operations from `IMOD/libimod/autocont.c`.
//!
//! The full native unit also couples contour construction to `Islice`, `Iobj`,
//! and the legacy OpenMP pipeline.  These three operations are independent of
//! that UI/pipeline machinery and are shared by auto-contouring callers.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::imod::libimod::icont::{
    imod_contour_area, imod_contour_default, imod_contour_reduce, imod_contour_shave,
    imod_contour_strip,
};
use crate::imod::libimod::imodel::{Icont, Iobj, Ipoint};
use crate::imod::libimod::ipoint::imod_point_inside_cont;

/// Native boundary pre-mask used by `imodAutoContoursFromSlice`.
pub fn pixels_inside_boundaries(boundaries: &[Icont], xsize: usize, ysize: usize, z: f32) -> Vec<bool> {
    (0..xsize.saturating_mul(ysize)).map(|index| {
        let point = Ipoint { x: (index % xsize) as f32 + 0.5, y: (index / xsize) as f32 + 0.5, z };
        boundaries.iter().any(|contour| imod_point_inside_cont(contour, &point) != 0)
    }).collect()
}

// C's `sStopProcessing`; the full slice-to-contour operation resets this at
// its start and polls it between patches.
static STOP_PROCESSING: AtomicBool = AtomicBool::new(false);

/// `AUTOX_FLOOD` from `imodel.h`.
pub const AUTOX_FLOOD: u8 = 1;
/// `AUTOX_PATCH` from `imodel.h`.
pub const AUTOX_PATCH: u8 = 1 << 1;
/// `AUTOX_FILL` from `imodel.h`.
pub const AUTOX_FILL: u8 = AUTOX_FLOOD | AUTOX_PATCH;
/// `AUTOX_CHECK` from `imodel.h`.
pub const AUTOX_CHECK: u8 = 1 << 5;

/// C `imodAutoContourStop`: request cancellation of an in-progress contour
/// operation.
pub fn imod_auto_contour_stop() {
    STOP_PROCESSING.store(true, Ordering::Relaxed);
}

/// Poll the C `sStopProcessing` cancellation latch.  The eventual
/// `imodAutoContoursFromSlice` port resets it before its scan, as the C
/// function does.
pub fn auto_contour_stop_requested() -> bool {
    STOP_PROCESSING.load(Ordering::Relaxed)
}

fn reset_auto_contour_stop() {
    STOP_PROCESSING.store(false, Ordering::Relaxed);
}

/// Source parameters for the unfiltered core of `imodAutoContoursFromSlice`.
#[derive(Clone, Copy, Debug)]
pub struct AutoContourOptions {
    pub high_threshold: f64,
    pub low_threshold: f64,
    pub exact: Option<u8>,
    /// 1: absolute thresholds, 2: supplied overall mean, 3: slice mean.
    pub dimension_mode: i32,
    pub mean: f32,
    pub min_size: usize,
    pub max_size: Option<usize>,
    pub follow_diagonals: i32,
    pub smooth_flags: i32,
    pub shave: f64,
    pub tolerance: f32,
}

impl Default for AutoContourOptions {
    fn default() -> Self {
        Self {
            high_threshold: 256.,
            low_threshold: 0.,
            exact: None,
            dimension_mode: 1,
            mean: 0.,
            min_size: 1,
            max_size: None,
            follow_diagonals: 0,
            smooth_flags: 0,
            shave: 0.,
            tolerance: 0.,
        }
    }
}

/// C `imodAutoContoursFromSlice`, for its byte-slice contouring core.
///
/// Filtering (`ksigma`) and boundary-object clipping are deliberately separate
/// callers in Rust; all thresholding, region growth, smoothing, edge tracing,
/// and native contour post-processing are performed here.
pub fn imod_auto_contours_from_slice(
    image: &[u8],
    xsize: usize,
    ysize: usize,
    z: f32,
    options: AutoContourOptions,
) -> Vec<Icont> {
    if xsize == 0 || ysize == 0 || image.len() < xsize.saturating_mul(ysize) {
        return Vec::new();
    }
    reset_auto_contour_stop();
    let nxy = xsize * ysize;
    let mean = if options.dimension_mode == 3 {
        image
            .iter()
            .take(nxy)
            .map(|&value| value as f64)
            .sum::<f64>() as f32
            / nxy as f32
    } else {
        options.mean
    };
    let (mut low, mut high) = if options.dimension_mode == 1 {
        (options.low_threshold as i32, options.high_threshold as i32)
    } else {
        (
            (mean as f64 * options.low_threshold) as i32,
            (mean as f64 * options.high_threshold) as i32,
        )
    };
    if options.exact.is_none() {
        high += 1;
        if low == 0 {
            low = -1;
        }
    }
    let mut labels = vec![0i32; nxy];
    for index in 0..nxy {
        if image[index] as i32 > low
            && (image[index] as i32) < high
            && options.exact.is_none_or(|value| image[index] != value)
        {
            labels[index] = -1;
        }
    }
    let listsize = 4 * (xsize + ysize).max(1);
    let mut xlist = vec![0; listsize];
    let mut ylist = vec![0; listsize];
    let mut label = 1;
    for y in 0..ysize {
        for x in 0..xsize {
            if auto_contour_stop_requested() {
                return Vec::new();
            }
            if labels[x + y * xsize] != 0 {
                continue;
            }
            let pixel = image[x + y * xsize];
            let diagonal = options.follow_diagonals >= 3
                || (options.follow_diagonals == 1
                    && options.exact.is_none()
                    && pixel as i32 >= high)
                || (options.follow_diagonals == 2
                    && options.exact.is_none()
                    && pixel as i32 <= low);
            let count = imoda_object_bfill_2d(
                image,
                &mut labels,
                &mut xlist,
                &mut ylist,
                xsize,
                ysize,
                x,
                y,
                low,
                high,
                options.exact,
                diagonal,
                label,
            );
            if count >= options.min_size.max(1) {
                label += 1;
            } else {
                for value in &mut labels {
                    if *value == label {
                        *value = -1;
                    }
                }
            }
        }
    }
    let lines: Vec<&[u8]> = image.chunks(xsize).take(ysize).collect();
    let mut result = Vec::new();
    for current in 1..label {
        if auto_contour_stop_requested() {
            return Vec::new();
        }
        let mut mask = vec![0u8; nxy];
        let Some(seed) = labels.iter().position(|&value| value == current) else {
            continue;
        };
        for (index, value) in labels.iter().enumerate() {
            if *value == current {
                mask[index] = AUTOX_FLOOD;
            }
        }
        imod_auto_patch(&mut mask, &mut xlist, &mut ylist, xsize, ysize);
        let repeats = if options.smooth_flags > 3 {
            (options.smooth_flags >> 2).max(1)
        } else {
            1
        };
        for _ in 0..repeats {
            if options.smooth_flags & 2 != 0 {
                imod_auto_expand(&mut mask, xsize, ysize);
            }
            if options.smooth_flags & 1 != 0 {
                imod_auto_shrink(&mut mask, xsize, ysize);
            }
        }
        let mut used_threshold = -1.;
        if options.smooth_flags & 3 == 0 && options.exact.is_none() {
            let value = image[seed] as i32;
            if value <= low {
                used_threshold = low as f32 + 0.5;
            }
            if value >= high {
                used_threshold = high as f32 - 0.5;
            }
        } else if options.smooth_flags & 3 != 0 {
            imod_auto_patch(&mut mask, &mut xlist, &mut ylist, xsize, ysize);
        }
        let reverse = image[seed] as i32 <= low;
        let diagonal = options.follow_diagonals >= 3
            || (options.follow_diagonals == 1 && image[seed] as i32 >= high)
            || (options.follow_diagonals == 2 && image[seed] as i32 <= low);
        for mut contour in imod_contours_from_image_points(
            &mut mask,
            options.exact.is_none().then_some(&lines),
            xsize,
            ysize,
            z,
            AUTOX_FLOOD,
            diagonal,
            used_threshold,
            reverse,
        ) {
            let area = imod_contour_area(Some(&contour)).abs();
            if area < options.min_size as f32
                || options.max_size.is_some_and(|max| area > max as f32)
            {
                continue;
            }
            imod_contour_strip(&mut contour);
            if options.tolerance != 0. {
                imod_contour_reduce(Some(&mut contour), options.tolerance);
            }
            if options.shave != 0. {
                imod_contour_shave(&mut contour, options.shave);
            }
            result.push(contour);
        }
    }
    result
}

/// C `imodAutoPatch`.
///
/// Marks holes surrounded by `AUTOX_FLOOD` as flood. `xlist` and `ylist`
/// are the native caller-provided ring-buffer work arrays; their common usable
/// length is used as the C `listsize` argument.
pub fn imod_auto_patch(
    data: &mut [u8],
    xlist: &mut [i32],
    ylist: &mut [i32],
    xsize: usize,
    ysize: usize,
) {
    let listsize = xlist.len().min(ylist.len());
    if xsize == 0 || ysize == 0 || data.len() < xsize.saturating_mul(ysize) || listsize == 0 {
        return;
    }

    let mut xmax = -1isize;
    let mut xmin = xsize as isize;
    let mut ymax = -1isize;
    let mut ymin = ysize as isize;
    for y in 0..ysize {
        for x in 0..xsize {
            if data[x + y * xsize] & AUTOX_FLOOD != 0 {
                xmin = xmin.min(x as isize);
                xmax = xmax.max(x as isize);
                ymin = ymin.min(y as isize);
                ymax = ymax.max(y as isize);
            }
        }
    }
    // Native callers invoke this only on nonempty flood masks.  Avoid the
    // invalid native bounds in the public safe Rust API.
    if xmax < xmin || ymax < ymin {
        return;
    }
    let (xmin, xmax, ymin, ymax) = (xmin as usize, xmax as usize, ymin as usize, ymax as usize);
    for x in xmin..=xmax {
        auto_patch_fill_outside(data, xlist, ylist, xsize, xmin, xmax, ymin, ymax, x, ymin);
        auto_patch_fill_outside(data, xlist, ylist, xsize, xmin, xmax, ymin, ymax, x, ymax);
    }
    for y in ymin..=ymax {
        auto_patch_fill_outside(data, xlist, ylist, xsize, xmin, xmax, ymin, ymax, xmin, y);
        auto_patch_fill_outside(data, xlist, ylist, xsize, xmin, xmax, ymin, ymax, xmax, y);
    }
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            let index = x + y * xsize;
            if data[index] & AUTOX_FILL == 0 {
                data[index] |= AUTOX_FLOOD;
            }
        }
    }
    for pixel in data.iter_mut().take(xsize * ysize) {
        *pixel &= !AUTOX_PATCH;
    }
}

#[allow(clippy::too_many_arguments)]
fn auto_patch_fill_outside(
    data: &mut [u8],
    xlist: &mut [i32],
    ylist: &mut [i32],
    xsize: usize,
    xmin: usize,
    xmax: usize,
    ymin: usize,
    ymax: usize,
    x: usize,
    y: usize,
) {
    let listsize = xlist.len().min(ylist.len());
    let index = x + y * xsize;
    if data[index] & AUTOX_FILL != 0 {
        return;
    }
    let mut next = 0;
    let mut free = 1 % listsize;
    xlist[0] = x as i32;
    ylist[0] = y as i32;
    data[index] |= AUTOX_CHECK | AUTOX_PATCH;
    let neighbor_flags = AUTOX_FLOOD | AUTOX_CHECK | AUTOX_PATCH;
    while next != free {
        let x = xlist[next] as usize;
        let y = ylist[next] as usize;
        let index = x + y * xsize;
        let mut add = |nx: usize, ny: usize, neighbor: usize| {
            if data[neighbor] & neighbor_flags == 0 {
                xlist[free] = nx as i32;
                ylist[free] = ny as i32;
                free = (free + 1) % listsize;
                data[neighbor] |= AUTOX_CHECK | AUTOX_PATCH;
            }
        };
        if x > xmin {
            add(x - 1, y, index - 1);
        }
        if x < xmax {
            add(x + 1, y, index + 1);
        }
        if y > ymin {
            add(x, y - 1, index - xsize);
        }
        if y < ymax {
            add(x, y + 1, index + xsize);
        }
        data[index] &= !AUTOX_CHECK;
        next = (next + 1) % listsize;
    }
}

/// C `imodAutoShrink`.
pub fn imod_auto_shrink(data: &mut [u8], imax: usize, jmax: usize) {
    if imax == 0 || jmax == 0 || data.len() < imax.saturating_mul(jmax) {
        return;
    }
    for j in 0..jmax {
        for i in 0..imax {
            let index = i + j * imax;
            if data[index] & AUTOX_FLOOD == 0 {
                continue;
            }
            let mut count = 0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let x = i as isize + dx;
                    let y = j as isize + dy;
                    if x >= 0
                        && y >= 0
                        && x < imax as isize
                        && y < jmax as isize
                        && data[x as usize + y as usize * imax] & AUTOX_FLOOD != 0
                    {
                        count += 1;
                    }
                }
            }
            if count < 7 {
                data[index] |= AUTOX_CHECK;
            }
        }
    }
    for pixel in data.iter_mut().take(imax * jmax) {
        if *pixel & AUTOX_CHECK != 0 {
            *pixel &= !(AUTOX_FLOOD | AUTOX_CHECK);
        }
    }
}

/// C `imodAutoExpand`.
pub fn imod_auto_expand(data: &mut [u8], imax: usize, jmax: usize) {
    if imax == 0 || jmax == 0 || data.len() < imax.saturating_mul(jmax) {
        return;
    }
    for j in 0..jmax {
        for i in 0..imax {
            if data[i + j * imax] & AUTOX_FILL == 0 {
                continue;
            }
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let x = i as isize + dx;
                    let y = j as isize + dy;
                    if x >= 0 && y >= 0 && x < imax as isize && y < jmax as isize {
                        data[x as usize + y as usize * imax] |= AUTOX_CHECK;
                    }
                }
            }
        }
    }
    for pixel in data.iter_mut().take(imax * jmax) {
        if *pixel & AUTOX_CHECK != 0 {
            *pixel |= AUTOX_FLOOD;
            *pixel &= !AUTOX_CHECK;
        }
    }
}

/// C `imoda_object_bfill_2d`.
///
/// Labels the threshold-connected patch beginning at `(x, y)` with
/// `cont_label`, returning its pixel count.  A zero in `labels` is unvisited;
/// nonzero labels are left intact, precisely as in the native contour scan.
#[allow(clippy::too_many_arguments)]
pub fn imoda_object_bfill_2d(
    image: &[u8],
    labels: &mut [i32],
    xlist: &mut [i32],
    ylist: &mut [i32],
    xsize: usize,
    ysize: usize,
    x: usize,
    y: usize,
    t1: i32,
    t2: i32,
    exact: Option<u8>,
    diagonal: bool,
    cont_label: i32,
) -> usize {
    let listsize = xlist.len().min(ylist.len());
    if xsize == 0
        || ysize == 0
        || x >= xsize
        || y >= ysize
        || listsize == 0
        || image.len() < xsize.saturating_mul(ysize)
        || labels.len() < xsize * ysize
    {
        return 0;
    }
    let start = x + y * xsize;
    let (threshold, direction, test_exact) = match exact {
        Some(value) => (0, 0, image[start] == value),
        None if image[start] as i32 <= t1 => (t1, -1, false),
        None => (t2, 1, false),
    };
    let mut next = 0;
    let mut free = 1 % listsize;
    xlist[0] = x as i32;
    ylist[0] = y as i32;
    labels[start] = -2;
    let mut added = 0;
    while next != free {
        let x = xlist[next] as usize;
        let y = ylist[next] as usize;
        let index = x + y * xsize;
        let passes = match exact {
            None => direction * (image[index] as i32 - threshold) >= 0,
            Some(value) => {
                (test_exact && image[index] == value)
                    || (!test_exact && (image[index] as i32 <= t1 || image[index] as i32 >= t2))
            }
        };
        if passes {
            labels[index] = cont_label;
            added += 1;
            let mut enqueue = |nx: usize, ny: usize| {
                let neighbor = nx + ny * xsize;
                if labels[neighbor] == 0 {
                    xlist[free] = nx as i32;
                    ylist[free] = ny as i32;
                    free = (free + 1) % listsize;
                    labels[neighbor] = -2;
                }
            };
            if x > 0 {
                enqueue(x - 1, y);
            }
            if x + 1 < xsize {
                enqueue(x + 1, y);
            }
            if y > 0 {
                enqueue(x, y - 1);
            }
            if y + 1 < ysize {
                enqueue(x, y + 1);
            }
            if diagonal {
                if x > 0 && y > 0 {
                    enqueue(x - 1, y - 1);
                }
                if x + 1 < xsize && y > 0 {
                    enqueue(x + 1, y - 1);
                }
                if x > 0 && y + 1 < ysize {
                    enqueue(x - 1, y + 1);
                }
                if x + 1 < xsize && y + 1 < ysize {
                    enqueue(x + 1, y + 1);
                }
            }
        }
        if labels[index] == -2 {
            labels[index] = 0;
        }
        next = (next + 1) % listsize;
    }
    added
}

/// C `imodContoursFromImagePoints`.
///
/// Walks the exposed sides of pixels selected by `testmask` and returns ordered
/// native contours.  `image_lines`, when supplied, is the C `unsigned char
/// **imdata` input and enables edge-position interpolation.
#[allow(clippy::too_many_arguments)]
pub fn imod_contours_from_image_points(
    data: &mut [u8],
    image_lines: Option<&[&[u8]]>,
    xsize: usize,
    ysize: usize,
    z: f32,
    testmask: u8,
    diagonal: bool,
    mut threshold: f32,
    reverse: bool,
) -> Vec<Icont> {
    if xsize == 0 || ysize == 0 || data.len() < xsize.saturating_mul(ysize) {
        return Vec::new();
    }
    let right = 16u8;
    let top = right << 1;
    let left = right << 2;
    let bottom = right << 3;
    let any_edge = right | top | left | bottom;
    let edge_masks = [right, top, left, bottom];
    // Direction along each edge, corner into the adjacent pixel, and pixel
    // across the edge respectively; these are the C static arrays verbatim.
    let next_x = [0isize, -1, 0, 1];
    let next_y = [1isize, 0, -1, 0];
    let corner_x = [1isize, -1, -1, 1];
    let corner_y = [1isize, 1, -1, -1];
    let other_x = [1isize, 0, -1, 0];
    let other_y = [0isize, 1, 0, -1];
    let valid_image = image_lines.is_some_and(|lines| {
        lines.len() >= ysize && lines.iter().take(ysize).all(|line| line.len() >= xsize)
    });
    if !valid_image {
        threshold = -1.;
    }
    let mut edge_sum = 0f64;
    let mut edge_count = 0usize;
    for y in 0..ysize {
        for x in 0..xsize {
            let index = x + y * xsize;
            if data[index] & testmask == 0 {
                continue;
            }
            let mut side = 0;
            let mut count_edge = |nx: usize, ny: usize| {
                if threshold < 0. && valid_image {
                    let lines = image_lines.expect("validated image lines");
                    edge_sum += (lines[y][x] as f64) + (lines[ny][nx] as f64);
                    edge_count += 1;
                }
            };
            if x + 1 == xsize {
                side |= right;
            } else if data[index + 1] & testmask == 0 {
                side |= right;
                count_edge(x + 1, y);
            }
            if y + 1 == ysize {
                side |= top;
            } else if data[index + xsize] & testmask == 0 {
                side |= top;
                count_edge(x, y + 1);
            }
            if x == 0 {
                side |= left;
            } else if data[index - 1] & testmask == 0 {
                side |= left;
                count_edge(x - 1, y);
            }
            if y == 0 {
                side |= bottom;
            } else if data[index - xsize] & testmask == 0 {
                side |= bottom;
                count_edge(x, y - 1);
            }
            data[index] |= side;
        }
    }
    if edge_count != 0 {
        threshold = (0.5 * edge_sum / edge_count as f64) as f32;
    }

    let in_bounds =
        |x: isize, y: isize| x >= 0 && y >= 0 && x < xsize as isize && y < ysize as isize;
    let mut contours = Vec::new();
    loop {
        let Some((mut x, mut y)) = (0..ysize).find_map(|y| {
            (0..xsize)
                .find(|&x| data[x + y * xsize] & any_edge != 0)
                .map(|x| (x, y))
        }) else {
            break;
        };
        let mut edge = edge_masks
            .iter()
            .position(|mask| data[x + y * xsize] & mask != 0)
            .expect("pixel selected from edge mask");
        let (start_x, start_y, start_edge) = (x, y, edge);
        let mut contour = Icont::default();
        imod_contour_default(&mut contour);
        // Each iteration clears exactly one edge.  The bound turns malformed
        // flag input into a partial contour rather than an infinite walk.
        for _ in 0..=xsize.saturating_mul(ysize).saturating_mul(4) {
            if !contour.pts.is_empty() && (x, y, edge) == (start_x, start_y, start_edge) {
                break;
            }
            if !diagonal {
                if data[x + y * xsize] & edge_masks[(edge + 1) % 4] != 0 {
                    edge = (edge + 1) % 4;
                } else {
                    let nx = x as isize + next_x[edge];
                    let ny = y as isize + next_y[edge];
                    if in_bounds(nx, ny)
                        && data[nx as usize + ny as usize * xsize] & edge_masks[edge] != 0
                    {
                        x = nx as usize;
                        y = ny as usize;
                    } else {
                        let nx = x as isize + corner_x[edge];
                        let ny = y as isize + corner_y[edge];
                        if !in_bounds(nx, ny) {
                            break;
                        }
                        x = nx as usize;
                        y = ny as usize;
                        edge = (edge + 3) % 4;
                    }
                }
            } else {
                let cx = x as isize + corner_x[edge];
                let cy = y as isize + corner_y[edge];
                if in_bounds(cx, cy)
                    && data[cx as usize + cy as usize * xsize] & edge_masks[(edge + 3) % 4] != 0
                {
                    x = cx as usize;
                    y = cy as usize;
                    edge = (edge + 3) % 4;
                } else {
                    let nx = x as isize + next_x[edge];
                    let ny = y as isize + next_y[edge];
                    if in_bounds(nx, ny)
                        && data[nx as usize + ny as usize * xsize] & edge_masks[edge] != 0
                    {
                        x = nx as usize;
                        y = ny as usize;
                    } else {
                        edge = (edge + 1) % 4;
                        if data[x + y * xsize] & edge_masks[edge] == 0 {
                            break;
                        }
                    }
                }
            }
            let across_x = x as isize + other_x[edge];
            let across_y = y as isize + other_y[edge];
            let mut fraction = 0.45;
            if threshold > 0. && valid_image && in_bounds(across_x, across_y) {
                let lines = image_lines.expect("validated image lines");
                let diff = lines[across_y as usize][across_x as usize] as f32 - lines[y][x] as f32;
                let polarity = if reverse { -1. } else { 1. };
                if polarity * diff < 0. {
                    fraction = ((threshold - lines[y][x] as f32) / diff).clamp(0.01, 0.99);
                }
            }
            contour.pts.push(Ipoint {
                x: x as f32 + 0.5 + fraction * other_x[edge] as f32,
                y: y as f32 + 0.5 + fraction * other_y[edge] as f32,
                z,
            });
            data[x + y * xsize] &= !edge_masks[edge];
        }
        if !contour.pts.is_empty() {
            contours.push(contour);
        }
    }
    contours
}

/// C `findBoundaryConts`.
///
/// Replaces the C `Ilist` of integer pointers with a caller-owned list of
/// contour indices. With `nearest` false, it returns no indices unless a
/// boundary contour lies exactly at `z`; otherwise it selects every valid
/// contour at the nearest Z plane.
pub fn find_boundary_conts(
    z: i32,
    boundary_object: &Iobj,
    nearest: bool,
    contour_indices: &mut Vec<usize>,
) {
    contour_indices.clear();
    let mut min_difference = i32::MAX;
    let mut nearest_z = 0;
    for contour in &boundary_object.cont {
        if contour.pts.len() < 3 {
            continue;
        }
        let contour_z = contour.pts[0].z.round() as i32;
        let difference = (contour_z - z).abs();
        if difference < min_difference {
            min_difference = difference;
            nearest_z = contour_z;
        }
    }
    if !nearest && min_difference > 0 {
        return;
    }
    for (index, contour) in boundary_object.cont.iter().enumerate() {
        if contour.pts.len() >= 3 && contour.pts[0].z.round() as i32 == nearest_z {
            contour_indices.push(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_fills_an_enclosed_hole_and_clears_work_flags() {
        let mut data = vec![AUTOX_FLOOD; 9];
        data[4] = 0;
        let mut xlist = [0; 16];
        let mut ylist = [0; 16];
        imod_auto_patch(&mut data, &mut xlist, &mut ylist, 3, 3);
        assert!(data.iter().all(|&pixel| pixel == AUTOX_FLOOD));
    }

    #[test]
    fn expand_then_shrink_match_native_eight_neighbor_rules() {
        let mut data = vec![0; 9];
        data[4] = AUTOX_FLOOD;
        imod_auto_expand(&mut data, 3, 3);
        assert!(data.iter().all(|&pixel| pixel == AUTOX_FLOOD));
        imod_auto_shrink(&mut data, 3, 3);
        assert_eq!(data, vec![0, 0, 0, 0, AUTOX_FLOOD, 0, 0, 0, 0]);
    }

    #[test]
    fn threshold_fill_keeps_four_and_eight_connectivity_distinct() {
        let image = [10, 0, 0, 0, 10, 0, 0, 0, 10];
        let mut four_way = [0; 9];
        let mut xlist = [0; 32];
        let mut ylist = [0; 32];
        assert_eq!(
            imoda_object_bfill_2d(
                &image,
                &mut four_way,
                &mut xlist,
                &mut ylist,
                3,
                3,
                0,
                0,
                2,
                8,
                None,
                false,
                7,
            ),
            1
        );
        let mut diagonal = [0; 9];
        assert_eq!(
            imoda_object_bfill_2d(
                &image,
                &mut diagonal,
                &mut xlist,
                &mut ylist,
                3,
                3,
                0,
                0,
                2,
                8,
                None,
                true,
                7,
            ),
            3
        );
        assert_eq!(diagonal, [7, 0, 0, 0, 7, 0, 0, 0, 7]);
    }

    #[test]
    fn contour_stop_latch_is_observable_and_resettable_by_a_new_scan() {
        reset_auto_contour_stop();
        assert!(!auto_contour_stop_requested());
        imod_auto_contour_stop();
        assert!(auto_contour_stop_requested());
        reset_auto_contour_stop();
    }

    #[test]
    fn contour_walker_turns_one_selected_pixel_into_four_edges() {
        let mut data = [AUTOX_FLOOD];
        let contours = imod_contours_from_image_points(
            &mut data,
            None,
            1,
            1,
            4.,
            AUTOX_FLOOD,
            false,
            -1.,
            false,
        );
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].pts.len(), 4);
        assert!(contours[0].pts.iter().all(|point| point.z == 4.));
        assert_eq!(data, [AUTOX_FLOOD]);
    }

    #[test]
    fn boundary_selector_honors_exact_and_nearest_planes() {
        let contour_at = |z| Icont {
            pts: vec![Ipoint { x: 0., y: 0., z }; 3],
            ..Icont::default()
        };
        let object = Iobj {
            cont: vec![contour_at(2.), contour_at(5.), Icont::default()],
            ..Iobj::default()
        };
        let mut selected = vec![99];
        find_boundary_conts(4, &object, false, &mut selected);
        assert!(selected.is_empty());
        find_boundary_conts(4, &object, true, &mut selected);
        assert_eq!(selected, vec![1]);
        find_boundary_conts(2, &object, false, &mut selected);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn slice_orchestrator_extracts_a_high_threshold_component() {
        let image = [10u8];
        let contours = imod_auto_contours_from_slice(
            &image,
            1,
            1,
            3.,
            AutoContourOptions {
                low_threshold: 0.,
                high_threshold: 5.,
                min_size: 0,
                ..AutoContourOptions::default()
            },
        );
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].pts.len(), 4);
    }
}
