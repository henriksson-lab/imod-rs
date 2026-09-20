//! Translation of `IMOD/libmesh/skeletonize.c` -- creates a cap based on the
//! medial axis (skeleton) of the previous slice.

use crate::imod::libimod::icont::{imod_contour_get_bbox, imodel_contour_scan};
use crate::imod::libimod::imodel::{Icont, Ipoint};

/*
Description of the algorithm by T.Y.Zhang and C.Y.Suen,
  'A fast parallel algorithm for thinning digital patterns',
  Communications of the ACM, 27,(3),236-239,1984.)

Keep deleting pixels in 2 passes until nothing changed. For all pixels,
start with counting the number of neighbor pixels set to 1 (B(p1)) and count
the transitions from 0 to 1 in a clockwise direction (A(p1)).
*/

/// Original: `DeleteInPass` (`skeletonize.c:81`).
#[rustfmt::skip]
pub const DELETE_IN_PASS: [[i32; 256]; 2] = [
  [
/*  0*/ 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
/* 30*/ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
/* 60*/ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/* 90*/ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
/*120*/ 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
/*150*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*180*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
/*210*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
/*240*/ 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ],
  [
/*  0*/ 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
/* 30*/ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
/* 60*/ 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/* 90*/ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
/*120*/ 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*150*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*180*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*210*/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
/*240*/ 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
  ],
];

/* steps for freeman chain codes:
     3 2 1
      \|/
     4-.-0
      /|\
     5 6 7
*/
/// Original static `freeman_dx` (`skeletonize.c:110`).
const FREEMAN_DX: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
/// Original static `freeman_dy` (`skeletonize.c:111`).
const FREEMAN_DY: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];

/// Original static `Skeletonize` (`skeletonize.c:113`).
pub fn skeletonize_pixels(size_x: i32, size_y: i32, pc_out: &mut [u8], border_size: i32) {
    let i_nb_pixels = size_x * size_y;

    let size_x_pred = size_x - 1;
    let size_x_succ = size_x + 1;
    let size_y_pred = size_y - 1;

    let low_x = border_size;
    let high_x = size_x_pred - border_size;
    let low_y = border_size;
    let high_y = size_y_pred - border_size;

    let mut b_changes = [1i32, 1i32];

    while b_changes[0] != 0 || b_changes[1] != 0 {
        /* Assume we are done */
        b_changes[0] = 0;
        b_changes[1] = 0;
        for pass in 0..2usize {
            let delete_in_this_pass = &DELETE_IN_PASS[pass];
            for j in low_y..=high_y {
                for i in low_x..=high_x {
                    let i_index = (j * size_x + i) as usize;
                    if pc_out[i_index] == 1 {
                        /* Only evaluate foreground, non-deleted pixels */
                        let mut byteval: u8 = 0;
                        if i > 0 {
                            byteval |= (pc_out[i_index - 1] & 0x01) << 1; // p8
                            if j > 0 {
                                byteval |= pc_out[i_index - size_x_succ as usize] & 0x01; // p9
                            }
                            if j < size_y_pred {
                                byteval |= (pc_out[i_index + size_x_pred as usize] & 0x01) << 2;
                                // p7
                            }
                        }
                        if i < size_x_pred {
                            byteval |= (pc_out[i_index + 1] & 0x01) << 5; // p4
                            if j > 0 {
                                byteval |= (pc_out[i_index - size_x_pred as usize] & 0x01) << 6;
                                // p3
                            }
                            if j < size_y_pred {
                                byteval |= (pc_out[i_index + size_x_succ as usize] & 0x01) << 4;
                                // p5
                            }
                        }
                        if j > 0 {
                            byteval |= (pc_out[i_index - size_x as usize] & 0x01) << 7; // p2
                        }
                        if j < size_y_pred {
                            byteval |= (pc_out[i_index + size_x as usize] & 0x01) << 3; // p6
                        }
                        if delete_in_this_pass[byteval as usize] != 0 {
                            pc_out[i_index] |= 0x02;
                            b_changes[pass] = 1;
                        }
                    }
                }
            }
            /* Delete all marked pixels (i.e. only keep pixels with a value of 1) */
            if b_changes[pass] != 0 {
                for i in 0..i_nb_pixels as usize {
                    pc_out[i] = if pc_out[i] == 1 { 1 } else { 0 };
                }
            }
        }
    }
}

/// Original static `SliceMedialAxis` (`skeletonize.c:173`).
///
/// Copied from `avs_medialaxis.cpp` (see AVS module MEDIAL_AXIS for
/// documentation).
pub fn slice_medial_axis(
    p_in: &[u8],
    pp_out: &mut Option<Vec<u8>>,
    i_size_x: i32,
    i_size_y: i32,
) -> i32 {
    /* Free the output */
    *pp_out = None;
    *pp_out = Some(vec![0u8; (i_size_x * i_size_y) as usize]);

    /* Copy pixel values: 0 or 1 */
    let pc_out = pp_out.as_mut().unwrap();
    pc_out.copy_from_slice(&p_in[..(i_size_x * i_size_y) as usize]);

    // Check if border check can be skipped by inspecting bounding box
    let mut min_x = i_size_x;
    let mut max_x = 0;
    let mut min_y = i_size_y;
    let mut max_y = 0;

    for j in 0..i_size_y {
        for i in 0..i_size_x {
            if pc_out[(i + j * i_size_x) as usize] != 0 {
                if j < min_y {
                    min_y = j;
                }
                if j > max_y {
                    max_y = j;
                }
                if i < min_x {
                    min_x = i;
                }
                if i > max_x {
                    max_x = i;
                }
            }
        }
    }

    let mut border_size = min_x;
    if i_size_x - 1 - max_x < border_size {
        border_size = i_size_x - 1 - max_x;
    }
    if min_y < border_size {
        border_size = min_y;
    }
    if i_size_y - 1 - max_y < border_size {
        border_size = i_size_y - 1 - max_y;
    }

    skeletonize_pixels(i_size_x, i_size_y, pc_out, border_size);

    1
}

/// Original static `AllPixelsProcessed` (`skeletonize.c:241`).
///
/// Dead in the source as well; translated because whole source units are
/// translated.
pub fn all_pixels_processed(use_count: &[i32], size: i32) -> i32 {
    for i in 0..size as usize {
        if use_count[i] == 0 {
            return 0;
        }
    }
    1
}

/// Original static `FindStartPoint` (`skeletonize.c:250`).
pub fn find_start_point(
    height: i32,
    width: i32,
    axis_field: &[u8],
    startx: &mut i32,
    starty: &mut i32,
) -> i32 {
    *startx = -1;
    *starty = -1;

    for j in 0..height {
        for i in 0..width {
            if axis_field[(j * width + i) as usize] != 0 {
                // p5 p3 p6
                // p1  x p2
                // p7 p4 p8
                let at = |jj: i32, ii: i32| -> i32 {
                    i32::from(axis_field[(jj * width + ii) as usize] != 0)
                };
                let p1 = at(j, i - 1);
                let p2 = at(j, i + 1);
                let p3 = at(j - 1, i);
                let p4 = at(j + 1, i);
                let p5 = at(j - 1, i - 1);
                let p6 = at(j - 1, i + 1);
                let p7 = at(j + 1, i - 1);
                let p8 = at(j + 1, i + 1);
                let sum = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8;
                if sum == 1
                    || (sum == 2
                        && ((p1 != 0 && p5 != 0)
                            || (p5 != 0 && p3 != 0)
                            || (p3 != 0 && p6 != 0)
                            || (p6 != 0 && p2 != 0)
                            || (p2 != 0 && p8 != 0)
                            || (p8 != 0 && p4 != 0)
                            || (p4 != 0 && p7 != 0)
                            || (p7 != 0 && p1 != 0)))
                {
                    *startx = i;
                    *starty = j;
                    return 1;
                }
            }
        }
    }

    0
}

/// Original static `EliminateLinearPart` (`skeletonize.c:287`).
pub fn eliminate_linear_part(
    startpoint: i32,
    startpoint2: i32,
    points_x: &mut Vec<i32>,
    points_y: &mut Vec<i32>,
    num_cap_points: &mut i32,
) -> i32 {
    let mut offset: i32 = 1;
    let mut n: i32 = 0;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    let del: f64;
    let aa: f64;
    let bb: f64;
    let mut syy = 0.0f64;
    let mut rss = 0.0f64;
    let mut rs: f64;
    let r_squared: f64;
    // points_x and points_y do not contain the first point also at the end, while
    // CapPoints does have this point twice.  We start checking with an offset of 1 to
    // compensate for this.
    let np: i32 = *num_cap_points - 1;

    while offset < np
        && points_x[((startpoint + offset) % np) as usize]
            == points_x[((np + (startpoint2 - offset)) % np) as usize]
        && points_y[((startpoint + offset) % np) as usize]
            == points_y[((np + (startpoint2 - offset)) % np) as usize]
    {
        let idx = ((startpoint + offset) % np) as usize;
        offset += 1;

        /* perform linear fit */

        n += 1;

        sx += points_x[idx] as f64;
        sy += points_y[idx] as f64;
        sxy += (points_y[idx] * points_x[idx]) as f64;
        sxx += (points_x[idx] as f32 * points_x[idx] as f32) as f64;

        syy += (points_y[idx] as f32 * points_y[idx] as f32) as f64;
    }
    del = n as f64 * sxx - sx * sx;
    aa = (sxx * sy - sx * sxy) / del; // intercept
    bb = (n as f64 * sxy - sx * sy) / del; // slope

    // Compute r-squared value for y[x] = aa + bb * x

    offset = 1;
    n = 0;
    while offset < np
        && points_x[((startpoint + offset) % np) as usize]
            == points_x[((np + (startpoint2 - offset)) % np) as usize]
        && points_y[((startpoint + offset) % np) as usize]
            == points_y[((np + (startpoint2 - offset)) % np) as usize]
    {
        let idx = ((startpoint + offset) % np) as usize;
        offset += 1;
        n += 1;

        rs = (points_y[idx] as f32) as f64 - (aa + bb * points_x[idx] as f64);
        rss += rs * rs;
    }

    if n > 1 {
        syy -= sy * sy / n as f64;
        if syy > 1e-8 {
            r_squared = 1.0f32 as f64 - (rss / syy);
        } else {
            r_squared = 1.0f32 as f64;
        }
        if r_squared > 0.99f32 as f64 {
            let mut newpoints_x: Vec<i32> = vec![0; (np - 2 * n).max(0) as usize];
            let mut newpoints_y: Vec<i32> = vec![0; (np - 2 * n).max(0) as usize];
            // delete startpoint, all 'n' points stored twice but keep one point at the
            // branch to keep contour organized

            // delete n points from startpoint forwards and n points from startpoint2
            // backwards
            offset = 0;
            for i in 0..np {
                if (i < startpoint || i >= startpoint + n)
                    && (i < startpoint2 - n || i > startpoint2)
                {
                    newpoints_x[offset as usize] = points_x[i as usize];
                    newpoints_y[offset as usize] = points_y[i as usize];
                    offset += 1;
                }
            }
            *points_x = newpoints_x;
            *points_y = newpoints_y;

            *num_cap_points -= 2 * n;
        }
    } else {
        return 0;
    }

    1
}

/// Original static `EliminateStraightSegments` (`skeletonize.c:376`).
pub fn eliminate_straight_segments(
    points_x: &mut Vec<i32>,
    points_y: &mut Vec<i32>,
    endpoints_x: &[i32],
    endpoints_y: &[i32],
    numendpoints: i32,
    num_cap_points: &mut i32,
) -> i32 {
    for i in 0..numendpoints as usize {
        let mut j: i32 = 0;
        while j < *num_cap_points - 1 {
            if endpoints_x[i] == points_x[j as usize] && endpoints_y[i] == points_y[j as usize] {
                let startpoint = j;

                // only the first point is replicated at the end, the other end points are
                // present once
                if startpoint == 0 {
                    let k_prime = *num_cap_points - 1;
                    eliminate_linear_part(startpoint, k_prime, points_x, points_y, num_cap_points);
                    break;
                } else {
                    eliminate_linear_part(
                        startpoint,
                        startpoint,
                        points_x,
                        points_y,
                        num_cap_points,
                    );
                    break;
                }
            }
            j += 1;
        }
    }
    1
}

/// Original static `TakeFreemanStep` (`skeletonize.c:404`).
pub fn take_freeman_step(freeman: i32, i: &mut i32, j: &mut i32) {
    *i += FREEMAN_DX[freeman as usize];
    *j += FREEMAN_DY[freeman as usize];
}

/// Original static `SkeletonToContour` (`skeletonize.c:410`).
#[allow(clippy::too_many_arguments)]
pub fn skeleton_to_contour(
    axis_field: &mut [u8],
    width: i32,
    height: i32,
    min_x: i32,
    min_z: i32,
    rangex: i32,
    rangey: i32,
    co_m: &Ipoint,
    cap_points: &mut Icont,
) -> i32 {
    let mut newpoints_x: Vec<i32> = vec![0; 1 + 100]; // estimated size of chunk: 100
    let mut newpoints_y: Vec<i32> = vec![0; 1 + 100];

    let mut endpoints_x: Vec<i32> = vec![0; 100]; // estimated size of chunk: 100
    let mut endpoints_y: Vec<i32> = vec![0; 100];
    let mut numendpoints: i32 = 0;
    let mut numpoints: i32;

    /* freeman codes:
        3 2 1
         \|/
        4-.-0
         /|\
        5 6 7
    */

    let mut current_point: i32 = 0;

    // black borders
    for i in 0..width {
        axis_field[i as usize] = 0;
    }
    for i in 0..width {
        axis_field[((height - 1) * width + i) as usize] = 0;
    }
    for i in 0..height {
        axis_field[(i * width) as usize] = 0;
    }
    for i in 0..height {
        axis_field[(i * width + (width - 1)) as usize] = 0;
    }

    let mut startx: i32 = 0;
    let mut starty: i32 = 0;
    if find_start_point(height, width, axis_field, &mut startx, &mut starty) != 0 {
        let mut i = startx;
        let mut j = starty;
        let mut freeman: i32 = 0;

        endpoints_x[0] = startx;
        endpoints_y[0] = starty;
        numendpoints += 1;

        while current_point == 0 || i != startx || j != starty {
            let mut candidates = [0i32; 2];
            let mut num_candidate: usize = 0;

            newpoints_x[current_point as usize] = i;
            newpoints_y[current_point as usize] = j;

            current_point += 1;

            if current_point % 100 == 0 {
                newpoints_x.resize((current_point + 100 + 1) as usize, 0);
                newpoints_y.resize((current_point + 100 + 1) as usize, 0);
            }

            // Scan clockwise, start one step from opposite direction
            let mut dir_counter: i32 = 0;

            // borders are set to zero, so no boundary checks are needed
            let mut new_freeman = (freeman + 2) % 8;
            while dir_counter < 8 && num_candidate < 2 {
                if axis_field[((j + FREEMAN_DY[new_freeman as usize]) * width
                    + i
                    + FREEMAN_DX[new_freeman as usize]) as usize]
                    != 0
                {
                    candidates[num_candidate] = new_freeman;
                    num_candidate += 1;
                }
                new_freeman = (new_freeman + 7) % 8;
                dir_counter += 1;
            }

            // favor straigth steps (even freeman code) over diagonal ones
            freeman = candidates[0];
            if num_candidate > 1
                && (candidates[0] % 2) != 0
                && ((8 + candidates[0] - candidates[1]) % 8) == 1
            {
                freeman = candidates[1];
            }

            take_freeman_step(freeman, &mut i, &mut j);
        }
    }

    // compare CapPoints with reversed list: matching parts indicate a branch (or the
    // entire contour).  For each branch, check if all points are (more or less) on a
    // line. Remove those branches as these are not needed for the caps.

    numpoints = if current_point != 0 {
        current_point + 1
    } else {
        0
    };

    // close complex cap by adding first point at the end
    if current_point > 0 {
        newpoints_x[current_point as usize] = newpoints_x[0];
        newpoints_y[current_point as usize] = newpoints_y[0];
    }

    if numendpoints >= 3 {
        eliminate_straight_segments(
            &mut newpoints_x,
            &mut newpoints_y,
            &endpoints_x,
            &endpoints_y,
            numendpoints,
            &mut numpoints,
        );
    }

    let psize = numpoints;
    if psize <= 0 {
        return 0;
    }
    let mut newpoints: Vec<Ipoint> = vec![Ipoint::default(); psize as usize];
    let mut scal_x: f32 = 1.0;
    let mut scal_y: f32 = 1.0;
    if width - 3 > 0 {
        scal_x = ((rangex - 1) / (width - 3)) as f32;
    }
    if height - 3 > 0 {
        scal_y = ((rangey - 1) / (height - 3)) as f32;
    }
    for i in 0..(numpoints - 1) as usize {
        newpoints[i].x = min_x as f32 + (newpoints_x[i] - 1) as f32 * scal_x; // black borders
        newpoints[i].y = min_z as f32 + (newpoints_y[i] - 1) as f32 * scal_y;
        newpoints[i].z = co_m.z;
    }

    // close complex cap by adding first point at the end
    if numpoints - 1 > 0 {
        newpoints[(numpoints - 1) as usize].x = newpoints[0].x;
        newpoints[(numpoints - 1) as usize].y = newpoints[0].y;
        newpoints[(numpoints - 1) as usize].z = co_m.z;
    }

    cap_points.pts = newpoints;
    1
}

/// Original: `skeletonize` (`skeletonize.c:573`).
pub fn skeletonize(
    cont: &Icont,
    scan_cont: Option<&Icont>,
    co_m: &Ipoint,
    cap_points: &mut Icont,
) -> i32 {
    let mut low_left = Ipoint::default();
    let mut up_right = Ipoint::default();
    let mut rc: i32;

    let owned_scan;
    let scan_cont: &Icont = match scan_cont {
        Some(sc) => sc,
        None => {
            owned_scan = imodel_contour_scan(Some(cont));
            match owned_scan.as_ref() {
                Some(sc) => sc,
                None => return 0,
            }
        }
    };

    imod_contour_get_bbox(Some(scan_cont), &mut low_left, &mut up_right);

    let fimin = low_left.x;
    let fimax = up_right.x;
    let fjmin = low_left.y;
    let fjmax = up_right.y;

    let mut sizex = fimax - fimin;
    let mut sizey = fjmax - fjmin;
    if sizex < 0. {
        sizex = -sizex;
    }
    if sizey < 0. {
        sizey = -sizey;
    }

    let isizex = ((sizex as f64).ceil() + 1.) as i32;
    let isizey = ((sizey as f64).ceil() + 1.) as i32;

    let imin = (fimin as f64).floor() as i32;
    let jmin = (fjmin as f64).floor() as i32;

    let mut divisor: i32 = 1;
    let mut downsized_x = (isizex + (divisor - 1)) / divisor;
    let mut downsized_y = (isizey + (divisor - 1)) / divisor;

    // Downsize with a factor of 2 when the image is larger than 100x100
    if downsized_x > 100 && downsized_y > 100 {
        divisor += 1;
        downsized_x = (isizex + (divisor - 1)) / divisor;
        downsized_y = (isizey + (divisor - 1)) / divisor;
    }

    downsized_x += 2; // black borders
    downsized_y += 2;
    let mut bitmap: Vec<u8> = vec![0u8; (downsized_x * downsized_y) as usize];

    for i in 0..(scan_cont.pts.len() as i32 / 2) as usize {
        let numpix = 1 + lroundf(scan_cont.pts[2 * i + 1].x) - lroundf(scan_cont.pts[2 * i].x);
        let scanx = 1 + lroundf(scan_cont.pts[2 * i].x) - imin; // black borders
        let scany = 1 + lroundf(scan_cont.pts[2 * i].y) - jmin;
        for j in 0..numpix / divisor {
            bitmap[(scany / divisor * downsized_x + scanx / divisor + j) as usize] = 1;
        }
    }

    let mut skeleton: Option<Vec<u8>> = None;
    rc = slice_medial_axis(&bitmap, &mut skeleton, downsized_x, downsized_y);

    match skeleton.as_mut() {
        Some(skel) => {
            rc = skeleton_to_contour(
                skel,
                downsized_x,
                downsized_y,
                imin,
                jmin,
                isizex,
                isizey,
                co_m,
                cap_points,
            );
        }
        None => {
            bitmap.clear();
            return 0;
        }
    }

    rc
}

/// C `lroundf`: round to nearest, halfway away from zero -- which is exactly
/// what `f32::round` does.
fn lroundf(value: f32) -> i32 {
    value.round() as i32
}
