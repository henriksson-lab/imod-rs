//! Translation of `IMOD/libcfshr/autocorrpeaks.c`.
#![allow(dead_code)]

use super::b3dutil::{CArg, c_format};

pub const FIND_ACPK_NO_WAFFLE: i32 = 0x1;
pub const FIND_ACPK_BOTH_GEOMS: i32 = 0x2;
pub const FIND_ACPK_HEX_GRID: i32 = 0x4;
pub const FIND_ACPK_TILT_IN_VEC: i32 = 0x8;

/// Matches C \`findAutoCorrPeaks\` (\`autocorrpeaks.c:68\`).
pub fn find_auto_corr_peaks(
    array: &[f32],
    nx_pad: i32,
    ny_pad: i32,
    x_peaks: &mut [f32],
    y_peaks: &mut [f32],
    peak: &mut [f32],
    num_peaks: usize,
    max_scan: i32,
    mut catal_fac: f32,
    flags: i32,
    marked_x: f32,
    marked_y: f32,
    dist1_ptr: &mut f32,
    dist2_ptr: &mut f32,
    angle: &mut f32,
    vectors: &mut [f32],
    num: &mut [i32],
    near_ind: &mut i32,
    mess_buf: &mut String,
    buf_size: i32,
) -> i32 {
    assert!(array.len() >= ((nx_pad + 2) * ny_pad) as usize);
    // The original unconditionally samples the first nine detected peaks.
    assert!(num_peaks >= 9);
    assert!(x_peaks.len() >= num_peaks);
    assert!(y_peaks.len() >= num_peaks);
    assert!(peak.len() >= num_peaks);
    assert!(vectors.len() >= 6);
    assert!(num.len() >= 3);
    let no_perp_median = catal_fac != 1. || flags & FIND_ACPK_NO_WAFFLE != 0;
    let num_facs = if flags & (FIND_ACPK_BOTH_GEOMS | FIND_ACPK_HEX_GRID) != 0 {
        3
    } else {
        1
    };
    let adjust_for_tilt = flags & FIND_ACPK_TILT_IN_VEC != 0;
    let (mut fac_start, mut both_geom, mut crit_add) = (0, false, 0.02_f32);
    let mut sin60 = 0.;
    if num_facs > 1 {
        sin60 = (3_f64.sqrt() / 2.) as f32;
        if flags & FIND_ACPK_BOTH_GEOMS != 0 {
            both_geom = true;
            crit_add = 0.03;
        } else {
            fac_start = 1;
        }
    }
    let mut num_found = num_peaks as i32;
    let array_len = ((nx_pad + 2) * ny_pad) as usize;
    if crate::imod::libcfshr::filtxcorr::find_many_xcorr_peaks(
        &array[..array_len],
        nx_pad + 2,
        ny_pad,
        -3,
        -1,
        &mut x_peaks[..num_peaks],
        &mut y_peaks[..num_peaks],
        &mut peak[..num_peaks],
        num_peaks as i32,
        (1.5 * num_peaks as f64) as i32,
        &mut num_found,
    ) != 0
    {
        crate::imod::libcfshr::filtxcorr::xcorr_peak_find(
            &array[..array_len],
            nx_pad + 2,
            ny_pad,
            &mut x_peaks[..num_peaks],
            &mut y_peaks[..num_peaks],
            &mut peak[..num_peaks],
            num_peaks as i32,
        );
    }
    let (tilt_angle, axis_angle) = if adjust_for_tilt {
        (vectors[0], vectors[1])
    } else {
        (0., 0.)
    };
    let mut del_x = [0_f32; 4];
    let mut del_y = [0_f32; 4];
    let mut del_x_zt = [0_f32; 4];
    let mut del_y_zt = [0_f32; 4];
    let mut ind_far = [0_i32; 4];
    let mut ind2_for_fac = [0_i32; 3];
    let mut num_for_fac = [0_i32; 3];
    let mut angle_for_fac = [0_f32; 3];
    let mut dist2_for_fac = [0_f32; 3];
    let mut angle_err = [0_f32; 3];
    let mut dist_err = [0_f32; 3];
    let mut len_ratio_for_fac = [0_f32; 3];
    let mut ind1;
    let mut dist1 = 0.0_f64;
    if marked_x == 0. && marked_y == 0. {
        let (mut count, mut mean_dist) = (0, 0_f64);
        for index in 0..9_usize {
            if peak[index] > -1e29 {
                let distance = ((x_peaks[index] * x_peaks[index] + y_peaks[index] * y_peaks[index])
                    as f64)
                    .sqrt();
                if distance > 1. {
                    count += 1;
                    mean_dist += distance;
                }
            }
        }
        let mut num_scan = 8;
        if count > 0 {
            num_scan = (1.3 * nx_pad.min(ny_pad) as f64 / (mean_dist / count as f64)) as i32;
            num_scan = 8.max(max_scan.min(num_scan));
        }
        num_scan = num_scan.min(num_found);
        let mut delp_x = [0_f32; 64];
        let mut delp_y = [0_f32; 64];
        let mut perp_med = [0_f32; 64];
        let mut indfarp = [0_i32; 64];
        let mut indp = [0_i32; 64];
        let mut nump = [0_i32; 64];
        let mut distp = [0_f64; 64];
        let (mut isc, mut maximum_total, mut minimum_distance, mut maximum_perp) =
            (-1_i32, 0_f64, 1e10_f64, -1e37_f32);
        ind1 = -1;
        for index in 0..num_scan {
            if !(peak[index as usize] > -1e29
                && x_peaks[index as usize] >= 0.
                && ((x_peaks[index as usize] as f64).abs() > 2.5
                    || (y_peaks[index as usize] as f64).abs() > 2.5))
            {
                continue;
            }
            isc += 1;
            if isc >= 64 {
                isc -= 1;
                break;
            }
            let scan = isc as usize;
            let mut ind2 = 0;
            let mut variation = 0.;
            delp_x[scan] = x_peaks[index as usize];
            delp_y[scan] = y_peaks[index as usize];
            indfarp[scan] = index;
            indp[scan] = index;
            find_peak_series(
                x_peaks,
                y_peaks,
                peak,
                &mut delp_x[scan],
                &mut delp_y[scan],
                &mut indfarp[scan],
                &mut nump[scan],
                &mut ind2,
                &mut variation,
                crit_add,
            );
            if !no_perp_median {
                perp_med[scan] =
                    perpendicular_line_median(array, nx_pad, ny_pad, delp_x[scan], delp_y[scan])
                        as f32;
            }
            distp[scan] =
                ((delp_x[scan] * delp_x[scan] + delp_y[scan] * delp_y[scan]) as f64).sqrt();
            let total_distance = nump[scan] as f64 * distp[scan];
            for divisor in 2..=((distp[scan] / 3.) as i32) {
                let (mut try_x, mut try_y) = (
                    x_peaks[index as usize] / divisor as f32,
                    y_peaks[index as usize] / divisor as f32,
                );
                let (mut ind_try, mut num_try, mut var_try) = (-1, 0, 0.);
                find_peak_series(
                    x_peaks,
                    y_peaks,
                    peak,
                    &mut try_x,
                    &mut try_y,
                    &mut ind_try,
                    &mut num_try,
                    &mut ind2,
                    &mut var_try,
                    crit_add,
                );
                let try_median = if ind_try >= 0 && !no_perp_median {
                    perpendicular_line_median(array, nx_pad, ny_pad, try_x, try_y) as f32
                } else {
                    0.
                };
                let try_distance = ((try_x * try_x + try_y * try_y) as f64).sqrt();
                if ind_try >= 0
                    && num_try >= divisor
                    && try_distance * num_try as f64 >= 0.75 * total_distance
                    && var_try < 2.5 * variation
                    && (no_perp_median || try_median > 0.2 * maximum_perp.max(perp_med[scan]))
                {
                    distp[scan] = try_distance;
                    delp_x[scan] = try_x;
                    delp_y[scan] = try_y;
                    indfarp[scan] = ind_try;
                    nump[scan] = num_try;
                    indp[scan] = ind2;
                    if !no_perp_median {
                        perp_med[scan] = try_median;
                    }
                }
            }
            if indp[scan] >= 0 {
                maximum_total = maximum_total.max(nump[scan] as f64 * distp[scan]);
                if !no_perp_median {
                    maximum_perp = maximum_perp.max(perp_med[scan]);
                }
            }
        }
        if isc >= 0 {
            for scan in 0..=isc as usize {
                let total_distance = nump[scan] as f64 * distp[scan];
                if distp[scan] <= minimum_distance
                    && total_distance >= 0.5 * maximum_total
                    && (no_perp_median || perp_med[scan] > 0.2 * maximum_perp)
                {
                    dist1 = distp[scan];
                    del_x[0] = delp_x[scan];
                    del_y[0] = delp_y[scan];
                    ind_far[0] = indfarp[scan];
                    num[0] = nump[scan];
                    ind1 = indp[scan];
                    minimum_distance = distp[scan];
                }
            }
        } else {
            dist1 = 0.;
        }
    } else {
        ind1 = closest_right_side_peak(x_peaks, y_peaks, peak, marked_x, marked_y, -1e29);
        if ind1 >= 0 {
            ind_far[0] = ind1;
            del_x[0] = x_peaks[ind1 as usize];
            del_y[0] = y_peaks[ind1 as usize];
            let mut variation = 0.;
            find_peak_series(
                x_peaks,
                y_peaks,
                peak,
                &mut del_x[0],
                &mut del_y[0],
                &mut ind_far[0],
                &mut num[0],
                &mut ind1,
                &mut variation,
                crit_add,
            );
        }
        dist1 = ((del_x[0] * del_x[0] + del_y[0] * del_y[0]) as f64).sqrt();
    }
    if ind1 < 0 {
        /* snprintf(messBuf, bufSize, ...) truncates at bufSize - 1 bytes. */
        *mess_buf = c_format(
            "Did not find a peak away from the center of the autocorrelation",
            &[],
        );
        mess_buf.truncate((buf_size.max(1) as usize - 1).min(mess_buf.len()));
        return -1;
    }
    *near_ind = ind1;
    del_x_zt[0] = del_x[0];
    del_y_zt[0] = del_y[0];
    if adjust_for_tilt {
        adjust_vector_for_tilt(
            del_x[0],
            del_y[0],
            tilt_angle,
            axis_angle,
            0,
            &mut del_x_zt[0],
            &mut del_y_zt[0],
        );
    }
    let dist1_zt = ((del_x_zt[0] * del_x_zt[0] + del_y_zt[0] * del_y_zt[0]) as f64).sqrt() as f32;
    let min_peak = if num_facs > 1 {
        peak[ind1 as usize] * 0.25
    } else {
        -1e29
    };
    let aspect_ratio = (nx_pad as f32 / ny_pad as f32).min(ny_pad as f32 / nx_pad as f32);
    for ifac in fac_start..num_facs {
        let (mut vec_x, mut vec_y) = if ifac != 0 {
            let direction = (3 - 2 * ifac) as f32;
            (
                0.5 * del_x_zt[0] - direction * sin60 * del_y_zt[0],
                direction * sin60 * del_x_zt[0] + 0.5 * del_y_zt[0],
            )
        } else {
            (catal_fac * del_y_zt[0], -catal_fac * del_x_zt[0])
        };
        if adjust_for_tilt {
            adjust_vector_for_tilt(
                vec_x, vec_y, tilt_angle, axis_angle, 1, &mut vec_x, &mut vec_y,
            );
        }
        let slot = ifac as usize;
        ind2_for_fac[slot] =
            closest_right_side_peak(x_peaks, y_peaks, peak, vec_x, vec_y, min_peak);
        let ind2 = ind2_for_fac[slot];
        if ind2 < 0 {
            continue;
        }
        vec_x = x_peaks[ind2 as usize];
        vec_y = y_peaks[ind2 as usize];
        if adjust_for_tilt {
            adjust_vector_for_tilt(
                vec_x, vec_y, tilt_angle, axis_angle, 0, &mut vec_x, &mut vec_y,
            );
        }
        let dist2 = ((x_peaks[ind2 as usize] * x_peaks[ind2 as usize]
            + y_peaks[ind2 as usize] * y_peaks[ind2 as usize]) as f64)
            .sqrt();
        let mut dist2_zt = ((vec_x * vec_x + vec_y * vec_y) as f64).sqrt() as f32;
        let dot = del_x_zt[0] * vec_x + del_y_zt[0] * vec_y;
        let mut angle_fac = (dot / (dist1_zt * dist2_zt)).acos() / (core::f32::consts::PI / 180.);
        let mut angle_error = if ifac != 0 {
            (angle_fac - 60.).abs().min((angle_fac - 120.).abs())
        } else {
            (angle_fac - 90.).abs()
        };
        let mut distance_error =
            (catal_fac * dist1_zt - dist2_zt).abs() / (catal_fac * dist1_zt).max(dist2_zt);
        del_x[(1 + ifac) as usize] = x_peaks[ind2 as usize];
        del_y[(1 + ifac) as usize] = y_peaks[ind2 as usize];
        ind_far[(1 + ifac) as usize] = ind2;
        let mut indst = 0;
        let mut variation = 0.;
        find_peak_series(
            x_peaks,
            y_peaks,
            peak,
            &mut del_x[(1 + ifac) as usize],
            &mut del_y[(1 + ifac) as usize],
            &mut ind_far[(1 + ifac) as usize],
            &mut num_for_fac[slot],
            &mut indst,
            &mut variation,
            crit_add,
        );
        let len_ratio = (num_for_fac[slot] as f64 * dist2
            / (aspect_ratio as f64 * num[0] as f64 * dist1)) as f32;
        angle_for_fac[slot] = angle_fac;
        dist2_for_fac[slot] = dist2_zt;
        angle_err[slot] = angle_error;
        dist_err[slot] = distance_error;
        len_ratio_for_fac[slot] = len_ratio;
        catal_fac = 1.;
    }
    let mut ifac = fac_start as usize;
    if ind2_for_fac[ifac] < 0 && (!both_geom || ind2_for_fac[1] < 0 || ind2_for_fac[2] < 0) {
        *mess_buf = c_format(
            "Did not find another peak at proper angle away from best one",
            &[],
        );
        mess_buf.truncate((buf_size.max(1) as usize - 1).min(mess_buf.len()));
        return 1;
    }
    if both_geom && ind2_for_fac[0] < 0 {
        ifac = 1;
        both_geom = false;
    }
    if both_geom && (ind2_for_fac[1] < 0 || ind2_for_fac[2] < 0) {
        both_geom = false;
    }
    if both_geom
        && len_ratio_for_fac[0] < 0.5
        && (len_ratio_for_fac[1] + len_ratio_for_fac[2]) / 2. > 0.5
    {
        ifac = 1;
        both_geom = false;
    }
    if both_geom
        && (len_ratio_for_fac[1] + len_ratio_for_fac[2]) / 2. < 0.5
        && len_ratio_for_fac[0] > 0.5
    {
        both_geom = false;
    }
    if (angle_err[ifac] > 10. || (fac_start != 0 && angle_err[2] > 10.))
        && (!both_geom || angle_err[1] > 10. || angle_err[2] > 10.)
    {
        if fac_start != 0 && angle_err[2] > angle_err[1] {
            ifac = 2;
        }
        *mess_buf = c_format(
            "The angle between the two peaks that were found\nis %.1f degrees, not close enough to %d degrees",
            &[
                CArg::Dbl(angle_for_fac[ifac] as f64),
                CArg::Int(if fac_start != 0 { 60 } else { 90 }),
            ],
        );
        mess_buf.truncate((buf_size.max(1) as usize - 1).min(mess_buf.len()));
        return 1;
    }
    if angle_err[ifac] > 10. {
        ifac = 1;
        both_geom = false;
    }
    if both_geom && (angle_err[1] > 10. || angle_err[2] > 10.) {
        both_geom = false;
    }
    if (dist_err[ifac] > 0.2 || (fac_start != 0 && dist_err[2] > 0.2))
        && (!both_geom || dist_err[1] > 0.2 || dist_err[2] > 0.2)
    {
        if fac_start != 0 && dist_err[2] > dist_err[1] {
            ifac = 2;
        }
        *mess_buf = c_format(
            "The distances from the center to the two peaks that were found,\n%.1f and %.1f, differ by more than 20%%",
            &[CArg::Dbl(dist1), CArg::Dbl(dist2_for_fac[ifac] as f64)],
        );
        mess_buf.truncate((buf_size.max(1) as usize - 1).min(mess_buf.len()));
        return 1;
    }
    if dist_err[ifac] > 0.2 || (both_geom && (dist_err[1] + dist_err[2]) / 2. < dist_err[0]) {
        ifac = 1;
    }
    del_x[1] = del_x[1 + ifac];
    del_y[1] = del_y[1 + ifac];
    num[1] = num_for_fac[ifac];
    *dist1_ptr = dist1 as f32;
    *dist2_ptr = dist2_for_fac[ifac];
    if num_facs > 1 {
        num[2] = 0;
    }
    if ifac != 0 {
        num[2] = num_for_fac[2];
        vectors[4] = del_x[3];
        vectors[5] = del_y[3];
        *angle = del_y[0].atan2(del_x[0]) / (core::f32::consts::PI / 180.);
    } else {
        *angle = -0.5
            * ((del_y[0].atan2(del_x[0]) + del_y[1].atan2(del_x[1]))
                / (core::f32::consts::PI / 180.)
                - 90.);
    }
    vectors[0] = del_x[0];
    vectors[1] = del_y[0];
    vectors[2] = del_x[1];
    vectors[3] = del_y[1];
    0
}

/// `findPeakSeries` (`autocorrpeaks.c:426`).
fn find_peak_series(
    x: &[f32],
    y: &[f32],
    peak: &[f32],
    dx: &mut f32,
    dy: &mut f32,
    far: &mut i32,
    count: &mut i32,
    near: &mut i32,
    variation: &mut f32,
    crit_add: f32,
) {
    let original = (*dx * *dx + *dy * *dy).sqrt();
    let criterion = (1. + crit_add * original).powi(2);
    let mut list = [0_i32; 10];
    let mut nlist = 0;
    *count = 0;
    let (mut lo, mut hi, mut last) = (0., 0., 0.);
    let mut diff = 0.;
    if *far >= 0 {
        *count = 1;
        lo = peak[*far as usize];
        hi = lo;
        last = lo;
        list[0] = *far;
        nlist = 1;
    }
    *near = *far;
    loop {
        let (ex, ey) = ((*count + 1) as f32 * *dx, (*count + 1) as f32 * *dy);
        let mut found = -1;
        for i in 0..peak.len() as i32 {
            if peak[i as usize] < -1e-29
                || x[i as usize] < 0.
                || i == *far
                || (x[i as usize] < 1. && y[i as usize].abs() < 1.)
                || crate::imod::libcfshr::b3dutil::number_in_list(i, Some(&list), nlist, 0) != 0
            {
                continue;
            }
            let xx = x[i as usize] - ex;
            let yy = y[i as usize] - ey;
            if xx * xx + yy * yy < criterion {
                found = i;
                break;
            }
        }
        if found < 0 {
            break;
        }
        *far = found;
        if nlist < 10 {
            list[nlist as usize] = found;
            nlist += 1;
        }
        *count += 1;
        *dx = x[found as usize] / *count as f32;
        *dy = y[found as usize] / *count as f32;
        if *near < 0 {
            *near = *far;
            lo = peak[*far as usize];
            hi = lo;
            last = lo;
        } else {
            let p = peak[found as usize];
            lo = lo.min(p);
            hi = hi.max(p);
            diff += (p - last).abs();
            last = p;
        }
        if *dx * *dx + *dy * *dy < original * original / 2. {
            break;
        }
    }
    *variation = 1.;
    if *count > 2 && hi - lo > 1e-35 * diff {
        *variation = diff / (hi - lo);
    }
}
fn closest_right_side_peak(x: &[f32], y: &[f32], p: &[f32], dx: f32, dy: f32, minp: f32) -> i32 {
    let (mut out, mut best) = (-1, 1e30_f32);
    for i in 0..p.len() as i32 {
        if p[i as usize] < minp || x[i as usize] < 0. {
            continue;
        }
        let xx = x[i as usize];
        let yy = y[i as usize];
        if xx * xx + yy * yy < 2. {
            continue;
        }
        let a = (xx - dx).powi(2) + (yy - dy).powi(2);
        let b = (xx + dx).powi(2) + (yy + dy).powi(2);
        if a.min(b) < best {
            out = i;
            best = a.min(b);
        }
    }
    out
}
fn perpendicular_line_median(a: &[f32], nx: i32, ny: i32, x: f32, y: f32) -> f64 {
    let peak = (x * x + y * y).sqrt();
    let thick = 1. / 30.;
    let max = (0.72_f32).max(thick * peak);
    let maxsq = max * max;
    let size = (10_f32 / 2.).max(0.8 * peak) / peak;
    let vx = -size * y;
    let vy = size * x;
    let xs = x - vx;
    let ys = y - vy;
    let xl = 2. * vx;
    let yl = 2. * vy;
    let den = xl * xl + yl * yl;
    let mut v = Vec::with_capacity((maxsq / (thick * thick)) as usize);
    for iy in ((ys.min(y + vy) - max - 2. + 5e-1).floor() as i32)
        ..=((ys.max(y + vy) + max + 2. + 5e-1).floor() as i32)
    {
        for ix in ((xs.min(x + vx) - max - 2. + 5e-1).floor() as i32)
            ..=((xs.max(x + vx) + max + 2. + 5e-1).floor() as i32)
        {
            let t = ((xl * (ix as f32 - xs) + yl * (iy as f32 - ys)) / den).clamp(0., 1.);
            let xx = xl * t + xs - ix as f32;
            let yy = yl * t + ys - iy as f32;
            if xx * xx + yy * yy <= maxsq {
                let cx = if ix < 0 { ix + nx } else { ix };
                let cy = if iy < 0 { iy + ny } else { iy };
                v.push(a[(cx + cy * (nx + 2)) as usize]);
            }
        }
    }
    let mut median = -1e37;
    if v.len() > 4 {
        let n = v.len() as i32;
        crate::imod::libcfshr::robuststat::rs_fast_median_in_place(&mut v, n, &mut median);
    }
    median as f64
}
fn adjust_vector_for_tilt(
    dx: f32,
    dy: f32,
    tilt: f32,
    axis: f32,
    to: i32,
    x: &mut f32,
    y: &mut f32,
) {
    let c = (axis.to_radians()).cos();
    let s = (axis.to_radians()).sin();
    let xr = dx * c + dy * s;
    let mut yr = -dx * s + dy * c;
    if to != 0 {
        yr *= tilt.to_radians().cos()
    } else {
        yr /= tilt.to_radians().cos()
    }
    *x = xr * c - yr * s;
    *y = xr * s + yr * c;
}

#[cfg(test)]
mod tests {
    use super::{
        adjust_vector_for_tilt, closest_right_side_peak, find_auto_corr_peaks, find_peak_series,
    };

    #[test]
    fn peak_series_refines_a_regular_axis() {
        let x = [0., 4., 8., 12., 5.];
        let y = [0., 0., 0., 0., 4.];
        let peaks = [99., 8., 7., 6., 5.];
        let (mut dx, mut dy, mut far, mut count, mut near, mut variation) = (4., 0., 1, 0, -1, 0.);
        find_peak_series(
            &x,
            &y,
            &peaks,
            &mut dx,
            &mut dy,
            &mut far,
            &mut count,
            &mut near,
            &mut variation,
            0.02,
        );
        assert_eq!((far, count, near), (3, 3, 1));
        assert_eq!((dx, dy), (4., 0.));
        assert!((variation - 1.).abs() < 1.0e-6);
    }

    #[test]
    fn closest_peak_considers_the_mirrored_vector() {
        let x = [0., 4., 8., -4.];
        let y = [0., 0., 0., 0.];
        let peaks = [10., 2., 4., 9.];
        assert_eq!(closest_right_side_peak(&x, &y, &peaks, -4., 0., -1.0e29), 1);
    }

    #[test]
    fn tilt_adjustment_round_trips() {
        let (mut x, mut y) = (0., 0.);
        adjust_vector_for_tilt(7., -3., 40., 25., 0, &mut x, &mut y);
        let (mut round_x, mut round_y) = (0., 0.);
        adjust_vector_for_tilt(x, y, 40., 25., 1, &mut round_x, &mut round_y);
        assert!((round_x - 7.).abs() < 1.0e-5);
        assert!((round_y + 3.).abs() < 1.0e-5);
    }

    #[test]
    fn no_peaks_returns_the_source_error() {
        let mut image = [0_f32; 100];
        let mut x = [0_f32; 9];
        let mut y = [0_f32; 9];
        let mut peaks = [0_f32; 9];
        let (mut distance1, mut distance2, mut angle, mut near) = (0., 0., 0., 0);
        let mut vectors = [0_f32; 6];
        let mut counts = [0_i32; 3];
        let mut message = String::new();
        let status = find_auto_corr_peaks(
            &image,
            8,
            10,
            &mut x,
            &mut y,
            &mut peaks,
            9,
            8,
            1.,
            0,
            0.,
            0.,
            &mut distance1,
            &mut distance2,
            &mut angle,
            &mut vectors,
            &mut counts,
            &mut near,
            &mut message,
            200,
        );
        assert_eq!(status, -1);
        assert_eq!(
            message,
            "Did not find a peak away from the center of the autocorrelation"
        );
    }
}
