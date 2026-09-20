//! Translation of `IMOD/libcfshr/sdsearch.c`.

/// Matches `montSdCalc` (`IMOD/libcfshr/sdsearch.c:228`).
pub fn mont_sd_calc(
    array: &[f32],
    brray: &[f32],
    nx: usize,
    ny: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    dx: f32,
    dy: f32,
) -> (f32, f32) {
    let mut sum = 0_f64;
    let mut squares = 0_f64;
    let mut number = 0_i32;
    let xp = x1 as f32 + dx;
    let yp = y1 as f32 + dy;
    let dx_fraction = xp - xp.round();
    let dy_fraction = yp - yp.round();
    let ix = (dx - dx_fraction) as i32;
    let iy = (dy - dy_fraction) as i32;
    if dx_fraction == 0. && dy_fraction == 0. {
        for by in (y0 + iy).max(0)..=(y1 + iy).min(ny as i32 - 1) {
            for bx in (x0 + ix).max(0)..=(x1 + ix).min(nx as i32 - 1) {
                let difference = brray[(bx + by * nx as i32) as usize]
                    - array[(bx - ix + (by - iy) * nx as i32) as usize];
                sum += difference as f64;
                squares += difference as f64 * difference as f64;
                number += 1;
            }
        }
    } else {
        let (d2, x2) = (dy_fraction * dy_fraction, dx_fraction * dx_fraction);
        let (c8, c2, c6, c4, c5) = (
            0.5 * (d2 + dy_fraction),
            0.5 * (d2 - dy_fraction),
            0.5 * (x2 + dx_fraction),
            0.5 * (x2 - dx_fraction),
            1. - x2 - d2,
        );
        for by in (y0 + iy).max(1)..=(y1 + iy).min(ny as i32 - 2) {
            for bx in (x0 + ix).max(1)..=(x1 + ix).min(nx as i32 - 2) {
                let interpolation = c5 * brray[(bx + by * nx as i32) as usize]
                    + c4 * brray[(bx - 1 + by * nx as i32) as usize]
                    + c6 * brray[(bx + 1 + by * nx as i32) as usize]
                    + c2 * brray[(bx + (by - 1) * nx as i32) as usize]
                    + c8 * brray[(bx + (by + 1) * nx as i32) as usize];
                let difference = interpolation - array[(bx - ix + (by - iy) * nx as i32) as usize];
                sum += difference as f64;
                squares += difference as f64 * difference as f64;
                number += 1;
            }
        }
    }
    let density = if number > 0 {
        (sum / number as f64) as f32
    } else {
        0.
    };
    let sd = if number > 1 {
        ((squares - sum * sum / number as f64) / (number - 1) as f64).sqrt() as f32
    } else {
        9999.
    };
    (sd, density)
}
/// Matches `montsdcalc`.
pub fn montsdcalc(
    array: &[f32],
    brray: &[f32],
    nx: usize,
    ny: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    dx: f32,
    dy: f32,
) -> (f32, f32) {
    mont_sd_calc(
        array,
        brray,
        nx,
        ny,
        x0 as i32 - 1,
        y0 as i32 - 1,
        x1 as i32 - 1,
        y1 as i32 - 1,
        dx,
        dy,
    )
}
/// Matches `montBigSearch` (`IMOD/libcfshr/sdsearch.c:36`).
pub fn mont_big_search(
    array: &[f32],
    brray: &[f32],
    nx: usize,
    ny: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    dx: &mut f32,
    dy: &mut f32,
    sd: &mut f32,
    density: &mut f32,
    iterations: i32,
    limit: i32,
) {
    let (cx, cy) = (*dx, *dy);
    let steps = 2_i32.pow((iterations - 1).max(0) as u32);
    let unit = 1. / steps as f32;
    let bound = (steps * limit).min(100);
    (*sd, *density) = mont_sd_calc(array, brray, nx, ny, x0, y0, x1, y1, *dx, *dy);
    let (mut bx, mut by) = (0, 0);
    for ix in -bound..=bound {
        for iy in -bound..=bound {
            let (candidate, d) = mont_sd_calc(
                array,
                brray,
                nx,
                ny,
                x0,
                y0,
                x1,
                y1,
                cx + ix as f32 * unit,
                cy + iy as f32 * unit,
            );
            if candidate < *sd {
                *sd = candidate;
                *density = d;
                bx = ix;
                by = iy;
            }
        }
    }
    *dx = cx + bx as f32 * unit;
    *dy = cy + by as f32 * unit;
}
/// Matches `montbigsearch`.
pub fn montbigsearch(
    array: &[f32],
    brray: &[f32],
    nx: usize,
    ny: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    dx: &mut f32,
    dy: &mut f32,
    sd: &mut f32,
    density: &mut f32,
    iterations: i32,
    limit: i32,
) {
    mont_big_search(
        array,
        brray,
        nx,
        ny,
        x0 as i32 - 1,
        y0 as i32 - 1,
        x1 as i32 - 1,
        y1 as i32 - 1,
        dx,
        dy,
        sd,
        density,
        iterations,
        limit,
    )
}
