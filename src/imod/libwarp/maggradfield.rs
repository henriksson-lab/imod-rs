//! Translation of `IMOD/libwarp/maggradfield.c`.
//!
//! Converted to idiomatic Rust per `NATIVE.md`: every `float *` is a slice and
//! the returned-by-pointer scalars are `&mut`.  The arithmetic is untouched —
//! `DTOR`, `2.`, `10000.`, `0.01` and `1.` are all **double** literals in the
//! source, so each of those expressions is evaluated in double and rounds once
//! on the store into a `float`.

use crate::imod::libwarp::warputils::interpolate_grid;

/// C `DTOR` (`maggradfield.c:88`).
const DTOR: f64 = 0.017453293;

/// Original `magGradientShift` (`maggradfield.c:100`).
#[allow(clippy::too_many_arguments)]
pub fn mag_gradient_shift(
    xx: f32,
    yy: f32,
    image_nx: i32,
    image_ny: i32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
    dx: &mut f32,
    dy: &mut f32,
) {
    /* get trig values for the rotation to axis and tilt angle */
    let cosphi = ((DTOR * axis_rot as f64).cos() * pixel_size as f64 / 10000.) as f32;
    let sinphi = ((DTOR * axis_rot as f64).sin() * pixel_size as f64 / 10000.) as f32;
    let tantheta = (DTOR * tilt as f64).tan() as f32;

    /* compute the location of this point relative to the mag center and its vertical
    height and thus rotation and mag */
    let mut xrel = xx - xcen;
    let mut yrel = yy - ycen;
    let zh = tantheta * (xrel * cosphi + yrel * sinphi);
    let sinrz = (DTOR * rot_per_um as f64 * zh as f64).sin() as f32;
    let cosrz = (DTOR * rot_per_um as f64 * zh as f64).cos() as f32;
    let gmag = (1. + 0.01 * dmag_per_um as f64 * zh as f64) as f32;

    /* have to mag around the center of this picture so do transform relative to that to
    get dx, dy */
    xrel = (xx as f64 - image_nx as f64 / 2.) as f32;
    yrel = (yy as f64 - image_ny as f64 / 2.) as f32;
    *dx = (((xrel * cosrz - yrel * sinrz) * gmag) as f64 + image_nx as f64 / 2. - xx as f64) as f32;
    *dy = (((xrel * sinrz + yrel * cosrz) * gmag) as f64 + image_ny as f64 / 2. - yy as f64) as f32;
}

/// Original `addMagGradField` (`maggradfield.c:62`).
#[allow(clippy::too_many_arguments)]
pub fn add_mag_grad_field(
    idf_dx: &[f32],
    idf_dy: &[f32],
    grad_dx: &mut [f32],
    grad_dy: &mut [f32],
    lm_grid: i32,
    image_nx: i32,
    image_ny: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_strt: f32,
    y_grid_strt: f32,
    x_grid_intrv: f32,
    y_grid_intrv: f32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
) {
    for ix in 0..nx_grid {
        for iy in 0..ny_grid {
            let xx = x_grid_strt + ix as f32 * x_grid_intrv;
            let yy = y_grid_strt + iy as f32 * y_grid_intrv;

            /* Get the shift due to the mag gradient, then look up the distortion field at
            this point and add the two shifts to get the total field */
            let mut dx = 0.0_f32;
            let mut dy = 0.0_f32;
            mag_gradient_shift(
                xx,
                yy,
                image_nx,
                image_ny,
                xcen,
                ycen,
                pixel_size,
                axis_rot,
                tilt,
                dmag_per_um,
                rot_per_um,
                &mut dx,
                &mut dy,
            );
            let mut dx2 = 0.0_f32;
            let mut dy2 = 0.0_f32;
            interpolate_grid(
                xx + dx,
                yy + dy,
                idf_dx,
                idf_dy,
                lm_grid,
                nx_grid,
                ny_grid,
                x_grid_strt,
                y_grid_strt,
                x_grid_intrv,
                y_grid_intrv,
                &mut dx2,
                &mut dy2,
            );
            grad_dx[(ix + iy * lm_grid) as usize] = dx + dx2;
            grad_dy[(ix + iy * lm_grid) as usize] = dy + dy2;
        }
    }
}

/// Original `makeMagGradField` (`maggradfield.c:34`).
#[allow(clippy::too_many_arguments)]
pub fn make_mag_grad_field(
    idf_dx: &mut [f32],
    idf_dy: &mut [f32],
    grad_dx: &mut [f32],
    grad_dy: &mut [f32],
    lm_grid: i32,
    image_nx: i32,
    image_ny: i32,
    nx_grid: &mut i32,
    ny_grid: &mut i32,
    x_grid_strt: &mut f32,
    y_grid_strt: &mut f32,
    x_grid_intrv: &mut f32,
    y_grid_intrv: &mut f32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
) {
    /* Set up a grid that has maximum resolution for the array size */
    *nx_grid = lm_grid;
    *ny_grid = lm_grid;
    *x_grid_strt = 1.;
    *y_grid_strt = 1.;
    // `(imageNx - 1.) / (lmGrid - 1.)` is a double quotient.
    *x_grid_intrv = ((image_nx as f64 - 1.) / (lm_grid as f64 - 1.)) as f32;
    *y_grid_intrv = ((image_ny as f64 - 1.) / (lm_grid as f64 - 1.)) as f32;
    for i in 0..(lm_grid * lm_grid) as usize {
        idf_dx[i] = 0.;
        idf_dy[i] = 0.;
    }
    add_mag_grad_field(
        idf_dx,
        idf_dy,
        grad_dx,
        grad_dy,
        lm_grid,
        image_nx,
        image_ny,
        *nx_grid,
        *ny_grid,
        *x_grid_strt,
        *y_grid_strt,
        *x_grid_intrv,
        *y_grid_intrv,
        xcen,
        ycen,
        pixel_size,
        axis_rot,
        tilt,
        dmag_per_um,
        rot_per_um,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_gradient_makes_a_zero_source_field() {
        let mut i = [7.0_f32; 9];
        let mut j = [8.0_f32; 9];
        let mut x = [-1.0_f32; 9];
        let mut y = [-1.0_f32; 9];
        let (mut nx, mut ny, mut xs, mut ys, mut xi, mut yi) = (0, 0, 0., 0., 0., 0.);
        make_mag_grad_field(
            &mut i, &mut j, &mut x, &mut y, 3, 101, 101, &mut nx, &mut ny, &mut xs, &mut ys,
            &mut xi, &mut yi, 50., 50., 1., 0., 0., 0., 0.,
        );
        assert_eq!((nx, ny, xs, ys, xi, yi), (3, 3, 1., 1., 50., 50.));
        assert_eq!(x, [0.; 9]);
        assert_eq!(y, [0.; 9]);
    }
}
