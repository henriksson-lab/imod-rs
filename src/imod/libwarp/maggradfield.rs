//! Translation of `IMOD/libwarp/maggradfield.c`.
#![allow(dead_code)]

use crate::imod::libwarp::warputils::interpolate_grid;

/// Original `magGradientShift` (`maggradfield.c:112`).
pub unsafe fn mag_gradient_shift(
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
    dx: *mut f32,
    dy: *mut f32,
) {
    unsafe {
        // `#define DTOR 0.017453293` (`maggradfield.c:88`) is a **double**
        // literal and `cos`/`sin`/`tan` are the double libm functions, so
        // every trig value here is computed in double and rounded once on
        // store; `imageNx / 2.` is likewise a double.
        let dtor = 0.017453293_f64;
        // get trig values for the rotation to axis and tilt angle
        let cosphi = ((dtor * axis_rot as f64).cos() * pixel_size as f64 / 10000.) as f32;
        let sinphi = ((dtor * axis_rot as f64).sin() * pixel_size as f64 / 10000.) as f32;
        let tantheta = (dtor * tilt as f64).tan() as f32;
        //
        // compute the location of this point relative to the mag center and
        // its vertical height and thus rotation and mag
        //
        let mut xrel = xx - xcen;
        let mut yrel = yy - ycen;
        let zh = tantheta * (xrel * cosphi + yrel * sinphi);
        let sinrz = (dtor * rot_per_um as f64 * zh as f64).sin() as f32;
        let cosrz = (dtor * rot_per_um as f64 * zh as f64).cos() as f32;
        let gmag = (1. + 0.01 * dmag_per_um as f64 * zh as f64) as f32;
        //
        // have to mag around the center of this picture so do transform
        // relative to that to get dx, dy
        //
        xrel = (xx as f64 - image_nx as f64 / 2.) as f32;
        yrel = (yy as f64 - image_ny as f64 / 2.) as f32;
        *dx = (((xrel * cosrz - yrel * sinrz) * gmag) as f64 + image_nx as f64 / 2. - xx as f64)
            as f32;
        *dy = (((xrel * sinrz + yrel * cosrz) * gmag) as f64 + image_ny as f64 / 2. - yy as f64)
            as f32;
    }
}

/// Original `addMagGradField` (`maggradfield.c:62`).
pub unsafe fn add_mag_grad_field(
    idf_dx: *mut f32,
    idf_dy: *mut f32,
    grad_dx: *mut f32,
    grad_dy: *mut f32,
    lm_grid: i32,
    image_nx: i32,
    image_ny: i32,
    nx_grid: i32,
    ny_grid: i32,
    x_grid_start: f32,
    y_grid_start: f32,
    x_grid_interval: f32,
    y_grid_interval: f32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
) {
    unsafe {
        for ix in 0..nx_grid {
            for iy in 0..ny_grid {
                let xx = x_grid_start + ix as f32 * x_grid_interval;
                let yy = y_grid_start + iy as f32 * y_grid_interval;
                let mut dx = 0.;
                let mut dy = 0.;
                let mut dx_two = 0.;
                let mut dy_two = 0.;
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
                interpolate_grid(
                    xx + dx,
                    yy + dy,
                    idf_dx,
                    idf_dy,
                    lm_grid,
                    nx_grid,
                    ny_grid,
                    x_grid_start,
                    y_grid_start,
                    x_grid_interval,
                    y_grid_interval,
                    &mut dx_two,
                    &mut dy_two,
                );
                *grad_dx.add((ix + iy * lm_grid) as usize) = dx + dx_two;
                *grad_dy.add((ix + iy * lm_grid) as usize) = dy + dy_two;
            }
        }
    }
}

/// Original `makeMagGradField` (`maggradfield.c:34`).
pub unsafe fn make_mag_grad_field(
    idf_dx: *mut f32,
    idf_dy: *mut f32,
    grad_dx: *mut f32,
    grad_dy: *mut f32,
    lm_grid: i32,
    image_nx: i32,
    image_ny: i32,
    nx_grid: *mut i32,
    ny_grid: *mut i32,
    x_grid_start: *mut f32,
    y_grid_start: *mut f32,
    x_grid_interval: *mut f32,
    y_grid_interval: *mut f32,
    xcen: f32,
    ycen: f32,
    pixel_size: f32,
    axis_rot: f32,
    tilt: f32,
    dmag_per_um: f32,
    rot_per_um: f32,
) {
    unsafe {
        *nx_grid = lm_grid;
        *ny_grid = lm_grid;
        *x_grid_start = 1.;
        *y_grid_start = 1.;
        // `(imageNx - 1.) / (lmGrid - 1.)` is a double quotient.
        *x_grid_interval = ((image_nx as f64 - 1.) / (lm_grid as f64 - 1.)) as f32;
        *y_grid_interval = ((image_ny as f64 - 1.) / (lm_grid as f64 - 1.)) as f32;
        for index in 0..lm_grid * lm_grid {
            *idf_dx.add(index as usize) = 0.;
            *idf_dy.add(index as usize) = 0.;
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
            *x_grid_start,
            *y_grid_start,
            *x_grid_interval,
            *y_grid_interval,
            xcen,
            ycen,
            pixel_size,
            axis_rot,
            tilt,
            dmag_per_um,
            rot_per_um,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_gradient_makes_a_zero_source_field() {
        let mut i = [7.; 9];
        let mut j = [8.; 9];
        let mut x = [-1.; 9];
        let mut y = [-1.; 9];
        let (mut nx, mut ny, mut xs, mut ys, mut xi, mut yi) = (0, 0, 0., 0., 0., 0.);
        unsafe {
            make_mag_grad_field(
                i.as_mut_ptr(),
                j.as_mut_ptr(),
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                3,
                101,
                101,
                &mut nx,
                &mut ny,
                &mut xs,
                &mut ys,
                &mut xi,
                &mut yi,
                50.,
                50.,
                1.,
                0.,
                0.,
                0.,
                0.,
            );
        }
        assert_eq!((nx, ny, xs, ys, xi, yi), (3, 3, 1., 1., 50., 50.));
        assert_eq!(x, [0.; 9]);
        assert_eq!(y, [0.; 9]);
    }
}
