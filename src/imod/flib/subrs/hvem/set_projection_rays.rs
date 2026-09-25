//! Translation of `IMOD/flib/subrs/hvem/set_projection_rays.f90`.

/// Original `set_projection_rays` (`set_projection_rays.f90:14`).
///
/// Sets up information about projection rays at an angle whose sine and
/// cosine are [sinang] and [cosang], through
/// an image slice whose size is [nxslice] by [nyslice].  The number of
/// rays, i.e. the size of centered output in X, is given by [nxout].
/// The arrays [xraystr] and [yraystr] are returned with the coordinates
/// at which to start each ray, the array [nrayinc] is returned with the
/// number of points in each ray, and [nraymax] is returned with the
/// maximum number of points.  [sinang] and [cosang] may be modified for
/// projections very near vertical or horizontal.
///
/// `ixOut` runs 1-based as in the source; the three output arrays are
/// indexed `ixOut - 1`.  `xgood`/`ygood` are local arrays without `SAVE`,
/// so they keep their contents from one ray to the next within a call, and
/// the source reads `ygood(2)` (and `xgood(2)`) even when only one
/// intersection was stored.  They are declared once per call here, which
/// preserves the carry-over between rays; the values before the first
/// store, uninitialised stack in the source, start as zero.
#[allow(clippy::too_many_arguments)]
pub fn set_projection_rays(
    sin_ang: &mut f32,
    cos_ang: &mut f32,
    nx_slice: i32,
    ny_slice: i32,
    nx_out: i32,
    xray_start: &mut [f32],
    yray_start: &mut [f32],
    num_ray_increm: &mut [i32],
    nray_max: &mut i32,
) {
    let mut xgood = [0f32; 4];
    let mut ygood = [0f32; 4];
    //
    *nray_max = 0;
    for ix_out in 1..=nx_out {
        let mut nray_tmp: i32 = 0;
        let mut xray_tmp: f32 = 0.;
        let mut yray_tmp: f32 = 0.;
        //
        // if a near-vertical projection, set up to be exactly vertical
        // (limit was 0.01, try it at 0.001)
        if sin_ang.abs() < 0.001 {
            *sin_ang = 0.;
            *cos_ang = 1f32.copysign(*cos_ang);
            if *cos_ang > 0. {
                yray_tmp = 2.;
                xray_tmp = (nx_slice / 2 + ix_out - nx_out / 2) as f32;
            } else {
                yray_tmp = ny_slice as f32 - 1.;
                xray_tmp = (nx_slice / 2 + 1 + nx_out / 2 - ix_out) as f32;
            }
            if xray_tmp >= 1. && xray_tmp <= nx_slice as f32 {
                nray_tmp = ny_slice - 2;
            }
            //
            // if a near-horizontal projection, set up to be exactly horizontal
            //
        } else if cos_ang.abs() < 0.001 {
            *sin_ang = 1f32.copysign(*sin_ang);
            *cos_ang = 0.;
            if *sin_ang > 0. {
                xray_tmp = 2.;
                yray_tmp = (ny_slice / 2 + 1 + nx_out / 2 - ix_out) as f32;
            } else {
                xray_tmp = nx_slice as f32 - 1.;
                yray_tmp = (ny_slice / 2 + ix_out - nx_out / 2) as f32;
            }
            if yray_tmp >= 1. && yray_tmp <= ny_slice as f32 {
                nray_tmp = nx_slice - 2;
            }
            //
            // otherwise need to look at intersections with slice box
            //
        } else {
            let mut num_good_inter: usize = 0;
            let tan_ang = *sin_ang / *cos_ang;
            let ray_intercept = -(ix_out as f32 - 0.5 - (nx_out / 2) as f32) / *sin_ang;
            //
            // coordinates of edges of slice box
            //
            let xleft = 1.01f32 - (nx_slice / 2) as f32;
            let xright = nx_slice as f32 - 1.01 - (nx_slice / 2) as f32;
            let ybot = 1.01f32 - (ny_slice / 2) as f32;
            let ytop = ny_slice as f32 - 1.01 - (ny_slice / 2) as f32;
            //
            // corresponding intersections of the ray with extended edges
            //
            let yleft = xleft / tan_ang + ray_intercept;
            let yright = xright / tan_ang + ray_intercept;
            let xbot = (ybot - ray_intercept) * tan_ang;
            let xtop = (ytop - ray_intercept) * tan_ang;
            //
            // make list of intersections that are actually within slice box
            //
            if yleft >= ybot && yleft <= ytop {
                num_good_inter += 1;
                xgood[num_good_inter - 1] = xleft;
                ygood[num_good_inter - 1] = yleft;
            }
            if yright >= ybot && yright <= ytop {
                num_good_inter += 1;
                xgood[num_good_inter - 1] = xright;
                ygood[num_good_inter - 1] = yright;
            }
            if xbot >= xleft && xbot <= xright {
                num_good_inter += 1;
                xgood[num_good_inter - 1] = xbot;
                ygood[num_good_inter - 1] = ybot;
            }
            if xtop >= xleft && xtop <= xright {
                num_good_inter += 1;
                xgood[num_good_inter - 1] = xtop;
                ygood[num_good_inter - 1] = ytop;
            }
            //
            // if there are real intersections, use them to set up ray start
            //
            if num_good_inter > 0 {
                let mut ind_good = 1;
                if (ygood[1] < ygood[0]) != (*cos_ang < 0.) {
                    ind_good = 2;
                }
                xray_tmp = xgood[ind_good - 1] + (nx_slice / 2) as f32 + 0.5;
                yray_tmp = ygood[ind_good - 1] + (ny_slice / 2) as f32 + 0.5;
                let dx = xgood[0] - xgood[1];
                let dy = ygood[0] - ygood[1];
                nray_tmp = (1. + (dx * dx + dy * dy).sqrt()) as i32;
                if nray_tmp < 3 {
                    nray_tmp = 0;
                }
            }
        }
        //
        // store final ray information
        //
        xray_start[(ix_out - 1) as usize] = xray_tmp;
        yray_start[(ix_out - 1) as usize] = yray_tmp;
        num_ray_increm[(ix_out - 1) as usize] = nray_tmp;
        *nray_max = (*nray_max).max(nray_tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::set_projection_rays;

    /// Expected bits captured from the reference `libhvem.so`.
    #[test]
    fn thirty_degree_rays_match_reference() {
        let mut sin_a = f32::from_bits(0x3F000000);
        let mut cos_a = f32::from_bits(0x3F5DB3D7);
        let (mut xs, mut ys, mut nr, mut nrmax) = ([0f32; 6], [0f32; 6], [0i32; 6], 0);
        set_projection_rays(
            &mut sin_a, &mut cos_a, 20, 10, 6, &mut xs, &mut ys, &mut nr, &mut nrmax,
        );
        assert_eq!(nrmax, 10);
        assert_eq!(nr, [10; 6]);
        let xbits: Vec<u32> = xs.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            xbits,
            [
                0x40A9E86B, 0x40CEDBB9, 0x40F3CF08, 0x410C612B, 0x411EDAD2, 0x4131547A
            ]
        );
        assert!(ys.iter().all(|v| v.to_bits() == 0x3FC147AE));
    }

    #[test]
    fn near_vertical_snaps_to_exact_vertical() {
        let mut sin_a = 0.0005f32;
        let mut cos_a = -0.99f32;
        let (mut xs, mut ys, mut nr, mut nrmax) = ([0f32; 3], [0f32; 3], [0i32; 3], 0);
        set_projection_rays(
            &mut sin_a, &mut cos_a, 10, 8, 3, &mut xs, &mut ys, &mut nr, &mut nrmax,
        );
        assert_eq!((sin_a, cos_a), (0., -1.));
        assert_eq!(xs, [6., 5., 4.]);
        assert_eq!(ys, [7.; 3]);
        assert_eq!(nr, [6; 3]);
    }
}
