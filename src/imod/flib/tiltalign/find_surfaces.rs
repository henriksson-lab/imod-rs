//! Translation of `IMOD/flib/tiltalign/find_surfaces.cpp` — functions for
//! assigning beads to surfaces and getting angles.
//!
//! # Arithmetic
//!
//! `RADIANS_PER_DEGREE` is the `double` literal `0.01745329252`
//! (`b3dutil.h:68`), so `alpha * RADIANS_PER_DEGREE` and everything computed
//! from it is `double`, rounded to `float` only on assignment.  `atan(bSlope)`
//! on a `float` is the C++ `float` overload (`atanf`); its result is widened
//! for the division.  `B3DMIN`/`B3DMAX`/`B3DABS`/`B3DSIGN` are written as
//! the source's conditional expressions (`CLAUDE.md`, NaN operand order).

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libcfshr::parse_params::exit_error;
use crate::imod::libcfshr::regression::mult_regress;
use crate::imod::libcfshr::simplestat::{ls_fit, ls_fit2};
use crate::imod::libcfshr::surfacesort::surface_sort;

/// `#define RADIANS_PER_DEGREE 0.01745329252` (`b3dutil.h:68`).
const RADIANS_PER_DEGREE: f64 = 0.01745329252;

/// Original: `find_surfaces` (`find_surfaces.cpp:51`).
///
/// Analyzes a set of `numRealPt` points with coordinates in `xyz`
/// (dimensioned (3,*)), correlates the Z with the X and Y coordinates, and
/// determines the angles the points would have to be tilted first around the
/// X axis then around the Y axis to lie parallel to the X-Y plane.  If
/// `numSurface` is 1 it fits a plane to all of the points; if 2 it divides the
/// points into two surfaces and estimates the new tilt angle from the slope of
/// two parallel planes.  `igroup` is returned with 1 for a point on the lower
/// surface and 2 on the upper, only if 2-surface analysis is done.  If
/// `ifComp` is non-zero it assumes data at a single tilt angle `tiltMax` and at
/// zero tilt and estimates the true tilt angle, returned in `tiltNew`.
/// `tiltAdd` is an existing change in tilt angles; `znew`, `znewInput` and
/// `imageBinned` let it report the unbinned thickness and centering shift.
///
/// Deviation (uninitialised memory): `truePlus` is only assigned by
/// `calcTiltNew` when `ifComp` is non-zero and `|cosNew| <= 1`, so with one
/// surface and `ifComp == 0` the source returns stack residue in `tiltNew`
/// (`find_surfaces.cpp:130`; measured `0x00007FFF`, `0x47AB3652`, varying by
/// build).  It starts at 0 here.  `tiltalign.cpp:714` never reads `tiltNew`
/// afterwards.
#[allow(clippy::too_many_arguments)]
pub fn find_surfaces(
    xyz: &[f32],
    num_real_pt: i32,
    num_surface: i32,
    tilt_max: f32,
    tilt_new: &mut f32,
    igroup: &mut [i32],
    if_comp: i32,
    tilt_add: f32,
    znew: f32,
    znew_input: f32,
    image_binned: i32,
) {
    let mut bintcp_minus: f32 = 0.;
    let mut bintcp: f32 = 0.;
    let n = num_real_pt as usize;
    let mut xmat: Vec<f32> = vec![0.; 4 * n];
    let mut num_pnt_minus: i32 = 0;
    let mut num_pnt_plus: i32 = 0;
    let mut a_slope: f32 = 0.;
    let mut b_slope: f32 = 0.;
    let mut alpha: f32 = 0.;
    let mut slope: f32 = 0.;
    let mut resid: f32 = 0.;
    let mut true_plus: f32 = 0.;
    let mut resid_minus: f32 = 0.;
    let mut resid_plus: f32 = 0.;
    let mut thick: f32;
    let shift_inc: f32;
    let mut shift_tot: f32;
    let mut bot_extreme: f32 = 0.;
    let mut top_extreme: f32 = 0.;
    //
    // first fit a line to all of the points to get starting angle
    //
    for i in 1..=n {
        xmat[i - 1] = xyz[i * 3 - 3];
        xmat[n + i - 1] = xyz[i * 3 - 2];
        xmat[2 * n + i - 1] = xyz[i * 3 - 1];
        igroup[i - 1] = 0;
    }
    lsfit2_resid(
        &xmat[(1 - 1) * n..],
        &xmat[(2 - 1) * n..],
        &xmat[(3 - 1) * n..],
        num_real_pt,
        &mut a_slope,
        &mut b_slope,
        &mut bintcp,
        &mut alpha,
        &mut slope,
        &mut resid,
        &mut bot_extreme,
        &mut top_extreme,
    );
    //
    // Batchruntomo is looking for '# of points' and the number after '=' and expects
    // either one number for all, or 3 numbers for all, bottom, top
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(&c_format_bytes(
        "\n SURFACE ANALYSIS:\n\n The following parameters are appropriate if fiducials are \
         NOT on two surfaces:\n\n Fit of one plane to all fiducials:\n # of points = %16d\n \
         Mean residual =    %11.2f\n",
        &[CArg::Int(num_real_pt as i64), CArg::Dbl(resid as f64)],
    ));
    if num_real_pt < 3 {
        let _ = out.write_all(b" Too few points to estimate tilt angle change or X-axis tilt\n");
        true_plus = 0.;
    } else {
        let _ = out.write_all(&c_format_bytes(
            " Adjusted slope =      %8.4f\n",
            &[CArg::Dbl(slope as f64)],
        ));
        if num_real_pt < 4 {
            let _ = out.write_all(b" Too few points to estimate X-axis tilt");
        } else {
            let _ = out.write_all(&c_format_bytes(
                " X axis tilt needed =%10.2f\n",
                &[CArg::Dbl(-alpha as f64)],
            ));
        }
        calc_tilt_new(slope, tilt_max, &mut true_plus, if_comp, tilt_add);
    }
    //
    // if there are supposed to be 2 surfaces, call the surfaceSort routine to get them
    // sorted out.  Then fit a pair of parallel planes to both surfaces
    //
    if num_surface > 1 && num_real_pt > 3 {
        if surface_sort(xyz, num_real_pt, 0, igroup) != 0 {
            exit_error(b"Allocating memory in surfaceSort");
        }

        two_surface_fits(
            xyz,
            igroup,
            num_real_pt,
            &mut xmat,
            &mut num_pnt_minus,
            &mut bintcp_minus,
            &mut resid_minus,
            &mut num_pnt_plus,
            &mut a_slope,
            &mut b_slope,
            &mut bintcp,
            &mut resid_plus,
            &mut alpha,
            &mut slope,
            &mut resid,
            &mut bot_extreme,
            &mut top_extreme,
        );

        let _ = out.write_all(
            b" The following parameters are appropriate if fiducials ARE on two surfaces,\n \
              based on fit of two parallel planes to fiducials sorted onto two surfaces:\n",
        );
        let _ = out.write_all(&c_format_bytes(
            "\n On bottom surface:\n # of points = %12d\n Mean residual =%11.2f\n Z axis \
             intercept =%8.1f\n",
            &[
                CArg::Int(num_pnt_minus as i64),
                CArg::Dbl(resid_minus as f64),
                CArg::Dbl(bintcp_minus as f64),
            ],
        ));
        let _ = out.write_all(&c_format_bytes(
            "\n On top surface:\n # of points = %12d\n Mean residual =%11.2f\n Z axis \
             intercept =%8.1f\n",
            &[
                CArg::Int(num_pnt_plus as i64),
                CArg::Dbl(resid_plus as f64),
                CArg::Dbl(bintcp as f64),
            ],
        ));
        thick = bintcp - bintcp_minus;
        let _ = out.write_all(&c_format_bytes(
            "\n Overall mean residual =%15.2f\n Thickness at Z intercepts =%11.1f\n Adjusted \
             slope =%22.4f\n",
            &[
                CArg::Dbl(resid as f64),
                CArg::Dbl(thick as f64),
                CArg::Dbl(slope as f64),
            ],
        ));
        if num_real_pt > 4 {
            let _ = out.write_all(&c_format_bytes(
                " X axis tilt needed =%18.2f\n",
                &[CArg::Dbl(-alpha as f64)],
            ));
        } else {
            let _ = out.write_all(b" Too few points to estimate X-axis tilt\n");
        }
        calc_tilt_new(slope, tilt_max, tilt_new, if_comp, tilt_add);
    } else {
        num_pnt_plus = num_real_pt;
        *tilt_new = true_plus;
        if num_surface > 1 {
            let _ = out.write_all(
                b"\n There are too few points to analyze for distribution on two surfaces\n\n",
            );
        }
    }
    let _ = num_pnt_plus;
    //
    // Get the unbinned thickness and shifts needed to center the gold
    // The direction is opposite to expected because the positive Z points
    // come out on the bottom of the tomogram, presumably due to rotation
    thick = image_binned as f32 * (top_extreme - bot_extreme);
    shift_tot = ((image_binned as f32 * (top_extreme + bot_extreme)) as f64 / 2.) as f32;
    shift_inc = shift_tot - image_binned as f32 * znew;
    shift_tot = shift_inc + image_binned as f32 * znew_input;
    let _ = out.write_all(&c_format_bytes(
        " Unbinned thickness needed to contain centers of all fiducials =%12.0f\n Incremental \
         unbinned shift needed to center range of fiducials in Z =%8.1f\n Total unbinned shift \
         needed to center range of fiducials in Z =%14.1f\n",
        &[
            CArg::Dbl(thick as f64),
            CArg::Dbl(shift_inc as f64),
            CArg::Dbl(shift_tot as f64),
        ],
    ));
    let _ = out.flush();
}

/// Original: `twoSurfaceFits` (`find_surfaces.cpp:151`, file `static`).
#[allow(clippy::too_many_arguments)]
fn two_surface_fits(
    xyz: &[f32],
    igroup: &[i32],
    num_real_pt: i32,
    xmat: &mut [f32],
    num_pnt_minus: &mut i32,
    bintcp_minus: &mut f32,
    resid_minus: &mut f32,
    num_pnt_plus: &mut i32,
    a_slope: &mut f32,
    b_slope: &mut f32,
    bintcp: &mut f32,
    resid_plus: &mut f32,
    alpha: &mut f32,
    slope: &mut f32,
    resid: &mut f32,
    bot_extreme: &mut f32,
    top_extreme: &mut f32,
) {
    let mut work = [0f32; 16];
    let mut coeff = [0f32; 4];
    let mut xmean = [0f32; 4];
    let mut xsd = [0f32; 4];
    let mut dev: f32;
    let mut num_col: i32;
    let n = num_real_pt as usize;
    //
    // first fit a plane to points in the first group
    //
    num_col = 3;
    if num_real_pt < 4 {
        num_col = 2;
    }
    let nc = num_col as usize;
    *num_pnt_minus = 0;
    *num_pnt_plus = 0;

    // Load the data matrix, putting the Y data in column 2 and overlaying with group
    // number if there are only two columns
    for ipt in 1..=n {
        if igroup[ipt - 1] == 1 {
            *num_pnt_minus += 1;
        } else {
            *num_pnt_plus += 1;
        }
        xmat[(1 - 1) * n + ipt - 1] = xyz[ipt * 3 - 3];
        xmat[(2 - 1) * n + ipt - 1] = xyz[ipt * 3 - 2];
        xmat[(nc - 1) * n + ipt - 1] = (igroup[ipt - 1] - 1) as f32;
        xmat[(nc + 1 - 1) * n + ipt - 1] = xyz[ipt * 3 - 1];
    }

    if mult_regress(
        xmat,
        num_real_pt,
        0,
        num_col,
        num_real_pt,
        1,
        0,
        &mut coeff,
        4,
        Some(std::slice::from_mut(bintcp_minus)),
        &mut xmean,
        &mut xsd,
        &mut work,
    ) != 0
    {
        exit_error(b"In matrix inversion for fitting planes to 3-D points");
    }

    // Compute a single slope and angles
    *bintcp = *bintcp_minus + coeff[nc - 1];
    coeff[nc - 1] = 0.;
    *a_slope = coeff[0];
    *b_slope = 0.;
    if num_col == 3 {
        *b_slope = coeff[1];
    }
    *alpha = (b_slope.atan() as f64 / RADIANS_PER_DEGREE) as f32;
    *slope = (coeff[0] as f64
        / ((*alpha as f64 * RADIANS_PER_DEGREE).cos()
            - *b_slope as f64 * (*alpha as f64 * RADIANS_PER_DEGREE).sin())) as f32;

    // Get averall mean residual and mean for each surface, and the extreme residual
    // above top and below bottom assuming pitch gets corrected
    *resid = 0.;
    *resid_plus = 0.;
    *resid_minus = 0.;
    *bot_extreme = 1.0e30;
    *top_extreme = -1.0e30;
    for ipt in 1..=n {
        if igroup[ipt - 1] == 1 {
            dev = xyz[ipt * 3 - 1]
                - (xyz[ipt * 3 - 3] * *a_slope + xyz[ipt * 3 - 2] * *b_slope + *bintcp_minus);
            *resid_minus += if dev >= 0. { dev } else { -dev };
            *bot_extreme = if *bot_extreme < *bintcp_minus + dev {
                *bot_extreme
            } else {
                *bintcp_minus + dev
            };
        } else {
            dev = xyz[ipt * 3 - 1]
                - (xyz[ipt * 3 - 3] * *a_slope + xyz[ipt * 3 - 2] * *b_slope + *bintcp);
            *resid_plus += if dev >= 0. { dev } else { -dev };
            *top_extreme = if *top_extreme > *bintcp + dev {
                *top_extreme
            } else {
                *bintcp + dev
            };
        }
        *resid += if dev >= 0. { dev } else { -dev };
    }
    *resid /= num_real_pt as f32;
    *resid_plus /= *num_pnt_plus as f32;
    *resid_minus /= *num_pnt_minus as f32;
}

/// Original: `calcTiltNew` (`find_surfaces.cpp:229`, file `static`) — gives the
/// new maximum tilt angle only for compression solutions.
///
/// `acos(cosNew / RADIANS_PER_DEGREE)` divides *inside* the `acos`
/// (`find_surfaces.cpp:241`), so any `|cosNew| > 0.0175` gives NaN; kept as
/// written (upstream defect, reached only with `ifComp`).
fn calc_tilt_new(
    slope_minus: f32,
    tilt_max: f32,
    true_minus: &mut f32,
    if_comp: i32,
    tilt_add: f32,
) {
    let cos_new: f32 = (slope_minus as f64 * (tilt_max as f64 * RADIANS_PER_DEGREE).sin()
        + (tilt_max as f64 * RADIANS_PER_DEGREE).cos()) as f32;
    let global_delta: f32 = ((-slope_minus).atan() as f64 / RADIANS_PER_DEGREE) as f32;
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(&c_format_bytes(
        " Incremental tilt angle change =%7.2f\n Total tilt angle change =%13.2f\n",
        &[
            CArg::Dbl(global_delta as f64),
            CArg::Dbl((global_delta + tilt_add) as f64),
        ],
    ));
    if if_comp != 0 {
        if (if cos_new >= 0. { cos_new } else { -cos_new }) <= 1. {
            let ac = (cos_new as f64 / RADIANS_PER_DEGREE).acos();
            *true_minus = (if tilt_max < 0. { -ac } else { ac }) as f32;
            let _ = out.write_all(&c_format_bytes(
                "     or, change a fixed maximum tilt angle from %7.2f to %7.2f\n\n",
                &[CArg::Dbl(tilt_max as f64), CArg::Dbl(*true_minus as f64)],
            ));
        } else {
            let _ =
                out.write_all(b"      or. . . But cannot derive implied tilt angle: |cos| > 1\n\n");
        }
    } else {
        let _ = out.write_all(b" \n");
    }
}

/// Original: `lsfit2Resid` (`find_surfaces.cpp:253`, file `static`).
#[allow(clippy::too_many_arguments)]
fn lsfit2_resid(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    n: i32,
    a: &mut f32,
    b: &mut f32,
    c: &mut f32,
    alpha: &mut f32,
    slope: &mut f32,
    resid: &mut f32,
    dev_min: &mut f32,
    dev_max: &mut f32,
) {
    let mut ro: f32 = 0.;
    let mut dev: f32;

    if n > 3 {
        ls_fit2(x, y, z, n, a, b, Some(c));
    } else if n > 2 {
        *b = 0.;
        ls_fit(x, z, n, a, c, &mut ro);
    } else {
        *a = 0.;
        *b = 0.;
        *c = ((z[0] + z[(n - 1) as usize]) as f64 / 2.) as f32;
    }
    *resid = 0.;
    *dev_min = 1.0e10;
    *dev_max = -1.0e10;
    for i in 1..=n as usize {
        dev = z[i - 1] - (x[i - 1] * *a + y[i - 1] * *b + *c);
        *resid += if dev >= 0. { dev } else { -dev };
        *dev_min = if *dev_min < dev { *dev_min } else { dev };
        *dev_max = if *dev_max > dev { *dev_max } else { dev };
    }
    *resid /= n as f32;
    *alpha = (b.atan() as f64 / RADIANS_PER_DEGREE) as f32;
    *slope = (*a as f64
        / ((*alpha as f64 * RADIANS_PER_DEGREE).cos()
            - *b as f64 * (*alpha as f64 * RADIANS_PER_DEGREE).sin())) as f32;
}
