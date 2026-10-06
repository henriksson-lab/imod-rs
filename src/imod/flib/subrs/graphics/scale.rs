//! Translation of `IMOD/flib/subrs/graphics/scale.f90`: "nice" axis limits
//! for a range of data values.

use crate::imod::flib::subrs::compat::gfortran_rt::{cvttss2si, powi_r4};

/// Original `scale` (`scale.f90:7`): the lower limit `xlo` and the increment
/// `dx` (1/10 of the axis) of nice limits containing `xmin..xmax`.
pub fn scale(xmin: f32, xmax: &mut f32, dx: &mut f32, xlo: &mut f32) {
    let good_range: [f32; 10] = [1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10.];
    let num_good_divs: [i32; 10] = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
    let mut ind_range = 0;
    scale_common(
        xmin,
        xmax,
        &good_range,
        &num_good_divs,
        10,
        dx,
        xlo,
        &mut ind_range,
    );
}

/// Original `scaleMultiDiv` (`scale.f90:23`): as [`scale`], with one more
/// range (1.2), returning the number of divisions (8, 10 or 12) in
/// `num_div` and their size in `dx_div`.
pub fn scale_multi_div(
    xmin: f32,
    xmax: &mut f32,
    dx: &mut f32,
    xlo: &mut f32,
    num_div: &mut i32,
    dx_div: &mut f32,
) {
    let good_range: [f32; 12] = [1., 1.2, 1.5, 1.6, 2., 2.5, 3., 4., 5., 6., 8., 10.];
    let num_good_divs: [i32; 12] = [10, 12, 10, 8, 10, 10, 10, 8, 10, 12, 8, 10];
    let mut ind_range = 0;
    scale_common(
        xmin,
        xmax,
        &good_range,
        &num_good_divs,
        11,
        dx,
        xlo,
        &mut ind_range,
    );
    *num_div = num_good_divs[(ind_range - 1) as usize];
    *dx_div = (10. * *dx) / *num_div as f32;
}

/// Original `scaleCommon` (`scale.f90:39`): returns the index (1-based) of
/// the selected range.  `xmax` is the caller's variable: it is raised when
/// the range is empty, as the source does.
///
/// Fixed in translation (BUGS.md, `scaleCommon`): when no range fits, the
/// source's loop leaves `indRange` one past `numRange`, which reads past
/// `scale`'s 10-element tables; here the index stops at the tables' last
/// entry.
#[allow(clippy::too_many_arguments)]
pub fn scale_common(
    xmin: f32,
    xmax: &mut f32,
    good_range: &[f32],
    num_good_divs: &[i32],
    num_range: i32,
    dx: &mut f32,
    xlo: &mut f32,
    ind_range: &mut i32,
) {
    *xlo = xmin;
    let mut xlo_last: f32 = 1.0e30;
    let mut dx_last: f32 = 1.0e30;
    for _loop in 1..=100 {
        //
        // If there is single point or no range, set range so point does not end up on an axis
        if *xmax <= *xlo {
            *xlo -= 0.5;
            *xmax = *xlo + 1.;
        }
        let dx_log = (*xmax - *xlo).log10();
        let mut log_exp = cvttss2si(dx_log);
        if dx_log < 0. {
            log_exp -= 1;
        }
        let fract = (*xmax - *xlo) / powi_r4(10., log_exp);
        *ind_range = 1;
        while *ind_range <= num_range {
            if fract <= good_range[(*ind_range - 1) as usize] {
                break;
            }
            *ind_range += 1;
        }
        *ind_range = (*ind_range).min(good_range.len() as i32);
        //
        // Callers expect dx to still be based on 10 divisions, but do the refinements
        // based on the actual number of divisions
        let ndivs = num_good_divs[(*ind_range - 1) as usize];
        *dx = good_range[(*ind_range - 1) as usize] * powi_r4(10., log_exp - 1);
        let dx_div = 10. * *dx / ndivs as f32;
        *xlo = dx_div * cvttss2si(xmin / dx_div) as f32;
        if *xlo > xmin {
            *xlo -= dx_div;
        }
        if *xlo == xmin && *xlo + (ndivs as f32 - 1.) * dx_div > *xmax {
            *xlo -= dx_div;
        }
        if (*xlo == xlo_last && *dx == dx_last) || *xlo + ndivs as f32 * dx_div >= *xmax {
            while *xlo >= dx_div && *xlo + (ndivs as f32 - 1.) * dx_div > *xmax {
                *xlo -= dx_div;
            }
            return;
        }
        xlo_last = *xlo;
        dx_last = *dx;
    }
}
