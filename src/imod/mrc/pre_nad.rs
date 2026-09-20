//! Translation of `IMOD/mrc/preNAD.cpp` — the Pre-NAD (non-linear anisotropic
//! diffusion) filter contributed by Mauro Maiorca, adapted for IMOD.
//!
//! The source's `float **` images come from `nrutil.c`'s `matrix()`, which is
//! `#include`d directly into this translation unit; they become
//! `nrutil::NrMatrix<f32>` here, with the same
//! `matrix(0, nx + 2*padding, 0, ny + 2*padding)` extent and the same
//! subscripts.
//!
//! Three properties of the source are load-bearing and easy to lose:
//!
//! * **`preNAD1Tilt` passes `float *` to a `double *` parameter.**
//!   `preNAD.cpp:508` is
//!   `dlaev2((double *) &(a[i][j]), (double *) &(b[i][j]), (double *) &(d[i][j]), …)`,
//!   and `dlaev2` (`flib/subrs/lapack/dlaev2.f`, reached through
//!   `include/lapackc.h`) takes `DOUBLE PRECISION` by reference.  Each argument
//!   is therefore an **8-byte load from a 4-byte float**: the two adjacent
//!   floats `x[i][j]` and `x[i][j+1]` reinterpreted as one little-endian
//!   `double`.  That is what the whole eigen-analysis runs on, so it is
//!   reproduced exactly; see `dlaev2_arg` below.  The last column
//!   (`j == ny + 2*padding`) is never written by anything in the program, so
//!   the high half of the final double comes from the `malloc` block — zero in
//!   a fresh allocation, which is what `NrMatrix` gives it.
//!
//! * **Three images are deliberate aliases.**  `lEED1` is `dx`, `lCED2` is
//!   `dy`, and `edgeConditionImage` is `b` (`preNAD.cpp:378-391`).  Because `b`
//!   is later overwritten with the diffusion tensor and re-smoothed, the
//!   `edgeConditionImage[i][j] > mean` test in the evolution loop is comparing
//!   a diffusion-tensor entry against an edge-condition mean, so `counterMVD`
//!   stays 0 and the returned MVD is always 0.  Native prints `MVD=0.000000`
//!   throughout; this is upstream behaviour, not a translation shortfall.
//!
//! * **`automaticPreNAD` reads one element past both arrays.**
//!   `preNAD.cpp:838-852` uses `ii` after its `for (ii = 0; ii < nz; ii++)`
//!   loop has left it equal to `nz`, so `stackIn[nz].skipMe`,
//!   `stackOut[nz].angleDegrees`, `stackOut[nz].MVD` and
//!   `stackOut[nz].angleRadiant` are all read out of bounds.  The stacks here
//!   are `nz + 1` long with a zeroed extra element, which is what the
//!   reference binary observes.  For the same reason `stackIn[ii].skipMe` is
//!   only ever assigned when `-views` was given, so without it the source
//!   reads uninitialised `malloc` memory for every projection.

use crate::imod::ctfplotter::ctfutils::read_tilt_angles;
use crate::imod::flib::subrs::lapack::dlaev2::dlaev2;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_output_file_type, c_format_bytes, imod_prog_name,
};
use crate::imod::libcfshr::islice::{slice_create, slice_mode_if_real};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_string, pip_print_help,
    pip_read_or_parse_options,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libiimod::iimage::{IIFILE_MRC, ii_fclose, ii_fopen, ii_lookup_file_from_fp};
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_copy_extra_header, mrc_head_label, mrc_head_read, mrc_head_write,
    mrc_init_output_header, mrc_read_slice, mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{slice_mmm, slice_new_mode};
use crate::imod::mrc::nrutil::{
    NrMatrix, free_ivector, free_matrix, free_vector, ivector, matrix, vector,
};
use crate::imod::mrc::recline::{
    DerivativeOrder, RecursiveFilterType, init_recursive_coefficients, recursive_filter_1d,
};
use std::io::Write as _;

const MY_PI: f64 = 3.141592653589;
const SLICE_MODE_FLOAT: i32 = 2;

/// C `TiltProjectionType`: support structure for iterating through slices.
#[derive(Clone, Debug, Default)]
pub struct TiltProjectionType {
    /// Current iteration
    pub current_iteration: i32,
    /// Masked Variance Difference at the current iteration
    pub mvd: f64,
    /// angle in radiants
    pub angle_radiant: f64,
    /// angle in degrees
    pub angle_degrees: f64,
    /// sigma for the current tilt projection
    pub sigma: f64,
    /// the current tilt projection won't be processed if this is set to true
    pub skip_me: bool,
    /// Current tilt projection
    pub tilt_number: i32,
    /// the processed tilt projection data at the current iteration
    pub data: Option<NrMatrix<f32>>,
    /// the original tilt projection data (prior to processing)
    pub data_original: Option<NrMatrix<f32>>,
}

/// C `FillingPadding`: fills the border with a padding.
pub fn filling_padding(i_image: &mut NrMatrix<f32>, nx: u32, ny: u32, padding_size: u32) {
    let (nx, ny, padding_size) = (nx as usize, ny as usize, padding_size as usize);
    if nx > padding_size + 1 && ny > padding_size + 1 {
        for p in 0..padding_size {
            for i in 0..nx + 2 * padding_size {
                i_image[(i, padding_size + ny + p)] = i_image[(i, ny - 1 - p)];
            }
            for i in 0..nx + 2 * padding_size {
                i_image[(i, p)] = i_image[(i, 2 * padding_size - p)];
            }
            for j in 0..ny + 2 * padding_size {
                i_image[(padding_size + nx + p, j)] = i_image[(nx - 1 - p, j)];
            }
            for j in 0..ny + 2 * padding_size {
                i_image[(p, j)] = i_image[(2 * padding_size - p, j)];
            }
        }
    }
}

/// C `getQuartileMVDIndex`: the index of the mean Masked Variance Difference.
pub fn get_quartile_mvd_index(array: &[TiltProjectionType], size: i32) -> u32 {
    let median_index: u32;
    let mut temp: f32;
    let mut tmp_index: f32;

    let mut values = vector(0, size as isize);
    let mut indexes = ivector(0, size as isize);

    //copy the vector and the index
    for i in 0..size as usize {
        values[i] = array[i].mvd as f32;
        indexes[i] = i as i32;
    }

    /* The function uses bubblesort algorithm to order the MVD values. */
    let mut i = size - 1;
    while i > 0 {
        for j in 1..=i as usize {
            if values[j - 1] > values[j] {
                temp = values[j - 1];
                values[j - 1] = values[j];
                values[j] = temp;
                tmp_index = indexes[j - 1] as f32;
                indexes[j - 1] = indexes[j];
                indexes[j] = tmp_index as i32;
            }
        }
        i -= 1;
    }

    /* The median is then extracted from the ordered array. */
    //get the first quartileInstead
    median_index = indexes[(size as f64 / 4.0) as u32 as usize] as u32;

    free_vector(values, 0, size as isize);
    free_ivector(indexes, 0, size as isize);

    median_index
}

/// C `gaussRecursiveDerivatives1D`: 1D zero- and first-order Gaussian
/// derivatives of a 2D image.
///
/// `fo == NULL` in the source means "overwrite the input"; the smoothing call
/// sites pass the same pointer for `f` and `fo`, which is the same thing, so
/// both are `None` here.
#[allow(clippy::too_many_arguments)]
pub fn gauss_recursive_derivatives_1d(
    sigma: f64,
    nx: u32,
    ny: u32,
    padding: u32,
    _hx: f64,
    _hy: f64,
    direction: u32,
    derivative_order: u32,
    f: &mut NrMatrix<f32>,
    mut fo: Option<&mut NrMatrix<f32>>,
) {
    let (nxu, nyu, pad) = (nx as usize, ny as usize, padding as usize);

    filling_padding(f, nx, ny, padding);

    //if less then a certain threshold, then just don't smooth, copy the image
    let sigma_threshold: f64 = 0.1;
    if sigma < sigma_threshold {
        for i in pad..nxu + pad {
            for j in pad..nyu + pad {
                let value = f[(i, j)];
                match fo {
                    Some(ref mut o) => o[(i, j)] = value,
                    None => f[(i, j)] = value,
                }
            }
        }
        match fo {
            Some(ref mut o) => filling_padding(o, nx, ny, padding),
            None => filling_padding(f, nx, ny, padding),
        }
        return;
    }

    //for each line, put the line in a buffer and process it.
    let rec_filter = RecursiveFilterType::GaussianDeriche;
    let mut deriv_order = DerivativeOrder::None;
    if derivative_order == 0 {
        deriv_order = DerivativeOrder::Zero;
    } else if derivative_order == 1 {
        deriv_order = DerivativeOrder::One;
    } else if derivative_order == 2 {
        deriv_order = DerivativeOrder::Two;
    } else if derivative_order == 3 {
        deriv_order = DerivativeOrder::Three;
    }
    let Some(rfc) = init_recursive_coefficients(sigma, rec_filter, deriv_order) else {
        exit_error(b"Allocation structure for recursive filter");
    };

    //fill the buffer
    if direction == 0 {
        //along X
        let mut buffer_in = vec![0f64; nxu + 2 * pad];
        let mut buffer_out = vec![0f64; nxu + 2 * pad];
        let mut buffer_tmp0 = vec![0f64; nxu + 2 * pad];
        let mut buffer_tmp1 = vec![0f64; nxu + 2 * pad];

        for j in 0..nyu + 2 * pad {
            for k in 0..nxu + 2 * pad {
                buffer_in[k] = f[(k, j)] as f64;
                buffer_out[k] = f[(k, j)] as f64;
                buffer_tmp0[k] = f[(k, j)] as f64;
                buffer_tmp1[k] = f[(k, j)] as f64;
            }
            recursive_filter_1d(
                &rfc,
                &buffer_in,
                &mut buffer_out,
                &mut buffer_tmp0,
                &mut buffer_tmp1,
                (nxu + 2 * pad) as i32,
            );

            for k in 0..nxu + 2 * pad {
                let value = buffer_out[k] as f32;
                match fo {
                    Some(ref mut o) => o[(k, j)] = value,
                    None => f[(k, j)] = value,
                }
            }
        }
    }
    //along Y
    if direction == 1 {
        let mut buffer_in = vec![0f64; nyu + 2 * pad];
        let mut buffer_out = vec![0f64; nyu + 2 * pad];
        let mut buffer_tmp0 = vec![0f64; nyu + 2 * pad];
        let mut buffer_tmp1 = vec![0f64; nyu + 2 * pad];

        for j in 0..nxu + 2 * pad {
            for k in 0..nyu + 2 * pad {
                buffer_in[k] = f[(j, k)] as f64;
                buffer_out[k] = f[(j, k)] as f64;
                buffer_tmp0[k] = f[(j, k)] as f64;
                buffer_tmp1[k] = f[(j, k)] as f64;
            }
            recursive_filter_1d(
                &rfc,
                &buffer_in,
                &mut buffer_out,
                &mut buffer_tmp0,
                &mut buffer_tmp1,
                (nyu + 2 * pad) as i32,
            );

            for k in 0..nyu + 2 * pad {
                let value = buffer_out[k] as f32;
                match fo {
                    Some(ref mut o) => o[(j, k)] = value,
                    None => f[(j, k)] = value,
                }
            }
        }
    }

    match fo {
        Some(ref mut o) => filling_padding(o, nx, ny, padding),
        None => filling_padding(f, nx, ny, padding),
    }
}

/// The `(double *) &(x[i][j])` argument `preNAD.cpp:508` hands `dlaev2`: the
/// eight bytes starting at a `float` element, which on a little-endian machine
/// are `x[i][j]` in the low half and `x[i][j+1]` in the high half.
fn dlaev2_arg(x: &NrMatrix<f32>, i: usize, j: usize) -> f64 {
    let low = x[(i, j)].to_bits() as u64;
    let high = x[(i, j + 1)].to_bits() as u64;
    f64::from_bits((high << 32) | low)
}

/// C `preNAD1Tilt`: the core of the algorithm; returns the Masked Variance
/// Difference between `Iout` and `I0`.
///
/// Both call sites pass the same image for `I` and `Iout`, which is also what
/// the source's `Iout == NULL` default means, so the evolution below updates
/// in place exactly as the C does — later pixels see the new values of their
/// already-visited neighbours.
#[allow(clippy::too_many_arguments)]
pub fn pre_nad_1_tilt(
    nx: u32,
    ny: u32,
    padding: u32,
    spacing_x: f64,
    spacing_y: f64,
    sigma: f64,
    _lambda_e: f64,
    _lambda_c: f64,
    _lambda_h: f64,
    _angle_radiant: f64,
    i_image: &mut NrMatrix<f32>,
    i0: &NrMatrix<f32>,
) -> f64 {
    let (nxu, nyu, pad) = (nx as usize, ny as usize, padding as usize);
    let hi = (nxu + 2 * pad) as isize;
    let hj = (nyu + 2 * pad) as isize;

    let mut dx = matrix(0, hi, 0, hj);
    let mut dy = matrix(0, hi, 0, hj);
    let mut gradient_magnitude_square = matrix(0, hi, 0, hj);

    //memory for eigenvectors
    let mut e_v1x = matrix(0, hi, 0, hj);
    let mut e_v1y = matrix(0, hi, 0, hj);
    let mut e_v2x = matrix(0, hi, 0, hj);
    let mut e_v2y = matrix(0, hi, 0, hj);

    //memory for The diffusion tensor
    let mut a = matrix(0, hi, 0, hj);
    let mut b = matrix(0, hi, 0, hj);
    let mut d = matrix(0, hi, 0, hj);

    let contrast_parameter_lambda_eed: f64 = 30.;
    let contrast_parameter_lambda_ced: f64 = 30.;
    let contrast_parameter_lambda_hybrid: f64 = 30.;
    let contrast_parameter_lambda_ced_square: f64 = contrast_parameter_lambda_ced.powf(2.0);
    let threshold_parameter_c: f64 = 3.31488;
    let zero_value_tolerance: f64 = 1e-15;
    let alpha: f64 = 0.001;
    let time_step: f64 = 0.125;
    let rxx = time_step / (2.0 * spacing_x * spacing_x);
    let ryy = time_step / (2.0 * spacing_y * spacing_y);
    let rxy = time_step / (4.0 * spacing_x * spacing_y);

    //structure tensor enhancers
    let convolution_precision: u32 = (sigma + 1.0) as u32;
    let magnify_gradient_ratio: f64 =
        (1.0 / convolution_precision as f64) * (convolution_precision as f64).ln();

    //gaussian first order derivative
    gauss_recursive_derivatives_1d(
        sigma,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        0,
        1,
        i_image,
        Some(&mut dx),
    );
    gauss_recursive_derivatives_1d(
        sigma,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        1,
        1,
        i_image,
        Some(&mut dy),
    );

    let mut mean_squared: f64 = 0.0;
    let counter: f64 = (nxu * nyu) as f64;

    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            // D= | a b |
            //    | b d | (Hermitian Matrix)
            a[(i, j)] = (dx[(i, j)] as f64).powf(2.0) as f32;
            b[(i, j)] = dx[(i, j)] * dy[(i, j)];
            d[(i, j)] = (dy[(i, j)] as f64).powf(2.0) as f32;
            gradient_magnitude_square[(i, j)] =
                (magnify_gradient_ratio * (a[(i, j)] + d[(i, j)]) as f64) as f32;
        }
    }

    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma, nx, ny, padding, spacing_x, spacing_y, direction, order, &mut a, None,
        );
    }
    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma, nx, ny, padding, spacing_x, spacing_y, direction, order, &mut b, None,
        );
    }
    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma, nx, ny, padding, spacing_x, spacing_y, direction, order, &mut d, None,
        );
    }
    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma,
            nx,
            ny,
            padding,
            spacing_x,
            spacing_y,
            direction,
            order,
            &mut gradient_magnitude_square,
            None,
        );
    }

    let mut mean: f64 = 0.;
    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            //note: dlaev2 already sorts the eigenvalues by absolute value
            let mut mu1: f64 = 0.0;
            let mut mu2: f64 = 0.0;
            let mut cs1: f64 = 0.0;
            let mut sn1: f64 = 0.0;
            dlaev2(
                dlaev2_arg(&a, i, j),
                dlaev2_arg(&b, i, j),
                dlaev2_arg(&d, i, j),
                &mut mu1,
                &mut mu2,
                &mut cs1,
                &mut sn1,
            );
            mu1 = mu1.abs();
            mu2 = mu2.abs();
            e_v1x[(i, j)] = cs1 as f32;
            e_v1y[(i, j)] = sn1 as f32;
            e_v2x[(i, j)] = -sn1 as f32;
            e_v2y[(i, j)] = cs1 as f32;

            //compute Lambdas for EED
            let mut lambda_eed1: f64 = 1.0;

            let gradient_magnitude = (gradient_magnitude_square[(i, j)] as f64).sqrt();
            let ratio: f64;
            let mut exp_val: f64;
            let kappa: f64;

            if gradient_magnitude > zero_value_tolerance {
                ratio = gradient_magnitude_square[(i, j)] as f64
                    / contrast_parameter_lambda_eed.powf(2.0);
                exp_val = ((-1.0 * threshold_parameter_c) / ratio.powf(4.0)).exp();
                lambda_eed1 = 1.0 - exp_val;
            }

            //compute Lambda's for CED
            let mut lambda_ced2: f64 = 1.0;
            let tmp: f64;

            //(always rememember: mu2>mu1)
            if mu1.abs() > zero_value_tolerance {
                kappa = (mu1 / (alpha + mu2)).powf(1.0);
                tmp = -1.0 * (2.0f64.ln() * contrast_parameter_lambda_ced_square) / kappa;
                exp_val = tmp.exp();
                lambda_ced2 = alpha + (1.0 - alpha) * exp_val;
            }

            // Compute the final lambdas for the continous switch
            let edge_condition = 4.0 * (1.0 + alpha + (mu2 - mu1).abs()).ln()
                + 4.0 * (1.0 + alpha + mu2 / (alpha + mu1)).ln();

            //fill lambdas (EED, CED)
            if lambda_eed1 < alpha {
                lambda_eed1 = 0.0;
            }
            if lambda_ced2 < alpha {
                lambda_ced2 = 0.0;
            }
            /* lEED1 is dx and lCED2 is dy -- the source reuses the memory. */
            dx[(i, j)] = lambda_eed1 as f32;
            dy[(i, j)] = lambda_ced2 as f32;

            /* edgeConditionImage is b -- likewise.  `dlaev2` above has already
            read `b[i][j]` and `b[i][j+1]` for this pixel, and `b[i][j+1]` is
            not written until the next iteration, so the eigen-analysis still
            sees the smoothed structure tensor. */
            b[(i, j)] = edge_condition as f32;

            //some statistics for computing the final epsilon
            mean += edge_condition;
            mean_squared = edge_condition * edge_condition;
        }
    }
    let _ = mean_squared;

    mean /= 1.0 + counter;
    let mut variance: f64 = 0.;

    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            variance += (b[(i, j)] as f64 - mean).abs();
        }
    }
    variance /= (nxu * nyu) as f64 - 1.0;
    let _ = variance;

    let mut counter_mvd: f64 = 0.;
    let mut mean_mvd: f64 = 0.;
    let mut variance_mvd: f64;

    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            let mut t = b[(i, j)] as f64;

            //note: K little => more edges. Use little K with noisy images (ET)
            t = (t - mean) / contrast_parameter_lambda_hybrid;
            let mut continuous_switch = 1.0f64.exp() / (1.0f64.exp() + (-1.0 * t).exp());

            if t > mean {
                mean_mvd += t;
                counter_mvd += 1.;
            }

            if continuous_switch >= 1.0 {
                continuous_switch = 1.0;
            }
            if continuous_switch < alpha {
                continuous_switch = 0.0;
            }

            //1 are edges, 0 are coherence
            let l_eed2: f64 = 1.0;
            let l_ced1: f64 = alpha;
            let mut lambda1 =
                (1.0 - continuous_switch) * l_ced1 + continuous_switch * dx[(i, j)] as f64;
            let mut lambda2 =
                (1.0 - continuous_switch) * dy[(i, j)] as f64 + continuous_switch * l_eed2;
            if lambda1 < alpha {
                lambda1 = alpha;
            }
            if lambda2 < alpha {
                lambda2 = alpha;
            }
            if lambda1 > 1.0 - alpha {
                lambda1 = 1.0;
            }
            if lambda2 > 1.0 - alpha {
                lambda2 = 1.0;
            }

            a[(i, j)] = (lambda1 * (e_v1x[(i, j)] as f64).powf(2.0)
                + lambda2 * (e_v1y[(i, j)] as f64).powf(2.0)) as f32;
            b[(i, j)] = (lambda1 * (e_v1x[(i, j)] * e_v2x[(i, j)]) as f64
                + lambda2 * (e_v1y[(i, j)] * e_v2y[(i, j)]) as f64) as f32;
            d[(i, j)] = (lambda1 * (e_v2x[(i, j)] as f64).powf(2.0)
                + lambda2 * (e_v2y[(i, j)] as f64).powf(2.0)) as f32;
        }
    }

    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma, nx, ny, padding, spacing_x, spacing_y, direction, order, &mut a, None,
        );
    }
    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma, nx, ny, padding, spacing_x, spacing_y, direction, order, &mut b, None,
        );
    }
    for (direction, order) in [(0u32, 0u32), (1, 0)] {
        gauss_recursive_derivatives_1d(
            sigma, nx, ny, padding, spacing_x, spacing_y, direction, order, &mut d, None,
        );
    }

    counter_mvd = 0.;
    for i in pad..nxu + pad {
        for j in pad..nyu + pad {
            //EVOLUTION
            let w_e = rxx * (a[(i + 1, j)] + a[(i, j)]) as f64
                - rxy
                    * ((b[(i + 1, j)] as f64 * b[(i + 1, j)] as f64).sqrt()
                        + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_w = rxx * (a[(i - 1, j)] + a[(i, j)]) as f64
                - rxy
                    * ((b[(i - 1, j)] as f64 * b[(i - 1, j)] as f64).sqrt()
                        + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_s = ryy * (d[(i, j + 1)] + d[(i, j)]) as f64
                - rxy
                    * ((b[(i, j + 1)] as f64 * b[(i, j + 1)] as f64).sqrt()
                        + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_n = ryy * (d[(i, j - 1)] + d[(i, j)]) as f64
                - rxy
                    * ((b[(i, j - 1)] as f64 * b[(i, j - 1)] as f64).sqrt()
                        + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_se = rxy
                * ((b[(i + 1, j + 1)] + b[(i, j)]) as f64
                    + (b[(i + 1, j + 1)] as f64 * b[(i + 1, j + 1)] as f64).sqrt()
                    + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_nw = rxy
                * ((b[(i - 1, j - 1)] + b[(i, j)]) as f64
                    + (b[(i - 1, j - 1)] as f64 * b[(i - 1, j - 1)] as f64).sqrt()
                    + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_ne = rxy
                * ((-b[(i + 1, j - 1)] - b[(i, j)]) as f64
                    + (b[(i + 1, j - 1)] as f64 * b[(i + 1, j - 1)] as f64).sqrt()
                    + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let w_sw = rxy
                * ((-b[(i - 1, j + 1)] - b[(i, j)]) as f64
                    + (b[(i - 1, j + 1)] as f64 * b[(i - 1, j + 1)] as f64).sqrt()
                    + (b[(i, j)] as f64 * b[(i, j)] as f64).sqrt());
            let centre = i_image[(i, j)];
            i_image[(i, j)] = (centre as f64
                + w_e * (i_image[(i + 1, j)] - centre) as f64
                + w_w * (i_image[(i - 1, j)] - centre) as f64
                + w_s * (i_image[(i, j + 1)] - centre) as f64
                + w_n * (i_image[(i, j - 1)] - centre) as f64
                + w_se * (i_image[(i + 1, j + 1)] - centre) as f64
                + w_nw * (i_image[(i - 1, j - 1)] - centre) as f64
                + w_sw * (i_image[(i - 1, j + 1)] - centre) as f64
                + w_ne * (i_image[(i + 1, j - 1)] - centre) as f64)
                as f32;

            //apply the mask
            //no edge image
            if b[(i, j)] as f64 > mean {
                mean_mvd += (i_image[(i, j)] - i0[(i, j)]).abs() as f64;
                counter_mvd += 1.;
            }
        }
    }
    mean_mvd /= counter_mvd + 1.;

    variance_mvd = 0.0;

    for i in pad..nxu + pad {
        for j in pad..nyu + pad {
            //there is no edge (coherence region)
            if b[(i, j)] as f64 > mean {
                variance_mvd += ((i_image[(i, j)] - i0[(i, j)]).abs() as f64 - mean_mvd).powf(2.);
            }
        }
    }
    variance_mvd = variance_mvd.sqrt() / (counter_mvd + 1.);

    //clear buffers
    free_matrix(dx, 0, hi, 0, hj);
    free_matrix(dy, 0, hi, 0, hj);
    free_matrix(gradient_magnitude_square, 0, hi, 0, hj);
    free_matrix(e_v1x, 0, hi, 0, hj);
    free_matrix(e_v1y, 0, hi, 0, hj);
    free_matrix(e_v2x, 0, hi, 0, hj);
    free_matrix(e_v2y, 0, hi, 0, hj);
    free_matrix(a, 0, hi, 0, hj);
    free_matrix(b, 0, hi, 0, hj);
    free_matrix(d, 0, hi, 0, hj);

    variance_mvd
}

/// C `automaticPreNAD`: automatic stop condition and iteration through the
/// whole dataset.
#[allow(clippy::too_many_arguments)]
pub fn automatic_pre_nad(
    stack_in: &[TiltProjectionType],
    stack_out: &mut [TiltProjectionType],
    nx: u32,
    ny: u32,
    nz: u32,
    padding: u32,
    min_iterations: u32,
    max_iterations: u32,
    spacing_x: f64,
    spacing_y: f64,
    lambda_e: f64,
    lambda_c: f64,
    lambda_h: f64,
) {
    let mut ii: u32;

    let _ = ImodFile::Stdout
        .write_all(b"START (t=tilt, a=angle, i=iteration, MVD=masked variance difference)\n");

    //We have to reach the min number of iterations for each tilt angles
    ii = 0;
    while ii < nz {
        let mut it: u32 = 0;
        while it < min_iterations && !stack_in[ii as usize].skip_me {
            let mut data = stack_out[ii as usize].data.take().unwrap();
            let sigma = stack_out[ii as usize].sigma;
            let angle_radiant = stack_out[ii as usize].angle_radiant;
            stack_out[ii as usize].mvd = pre_nad_1_tilt(
                nx,
                ny,
                padding,
                spacing_x,
                spacing_y,
                sigma,
                lambda_e,
                lambda_c,
                lambda_h,
                angle_radiant,
                &mut data,
                stack_in[ii as usize].data.as_ref().unwrap(),
            );
            stack_out[ii as usize].data = Some(data);
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "[t=%u,a=%f,i=%u,MVD=%f]",
                &[
                    CArg::Uint(ii as u64),
                    CArg::Dbl(stack_out[ii as usize].angle_degrees),
                    CArg::Uint(it as u64),
                    CArg::Dbl(stack_out[ii as usize].mvd),
                ],
            ));
            it += 1;
        }
        let _ = ImodFile::Stdout.write_all(b"\n");
        ii += 1;
    }

    let _ = ImodFile::Stdout.write_all(b"\n\nSearch for the maximum(minimum MVD)\n");

    let _ = ImodFile::Stdout.write_all(b"\n(iteration, tilt, MVD)=\n");
    let index_target_mvd = get_quartile_mvd_index(stack_out, nz as i32) as usize;

    let mut stop_mvd = stack_out[index_target_mvd].mvd;

    /* `ii` is `nz` here: the source reads one element past both stacks. */
    let mut it = min_iterations;
    while it < max_iterations && !stack_in[ii as usize].skip_me {
        stop_mvd = stack_out[index_target_mvd].mvd;
        let mut data = stack_out[index_target_mvd].data.take().unwrap();
        let sigma = stack_out[index_target_mvd].sigma;
        let angle_radiant = stack_out[ii as usize].angle_radiant;
        stack_out[index_target_mvd].mvd = pre_nad_1_tilt(
            nx,
            ny,
            padding,
            spacing_x,
            spacing_y,
            sigma,
            lambda_e,
            lambda_c,
            lambda_h,
            angle_radiant,
            &mut data,
            stack_in[index_target_mvd].data.as_ref().unwrap(),
        );
        stack_out[index_target_mvd].data = Some(data);
        stack_out[index_target_mvd].current_iteration = it as i32;
        stop_mvd = (stop_mvd + stack_out[index_target_mvd].mvd) / 2.;
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "[t=%u,a=%f,i=%u,MVD=%f] ",
            &[
                CArg::Uint(ii as u64),
                CArg::Dbl(stack_out[ii as usize].angle_degrees),
                CArg::Uint(it as u64),
                CArg::Dbl(stack_out[ii as usize].mvd),
            ],
        ));
        it += 1;
    }

    let _ = ImodFile::Stdout.write_all(b"\n");
    let _ = ImodFile::Stdout.write_all(&c_format_bytes("(stopMVD)=%f\n", &[CArg::Dbl(stop_mvd)]));

    let _ = ImodFile::Stdout.write_all(
        b"\n\n\nClassic computation (t=tilt, a=angle, i=iteration, MVD=masked variance difference):\n",
    );
    ii = 0;
    while ii < nz {
        let mut it = min_iterations;
        while it < max_iterations && !stack_in[ii as usize].skip_me {
            if stack_out[ii as usize].current_iteration < max_iterations as i32
                && stack_out[ii as usize].mvd <= stop_mvd
            {
                stack_out[ii as usize].current_iteration = it as i32;
                let mut data = stack_out[ii as usize].data.take().unwrap();
                let sigma = stack_out[ii as usize].sigma;
                let angle_radiant = stack_out[ii as usize].angle_radiant;
                stack_out[ii as usize].mvd = pre_nad_1_tilt(
                    nx,
                    ny,
                    padding,
                    spacing_x,
                    spacing_y,
                    sigma,
                    lambda_e,
                    lambda_c,
                    lambda_h,
                    angle_radiant,
                    &mut data,
                    stack_in[ii as usize].data.as_ref().unwrap(),
                );
                stack_out[ii as usize].data = Some(data);
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "[t=%u,a=%f,i=%u,MVD=%f] ",
                    &[
                        CArg::Uint(ii as u64),
                        CArg::Dbl(stack_out[ii as usize].angle_degrees),
                        CArg::Uint(it as u64),
                        CArg::Dbl(stack_out[ii as usize].mvd),
                    ],
                ));
            }
            it += 1;
        }
        let _ = ImodFile::Stdout.write_all(b"\n");
        ii += 1;
    }
    let _ = ImodFile::Stdout.write_all(b"\n");
}

/// C `main` in `preNAD.cpp`.
pub fn pre_nad(argv: &[String]) -> i32 {
    let argc = argv.len() as i32;
    let mut stack_fn: Vec<u8> = Vec::new();
    let mut angle_fn: Vec<u8> = Vec::new();
    let mut out_fn: Vec<u8> = Vec::new();
    let mut views_to_process: Vec<u8> = Vec::new();
    let mut starting_view: i32 = 0;
    let mut min_iterations: i32 = -1;
    let mut max_iterations: i32 = -1;
    let mut masked_variance_difference: f32 = -1.0;

    let progname = imod_prog_name(&argv[0]);

    let mut num_opt_args = 0i32;
    let mut num_non_opt_args = 0i32;
    let num_options = 8;
    let mut sigma: f32 = 1.0;
    let padding: u32 = 40;
    let mut range_requested = false;
    let mut no_tilt_angle = false;

    //parselist of views
    let mut angles_to_process: Vec<i32> = Vec::new();
    let mut n_angles_to_process: usize = 0;

    let options: [&[u8]; 8] = [
        b"input:InputStack:FN:",
        b"output:OutputFileName:FN:",
        b"angles:AnglesFile:FN:",
        b"MVD:MaskedVarianceDifference:F:",
        b"s:sigma:F:",
        b"minite:MinIterations:I:",
        b"maxite:MaxIterations:I:",
        b"views:ViewsToProcess:LI:",
    ];

    let mut out = ImodFile::Stdout;
    let _ = out.write_all(b"\n [Mauro Maiorca, of the Biochemistry & Molecular Biology Department, Bio21 institute, University of Melbourne, Australia, contributed the preNAD program (adapted for IMOD). ");
    let _ = out.write_all(b"It uses recursive line filter routines from Gregoire Malandain, covered by version 3 of the GPL (see GPL-3.0.txt). ");
    let _ = out.write_all(b"Please quote Maiorca, M., et al., J Struct Biol (2012) 180, 132-42. This work was supported by funding from the Australian Research Council and the National Health and Medical Research Council.]\n\n");
    let _ = out.write_all(b" examples of use:\n  standard:\n    preNAD -input myInputImage.mrc  -output myOutputImage.ali -angles myTiltAnglesFile.tlt -s 3 -minite 6 -maxite 8\n");
    let _ = out.write_all(b"  only tilt projections number 1 2 8 15 18 19 20:\n    preNAD -input myInputImage.mrc  -output myOutputImage.ali -angles myTiltAnglesFile.tlt -s 3 -minite 6 -maxite 8 -views 1-2,8,15,18-20\n\n\n");

    let argv_bytes: Vec<Vec<u8>> = argv.iter().map(|a| a.as_bytes().to_vec()).collect();
    pip_read_or_parse_options(
        argc,
        &argv_bytes,
        &options,
        num_options,
        progname.as_bytes(),
        1,
        0,
        0,
        &mut num_opt_args,
        &mut num_non_opt_args,
        None,
    );
    if pip_get_boolean(b"usage", &mut starting_view) == 0 {
        pip_print_help(progname.as_bytes(), 0, 0, 0);
        let _ = ImodFile::Stdout.flush();
        std::process::exit(0);
    }
    if pip_get_string(b"InputStack", &mut stack_fn) != 0 {
        exit_error(b"No stack specified");
    }
    if pip_get_string(b"OutputFileName", &mut out_fn) != 0 {
        exit_error(b"OutputFileName is not specified");
    }
    if pip_get_string(b"ViewsToProcess", &mut views_to_process) == 0 {
        range_requested = true;
        angles_to_process =
            parselist(&String::from_utf8_lossy(&views_to_process)).unwrap_or_default();
        n_angles_to_process = angles_to_process.len();
    }
    let mut have_angle_file = true;
    if pip_get_string(b"AnglesFile", &mut angle_fn) != 0 {
        have_angle_file = false;
        let _ = ImodFile::Stdout.write_all(
            b"[Warning] No angle file is specified, tilt angles are assumed to be all 0.0 degrees.\n",
        );
        no_tilt_angle = true;
    }
    if pip_get_float(b"sigma", &mut sigma) != 0 {
        sigma = 1.0;
        let _ = ImodFile::Stdout.write_all(b"No sigma is specified, sigma is assumed to be 1.0\n");
    }

    pip_get_integer(b"MinIterations", &mut min_iterations);
    pip_get_integer(b"MaxIterations", &mut max_iterations);
    pip_get_float(b"MaskedVarianceDifference", &mut masked_variance_difference);

    //manual
    if masked_variance_difference > 0.0 {
        if max_iterations < 0 {
            max_iterations = 6;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "MaxIterations not properly specified, is assumed to be %d\n",
                &[CArg::Int(max_iterations as i64)],
            ));
        }
    } else if !(max_iterations > 0 && min_iterations > 0 && max_iterations >= min_iterations) {
        if max_iterations < 0 && min_iterations < 0 {
            min_iterations = 3;
            max_iterations = 6;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "MaxIterations and MinIterations are not properly specified, assumed to be %d and %d\n",
                &[
                    CArg::Int(min_iterations as i64),
                    CArg::Int(max_iterations as i64),
                ],
            ));
        } else if max_iterations < 0 && min_iterations > 0 {
            max_iterations = min_iterations + 3;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "MaxIterations is not properly specified, assumed to be %d\n",
                &[CArg::Int(max_iterations as i64)],
            ));
        } else if max_iterations > 3 && min_iterations < 0 {
            min_iterations = max_iterations - 3;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "MinIterations is not properly specified, assumed to be %d\n",
                &[CArg::Int(min_iterations as i64)],
            ));
        } else if max_iterations >= 1 && min_iterations < 0 {
            min_iterations = 1;
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "MinIterations is not properly specified, assumed to be %d\n",
                &[CArg::Int(min_iterations as i64)],
            ));
        } else if max_iterations == 0 || min_iterations == 0 {
            let _ = ImodFile::Stdout.write_all(b"Warning: you might have no iterations for some tilt projections, I assume you know what you are doing\n");
        } else {
            exit_error(b"MaxIterations/MinIterations not properly specified\n");
        }
    }

    let _ = ImodFile::Stdout.write_all(b"parameters:\n");
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "       stackFn = \"%s\", outFn=\"%s\",  angleFn=\"%s\" ",
        &[
            CArg::Bytes(&stack_fn),
            CArg::Bytes(&out_fn),
            CArg::Bytes(if have_angle_file {
                angle_fn.as_slice()
            } else {
                b"(null)".as_slice()
            }),
        ],
    ));
    let _ = ImodFile::Stdout.write_all(b"\n       ");
    if min_iterations > 0 {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "minite=%d ",
            &[CArg::Int(min_iterations as i64)],
        ));
    }
    if max_iterations > 0 {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "maxite=%d ",
            &[CArg::Int(max_iterations as i64)],
        ));
    }
    if masked_variance_difference > 0. {
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "MaskedVarianceDifference=%f",
            &[CArg::Dbl(masked_variance_difference as f64)],
        ));
    }
    let _ = ImodFile::Stdout.write_all(b"\n");

    let Some(mut fp_stack) = ii_fopen(&stack_fn, "rb") else {
        let _ = ImodFile::Stdout.write_all(b"error 0.5\n");
        exit_error(&c_format_bytes(
            "could not open input file %s",
            &[CArg::Bytes(&stack_fn)],
        ));
    };

    let mut header = MrcHeader::default();
    let mut out_header = MrcHeader::default();
    let slice_mode: i32;
    /* read header */
    if mrc_head_read(&mut fp_stack, &mut header) != 0 {
        let _ = ImodFile::Stdout.write_all(b"Error 2\n");
        exit_error(&c_format_bytes(
            "reading header of input file %s",
            &[CArg::Bytes(&stack_fn)],
        ));
    }

    if mrc_head_read(&mut fp_stack, &mut out_header) != 0 {
        let _ = ImodFile::Stdout.write_all(b"Error 3\n");
        exit_error(&c_format_bytes(
            "reading header of input file %s",
            &[CArg::Bytes(&stack_fn)],
        ));
    }

    //check the file
    slice_mode = slice_mode_if_real(header.mode);
    if slice_mode < 0 {
        exit_error(&c_format_bytes(
            "File mode is %d; only byte, short integer, or real allowed",
            &[CArg::Int(header.mode as i64)],
        ));
    }
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "slice mode=%d\n",
        &[CArg::Int(slice_mode as i64)],
    ));

    let _ = starting_view;
    let num_slides = header.nz;
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "Number of slices=%d\n",
        &[CArg::Int(num_slides as i64)],
    ));

    //write the header
    let nx = header.nx as u32;
    let ny = header.ny as u32;

    let angle_sign: f32 = 1.;
    let mut min_angle: f32 = -20.;
    let mut max_angle: f32 = 20.;

    let mut tilt_angles: Vec<f32> = Vec::new();
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "reading angle file=%s\n",
        &[CArg::Bytes(if have_angle_file {
            angle_fn.as_slice()
        } else {
            b"(null)".as_slice()
        })],
    ));
    if !no_tilt_angle {
        tilt_angles = read_tilt_angles(
            &angle_fn,
            header.nz,
            angle_sign,
            &mut min_angle,
            &mut max_angle,
        );
    }

    let Some(mut foutput) = ii_fopen(&out_fn, "wb") else {
        exit_error(&c_format_bytes(
            "could not open output file %s",
            &[CArg::Bytes(&out_fn)],
        ));
    };

    // DNM: Set header correcty for a new output file and copy the extended header
    mrc_init_output_header(&mut out_header);
    out_header.fp = Some(foutput.clone());
    let ii_file = ii_lookup_file_from_fp(&fp_stack);
    if let Some(ii_file) = ii_file {
        /* `b3dOutputFileType() == OUTPUT_TYPE_MRC`, `b3dutil.c`'s value 2. */
        if unsafe { (*ii_file).file } == IIFILE_MRC && b3d_output_file_type() == 2 {
            let err = mrc_copy_extra_header(&mut header, &mut out_header);
            if err != 0 {
                exit_error(&c_format_bytes(
                    "Copying MRC extended header to output file (error # %d)",
                    &[CArg::Int(err as i64)],
                ));
            }
        }
    }

    mrc_head_label(&mut out_header, b"PreNAD filtered image");
    mrc_head_write(&mut foutput, &mut out_header);

    /* One element longer than the source's `malloc(header.nz * …)`, because
    `automaticPreNAD` reads index `nz`; see the module comment. */
    let mut stack_in: Vec<TiltProjectionType> =
        vec![TiltProjectionType::default(); header.nz as usize + 1];
    let mut stack_out: Vec<TiltProjectionType> =
        vec![TiltProjectionType::default(); header.nz as usize + 1];

    let spacing_x: f64 = 1.0;
    let spacing_y: f64 = 1.0;
    let lambda_e: f64 = 30.0;
    let lambda_c: f64 = 30.0;
    let lambda_h: f64 = 30.0;
    let mut ii: u32;

    //set the angles to process
    ii = 0;
    while ii < header.nz as u32 && range_requested {
        stack_in[ii as usize].skip_me = true;
        ii += 1;
    }

    if range_requested {
        for ii in 0..n_angles_to_process {
            if angles_to_process[ii] > 0 && angles_to_process[ii] <= header.nz {
                stack_in[(angles_to_process[ii] - 1) as usize].skip_me = false;
            } else {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "[WARNING] projection %d not in the views, ignored!\n",
                    &[CArg::Int(angles_to_process[ii] as i64)],
                ));
            }
        }
    }

    let hi = (nx as usize + padding as usize + padding as usize) as isize;
    let hj = (ny as usize + padding as usize + padding as usize) as isize;

    for ii in 0..header.nz as usize {
        let angle: f32 = if no_tilt_angle { 0.0 } else { tilt_angles[ii] };

        let Some(mut curr_slice) = slice_create(nx as i32, ny as i32, slice_mode) else {
            exit_error(b"Creating slice for input");
        };
        //get the type of data we are dealing with
        if mrc_read_slice(
            curr_slice.data.bytes_mut(),
            &mut fp_stack,
            &mut header,
            ii as i32,
            b'Z',
        ) != 0
        {
            exit_error(&c_format_bytes("Reading slice %d", &[CArg::Int(ii as i64)]));
        }

        // Convert slice to floats
        if slice_mode != SLICE_MODE_FLOAT && slice_new_mode(&mut curr_slice, SLICE_MODE_FLOAT) < 0 {
            exit_error(b"Converting slice to float");
        }

        let mut data_in = matrix(0, hi, 0, hj);
        let mut data_out = matrix(0, hi, 0, hj);
        stack_out[ii].angle_radiant = (angle as f64 * MY_PI) / 180.0;
        stack_out[ii].angle_degrees = angle as f64;
        stack_out[ii].mvd = 0.0;
        stack_out[ii].current_iteration = 0;
        stack_out[ii].tilt_number = ii as i32;
        /* `cosf` takes a float, so the double angle is narrowed first. */
        stack_out[ii].sigma = (sigma * (stack_out[ii].angle_radiant as f32).cos()) as f64;

        // Copy data into array
        for j in 0..ny as usize {
            for i in 0..nx as usize {
                data_in[(i + padding as usize, j + padding as usize)] =
                    curr_slice.data.f()[i + j * nx as usize];
                data_out[(i + padding as usize, j + padding as usize)] =
                    curr_slice.data.f()[i + j * nx as usize];
            }
        }

        filling_padding(&mut data_in, nx, ny, padding);
        filling_padding(&mut data_out, nx, ny, padding);
        stack_in[ii].data = Some(data_in);
        stack_out[ii].data = Some(data_out);
    }

    let _ = ImodFile::Stdout.write_all(b"start processing\n");
    // ********************************************************************
    // ********************** PRE NON-LINEAR ANISOTROPIC DIFFUSION
    // ********************************************************************

    automatic_pre_nad(
        &stack_in,
        &mut stack_out,
        nx,
        ny,
        header.nz as u32,
        padding,
        min_iterations as u32,
        max_iterations as u32,
        spacing_x,
        spacing_y,
        lambda_e,
        lambda_c,
        lambda_h,
    );

    // ********************************************************************
    // ****  END  ***************  PRE NON-LINEAR ANISOTROPIC DIFFUSION
    // ********************************************************************
    let _ = ImodFile::Stdout
        .write_all(b"\n***************\n write image data and close file \n***************\n");

    // DNM: get the new min/max/mean and write header at the end
    out_header.amin = 1.0e37;
    out_header.amax = -1.0e37;
    out_header.amean = 0.;
    for ii in 0..header.nz as usize {
        let Some(mut curr_slice) = slice_create(nx as i32, ny as i32, SLICE_MODE_FLOAT) else {
            exit_error(b"Creating slice for output");
        };
        {
            let data = stack_out[ii].data.as_ref().unwrap();
            for j in 0..ny as usize {
                for i in 0..nx as usize {
                    curr_slice.data.f_mut()[i + j * nx as usize] =
                        data[(i + padding as usize, j + padding as usize)];
                }
            }
        }

        // Convert if necessary and write slice
        if slice_mode != SLICE_MODE_FLOAT && slice_new_mode(&mut curr_slice, slice_mode) < 0 {
            exit_error(b"Converting slice to short");
        }
        if mrc_write_slice(
            curr_slice.data.bytes(),
            &mut foutput,
            &mut out_header,
            ii as i32,
            b'Z',
        ) != 0
        {
            exit_error(&c_format_bytes("Writing slice %d", &[CArg::Int(ii as i64)]));
        }

        slice_mmm(&mut curr_slice);
        /* ACCUM_MIN / ACCUM_MAX are `a = a < b ? a : b`, which keep the second
        operand when either is NaN. */
        if curr_slice.min < out_header.amin {
            out_header.amin = curr_slice.min;
        }
        if curr_slice.max > out_header.amax {
            out_header.amax = curr_slice.max;
        }
        out_header.amean += curr_slice.mean / out_header.nz as f32;

        if let Some(m) = stack_in[ii].data.take() {
            free_matrix(m, 0, hi, 0, hj);
        }
        if let Some(m) = stack_out[ii].data.take() {
            free_matrix(m, 0, hi, 0, hj);
        }
    }

    mrc_head_write(&mut foutput, &mut out_header);

    ii_fclose(&mut foutput);
    ii_fclose(&mut fp_stack);
    let _ = ImodFile::Stdout.flush();
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filling_padding_mirrors_the_frame() {
        let mut m = matrix(0, 6, 0, 6);
        for i in 0..6usize {
            for j in 0..6usize {
                m[(i, j)] = (i * 10 + j) as f32;
            }
        }
        /* nx = ny = 4, padding = 1 */
        filling_padding(&mut m, 4, 4, 1);
        assert_eq!(m[(0, 2)], m[(2, 2)]);
        assert_eq!(m[(5, 2)], m[(3, 2)]);
    }

    #[test]
    fn dlaev2_argument_is_two_adjacent_floats_as_one_double() {
        let mut m = matrix(0, 2, 0, 2);
        m[(0, 0)] = 1.5;
        m[(0, 1)] = 2.0;
        let expected = f64::from_bits(((2.0f32.to_bits() as u64) << 32) | 1.5f32.to_bits() as u64);
        assert_eq!(dlaev2_arg(&m, 0, 0), expected);
    }

    #[test]
    fn quartile_index_is_the_first_quarter_of_the_bubble_sorted_order() {
        let make = |mvd| TiltProjectionType {
            mvd,
            ..Default::default()
        };
        let stack = [make(3.), make(1.), make(2.), make(4.)];
        assert_eq!(get_quartile_mvd_index(&stack, 4), 2);
    }
}
