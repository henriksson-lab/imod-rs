//! Translation of `IMOD/mrc/preNID.cpp` — the Pre-NID (non-conservative
//! non-linear isotropic diffusion) filter contributed by Mauro Maiorca,
//! adapted for IMOD.
//!
//! As in `preNAD.cpp`, `nrutil.c` is `#include`d directly into this
//! translation unit, so the `float **` images are
//! `matrix(0, nx + 2*padding, 0, ny + 2*padding)` blocks; they are
//! `nrutil::NrMatrix<f32>` here with the source's subscripts.  `FillingPadding`
//! and `gaussRecursiveDerivatives1D` are this unit's own definitions, not
//! `preNAD.cpp`'s — the two programs each compile their own copy.
//!
//! One uninitialised read is reachable and is reproduced as a zero fill.
//! `CreateMaskedLocalSmooth` writes its local `mask` only over
//! `padding..nx+padding` before testing `mask[i][j] > 0.1` over the *whole*
//! padded extent (`preNID.cpp:437`), so on the first call the frame of that
//! `malloc` block is whatever the allocator handed over.  On the reference
//! binary that block comes off the top of the heap and reads as zero, and
//! every later call gets the same coalesced region back with the frame already
//! written to `0.0` by the `IMask` loop, so a zero-filled `Vec` is what the
//! reference observes.

use crate::imod::ctfplotter::ctfutils::read_tilt_angles;
use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, b3d_output_file_type, c_format_bytes, imod_prog_name,
};
use crate::imod::libcfshr::islice::{slice_create, slice_mode_if_real};
use crate::imod::libcfshr::parse_params::{
    exit_error, pip_get_boolean, pip_get_string, pip_print_help, pip_read_or_parse_options, strtod,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libiimod::iimage::{IIFILE_MRC, ii_fclose, ii_fopen, ii_lookup_file_from_fp};
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_copy_extra_header, mrc_head_label, mrc_head_read, mrc_head_write,
    mrc_init_output_header, mrc_read_slice, mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{slice_mmm, slice_new_mode};
use crate::imod::mrc::nrutil::{NrMatrix, free_matrix, matrix};
use crate::imod::mrc::recline::{
    DerivativeOrder, RecursiveFilterType, init_recursive_coefficients, recursive_filter_1d,
};
use std::io::Write as _;

const MY_PI: f64 = 3.141592653589;
const SLICE_MODE_FLOAT: i32 = 2;

/// C `PreLRTiltProjectionType`: support structure for iterating through
/// slices.
#[derive(Clone, Debug, Default)]
pub struct PreLRTiltProjectionType {
    /// angle in radiants
    pub angle_radiant: f64,
    /// angle in degrees
    pub angle_degrees: f64,
    /// sigma for the current tilt projection
    pub sigma: f64,
    /// The current tilt projection won't be processed if skipMe is set to true
    pub skip_me: bool,
    /// Current tilt projection number
    pub tilt_number: i32,
    /// the processed tilt projection data at the current iteration
    pub data: Option<NrMatrix<f32>>,
    /// the original tilt projection data (prior to processing)
    pub data_original: Option<NrMatrix<f32>>,
}

/// C `separateStringCommaValues`: parsing a comma separated string.
///
/// `std::getline(f, s, ',')` stops at end of file without producing a final
/// empty token, so a trailing comma adds no value — unlike `str::split`.
pub fn separate_string_comma_values(input_s: &[u8], values: &mut Vec<f32>) -> i32 {
    values.clear();
    if !input_s.is_empty() {
        let mut fields: Vec<&[u8]> = input_s.split(|c| *c == b',').collect();
        if input_s[input_s.len() - 1] == b',' {
            fields.pop();
        }
        for s in fields {
            let mut end = 0usize;
            values.push(strtod(s, &mut end) as f32);
        }
    }
    0
}

/// C `FillingPadding`: recompute the frame of an image as a mirror would do.
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

/// C `gaussRecursiveDerivatives1D`: 1D zero- and first-order Gaussian
/// derivatives of a 2D image.
///
/// `fo == NULL` means the input is overwritten; the smoothing call sites that
/// pass the same pointer twice are the same case, and are `None` here.
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

/// C `CreateMaskedLocalSmooth`: masks highly irregular regions on the image,
/// replaces them with a smoothed version, and grades the irregularity of each
/// pixel in `[0,1]`.
#[allow(clippy::too_many_arguments)]
pub fn create_masked_local_smooth(
    nx: u32,
    ny: u32,
    padding: u32,
    spacing_x: f64,
    spacing_y: f64,
    sigma_mask: f64,
    alpha_: f64,
    beta_: f64,
    tau: f64,
    _angle_radiant: f64,
    i_image: &mut NrMatrix<f32>,
    i_mask: &mut NrMatrix<f32>,
    i_grade_irragularity: &mut NrMatrix<f32>,
    masked_local_smoothed_image: &mut NrMatrix<f32>,
    normalized_dog: &mut NrMatrix<f32>,
) -> f64 {
    let (nxu, nyu, pad) = (nx as usize, ny as usize, padding as usize);
    let hi = (nxu + 2 * pad) as isize;
    let hj = (nyu + 2 * pad) as isize;

    //output mask file
    let mut mask = matrix(0, hi, 0, hj);

    //gaussian blurred images
    let mut i_blurred = matrix(0, hi, 0, hj);
    let mut i_blurred_a = matrix(0, hi, 0, hj);

    //Difference of Gaussians
    let mut dog = matrix(0, hi, 0, hj);
    let relevant_dog = matrix(0, hi, 0, hj);
    let mut d_dogdx = matrix(0, hi, 0, hj);
    let mut d_dogdy = matrix(0, hi, 0, hj);

    //gaussian blurred image
    gauss_recursive_derivatives_1d(
        sigma_mask,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        0,
        0,
        i_image,
        Some(&mut i_blurred),
    );
    gauss_recursive_derivatives_1d(
        sigma_mask,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        1,
        0,
        &mut i_blurred,
        None,
    );

    let mut mean_dog: f64 = (i_image[(pad, pad)] - i_blurred[(pad, pad)]) as f64;
    let mut max_dog: f64 = mean_dog;
    let mut min_dog: f64 = mean_dog;
    let counter: f64 = (nxu * nyu) as f64;

    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            dog[(i, j)] = i_image[(i, j)] - i_blurred[(i, j)];
        }
    }

    let sigma_mean = sigma_mask * alpha_;
    gauss_recursive_derivatives_1d(
        sigma_mean, nx, ny, padding, spacing_x, spacing_y, 0, 0, &mut dog, None,
    );
    gauss_recursive_derivatives_1d(
        sigma_mean, nx, ny, padding, spacing_x, spacing_y, 1, 0, &mut dog, None,
    );

    max_dog = 0.0;
    min_dog = 0.0;
    for i in pad..nxu + pad {
        for j in pad..nyu + pad {
            normalized_dog[(i, j)] = -dog[(i, j)];
            if normalized_dog[(i, j)] < 0. {
                normalized_dog[(i, j)] = 0.;
            }
            if normalized_dog[(i, j)] as f64 > max_dog {
                max_dog = normalized_dog[(i, j)] as f64;
            }
        }
    }
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "     minDoG=%f; maxDoG=%f\n",
        &[CArg::Dbl(min_dog), CArg::Dbl(max_dog)],
    ));

    for i in pad..nxu + pad {
        for j in pad..nyu + pad {
            normalized_dog[(i, j)] = (normalized_dog[(i, j)] as f64 / max_dog) as f32;

            if normalized_dog[(i, j)] > 0.0 {
                mask[(i, j)] = 1.0;
            } else {
                mask[(i, j)] = 0.0;
            }
        }
    }

    gauss_recursive_derivatives_1d(
        sigma_mean,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        0,
        1,
        &mut dog,
        Some(&mut d_dogdx),
    );
    gauss_recursive_derivatives_1d(
        sigma_mean,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        1,
        1,
        &mut dog,
        Some(&mut d_dogdy),
    );

    mean_dog = (d_dogdx[(pad, pad)] as f64).powf(2.0) + (d_dogdy[(pad, pad)] as f64).powf(2.0);
    max_dog = mean_dog;
    min_dog = mean_dog;
    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            dog[(i, j)] =
                ((d_dogdx[(i, j)] as f64).powf(2.0) + (d_dogdy[(i, j)] as f64).powf(2.0)) as f32;
            if dog[(i, j)] as f64 > max_dog {
                max_dog = dog[(i, j)] as f64;
            }
            if (dog[(i, j)] as f64) < min_dog {
                min_dog = dog[(i, j)] as f64;
            }
        }
    }
    mean_dog /= counter + 1.0;

    gauss_recursive_derivatives_1d(
        sigma_mean, nx, ny, padding, spacing_x, spacing_y, 0, 0, &mut dog, None,
    );
    gauss_recursive_derivatives_1d(
        sigma_mean, nx, ny, padding, spacing_x, spacing_y, 1, 0, &mut dog, None,
    );

    mean_dog = dog[(pad, pad)] as f64;
    max_dog = mean_dog;
    min_dog = mean_dog;
    let mut counter_mask: f64 = 0.;
    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            if mask[(i, j)] > 0.1 {
                if dog[(i, j)] as f64 > max_dog {
                    max_dog = dog[(i, j)] as f64;
                }
                if (dog[(i, j)] as f64) < min_dog {
                    min_dog = dog[(i, j)] as f64;
                }
                mean_dog += dog[(i, j)] as f64;
                counter_mask += 1.;
            }
        }
    }
    mean_dog /= counter_mask + 1.0;

    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            dog[(i, j)] = ((dog[(i, j)] as f64 - mean_dog) / (max_dog - min_dog)) as f32;
            if dog[(i, j)] < 0.0 {
                dog[(i, j)] = 0.0;
            }
        }
    }

    //gaussian blurred image with adapted sigma
    gauss_recursive_derivatives_1d(
        beta_,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        0,
        0,
        i_image,
        Some(&mut i_blurred_a),
    );
    gauss_recursive_derivatives_1d(
        beta_,
        nx,
        ny,
        padding,
        spacing_x,
        spacing_y,
        1,
        0,
        &mut i_blurred_a,
        None,
    );

    let mut irragularity_max: f64 = 0.0;
    let mut irragularity_min: f64 = 1.0;
    let irragularity_threshold: f64 = 0.005;
    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            if mask[(i, j)] > 0.01 && dog[(i, j)] as f64 > tau {
                i_mask[(i, j)] = dog[(i, j)];
                mask[(i, j)] = 1.0;
                masked_local_smoothed_image[(i, j)] = i_blurred_a[(i, j)];
            } else {
                masked_local_smoothed_image[(i, j)] = i_image[(i, j)];
                i_mask[(i, j)] = 0.0;
                mask[(i, j)] = 0.0;
            }
            if (i_mask[(i, j)] as f64) < irragularity_threshold {
                i_mask[(i, j)] = 0.;
            }
            if i_mask[(i, j)] as f64 > irragularity_max {
                irragularity_max = i_mask[(i, j)] as f64;
            }
            if (i_mask[(i, j)] as f64) < irragularity_min {
                irragularity_min = i_mask[(i, j)] as f64;
            }

            i_mask[(i, j)] = mask[(i, j)];
        }
    }

    let range_irregularitues: f64 = irragularity_max - irragularity_min;
    for i in 0..nxu + 2 * pad {
        for j in 0..nyu + 2 * pad {
            i_grade_irragularity[(i, j)] =
                ((i_mask[(i, j)] as f64 - irragularity_min) / range_irregularitues) as f32;
        }
    }

    //clear buffers
    free_matrix(i_blurred_a, 0, hi, 0, hj);
    free_matrix(i_blurred, 0, hi, 0, hj);
    free_matrix(mask, 0, hi, 0, hj);
    free_matrix(dog, 0, hi, 0, hj);
    free_matrix(relevant_dog, 0, hi, 0, hj);
    free_matrix(d_dogdx, 0, hi, 0, hj);
    free_matrix(d_dogdy, 0, hi, 0, hj);

    0.
}

/// C `LinearityEnhancingDiffusion`: the core of the algorithm.
#[allow(clippy::too_many_arguments)]
pub fn linearity_enhancing_diffusion(
    nx: u32,
    ny: u32,
    padding: u32,
    spacing_x: f64,
    spacing_y: f64,
    sigma: f64,
    _angle_radiant: f64,
    _iterations: i32,
    i_image: &NrMatrix<f32>,
    iout: &mut NrMatrix<f32>,
    mask: &NrMatrix<f32>,
    _mask_invariants: &NrMatrix<f32>,
) -> f64 {
    let (nxu, nyu, pad) = (nx as usize, ny as usize, padding as usize);

    let time_step: f64 = 0.1;
    let rxx = time_step / (1.0 * spacing_x * spacing_x);
    let ryy = time_step / (1.0 * spacing_y * spacing_y);
    let rxy = time_step / (1.4142 * spacing_x * spacing_y);

    //structure tensor enhancers
    let convolution_precision: u32 = (sigma + 1.0) as u32;
    let magnify_gradient_ratio: f64 =
        (1.0 / convolution_precision as f64) * (convolution_precision as f64).ln();
    let _ = magnify_gradient_ratio;

    for i in pad..nxu + pad {
        for j in pad..nyu + pad {
            //EVOLUTION
            let w_e = rxx * mask[(i, j)] as f64;
            let w_w = rxx * mask[(i, j)] as f64;
            let w_s = ryy * mask[(i, j)] as f64;
            let w_n = ryy * mask[(i, j)] as f64;
            let w_se = rxy * mask[(i, j)] as f64;
            let w_nw = rxy * mask[(i, j)] as f64;
            let w_ne = rxy * mask[(i, j)] as f64;
            let w_sw = rxy * mask[(i, j)] as f64;

            iout[(i, j)] = (i_image[(i, j)] as f64
                + w_e * (i_image[(i + 1, j)] - i_image[(i, j)]) as f64
                + w_w * (i_image[(i - 1, j)] - i_image[(i, j)]) as f64
                + w_s * (i_image[(i, j + 1)] - i_image[(i, j)]) as f64
                + w_n * (i_image[(i, j - 1)] - i_image[(i, j)]) as f64
                + w_se * (i_image[(i + 1, j + 1)] - i_image[(i, j)]) as f64
                + w_nw * (i_image[(i - 1, j - 1)] - i_image[(i, j)]) as f64
                + w_sw * (i_image[(i - 1, j + 1)] - i_image[(i, j)]) as f64
                + w_ne * (i_image[(i + 1, j - 1)] - i_image[(i, j)]) as f64)
                as f32;
        }
    }

    0.
}

/// C `automaticPreLR`: automatic stop condition and iteration through the
/// whole dataset.
#[allow(clippy::too_many_arguments)]
pub fn automatic_pre_lr(
    stack_in: &[PreLRTiltProjectionType],
    stack_out: &mut [PreLRTiltProjectionType],
    mask_in: &[PreLRTiltProjectionType],
    nx: u32,
    ny: u32,
    nz: u32,
    padding: u32,
    sigma_list: &[f32],
    alpha_sigma_list: &[f32],
    beta_sigma_list: &[f32],
    tau_list: &[f32],
    iterations_list: &[f32],
    spacing_x: f64,
    spacing_y: f64,
    _lambda_c: f64,
    mask_output: bool,
    abemus_mask: bool,
) {
    let (nxu, nyu, pad) = (nx as usize, ny as usize, padding as usize);
    let hi = (nxu + 2 * pad) as isize;
    let hj = (nyu + 2 * pad) as isize;

    let _ = ImodFile::Stdout.write_all(b"START (t=tilt, a=angle, i=iteration)\n");

    let mut j0 = matrix(0, hi, 0, hj);
    let mut j_image = matrix(0, hi, 0, hj);
    let mut j1 = matrix(0, hi, 0, hj);
    let j_smooth = matrix(0, hi, 0, hj);
    let mut mask = matrix(0, hi, 0, hj);
    let residual_image = matrix(0, hi, 0, hj);
    let mask_dilated = matrix(0, hi, 0, hj);
    let mut i_mask_dog = matrix(0, hi, 0, hj);
    // Image with irregularities graded: 0 none, 1 max
    let mut i_graded_irragularities = matrix(0, hi, 0, hj);

    //We have to reach the min number of iterations for each tilt angles
    for ii in 0..nz as usize {
        {
            let data = stack_in[ii].data.as_ref().unwrap();
            for i in 0..nxu + 2 * pad {
                for j in 0..nyu + 2 * pad {
                    j_image[(i, j)] = data[(i, j)];
                    j0[(i, j)] = data[(i, j)];
                    j1[(i, j)] = data[(i, j)];
                }
            }
        }

        for index in 0..sigma_list.len() {
            let sigma0 = sigma_list[index] as f64;
            let sigma_adapted = sigma0;
            let abs_angle = stack_out[ii].angle_radiant.abs();

            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "[ITERATE sigma=%f, alpha=%f, beta=%f, tau=%f,] ",
                &[
                    CArg::Dbl(sigma0),
                    CArg::Dbl(alpha_sigma_list[index] as f64),
                    CArg::Dbl(beta_sigma_list[index] as f64),
                    CArg::Dbl(tau_list[index] as f64),
                ],
            ));

            for i in 0..nxu + 2 * pad {
                for j in 0..nyu + 2 * pad {
                    j_image[(i, j)] = j1[(i, j)];
                }
            }

            if !abemus_mask {
                //create the mask
                create_masked_local_smooth(
                    nx,
                    ny,
                    padding,
                    spacing_x,
                    spacing_y,
                    sigma_adapted,
                    alpha_sigma_list[index] as f64,
                    beta_sigma_list[index] as f64,
                    tau_list[index] as f64,
                    abs_angle,
                    &mut j_image,
                    &mut mask,
                    &mut i_graded_irragularities,
                    &mut j1,
                    &mut i_mask_dog,
                );
            } else {
                //import the mask
                let data = mask_in[ii].data.as_ref().unwrap();
                for i in 0..nxu + 2 * pad {
                    for j in 0..nyu + 2 * pad {
                        mask[(i, j)] = data[(i, j)];
                    }
                }
            }

            let mut iii: i32 = 0;
            while (iii as f32) < iterations_list[index] {
                linearity_enhancing_diffusion(
                    nx,
                    ny,
                    padding,
                    spacing_x,
                    spacing_y,
                    sigma_list[index] as f64,
                    stack_out[ii].angle_radiant,
                    1,
                    &j1,
                    &mut j_image,
                    &mask,
                    &mask,
                );
                for i in 0..nxu + 2 * pad {
                    for j in 0..nyu + 2 * pad {
                        j1[(i, j)] = j_image[(i, j)];
                    }
                }
                iii += 1;
            }
        }

        let data = stack_out[ii].data.as_mut().unwrap();
        for i in pad..nxu + pad {
            for j in pad..nyu + pad {
                if mask_output {
                    data[(i, j)] = 100. * mask[(i, j)];
                } else {
                    data[(i, j)] = j1[(i, j)];
                }
            }
        }
    }

    free_matrix(i_graded_irragularities, 0, hi, 0, hj);
    free_matrix(j_smooth, 0, hi, 0, hj);
    free_matrix(residual_image, 0, hi, 0, hj);
    free_matrix(mask, 0, hi, 0, hj);
    free_matrix(j0, 0, hi, 0, hj);
    free_matrix(j_image, 0, hi, 0, hj);
    free_matrix(j1, 0, hi, 0, hj);
    free_matrix(mask_dilated, 0, hi, 0, hj);
    free_matrix(i_mask_dog, 0, hi, 0, hj);
}

/// C `main` in `preNID.cpp`.
pub fn pre_nid(argv: &[String]) -> i32 {
    let argc = argv.len() as i32;
    let mut stack_fn: Vec<u8> = Vec::new();
    let mut mask_fn: Vec<u8> = Vec::new();
    let mut angle_fn: Vec<u8> = Vec::new();
    let mut out_fn: Vec<u8> = Vec::new();
    let mut views_to_process: Vec<u8> = Vec::new();
    let mut sigma_text: Vec<u8> = Vec::new();
    let mut alpha_text: Vec<u8> = Vec::new();
    let mut beta_text: Vec<u8> = Vec::new();
    let mut tau_text: Vec<u8> = Vec::new();
    let mut iterations_text: Vec<u8> = Vec::new();
    let mut mask_output: i32 = 0;
    let mut starting_view: i32 = 0;

    let progname = imod_prog_name(&argv[0]);

    let mut num_opt_args = 0i32;
    let mut num_non_opt_args = 0i32;

    let padding: u32 = 40;
    let mut range_requested = false;
    let mut no_tilt_angle = false;
    let mut got_mask_output = false;
    let mut got_mask_input;

    let mut sigma_list: Vec<f32> = Vec::new();
    let mut alpha_sigma_list: Vec<f32> = Vec::new();
    let mut beta_sigma_list: Vec<f32> = Vec::new();
    let mut tau_list: Vec<f32> = Vec::new();
    let mut iterations_list: Vec<f32> = Vec::new();

    //parselist of views
    let mut angles_to_process: Vec<i32> = Vec::new();
    let mut n_angles_to_process: usize = 0;

    // Fallbacks from    ../manpages/autodoc2man 2 1 preNID
    let num_options = 11;
    let options: [&[u8]; 11] = [
        b"input:InputStack:FN:",
        b"output:OutputFileName:FN:",
        b"angles:AnglesFile:FN:",
        b"s:Sigma:FA:",
        b"a:Alpha:FA:",
        b"b:Beta:FA:",
        b"t:Tau:FA:",
        b"ite:Iterations:IA:",
        b"im:InputMask:FN:",
        b"mask:MaskOutput:B:",
        b"views:ViewsToProcess:LI:",
    ];

    let mut out = ImodFile::Stdout;
    let _ = out.write_all(b"\n [Mauro Maiorca, of the Biochemistry & Molecular Biology Department, Bio21 institute, University of Melbourne, Australia, contributed the preNID program (adapted for IMOD). ");
    let _ = out.write_all(b"It uses recursive line filter routines from Gregoire Malandain, covered by version 3 of the GPL (see GPL-3.0.txt). ");
    let _ = out.write_all(b" examples of use:\n\n");

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
    if pip_get_string(b"InputMask", &mut mask_fn) != 0 {
        mask_fn = b"none".to_vec();
        got_mask_input = false;
    } else {
        got_mask_input = true;
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
    if pip_get_string(b"Sigma", &mut sigma_text) == 0 {
        separate_string_comma_values(&sigma_text, &mut sigma_list);
        if sigma_list.is_empty() {
            exit_error(b"No sigma is specified, aborting...");
        }
    } else {
        exit_error(b"No sigma is specified, aborting...");
    }
    if pip_get_string(b"Alpha", &mut alpha_text) == 0 {
        separate_string_comma_values(&alpha_text, &mut alpha_sigma_list);
        if alpha_sigma_list.len() != sigma_list.len() {
            exit_error(
                b"Error with '--Alpha' option: be sure it is consistent with the '--Sigma' option. Aborting...",
            );
        }
    } else {
        alpha_sigma_list = vec![0.5; sigma_list.len()];
        let _ = ImodFile::Stdout.write_all(b"No alpha inserted, using default value 0.5\n");
    }
    if pip_get_boolean(b"MaskOutput", &mut mask_output) == 0 {
        got_mask_output = true;
    }
    if pip_get_string(b"Beta", &mut beta_text) == 0 {
        separate_string_comma_values(&beta_text, &mut beta_sigma_list);
        if beta_sigma_list.len() != sigma_list.len() {
            exit_error(
                b"Error with '--Beta' option: be sure it is consistent with the '--Sigma' option. Aborting...",
            );
        }
    } else {
        beta_sigma_list = vec![0.5; sigma_list.len()];
        let _ = ImodFile::Stdout.write_all(b"No beta inserted, using default value 0.5\n");
    }
    if pip_get_string(b"Tau", &mut tau_text) == 0 {
        separate_string_comma_values(&tau_text, &mut tau_list);
        if tau_list.len() != sigma_list.len() {
            exit_error(
                b"Error with '--Tau' option: be sure it is consistent with the '--Sigma' option. Aborting...",
            );
        }
    } else {
        tau_list = vec![0.1; sigma_list.len()];
        let _ = ImodFile::Stdout.write_all(b"No tau inserted, using default value 0\n");
    }

    if pip_get_string(b"Iterations", &mut iterations_text) == 0 {
        separate_string_comma_values(&iterations_text, &mut iterations_list);
        if iterations_list.len() != sigma_list.len() {
            exit_error(
                b"Error with '--Iterations' option: be sure it is consistent with the '--sigma' option. Aborting...",
            );
        }
    } else {
        iterations_list = vec![1.0; sigma_list.len()];
        let _ = ImodFile::Stdout.write_all(b"No Iterations inserted, using default value 1.0\n");
    }

    let _ = ImodFile::Stdout.write_all(b"parameters:\n");
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "       stackFn = \"%s\", outFn=\"%s\",  angleFn=\"%s InputMask=\"%s\"",
        &[
            CArg::Bytes(&stack_fn),
            CArg::Bytes(&out_fn),
            CArg::Bytes(if have_angle_file {
                angle_fn.as_slice()
            } else {
                b"(null)".as_slice()
            }),
            CArg::Bytes(&mask_fn),
        ],
    ));
    let _ = ImodFile::Stdout.write_all(b"\n       ");

    let Some(mut fp_stack) = ii_fopen(&stack_fn, "rb") else {
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
    let num_slices = header.nz;
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "Number of slices=%d\n",
        &[CArg::Int(num_slices as i64)],
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

    mrc_head_label(&mut out_header, b"PreNID filtered image");
    mrc_head_write(&mut foutput, &mut out_header);

    let mut stack_in: Vec<PreLRTiltProjectionType> =
        vec![PreLRTiltProjectionType::default(); header.nz as usize];
    let mut stack_out: Vec<PreLRTiltProjectionType> =
        vec![PreLRTiltProjectionType::default(); header.nz as usize];
    let mut stack_mask: Vec<PreLRTiltProjectionType> =
        vec![PreLRTiltProjectionType::default(); header.nz as usize];

    let spacing_x: f64 = 1.0;
    let spacing_y: f64 = 1.0;
    let lambda_e: f64 = 30.0;
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
        stack_out[ii].tilt_number = ii as i32;

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

    //Get mask if any
    if got_mask_input {
        let slice_mask_mode: i32;
        let Some(mut fp_mask) = ii_fopen(&mask_fn, "rb") else {
            exit_error(&c_format_bytes(
                "could not open input file %s",
                &[CArg::Bytes(&mask_fn)],
            ));
        };

        let mut mask_header = MrcHeader::default();
        if mrc_head_read(&mut fp_mask, &mut mask_header) != 0 {
            exit_error(&c_format_bytes(
                "reading header of input file %s",
                &[CArg::Bytes(&mask_fn)],
            ));
        }

        slice_mask_mode = slice_mode_if_real(mask_header.mode);
        if slice_mask_mode < 0 {
            exit_error(&c_format_bytes(
                "File mode is %d; only byte, short integer, or real allowed",
                &[CArg::Int(mask_header.mode as i64)],
            ));
        }

        if header.nz != mask_header.nz || header.nx != mask_header.nx || header.ny != mask_header.ny
        {
            exit_error(b"Mask size not compatible with input image size\n");
        }

        for ii in 0..header.nz as usize {
            let Some(mut curr_mask_slice) = slice_create(nx as i32, ny as i32, slice_mask_mode)
            else {
                exit_error(b"Creating slice for mask");
            };
            if mrc_read_slice(
                curr_mask_slice.data.bytes_mut(),
                &mut fp_mask,
                &mut mask_header,
                ii as i32,
                b'Z',
            ) != 0
            {
                exit_error(&c_format_bytes(
                    "Reading mask slide %d",
                    &[CArg::Int(ii as i64)],
                ));
            }

            // Convert slice to byte
            if slice_mask_mode != SLICE_MODE_FLOAT
                && slice_new_mode(&mut curr_mask_slice, SLICE_MODE_FLOAT) < 0
            {
                exit_error(b"Converting slice to float");
            }

            let mut data = matrix(0, hi, 0, hj);

            // erasing data
            for j in 0..ny as usize + 2 * padding as usize {
                for i in 0..nx as usize + 2 * padding as usize {
                    data[(i, j)] = 0.;
                }
            }

            // Copy data into array
            for j in 0..ny as usize {
                for i in 0..nx as usize {
                    let value: f32 = if curr_mask_slice.data.f()[i + j * nx as usize] > 0.000001 {
                        1.0
                    } else {
                        0.0
                    };

                    data[(i + padding as usize, j + padding as usize)] = value;
                }
            }
            stack_mask[ii].data = Some(data);
        }
        ii_fclose(&mut fp_mask);
    }

    let _ = ImodFile::Stdout.write_all(b"start processing\n");
    // ********************************************************************
    // ********************** PRE NON-LINEAR ANISOTROPIC DIFFUSION
    // ********************************************************************

    automatic_pre_lr(
        &stack_in,
        &mut stack_out,
        &stack_mask,
        nx,
        ny,
        header.nz as u32,
        padding,
        &sigma_list,
        &alpha_sigma_list,
        &beta_sigma_list,
        &tau_list,
        &iterations_list,
        spacing_x,
        spacing_y,
        lambda_e,
        got_mask_output,
        got_mask_input,
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
        if got_mask_input {
            if let Some(m) = stack_mask[ii].data.take() {
                free_matrix(m, 0, hi, 0, hj);
            }
        }
    }

    mrc_head_write(&mut foutput, &mut out_header);

    ii_fclose(&mut fp_stack);
    ii_fclose(&mut foutput);
    let _ = ImodFile::Stdout.flush();
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_values_drop_a_trailing_empty_field_as_getline_does() {
        let mut values = Vec::new();
        separate_string_comma_values(b"3,4.5,", &mut values);
        assert_eq!(values, vec![3.0, 4.5]);
        separate_string_comma_values(b"1,,2", &mut values);
        assert_eq!(values, vec![1.0, 0.0, 2.0]);
        separate_string_comma_values(b"", &mut values);
        assert!(values.is_empty());
    }

    #[test]
    fn filling_padding_mirrors_the_frame() {
        let mut m = matrix(0, 6, 0, 6);
        for i in 0..6usize {
            for j in 0..6usize {
                m[(i, j)] = (i * 10 + j) as f32;
            }
        }
        filling_padding(&mut m, 4, 4, 1);
        assert_eq!(m[(0, 2)], m[(2, 2)]);
        assert_eq!(m[(5, 2)], m[(3, 2)]);
    }

    #[test]
    fn a_zero_mask_leaves_the_image_unchanged() {
        let nx = 6u32;
        let ny = 6u32;
        let padding = 2u32;
        let hi = (nx + 2 * padding) as isize;
        let hj = (ny + 2 * padding) as isize;
        let mut source = matrix(0, hi, 0, hj);
        for i in 0..(nx + 2 * padding) as usize {
            for j in 0..(ny + 2 * padding) as usize {
                source[(i, j)] = (i + j) as f32;
            }
        }
        let mask = matrix(0, hi, 0, hj);
        let mut out = matrix(0, hi, 0, hj);
        linearity_enhancing_diffusion(
            nx, ny, padding, 1., 1., 3., 0., 1, &source, &mut out, &mask, &mask,
        );
        for i in padding as usize..(nx + padding) as usize {
            for j in padding as usize..(ny + padding) as usize {
                assert_eq!(out[(i, j)], source[(i, j)]);
            }
        }
    }
}
