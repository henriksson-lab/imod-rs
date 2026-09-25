//! Translation of `IMOD/flib/tiltalign/fill_matrices.cpp`.
//!
//! Routines that set up a matrix for each of the individual changes, and the
//! conversion of alpha, tilt and rotation for beam tilt.  Linked into both
//! `tiltalign` and `beadtrack` (`flib/beadtrack/Makefile:48`, compiled with no
//! `-DBEADTRACK`, so there is one build of it).
//!
//! # Arithmetic
//!
//! The source includes `<math.h>` from C++, where libstdc++ brings the `float`
//! overloads of `cos`/`sin` into the global namespace: `cos(skew)` on a `float`
//! is `cosf`, so these are `f32::cos`/`f32::sin` here.  `convert_for_beamtilt`'s
//! `dtor` is the truncated literal `0.0174532` the source writes, not the exact
//! constant.  `mat_product` multiplies a `float` by a `double`, so each term is
//! widened before the product.
//!
//! # Not translated
//!
//! The six Fortran wrappers `mat_productfw`, `convert_for_beamtiltfw`
//! (`convert_for_beamtilt_`), `fill_dist_matrixfw`, `fill_xtilt_matrixfw`,
//! `fill_beam_matricesfw`, `fill_ytilt_matrixfw`, `fill_proj_matrixfw` and
//! `fill_rot_matrixfw` (`fill_matrices.cpp:66,129,172,202,231,251,274,294`)
//! only dereference their by-reference arguments.  Their one Fortran caller is
//! `tiltalign/xfforfidless.f:87`, a different program outside the ORDER.md §9
//! scope; per `TOFIX.md` "Stays deleted" a translated caller calls the C entry
//! directly.

use crate::imod::libcfshr::linearxforms::{icalc_angles, invert_matrix};

/// Original: `zero_matrix` (`fill_matrices.cpp:31`) — fills a matrix with zeros.
pub fn zero_matrix(rmat: &mut [f32], n: i32) {
    let mut i = 1;
    while i <= n {
        rmat[(i - 1) as usize] = 0.;
        i += 1;
    }
}

/// Original: `mat_product` (`fill_matrices.cpp:46`) — takes the product of
/// `rmat`, a `nmRows x nmCols` matrix, and `prod`, a `npRows x npCols` matrix
/// applied first, and places the resulting `nmRows x npCols` matrix back into
/// `prod`.  Matrices progress across rows.
#[inline]
pub fn mat_product(
    prod: &mut [f64],
    np_rows: i32,
    np_cols: i32,
    rmat: &[f32],
    nm_rows: i32,
    nm_cols: i32,
) {
    let _ = np_rows;
    let mut tmp = [[0f64; 3]; 3];
    for irow in 0..nm_rows {
        for icol in 0..np_cols {
            // `tmp[irow][icol]` is accumulated in a local, as gcc keeps it in
            // a register: the same `0. + t0 + t1 + ...` sequence of double
            // sums in the same order, without a store-to-load round trip on
            // `tmp` per term (which made this ~1.4x native).
            let mut sum = 0.;
            for i in 0..nm_cols {
                sum += rmat[(irow * nm_cols + i) as usize] as f64
                    * prod[(i * np_cols + icol) as usize];
            }
            tmp[irow as usize][icol as usize] = sum;
        }
    }
    for irow in 0..nm_rows {
        for icol in 0..np_cols {
            prod[(irow * np_cols + icol) as usize] = tmp[irow as usize][icol as usize];
        }
    }
}

/// Original: `convert_for_beamtilt` (`fill_matrices.cpp:78`) — converts the
/// given alpha, tilt angle, and rotation for the beam tilt angle.  `alpha`,
/// `tilt` and `rot` are in degrees, `beam_tilt` in radians.
pub fn convert_for_beamtilt(
    alpha: &mut f32,
    tilt: &mut f32,
    rot: &mut f32,
    beam_tilt: f32,
    if_any_alpha: i32,
) {
    let mut beam_mat = [0f32; 9];
    let mut beam_inv = [0f32; 9];
    let mut xtmat = [0f32; 9];
    let mut ytmat = [0f32; 9];
    let mut rmat = [0f32; 9];
    let mut angles = [0f32; 3];
    let (mut cos_alpha, mut sin_alpha, mut cos_beta, mut sin_beta) = (0f32, 0f32, 0f32, 0f32);
    let (mut cos_beam, mut sin_beam, mut cos_tmp, mut sin_tmp) = (0f32, 0f32, 0f32, 0f32);
    let mut pmat = [0f64; 9];
    let dtor: f32 = 0.0174532;
    //
    // Get a matrix that includes X tilt, beam tilt, tilt, and rotation, then
    // derive 3 rotation angles from that

    fill_beam_matrices(
        beam_tilt,
        &mut beam_inv,
        &mut beam_mat,
        &mut cos_beam,
        &mut sin_beam,
    );
    beam_mat[6] = 0.;
    beam_mat[7] = sin_beam;
    beam_mat[8] = cos_beam;
    fill_xtilt_matrix(
        *alpha * dtor,
        if_any_alpha,
        &mut xtmat,
        &mut cos_alpha,
        &mut sin_alpha,
    );
    fill_ytilt_matrix(*tilt * dtor, &mut ytmat, &mut cos_beta, &mut sin_beta);
    zero_matrix(&mut rmat, 9);
    fill_rot_matrix(*rot * dtor, &mut rmat, &mut cos_tmp, &mut sin_tmp);
    rmat[2] = 0.;
    rmat[3] = sin_tmp;
    rmat[4] = cos_tmp;
    rmat[8] = 1.;
    for i in 1..=9usize {
        pmat[i - 1] = xtmat[i - 1] as f64;
    }
    mat_product(&mut pmat, 3, 3, &beam_inv, 3, 3);
    mat_product(&mut pmat, 3, 3, &ytmat, 3, 3);
    mat_product(&mut pmat, 3, 3, &beam_mat, 3, 3);
    mat_product(&mut pmat, 3, 3, &rmat, 3, 3);
    //
    // transpose the matrix to go into the stock routines
    //
    for i in 1..=3usize {
        for j in 1..=3usize {
            xtmat[i + 3 * (j - 1) - 1] = pmat[j + 3 * (i - 1) - 1] as f32;
        }
    }
    invert_matrix(&xtmat, &mut ytmat);
    icalc_angles(&mut angles, &ytmat);
    //
    // replace the angles - then assume beam tilt 0 below
    //
    *rot = -angles[2];
    *tilt = -angles[1];
    *alpha = -angles[0];
}

/// Original: `fill_dist_matrix` (`fill_matrices.cpp:145`) — fills a 3x3
/// matrix `dmat` for in-plane distortions, given mag `gmag`, x-stretch `dmag`,
/// X-axis rotation angle `skew`, compression `comp`, and stretch type
/// `istrType`.
#[allow(clippy::too_many_arguments)]
pub fn fill_dist_matrix(
    gmag: f32,
    dmag: f32,
    skew: f32,
    comp: f32,
    istr_type: i32,
    dmat: &mut [f32],
    cos_delta: &mut f32,
    sin_delta: &mut f32,
) {
    let xmag: f32 = gmag + dmag;
    *cos_delta = skew.cos();
    *sin_delta = skew.sin();
    zero_matrix(dmat, 9);
    if istr_type == 1 {
        dmat[0] = xmag * *cos_delta;
        dmat[3] = xmag * *sin_delta;
        dmat[4] = gmag;
    } else if istr_type == 2 {
        dmat[0] = xmag * *cos_delta;
        dmat[1] = -xmag * *sin_delta;
        dmat[3] = -gmag * *sin_delta;
        dmat[4] = gmag * *cos_delta;
    } else {
        dmat[0] = (gmag - dmag) * *cos_delta;
        dmat[1] = -xmag * *sin_delta;
        dmat[3] = -(gmag - dmag) * *sin_delta;
        dmat[4] = xmag * *cos_delta;
    }
    dmat[8] = gmag * comp;
}

/// Original: `fill_xtilt_matrix` (`fill_matrices.cpp:182`) — fills 3x3 matrix
/// `xtmat` given X-axis tilt angle `alpha` if `ifAnyAlpha` is not zero;
/// otherwise it assumes alpha = 0.
pub fn fill_xtilt_matrix(
    alpha: f32,
    if_any_alpha: i32,
    xtmat: &mut [f32],
    cos_alpha: &mut f32,
    sin_alpha: &mut f32,
) {
    zero_matrix(xtmat, 9);
    xtmat[0] = 1.;
    xtmat[4] = 1.;
    xtmat[8] = 1.;
    if if_any_alpha != 0 {
        *cos_alpha = alpha.cos();
        *sin_alpha = alpha.sin();
        xtmat[4] = *cos_alpha;
        xtmat[5] = -*sin_alpha;
        xtmat[7] = *sin_alpha;
        xtmat[8] = *cos_alpha;
    } else {
        *cos_alpha = 1.;
        *sin_alpha = 0.;
    }
}

/// Original: `fill_beam_matrices` (`fill_matrices.cpp:212`) — fills 3x3 matrix
/// `beamInv` and 2x3 matrix `beamMat` given beam inclination angle `beamTilt`.
pub fn fill_beam_matrices(
    beam_tilt: f32,
    beam_inv: &mut [f32],
    beam_mat: &mut [f32],
    cos_beam: &mut f32,
    sin_beam: &mut f32,
) {
    //
    zero_matrix(beam_inv, 9);
    zero_matrix(beam_mat, 6);

    beam_inv[0] = 1.;
    beam_mat[0] = 1.;
    *cos_beam = beam_tilt.cos();
    *sin_beam = beam_tilt.sin();
    beam_inv[4] = *cos_beam;
    beam_inv[5] = *sin_beam;
    beam_inv[7] = -*sin_beam;
    beam_inv[8] = *cos_beam;
    beam_mat[4] = *cos_beam;
    beam_mat[5] = -*sin_beam;
}

/// Original: `fill_ytilt_matrix` (`fill_matrices.cpp:239`) — fills 3x3 matrix
/// `ytmat` given tilt angle `tilt`.
pub fn fill_ytilt_matrix(tilt: f32, ytmat: &mut [f32], cos_beta: &mut f32, sin_beta: &mut f32) {
    zero_matrix(ytmat, 9);
    *cos_beta = tilt.cos();
    *sin_beta = tilt.sin();
    ytmat[0] = *cos_beta;
    ytmat[2] = *sin_beta;
    ytmat[4] = 1.;
    ytmat[6] = -*sin_beta;
    ytmat[8] = *cos_beta;
}

/// Original: `fill_proj_matrix` (`fill_matrices.cpp:261`) — fills 2x2 matrix
/// `projMat` for projection skew given skew `projSkew` and the approximate
/// rotation angle of the axes, `projStrRot`.
#[allow(clippy::too_many_arguments)]
pub fn fill_proj_matrix(
    proj_str_rot: f32,
    proj_skew: f32,
    proj_mat: &mut [f32],
    cos_p_skew: &mut f32,
    sin_p_skew: &mut f32,
    cos2rot: &mut f32,
    sin2rot: &mut f32,
) {
    *cos_p_skew = proj_skew.cos();
    *sin_p_skew = proj_skew.sin();
    *cos2rot = (2. * proj_str_rot).cos();
    *sin2rot = (2. * proj_str_rot).sin();
    proj_mat[0] = *cos_p_skew + *sin_p_skew * *sin2rot;
    proj_mat[1] = -*sin_p_skew * *cos2rot;
    proj_mat[2] = proj_mat[1];
    proj_mat[3] = *cos_p_skew - *sin_p_skew * *sin2rot;
}

/// Original: `fill_rot_matrix` (`fill_matrices.cpp:284`) — fills 2x2 matrix
/// `rmat` given rotation angle `rot`.
pub fn fill_rot_matrix(rot: f32, rmat: &mut [f32], cos_gamma: &mut f32, sin_gamma: &mut f32) {
    *cos_gamma = rot.cos();
    *sin_gamma = rot.sin();
    rmat[0] = *cos_gamma;
    rmat[1] = -*sin_gamma;
    rmat[2] = *sin_gamma;
    rmat[3] = *cos_gamma;
}
