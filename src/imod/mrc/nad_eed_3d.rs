//! Translation of `IMOD/mrc/nad_eed_3d.c` — nonlinear anisotropic diffusion,
//! edge enhancing, 3D (A. Frangakis / R. Hegerl, adapted for IMOD).
//!
//! The source stores every field as a Numerical Recipes `float ***` with a
//! one-voxel periodic halo, allocated by `f3tensor(0, nx+1, 0, ny+1, 0, nz+1)`
//! and addressed `v[i][j][k]` for `i` in `1..=nx`.  `nrutil::NrTensor3` is that
//! same block owned; the subscripts below are the source's subscripts.
//!
//! Intermediate types are read off the C declarations rather than inferred:
//! `sqrt()` and `exp()` are the *double* functions, so every weight in `eed`
//! that subtracts a `sqrt` is evaluated in double and rounded back to `float`
//! on assignment, while the neighbour differences that follow are pure
//! `float`.  Likewise `analyse` divides a `float` by a `long`, which C
//! converts to a `float` division, not a double one.

use crate::imod::libcfshr::b3dutil::{
    CArg, ImodFile, c_format_bytes, imod_backup_file, imod_prog_name, pid_to_stderr,
    set_output_type_from_string,
};
use crate::imod::libcfshr::dsyevq3::dsyevq3;
use crate::imod::libcfshr::dsyevv3::dsyevv3;
use crate::imod::libcfshr::islice::{slice_create, slice_mode_if_real};
use crate::imod::libcfshr::parse_params::{exit_error, pip_exit_on_error, strtod, strtol};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libcfshr::simplestat::eigen_sort;
use crate::imod::libiimod::iimage::{ii_fclose, ii_fopen};
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_head_label, mrc_head_read, mrc_head_write, mrc_init_output_header,
    mrc_read_slice, mrc_write_slice,
};
use crate::imod::libiimod::mrcslice::{slice_mmm, slice_new_mode};
use crate::imod::mrc::nrutil::{NrTensor3, f3tensor, free_f3tensor, free_vector, vector};
use std::io::Write as _;

const SLICE_MODE_FLOAT: i32 = 2;

/// C `dummies`: creates dummy boundaries by periodical continuation.
pub fn dummies(v: &mut NrTensor3<f32>, nx: usize, ny: usize, nz: usize) {
    for i in 1..=nx {
        for j in 1..=ny {
            /* first level of the extended image is */
            /* equal to the last level of the       */
            /* original image                       */
            v[(i, j, 0)] = v[(i, j, nz)];
            v[(i, j, nz + 1)] = v[(i, j, 1)];
        }
    }

    for j in 1..=ny {
        for k in 0..=nz + 1 {
            v[(0, j, k)] = v[(nx, j, k)];
            v[(nx + 1, j, k)] = v[(1, j, k)];
        }
    }

    for k in 0..=nz + 1 {
        for i in 0..=nx + 1 {
            v[(i, 0, k)] = v[(i, ny, k)];
            v[(i, ny + 1, k)] = v[(i, 1, k)];
        }
    }
}

/// C `analyse`: minimum, maximum, mean and variance of an image `u`.
#[allow(clippy::too_many_arguments)]
pub fn analyse(
    u: &NrTensor3<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    min: &mut f32,
    max: &mut f32,
    mean: &mut f32,
    vari: &mut f32,
) {
    let mut help: f32;
    let mut help2: f64;

    *min = u[(1, 1, 1)];
    *max = u[(1, 1, 1)];
    help2 = 0.0;

    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                if u[(i, j, k)] < *min {
                    *min = u[(i, j, k)];
                }
                if u[(i, j, k)] > *max {
                    *max = u[(i, j, k)];
                }
                help2 += u[(i, j, k)] as f64;
            }
        }
    }
    /* `(float)help2 / (nx * ny * nz)` is a *float* division: C converts the
    `long` operand to `float`, it does not widen the left side to double. */
    *mean = (help2 as f32) / ((nx * ny * nz) as f32);

    *vari = 0.0;
    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                help = u[(i, j, k)] - *mean;
                *vari += help * help;
            }
        }
    }
    *vari /= (nx * ny * nz) as f32;
}

/// C `gauss_conv`: Gaussian convolution with periodic boundaries.
#[allow(clippy::too_many_arguments)]
pub fn gauss_conv(
    sigma: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: f32,
    hy: f32,
    hz: f32,
    precision: i32,
    f: &mut NrTensor3<f32>,
) {
    let mut length: usize;
    let mut sum: f32;

    /* ------------------------ diffusion in x direction -------------------- */

    /* calculate length of convolution vector */
    length = (precision + 1) as usize;
    if length > nx {
        let _ = ImodFile::Stdout.write_all(b"gauss_conv: sigma too large \n");
        let _ = ImodFile::Stdout.flush();
        std::process::exit(0);
    }

    /* allocate storage for convolution vector */
    let mut conv = vector(0, length as isize);

    /* calculate entries of convolution vector */
    for i in 0..=length {
        conv[i] = (1.0 / (sigma as f64 * (2.0 * 3.1415927_f64).sqrt())
            * ((-(((i * i) as f32) * hx * hx)) as f64 / (2.0 * sigma as f64 * sigma as f64)).exp())
            as f32;
    }

    /* normalisation */
    sum = conv[0];
    for i in 1..=length {
        sum = (sum as f64 + 2.0 * conv[i] as f64) as f32;
    }
    for i in 0..=length {
        conv[i] /= sum;
    }

    /* allocate storage for a row */
    let mut help = vector(0, (nx + length + length - 1) as isize);

    for j in 1..=ny {
        for k in 1..=nz {
            /* copy in row vector */
            for i in 1..=nx {
                help[i + length - 1] = f[(i, j, k)];
            }

            for p in 1..=length {
                help[length - p] = help[nx + length - p];
                help[nx + length - 1 + p] = help[length + p - 1];
            }

            /* convolution step */
            for i in length..=nx + length - 1 {
                /* calculate convolution */
                sum = conv[0] * help[i];
                for p in 1..=length {
                    sum += conv[p] * (help[i + p] + help[i - p]);
                }
                /* write back */
                f[(i - length + 1, j, k)] = sum;
            }
        }
    }

    free_vector(help, 0, (nx + length + length - 1) as isize);
    free_vector(conv, 0, length as isize);

    /* ------------------------ diffusion in y direction -------------------- */

    length = (precision + 1) as usize;
    if length > ny {
        let _ = ImodFile::Stdout.write_all(b"gauss_conv: sigma too large \n");
        let _ = ImodFile::Stdout.flush();
        std::process::exit(0);
    }

    let mut conv = vector(0, length as isize);

    for j in 0..=length {
        conv[j] = (1.0 / (sigma as f64 * (2.0 * 3.1415927_f64).sqrt())
            * ((-(((j * j) as f32) * hy * hy)) as f64 / (2.0 * sigma as f64 * sigma as f64)).exp())
            as f32;
    }

    /* normalization */
    sum = conv[0];
    for j in 1..=length {
        sum = (sum as f64 + 2.0 * conv[j] as f64) as f32;
    }
    for j in 0..=length {
        conv[j] /= sum;
    }

    let mut help = vector(0, (ny + length + length - 1) as isize);

    for i in 1..=nx {
        for k in 1..=nz {
            /* copy in column vector */
            for j in 1..=ny {
                help[j + length - 1] = f[(i, j, k)];
            }

            /* assign boundary conditions */
            for p in 1..=length {
                help[length - p] = help[ny + length - p];
                help[ny + length - 1 + p] = help[length + p - 1];
            }

            /* convolution step */
            for j in length..=ny + length - 1 {
                sum = conv[0] * help[j];
                for p in 1..=length {
                    sum += conv[p] * (help[j + p] + help[j - p]);
                }
                f[(i, j - length + 1, k)] = sum;
            }
        }
    }

    free_vector(help, 0, (ny + length + length - 1) as isize);
    free_vector(conv, 0, length as isize);

    /* ------------------------ diffusion in z direction -------------------- */

    /* The source's `length > nz` guard is commented out here. */
    length = (precision + 1) as usize;

    let mut conv = vector(0, length as isize);

    for k in 0..=length {
        conv[k] = (1.0 / (sigma as f64 * (2.0 * 3.1415927_f64).sqrt())
            * ((-(((k * k) as f32) * hz * hz)) as f64 / (2.0 * sigma as f64 * sigma as f64)).exp())
            as f32;
    }

    sum = conv[0];
    for k in 1..=length {
        sum = (sum as f64 + 2.0 * conv[k] as f64) as f32;
    }
    for k in 0..=length {
        conv[k] /= sum;
    }

    let mut help = vector(0, (nz + length + length - 1) as isize);

    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                help[k + length - 1] = f[(i, j, k)];
            }

            for p in 1..=length {
                help[length - p] = help[nz + length - p];
                help[nz + length - 1 + p] = help[length + p - 1];
            }

            for k in length..=nz + length - 1 {
                sum = conv[0] * help[k];
                for p in 1..=length {
                    sum += conv[p] * (help[k + p] + help[k - p]);
                }
                f[(i, j, k - length + 1)] = sum;
            }
        }
    }

    free_vector(help, 0, (nz + length + length - 1) as isize);
    free_vector(conv, 0, length as isize);
}

/// C `struct_tensor_eed`: calculates the structure tensor.
#[allow(clippy::too_many_arguments)]
pub fn struct_tensor_eed(
    v: &mut NrTensor3<f32>,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: f32,
    hy: f32,
    hz: f32,
    sigma: f32,
    dxx: &mut NrTensor3<f32>,
    dxy: &mut NrTensor3<f32>,
    dxz: &mut NrTensor3<f32>,
    dyy: &mut NrTensor3<f32>,
    dyz: &mut NrTensor3<f32>,
    dzz: &mut NrTensor3<f32>,
    grd: &mut NrTensor3<f32>,
) {
    let (mut dv_dx, mut dv_dy, mut dv_dz);
    let (two_hx, two_hy, two_hz): (f32, f32, f32);

    /* ---- building tensor product ---- */

    two_hx = (2.0 * hx as f64) as f32; /*norming element -depends on Pixel size */
    two_hy = (2.0 * hy as f64) as f32;
    two_hz = (2.0 * hz as f64) as f32;
    dummies(v, nx, ny, nz);

    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                dv_dx = (v[(i + 1, j, k)] - v[(i - 1, j, k)]) / two_hx;
                dv_dy = (v[(i, j + 1, k)] - v[(i, j - 1, k)]) / two_hy;
                dv_dz = (v[(i, j, k + 1)] - v[(i, j, k - 1)]) / two_hz;
                dxx[(i, j, k)] = dv_dx * dv_dx;
                dxy[(i, j, k)] = dv_dx * dv_dy;
                dxz[(i, j, k)] = dv_dz * dv_dx;
                dyy[(i, j, k)] = dv_dy * dv_dy;
                dyz[(i, j, k)] = dv_dz * dv_dy;
                dzz[(i, j, k)] = dv_dz * dv_dz;
                grd[(i, j, k)] =
                    ((dv_dx * dv_dx + dv_dy * dv_dy + dv_dz * dv_dz) as f64).sqrt() as f32;
            }
        }
    }

    /* Smooth the gradient */

    if sigma > 0.0 {
        gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, dxx);
        gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, dxy);
        gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, dxz);
        gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, dyy);
        gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, dyz);
        gauss_conv(sigma, nx, ny, nz, hx, hy, hz, 2, dzz);
    }
}

/// C `read_struct_tens`: load the upper triangle of the structure tensor.
///
/// The lower triangle is left as the caller's `double a[3][3]` had it — the
/// source never writes it and `dsyevq3`/`dsyevv3` never read it.
#[allow(clippy::too_many_arguments)]
pub fn read_struct_tens(
    v_dxx: f32,
    v_dxy: f32,
    v_dxz: f32,
    v_dyy: f32,
    v_dyz: f32,
    v_dzz: f32,
    a: &mut [[f64; 3]; 3],
) {
    /* DNM: This loads the upper triangle of matrix */
    a[0][0] = v_dxx as f64;
    a[0][1] = v_dxy as f64;
    a[0][2] = v_dxz as f64;
    a[1][1] = v_dyy as f64;
    a[1][2] = v_dyz as f64;
    a[2][2] = v_dzz as f64;
}

/// C `read_ei`: read the eigenvalues and eigenvectors.
#[allow(clippy::too_many_arguments)]
pub fn read_ei(
    w: &[f64; 3],
    a: &[[f64; 3]; 3],
    ev11: &mut f32,
    ev12: &mut f32,
    ev13: &mut f32,
    ev21: &mut f32,
    ev22: &mut f32,
    ev23: &mut f32,
    ev31: &mut f32,
    ev32: &mut f32,
    ev33: &mut f32,
    lam1: &mut f32,
    lam2: &mut f32,
    lam3: &mut f32,
) {
    /* Output */
    *ev11 = a[0][0] as f32;
    *ev12 = a[1][0] as f32;
    *ev13 = a[2][0] as f32;
    *ev21 = a[0][1] as f32;
    *ev22 = a[1][1] as f32;
    *ev23 = a[2][1] as f32;
    *ev31 = a[0][2] as f32;
    *ev32 = a[1][2] as f32;
    *ev33 = a[2][2] as f32;

    *lam1 = w[0] as f32;
    *lam2 = w[1] as f32;
    *lam3 = w[2] as f32;
}

/// C `PA_backtrans`: principal axis backtransformation of a symmetric
/// (3*3)-matrix.
#[allow(clippy::too_many_arguments)]
pub fn pa_backtrans(
    ev11: f32,
    ev12: f32,
    ev13: f32,
    ev21: f32,
    ev22: f32,
    ev23: f32,
    ev31: f32,
    ev32: f32,
    ev33: f32,
    lam1: f32,
    lam2: f32,
    lam3: f32,
    a11: &mut f32,
    a12: &mut f32,
    a13: &mut f32,
    a22: &mut f32,
    a23: &mut f32,
    a33: &mut f32,
) {
    *a11 = lam1 * ev11 * ev11 + lam2 * ev21 * ev21 + lam3 * ev31 * ev31;
    *a12 = lam1 * ev11 * ev12 + lam2 * ev21 * ev22 + lam3 * ev31 * ev32;
    *a13 = lam1 * ev11 * ev13 + lam2 * ev21 * ev23 + lam3 * ev31 * ev33;
    *a22 = lam1 * ev12 * ev12 + lam2 * ev22 * ev22 + lam3 * ev32 * ev32;
    *a23 = lam1 * ev12 * ev13 + lam2 * ev22 * ev23 + lam3 * ev32 * ev33;
    *a33 = lam1 * ev13 * ev13 + lam2 * ev23 * ev23 + lam3 * ev33 * ev33;
}

/// C `diff_tensor`: the diffusion tensor of CED from the structure tensor.
#[allow(clippy::too_many_arguments)]
pub fn diff_tensor(
    lambda: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    _hx: f32,
    _hy: f32,
    _hz: f32,
    _sigma: f32,
    fast_eigen: i32,
    grd: &NrTensor3<f32>,
    dxx: &mut NrTensor3<f32>,
    dxy: &mut NrTensor3<f32>,
    dxz: &mut NrTensor3<f32>,
    dyy: &mut NrTensor3<f32>,
    dyz: &mut NrTensor3<f32>,
    dzz: &mut NrTensor3<f32>,
) {
    let mut ev11 = 0f32;
    let mut ev12 = 0f32;
    let mut ev13 = 0f32;
    let mut ev21 = 0f32;
    let mut ev22 = 0f32;
    let mut ev23 = 0f32;
    let mut ev31 = 0f32;
    let mut ev32 = 0f32;
    let mut ev33 = 0f32;
    let mut mu1 = 0f32;
    let mut mu2 = 0f32;
    let mut mu3 = 0f32;
    let (mut lam1, mut lam2, mut lam3);
    let mut a = [[0f64; 3]; 3]; /* Matrix for real tensors */
    let mut q = [[0f64; 3]; 3]; /* Matrix for eigenvectors */
    let mut w = [0f64; 3]; /* Array with unordered eigenvalues */

    /* The source allocates and immediately frees a `v` tensor here that it
    never touches; an owned translation has nothing to allocate. */

    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                read_struct_tens(
                    dxx[(i, j, k)],
                    dxy[(i, j, k)],
                    dxz[(i, j, k)],
                    dyy[(i, j, k)],
                    dyz[(i, j, k)],
                    dzz[(i, j, k)],
                    &mut a,
                );
                if fast_eigen != 0 {
                    dsyevv3(&a, &mut q, &mut w);
                } else {
                    dsyevq3(&a, &mut q, &mut w);
                }
                eigen_sort(&mut w, q.as_flattened_mut(), 3, 3, 1, 0);
                read_ei(
                    &w, &q, &mut ev11, &mut ev12, &mut ev13, &mut ev21, &mut ev22, &mut ev23,
                    &mut ev31, &mut ev32, &mut ev33, &mut mu1, &mut mu2, &mut mu3,
                );

                if grd[(i, j, k)] > 0.0 {
                    lam1 = (1.0 - (-3.31488 / (grd[(i, j, k)] / lambda).powf_c(16.0)).exp()) as f32;
                    lam2 = lam1;
                } else {
                    lam1 = 1.0;
                    lam2 = 1.0;
                }

                lam3 = 1.0;
                let (mut r11, mut r12, mut r13, mut r22, mut r23, mut r33) =
                    (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
                pa_backtrans(
                    ev11, ev12, ev13, ev21, ev22, ev23, ev31, ev32, ev33, lam1, lam2, lam3,
                    &mut r11, &mut r12, &mut r13, &mut r22, &mut r23, &mut r33,
                );
                dxx[(i, j, k)] = r11;
                dxy[(i, j, k)] = r12;
                dxz[(i, j, k)] = r13;
                dyy[(i, j, k)] = r22;
                dyz[(i, j, k)] = r23;
                dzz[(i, j, k)] = r33;
            }
        }
    }
}

/// `pow(x, y)` on a `float` argument: the C promotes to double, and the two
/// `lam` assignments in `diff_tensor` are the only places this matters.
trait PowC {
    fn powf_c(self, exponent: f64) -> f64;
}
impl PowC for f32 {
    fn powf_c(self, exponent: f64) -> f64 {
        (self as f64).powf(exponent)
    }
}

/// C `eed`: coherence-enhancing anisotropic diffusion, explicit
/// discretization.
#[allow(clippy::too_many_arguments)]
pub fn eed(
    ht: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    hx: f32,
    hy: f32,
    hz: f32,
    sigma: f32,
    lambda: f32,
    fast_eigen: i32,
    u: &mut NrTensor3<f32>,
) {
    let (rxx, rxy, rxz, ryy, ryz, rzz): (f32, f32, f32, f32, f32, f32);
    let (mut w_n, mut w_ne, mut w_e, mut w_se): (f32, f32, f32, f32);
    let (mut w_s, mut w_sw, mut w_w, mut w_nw): (f32, f32, f32, f32);
    let (mut w_b, mut w_sb, mut w_nb, mut w_eb, mut w_wb): (f32, f32, f32, f32, f32);
    let (mut w_f, mut w_sf, mut w_nf, mut w_ef, mut w_wf): (f32, f32, f32, f32, f32);

    /* ---- allocate storage ---- */

    let mut f = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut grd = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut dxx = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut dxy = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut dxz = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut dyy = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut dyz = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    let mut dzz = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);

    /* ---- copy u into f ---- */

    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                f[(i, j, k)] = u[(i, j, k)];
            }
        }
    }

    /* ---- calculate entries of structure tensor for eed ---- */

    struct_tensor_eed(
        &mut f, nx, ny, nz, hx, hy, hz, sigma, &mut dxx, &mut dxy, &mut dxz, &mut dyy, &mut dyz,
        &mut dzz, &mut grd,
    );

    /* ---- calculate entries of diffusion tensor ---- */

    diff_tensor(
        lambda, nx, ny, nz, hx, hy, hz, sigma, fast_eigen, &grd, &mut dxx, &mut dxy, &mut dxz,
        &mut dyy, &mut dyz, &mut dzz,
    );

    /* ---- calculate explicit nonlinear diffusion of u ---- */

    rxx = (ht as f64 / (2.0 * hx as f64 * hx as f64)) as f32;
    ryy = (ht as f64 / (2.0 * hy as f64 * hy as f64)) as f32;
    rzz = (ht as f64 / (2.0 * hz as f64 * hz as f64)) as f32;
    rxy = (ht as f64 / (4.0 * hx as f64 * hy as f64)) as f32;
    rxz = (ht as f64 / (4.0 * hx as f64 * hz as f64)) as f32;
    ryz = (ht as f64 / (4.0 * hy as f64 * hz as f64)) as f32;

    dummies(&mut dxx, nx, ny, nz);
    dummies(&mut dxy, nx, ny, nz);
    dummies(&mut dxz, nx, ny, nz);
    dummies(&mut dyy, nx, ny, nz);
    dummies(&mut dyz, nx, ny, nz);
    dummies(&mut dzz, nx, ny, nz);

    /* copy u into f and assign dummy boundaries */
    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                f[(i, j, k)] = u[(i, j, k)];
            }
        }
    }
    dummies(&mut f, nx, ny, nz);

    /* `sqrt()` is the double routine, so each of these weights is a double
    expression truncated to `float` on assignment. */
    let abs = |x: f32| ((x as f64) * (x as f64)).sqrt();

    /* diffuse */
    for i in 1..=nx {
        for j in 1..=ny {
            for k in 1..=nz {
                /* weights */
                w_e = ((rxx * (dxx[(i + 1, j, k)] + dxx[(i, j, k)])) as f64
                    - rxy as f64 * (abs(dxy[(i + 1, j, k)]) + abs(dxy[(i, j, k)]))
                    - rxz as f64 * (abs(dxz[(i + 1, j, k)]) + abs(dxz[(i, j, k)])))
                    as f32;
                w_w = ((rxx * (dxx[(i - 1, j, k)] + dxx[(i, j, k)])) as f64
                    - rxy as f64 * (abs(dxy[(i - 1, j, k)]) + abs(dxy[(i, j, k)]))
                    - rxz as f64 * (abs(dxz[(i - 1, j, k)]) + abs(dxz[(i, j, k)])))
                    as f32;
                w_s = ((ryy * (dyy[(i, j + 1, k)] + dyy[(i, j, k)])) as f64
                    - rxy as f64 * (abs(dxy[(i, j + 1, k)]) + abs(dxy[(i, j, k)]))
                    - ryz as f64 * (abs(dyz[(i, j + 1, k)]) + abs(dyz[(i, j, k)])))
                    as f32;
                w_n = ((ryy * (dyy[(i, j - 1, k)] + dyy[(i, j, k)])) as f64
                    - rxy as f64 * (abs(dxy[(i, j - 1, k)]) + abs(dxy[(i, j, k)]))
                    - ryz as f64 * (abs(dyz[(i, j - 1, k)]) + abs(dyz[(i, j, k)])))
                    as f32;
                w_b = ((rzz * (dzz[(i, j, k - 1)] + dzz[(i, j, k)])) as f64
                    - ryz as f64 * (abs(dyz[(i, j, k - 1)]) + abs(dyz[(i, j, k)]))
                    - rxz as f64 * (abs(dxz[(i, j, k - 1)]) + abs(dxz[(i, j, k)])))
                    as f32;
                w_f = ((rzz * (dzz[(i, j, k + 1)] + dzz[(i, j, k)])) as f64
                    - ryz as f64 * (abs(dyz[(i, j, k + 1)]) + abs(dyz[(i, j, k)]))
                    - rxz as f64 * (abs(dxz[(i, j, k + 1)]) + abs(dxz[(i, j, k)])))
                    as f32;

                w_se = (rxy as f64
                    * (((dxy[(i + 1, j + 1, k)] + dxy[(i, j, k)]) as f64
                        + abs(dxy[(i + 1, j + 1, k)]))
                        + abs(dxy[(i, j, k)]))) as f32;
                w_nw = (rxy as f64
                    * (((dxy[(i - 1, j - 1, k)] + dxy[(i, j, k)]) as f64
                        + abs(dxy[(i - 1, j - 1, k)]))
                        + abs(dxy[(i, j, k)]))) as f32;
                w_ne = (rxy as f64
                    * (((-dxy[(i + 1, j - 1, k)] - dxy[(i, j, k)]) as f64
                        + abs(dxy[(i + 1, j - 1, k)]))
                        + abs(dxy[(i, j, k)]))) as f32;
                w_sw = (rxy as f64
                    * (((-dxy[(i - 1, j + 1, k)] - dxy[(i, j, k)]) as f64
                        + abs(dxy[(i - 1, j + 1, k)]))
                        + abs(dxy[(i, j, k)]))) as f32;
                w_sf = (ryz as f64
                    * (((dyz[(i, j + 1, k + 1)] + dyz[(i, j, k)]) as f64
                        + abs(dyz[(i, j + 1, k + 1)]))
                        + abs(dyz[(i, j, k)]))) as f32;
                w_nf = (ryz as f64
                    * (((-dyz[(i, j - 1, k + 1)] - dyz[(i, j, k)]) as f64
                        + abs(dyz[(i, j - 1, k + 1)]))
                        + abs(dyz[(i, j, k)]))) as f32;
                w_ef = (rxz as f64
                    * (((dxz[(i + 1, j, k + 1)] + dxz[(i, j, k)]) as f64
                        + abs(dxz[(i + 1, j, k + 1)]))
                        + abs(dxz[(i, j, k)]))) as f32;
                w_wf = (rxz as f64
                    * (((-dxz[(i - 1, j, k + 1)] - dxz[(i, j, k)]) as f64
                        + abs(dxz[(i - 1, j, k + 1)]))
                        + abs(dxz[(i, j, k)]))) as f32;
                w_sb = (ryz as f64
                    * (((-dyz[(i, j + 1, k - 1)] - dyz[(i, j, k)]) as f64
                        + abs(dyz[(i, j + 1, k - 1)]))
                        + abs(dyz[(i, j, k)]))) as f32;
                w_nb = (ryz as f64
                    * (((dyz[(i, j - 1, k - 1)] + dyz[(i, j, k)]) as f64
                        + abs(dyz[(i, j - 1, k - 1)]))
                        + abs(dyz[(i, j, k)]))) as f32;
                w_eb = (rxz as f64
                    * (((-dxz[(i + 1, j, k - 1)] - dxz[(i, j, k)]) as f64
                        + abs(dxz[(i + 1, j, k - 1)]))
                        + abs(dxz[(i, j, k)]))) as f32;
                w_wb = (rxz as f64
                    * (((dxz[(i - 1, j, k - 1)] + dxz[(i, j, k)]) as f64
                        + abs(dxz[(i - 1, j, k - 1)]))
                        + abs(dxz[(i, j, k)]))) as f32;

                /* modify weights to prevent flux across boundaries */
                if i == 1 {
                    /* set western weights zero */
                    w_sw = 0.0;
                    w_w = 0.0;
                    w_nw = 0.0;
                    w_wb = 0.0;
                    w_wf = 0.0;
                }
                if i == nx {
                    /* set eastern weights zero */
                    w_ne = 0.0;
                    w_e = 0.0;
                    w_se = 0.0;
                    w_eb = 0.0;
                    w_ef = 0.0;
                }
                if j == 1 {
                    /* set northern weights zero */
                    w_nw = 0.0;
                    w_n = 0.0;
                    w_ne = 0.0;
                    w_nb = 0.0;
                    w_nf = 0.0;
                }
                if j == ny {
                    /* set southern weights zero */
                    w_se = 0.0;
                    w_s = 0.0;
                    w_sw = 0.0;
                    w_sb = 0.0;
                    w_sf = 0.0;
                }
                if k == nz {
                    /* set forward weights zero -- the source really does zero
                    wNB here rather than wNF */
                    w_ef = 0.0;
                    w_f = 0.0;
                    w_wf = 0.0;
                    w_nb = 0.0;
                    w_sf = 0.0;
                }
                if k == 1 {
                    /* set backward weights zero */
                    w_sb = 0.0;
                    w_b = 0.0;
                    w_nb = 0.0;
                    w_eb = 0.0;
                    w_wb = 0.0;
                }

                /* evolution */
                u[(i, j, k)] = f[(i, j, k)]
                    + w_e * (f[(i + 1, j, k)] - f[(i, j, k)])
                    + w_w * (f[(i - 1, j, k)] - f[(i, j, k)])
                    + w_s * (f[(i, j + 1, k)] - f[(i, j, k)])
                    + w_n * (f[(i, j - 1, k)] - f[(i, j, k)])
                    + w_b * (f[(i, j, k - 1)] - f[(i, j, k)])
                    + w_f * (f[(i, j, k + 1)] - f[(i, j, k)])
                    + w_se * (f[(i + 1, j + 1, k)] - f[(i, j, k)])
                    + w_nw * (f[(i - 1, j - 1, k)] - f[(i, j, k)])
                    + w_sw * (f[(i - 1, j + 1, k)] - f[(i, j, k)])
                    + w_ne * (f[(i + 1, j - 1, k)] - f[(i, j, k)])
                    + w_nb * (f[(i, j - 1, k - 1)] - f[(i, j, k)])
                    + w_nf * (f[(i, j - 1, k + 1)] - f[(i, j, k)])
                    + w_eb * (f[(i + 1, j, k - 1)] - f[(i, j, k)])
                    + w_ef * (f[(i + 1, j, k + 1)] - f[(i, j, k)])
                    + w_wb * (f[(i - 1, j, k - 1)] - f[(i, j, k)])
                    + w_wf * (f[(i - 1, j, k + 1)] - f[(i, j, k)])
                    + w_sb * (f[(i, j + 1, k - 1)] - f[(i, j, k)])
                    + w_sf * (f[(i, j + 1, k + 1)] - f[(i, j, k)]);
            }
        }
    }

    /* ---- disallocate storage ---- */

    let bounds = (0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);
    free_f3tensor(
        f, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        grd, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        dxx, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        dxy, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        dxz, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        dyy, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        dyz, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
    free_f3tensor(
        dzz, bounds.0, bounds.1, bounds.2, bounds.3, bounds.4, bounds.5,
    );
}

// IMOD modifications all below here

/// C `usage`.
pub fn usage(progname: &str, ht: f32, pmax: i32, sigma: f32, lambda: f32) -> ! {
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        concat!(
            "%s by A. Frangakis and R. Hegerl (adapted for IMOD)\n",
            "Usage: %s [options] <input file> <output file>\nOptions:\n",
            "\t-k #\tK (lambda) value, threshold for gradients (default %.1f)\n",
            "\t-n #\tNumber of iterations (default %d)\n",
            "\t-i list\tList of iterations at which to write output\n",
            "\t-e ext\tExtension to put after iteration number (excluding dot)\n",
            "\t-o #\tOutput only the given Z slice (numbered from 1)\n",
            "\t-m #\tMode for output file, 0=byte, 1=int, 2=real, 6=unsigned int",
            "\n",
            "\t-s #\tSigma for smoothing of structure tensor (default %.1f)\n",
            "\t-t #\tTime step (default %.2f)\n",
            "\t-P  \tPrint PID to standard error\n"
        ),
        &[
            CArg::Bytes(progname.as_bytes()),
            CArg::Bytes(progname.as_bytes()),
            CArg::Dbl(lambda as f64),
            CArg::Int(pmax as i64),
            CArg::Dbl(sigma as f64),
            CArg::Dbl(ht as f64),
        ],
    ));
    let _ = ImodFile::Stdout.flush();
    std::process::exit(1);
}

/// C `testNumericEntry`.
pub fn test_numeric_entry(endptr: usize, argv: &str, option: &str) {
    if endptr == 0 {
        exit_error(&c_format_bytes(
            "Option %s must be followed by a number, not by %s",
            &[CArg::Bytes(option.as_bytes()), CArg::Bytes(argv.as_bytes())],
        ));
    }
}

const STRING_MAX: usize = 1024;

/// C `main` in `nad_eed_3d.c`.
pub fn nad_eed_3d(argv: &[String]) -> i32 {
    let argc = argv.len() as i32;
    let mut header = MrcHeader::default();
    let mut ht: f32 = 0.1; /* time step size */
    let mut pmax: i32 = 20; /* largest iteration number */
    let mut sigma: f32 = 0.; /* noise scale */
    let mut lambda: f32 = 1.; /* lamda Parameter */
    let mut max: f32 = 0.;
    let mut min: f32 = 0.;
    let mut mean: f32 = 0.;
    let mut vari: f32 = 0.;
    let mut fast_eigen: i32 = 0; /* Flag to use faster eigenvector call */
    let mut format_set: i32 = -2;
    let mut slice_mode;
    let progname = imod_prog_name(&argv[0]);
    let mut endptr: usize;
    let mut n_write: usize = 0;
    let mut write_arg: usize = 0;
    let mut write_list: Vec<i32> = Vec::new();
    let mut out_file: Vec<u8>;
    let mut one_slice: i32 = 0;
    let mut out_mode: i32 = -1;
    let mut ext_argv_ind: i32 = -1;
    let (mut minout, mut maxout, mut sumout): (f32, f32, f32);

    if argc < 3 {
        usage(&progname, ht, pmax, sigma, lambda);
    }

    pip_exit_on_error(0, b"ERROR: nad_eed_3d -");
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "\nProgram %s\n",
        &[CArg::Bytes(progname.as_bytes())],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "        started at: %s",
        &[CArg::Bytes(asctime_now().as_bytes())],
    ));

    let mut iarg: usize = 1;
    while iarg < argc as usize {
        if argv[iarg].as_bytes().first().copied().unwrap_or(0) == b'-' {
            match argv[iarg].as_bytes().get(1).copied().unwrap_or(0) {
                b'k' => {
                    iarg += 1;
                    endptr = 0;
                    lambda = strtod(argv[iarg].as_bytes(), &mut endptr) as f32;
                    test_numeric_entry(endptr, &argv[iarg], "-k");
                }
                b's' => {
                    iarg += 1;
                    endptr = 0;
                    sigma = strtod(argv[iarg].as_bytes(), &mut endptr) as f32;
                    test_numeric_entry(endptr, &argv[iarg], "-s");
                }
                b'n' => {
                    iarg += 1;
                    endptr = 0;
                    pmax = strtol(argv[iarg].as_bytes(), &mut endptr, 10) as i32;
                    test_numeric_entry(endptr, &argv[iarg], "-n");
                }
                b'o' => {
                    iarg += 1;
                    endptr = 0;
                    one_slice = strtol(argv[iarg].as_bytes(), &mut endptr, 10) as i32;
                    test_numeric_entry(endptr, &argv[iarg], "-o");
                }
                b'm' => {
                    iarg += 1;
                    endptr = 0;
                    out_mode = strtol(argv[iarg].as_bytes(), &mut endptr, 10) as i32;
                    test_numeric_entry(endptr, &argv[iarg], "-m");
                    if slice_mode_if_real(out_mode) < 0 {
                        exit_error(&c_format_bytes(
                            "Output mode %d not allowed\n",
                            &[CArg::Int(out_mode as i64)],
                        ));
                    }
                    out_mode =
                        crate::imod::libcfshr::b3dutil::set_float_output_for_entered_mode(out_mode);
                }
                b't' => {
                    iarg += 1;
                    endptr = 0;
                    ht = strtod(argv[iarg].as_bytes(), &mut endptr) as f32;
                    test_numeric_entry(endptr, &argv[iarg], "-t");
                }
                b'i' => {
                    iarg += 1;
                    match parselist(&argv[iarg]) {
                        Ok(list) => {
                            write_list = list;
                            n_write = write_list.len();
                        }
                        Err(_) => exit_error(b"Bad entry in iteration list"),
                    }
                    write_arg = iarg;
                }
                b'f' => {
                    fast_eigen = 1;
                }
                b'e' => {
                    iarg += 1;
                    ext_argv_ind = iarg as i32;
                }
                b'F' => {
                    iarg += 1;
                    format_set = set_output_type_from_string(&argv[iarg]);
                    if format_set < 0 {
                        exit_error(&c_format_bytes(
                            "Output file format entry %s is not %s.",
                            &[
                                CArg::Bytes(argv[iarg].as_bytes()),
                                CArg::Bytes(if format_set == -1 {
                                    b"recognized".as_slice()
                                } else {
                                    b"available in this copy of IMOD".as_slice()
                                }),
                            ],
                        ));
                    }
                }
                b'P' => {
                    pid_to_stderr();
                }
                _ => {
                    exit_error(&c_format_bytes(
                        "Invalid option %s",
                        &[CArg::Bytes(argv[iarg].as_bytes())],
                    ));
                }
            }
        } else {
            break;
        }
        iarg += 1;
    }

    // Override pmax with maximum to write
    if n_write > 0 {
        pmax = 0;
        for i in 0..n_write {
            pmax = if pmax > write_list[i] {
                pmax
            } else {
                write_list[i]
            };
        }
    }

    if iarg != argc as usize - 2 {
        exit_error(b"Command line should end with input and output files");
    }

    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "input file:          %s\n",
        &[CArg::Bytes(argv[iarg].as_bytes())],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "output file:         %s\n",
        &[CArg::Bytes(argv[iarg + 1].as_bytes())],
    ));

    /* ---- read input image ---- */
    let Some(mut fp_infile) = ii_fopen(argv[iarg].as_bytes(), "rb") else {
        exit_error(&c_format_bytes(
            "Could not open input file %s",
            &[CArg::Bytes(argv[iarg].as_bytes())],
        ));
    };

    /* read header */
    if mrc_head_read(&mut fp_infile, &mut header) != 0 {
        exit_error(&c_format_bytes(
            "Reading header of input file %s",
            &[CArg::Bytes(argv[iarg].as_bytes())],
        ));
    }

    // Check if it is the correct data type and set slice type
    slice_mode = slice_mode_if_real(header.mode);
    if slice_mode < 0 {
        exit_error(&c_format_bytes(
            "File mode is %d; only byte, short, integer allowed",
            &[CArg::Int(header.mode as i64)],
        ));
    }

    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "dimensions:          %d x %d x %d\n",
        &[
            CArg::Int(nx as i64),
            CArg::Int(ny as i64),
            CArg::Int(nz as i64),
        ],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "K (lambda):          %f\n\n",
        &[CArg::Dbl(lambda as f64)],
    ));
    if one_slice < 1 || one_slice > nz as i32 {
        one_slice = 0;
    }

    /* allocate storage */
    let mut u = f3tensor(0, nx as isize + 1, 0, ny as isize + 1, 0, nz as isize + 1);

    /* read image data */
    for k in 1..=nz {
        // Create a slice and read into it
        let Some(mut sl) = slice_create(nx as i32, ny as i32, slice_mode) else {
            exit_error(b"Creating slice for input");
        };
        if mrc_read_slice(
            sl.data.bytes_mut(),
            &mut fp_infile,
            &mut header,
            k as i32 - 1,
            b'Z',
        ) != 0
        {
            exit_error(&c_format_bytes("Reading slice %d", &[CArg::Int(k as i64)]));
        }

        // Convert slice to floats
        if slice_mode != SLICE_MODE_FLOAT && slice_new_mode(&mut sl, SLICE_MODE_FLOAT) < 0 {
            exit_error(b"Converting slice to float");
        }

        // Copy data into array
        for j in 0..ny {
            for i in 0..nx {
                u[(i + 1, j + 1, k)] = sl.data.f()[i + j * nx];
            }
        }
    }
    ii_fclose(&mut fp_infile);

    /* ---- Image ---- */
    analyse(&u, nx, ny, nz, &mut min, &mut max, &mut mean, &mut vari);
    print_stats(min, max, mean, vari);

    // Take care of header
    if one_slice != 0 && n_write != 0 {
        out_file = c_format_bytes(
            "%s: Z %d, iter %s",
            &[
                CArg::Bytes(progname.as_bytes()),
                CArg::Int(one_slice as i64),
                CArg::Bytes(argv[write_arg].as_bytes()),
            ],
        );
    } else {
        out_file = c_format_bytes(
            "%s: Edge-enhancing anisotropic diffusion",
            &[CArg::Bytes(progname.as_bytes())],
        );
    }
    mrc_head_label(&mut header, &out_file);

    // Adjust output mode if valid entry made (it was tested on arg processing)
    if out_mode >= 0 {
        slice_mode = slice_mode_if_real(out_mode);
        header.mode = out_mode;
    }

    // Fix things in header for an output file.  You have to set header size
    // not just set next to 0
    mrc_init_output_header(&mut header);

    /* ---- process image ---- */

    let mut nzout: i32 = 0;
    minout = 1.0e30;
    maxout = -minout;
    sumout = 0.;

    let mut fp_outfile: Option<ImodFile> = None;

    for p in 1..=pmax {
        /* perform one iteration */
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "iteration number: %5d / %d \n",
            &[CArg::Int(p as i64), CArg::Int(pmax as i64)],
        ));

        eed(
            ht, nx, ny, nz, 1.0, 1.0, 1.0, sigma, lambda, fast_eigen, &mut u,
        );

        /* check minimum, maximum, mean, variance */
        analyse(&u, nx, ny, nz, &mut min, &mut max, &mut mean, &mut vari);
        print_stats(min, max, mean, vari);

        // Write data if it is an iteration on list or the last iteration
        let mut do_write = 0;
        for i in 0..n_write {
            if p == write_list[i] {
                do_write = 1;
            }
        }
        if do_write != 0 || p == pmax {
            header.amean = mean;
            header.amin = min;
            header.amax = max;

            let mut max_copy = STRING_MAX - 10;
            if n_write != 0 && one_slice == 0 && ext_argv_ind > 0 {
                max_copy -= argv[ext_argv_ind as usize].len();
            }
            if max_copy < 5 {
                exit_error(b"Output filename would be too long for buffer");
            }
            let source = argv[iarg + 1].as_bytes();
            out_file = source[..source.len().min(max_copy)].to_vec();
            if n_write != 0 && one_slice == 0 {
                out_file.extend_from_slice(&c_format_bytes("-%03d", &[CArg::Int(p as i64)]));
                if ext_argv_ind > 0 {
                    out_file.extend_from_slice(b".");
                    out_file.extend_from_slice(argv[ext_argv_ind as usize].as_bytes());
                }
            }

            /* open output file if not open yet */
            if fp_outfile.is_none() {
                if std::env::var_os("IMOD_NO_IMAGE_BACKUP").is_none()
                    && imod_backup_file(&String::from_utf8_lossy(&out_file)) != 0
                {
                    let _ = ImodFile::Stderr.write_all(&c_format_bytes(
                        "WARNING: error renaming existing %s to %s~",
                        &[CArg::Bytes(&out_file), CArg::Bytes(&out_file)],
                    ));
                }

                let Some(file) = ii_fopen(&out_file, "wb") else {
                    exit_error(&c_format_bytes(
                        "Could not open output file %s",
                        &[CArg::Bytes(&out_file)],
                    ));
                };
                fp_outfile = Some(file);
            }

            /* write image data and close file */
            let kst = if one_slice != 0 {
                one_slice as usize
            } else {
                1
            };
            let knd = if one_slice != 0 {
                one_slice as usize
            } else {
                nz
            };
            for k in kst..=knd {
                // Create a slice and copy into it
                let Some(mut sl) = slice_create(nx as i32, ny as i32, SLICE_MODE_FLOAT) else {
                    exit_error(b"Creating slice for output");
                };
                for j in 0..ny {
                    for i in 0..nx {
                        sl.data.f_mut()[i + j * nx] = u[(i + 1, j + 1, k)];
                    }
                }

                // Convert if necessary and write slice
                if slice_mode != SLICE_MODE_FLOAT && slice_new_mode(&mut sl, slice_mode) < 0 {
                    exit_error(b"Converting slice to short");
                }
                let out = fp_outfile.as_mut().unwrap();
                if mrc_write_slice(
                    sl.data.bytes(),
                    out,
                    &mut header,
                    if one_slice != 0 { nzout } else { k as i32 - 1 },
                    b'Z',
                ) != 0
                {
                    exit_error(&c_format_bytes("Writing slice %d", &[CArg::Int(k as i64)]));
                }

                // If doing one slice, accumulate mmm
                if one_slice != 0 {
                    slice_mmm(&mut sl);
                    minout = if minout < sl.min { minout } else { sl.min };
                    maxout = if maxout > sl.max { maxout } else { sl.max };
                    sumout += sl.mean;
                }
            }

            nzout += 1;

            // Close the file if not doing one slice or end of run
            if one_slice == 0 || p == pmax {
                // Adjust size and mmm if did one slice
                if one_slice != 0 {
                    header.nz = nzout;
                    header.mz = (header.mz * nzout) / nz as i32;
                    header.zlen = (header.zlen * nzout as f32) / nz as f32;
                    header.amin = minout;
                    header.amax = maxout;
                    header.amean = sumout / nzout as f32;
                }

                // Write the MRC header.
                let out = fp_outfile.as_mut().unwrap();
                if mrc_head_write(out, &mut header) != 0 {
                    exit_error(b"Writing header");
                }

                ii_fclose(fp_outfile.as_mut().unwrap());
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "output image %s successfully written\n\n",
                    &[CArg::Bytes(&out_file)],
                ));
                fp_outfile = None;
            }
        }
        let _ = ImodFile::Stdout.flush();
    }

    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "program finished at: %s\n",
        &[CArg::Bytes(asctime_now().as_bytes())],
    ));

    /* ---- disallocate storage ---- */

    free_f3tensor(
        u,
        0,
        nx as isize + 1,
        0,
        ny as isize + 1,
        0,
        nz as isize + 1,
    );
    let _ = ImodFile::Stdout.flush();
    std::process::exit(0);
}

/// The four `printf("%1.10f")` statistics lines `main` writes after every call
/// to `analyse`, verbatim at both sites.
fn print_stats(min: f32, max: f32, mean: f32, vari: f32) {
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "minimum:       %1.10f \n",
        &[CArg::Dbl(min as f64)],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "maximum:       %1.10f \n",
        &[CArg::Dbl(max as f64)],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "mean:          %1.10f \n",
        &[CArg::Dbl(mean as f64)],
    ));
    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "variance:      %1.10f \n\n",
        &[CArg::Dbl(vari as f64)],
    ));
}

/// `asctime(localtime(&t))`, including its trailing newline.
fn asctime_now() -> String {
    format!("{}\n", chrono::Local::now().format("%a %b %e %H:%M:%S %Y"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummies_wraps_the_halo_periodically() {
        let mut v = f3tensor(0, 3, 0, 2, 0, 2);
        for i in 1..=2usize {
            for j in 1..=1usize {
                for k in 1..=1usize {
                    v[(i, j, k)] = i as f32;
                }
            }
        }
        dummies(&mut v, 2, 1, 1);
        assert_eq!(v[(0, 1, 1)], 2.0);
        assert_eq!(v[(3, 1, 1)], 1.0);
    }

    #[test]
    fn analyse_matches_the_source_definitions() {
        let mut v = f3tensor(0, 3, 0, 2, 0, 2);
        v[(1, 1, 1)] = 1.0;
        v[(2, 1, 1)] = 3.0;
        let (mut mn, mut mx, mut me, mut va) = (0., 0., 0., 0.);
        analyse(&v, 2, 1, 1, &mut mn, &mut mx, &mut me, &mut va);
        assert_eq!((mn, mx, me, va), (1.0, 3.0, 2.0, 1.0));
    }

    #[test]
    fn a_constant_volume_is_a_fixed_point_of_eed() {
        let mut v = f3tensor(0, 4, 0, 4, 0, 4);
        for i in 1..=3usize {
            for j in 1..=3usize {
                for k in 1..=3usize {
                    v[(i, j, k)] = 4.0;
                }
            }
        }
        eed(0.1, 3, 3, 3, 1., 1., 1., 0., 1., 0, &mut v);
        for i in 1..=3usize {
            for j in 1..=3usize {
                for k in 1..=3usize {
                    assert_eq!(v[(i, j, k)], 4.0);
                }
            }
        }
    }
}
