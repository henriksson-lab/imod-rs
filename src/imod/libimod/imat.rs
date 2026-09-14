//! Translation of `IMOD/libimod/imat.c` together with its paired header
//! `IMOD/include/imat.h`.
//!
//! "a quick and dirty way to get transforms on none gl machines."
#![allow(dead_code, unused_variables)]

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libcfshr::linearxforms::matrix_to_angles;
use crate::imod::libimod::imodel::Ipoint;

/// Original: `b3dX` (`include/hvemtypes.h:21`).
pub const B3D_X: i32 = 0;
/// Original: `b3dY` (`include/hvemtypes.h:22`).
pub const B3D_Y: i32 = 1;
/// Original: `b3dZ` (`include/hvemtypes.h:23`).
pub const B3D_Z: i32 = 2;

/// Original: `Imat` / `struct imodel_matrix` (`include/imat.h:11`).
#[derive(Clone, Debug, PartialEq)]
pub struct Imat {
    pub data: Vec<f32>,
    /// is 2D or 3D
    pub dim: i32,
    pub size: i32,
}

/// Original: `imodMatNew` (`imat.c:28`).
///
/// Creates a new matrix structure and sets it to the identity matrix.  The
/// input dimension `dim` can be 2 or 3.  Returns `None` for error.
pub fn imod_mat_new(dim: i32) -> Option<Imat> {
    let mut mat = Imat {
        data: Vec::new(),
        dim,
        size: 0,
    };
    match dim {
        2 => {
            mat.data = vec![0.; 9];
            mat.size = 9;
        }
        3 => {
            mat.data = vec![0.; 16];
            mat.size = 16;
        }
        _ => {}
    }
    if mat.data.is_empty() {
        return None;
    }
    imod_mat_id(&mut mat);
    Some(mat)
}

/// Original: `imodMatDelete` (`imat.c:55`).
///
/// Frees the matrix `mat` as well as its `data` member.
pub fn imod_mat_delete(mat: &mut Imat) {
    mat.data.clear();
}

/// Original: `imodMatId` (`imat.c:64`).
///
/// Sets the matrix `mat` to the identity matrix.
pub fn imod_mat_id(mat: &mut Imat) {
    for i in 0..mat.size as usize {
        mat.data[i] = 0.0f32;
    }
    mat.data[0] = 1.0f32;
    if mat.dim == 2 {
        mat.data[4] = 1.0f32;
        mat.data[8] = 1.0f32;
    } else {
        mat.data[5] = 1.0f32;
        mat.data[10] = 1.0f32;
        mat.data[15] = 1.0f32;
    }
}

/// Original: `imodMatPrint` (`imat.c:83`).
///
/// Prints matrix in `mat`.
pub fn imod_mat_print(mat: &Imat) {
    if mat.dim == 2 {
        for i in 0..2usize {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "%13.6f %13.6f %13.3f\n",
                    &[
                        CArg::Dbl(mat.data[i] as f64),
                        CArg::Dbl(mat.data[i + 3] as f64),
                        CArg::Dbl(mat.data[i + 6] as f64),
                    ],
                )
                .as_bytes(),
            );
        }
    } else {
        for i in 0..3usize {
            let _ = ImodFile::Stdout.write_all(
                c_format(
                    "%13.6f %13.6f %13.6f %13.3f\n",
                    &[
                        CArg::Dbl(mat.data[i] as f64),
                        CArg::Dbl(mat.data[i + 4] as f64),
                        CArg::Dbl(mat.data[i + 8] as f64),
                        CArg::Dbl(mat.data[i + 12] as f64),
                    ],
                )
                .as_bytes(),
            );
        }
    }
}

/// Original: `imodMatCopy` (`imat.c:100`).
///
/// Copies matrix `fmat` to `tomat`, provided they are the same dimension.
pub fn imod_mat_copy(fmat: &Imat, tomat: &mut Imat) {
    if fmat.dim != tomat.dim {
        return;
    }
    for i in 0..fmat.size as usize {
        tomat.data[i] = fmat.data[i];
    }
    tomat.dim = fmat.dim;
    tomat.size = fmat.size;
}

/// Original: `imodMatMult` (`imat.c:118`).
///
/// Forms the matrix product `mat1` x `mat2` and places it into `matout`; i.e.,
/// the input arguments are the matrix applied first and the matrix applied
/// second.
pub fn imod_mat_mult(mat2: &Imat, mat1: &Imat, matout: &mut Imat) {
    if mat2.dim != mat1.dim {
        return;
    }
    let m1 = &mat1.data;
    let m2 = &mat2.data;
    let out = &mut matout.data;
    if mat2.dim == 2 {
        out[0] = (m1[0] * m2[0]) + (m1[3] * m2[1]) + (m1[6] * m2[2]);
        out[1] = (m1[1] * m2[0]) + (m1[4] * m2[1]) + (m1[7] * m2[2]);
        out[2] = (m1[2] * m2[0]) + (m1[5] * m2[1]) + (m1[8] * m2[2]);
        out[3] = (m1[0] * m2[3]) + (m1[3] * m2[4]) + (m1[6] * m2[5]);
        out[4] = (m1[1] * m2[3]) + (m1[4] * m2[4]) + (m1[7] * m2[5]);
        out[5] = (m1[2] * m2[3]) + (m1[5] * m2[4]) + (m1[8] * m2[5]);
        out[6] = (m1[0] * m2[6]) + (m1[3] * m2[7]) + (m1[6] * m2[8]);
        out[7] = (m1[1] * m2[6]) + (m1[4] * m2[7]) + (m1[7] * m2[8]);
        out[8] = (m1[2] * m2[6]) + (m1[5] * m2[7]) + (m1[8] * m2[8]);
        return;
    }

    let mut i: usize = 0;
    let mut j: isize = 0;
    for n in 0..16usize {
        let ju = j as usize;
        out[n] = (m1[ju] * m2[i])
            + (m1[ju + 4] * m2[i + 1])
            + (m1[ju + 8] * m2[i + 2])
            + (m1[ju + 12] * m2[i + 3]);
        if j == 3 {
            j = -1;
            i += 4;
        }
        j += 1;
    }
}

/// Original: `imodMatTrans` (`imat.c:154`).
///
/// Adds the translation in `pt` to the transformation in `mat`.
pub fn imod_mat_trans(mat: &mut Imat, pt: &Ipoint) {
    if mat.dim == 2 {
        mat.data[6] += pt.x;
        mat.data[7] += pt.y;
        return;
    }
    mat.data[12] += pt.x;
    mat.data[13] += pt.y;
    mat.data[14] += pt.z;
}

/// Original: `imodMatScale` (`imat.c:169`).
///
/// Applies scaling by the factors in `pt` to the transformation in `mat`.
pub fn imod_mat_scale(mat: &mut Imat, pt: &Ipoint) -> i32 {
    let Some(mut smat) = imod_mat_new(mat.dim) else {
        return -1;
    };
    let Some(mut omat) = imod_mat_new(mat.dim) else {
        imod_mat_delete(&mut smat);
        return -1;
    };

    if mat.dim == 2 {
        smat.data[0] = pt.x;
        smat.data[4] = pt.y;
    } else {
        smat.data[0] = pt.x;
        smat.data[5] = pt.y;
        smat.data[10] = pt.z;
    }

    imod_mat_mult(mat, &smat, &mut omat);
    imod_mat_copy(&omat, mat);
    imod_mat_delete(&mut omat);
    imod_mat_delete(&mut smat);
    0
}

/// Original: `imodMatRot` (`imat.c:199`).
///
/// Applies rotation by `angle` in degrees around one axis to the
/// transformation in `mat`.  For a 3D matrix, `axis` must be one of `B3D_X`,
/// `B3D_Y`, or `B3D_Z`; for a 2D matrix `axis` is ignored.  Returns 1 for
/// memory error.
pub fn imod_mat_rot(mat: &mut Imat, mut angle: f64, axis: i32) -> i32 {
    angle *= 0.017453293;

    let cosa = angle.cos();
    let sina = angle.sin();

    let Some(mut rmat) = imod_mat_new(mat.dim) else {
        return -1;
    };
    let Some(mut omat) = imod_mat_new(mat.dim) else {
        imod_mat_delete(&mut rmat);
        return -1;
    };

    if mat.dim == 2 {
        rmat.data[0] = cosa as f32;
        rmat.data[1] = sina as f32;
        rmat.data[3] = (-sina) as f32;
        rmat.data[4] = cosa as f32;
    } else {
        match axis {
            B3D_X => {
                rmat.data[5] = cosa as f32;
                rmat.data[6] = sina as f32;
                rmat.data[9] = (-sina) as f32;
                rmat.data[10] = cosa as f32;
            }
            B3D_Y => {
                rmat.data[0] = cosa as f32;
                rmat.data[2] = (-sina) as f32;
                rmat.data[8] = sina as f32;
                rmat.data[10] = cosa as f32;
            }
            B3D_Z => {
                rmat.data[0] = cosa as f32;
                rmat.data[1] = sina as f32;
                rmat.data[4] = (-sina) as f32;
                rmat.data[5] = cosa as f32;
            }
            _ => {
                imod_mat_delete(&mut omat);
                imod_mat_delete(&mut rmat);
                return -1;
            }
        }
    }
    imod_mat_mult(mat, &rmat, &mut omat);
    imod_mat_copy(&omat, mat);
    imod_mat_delete(&mut omat);
    imod_mat_delete(&mut rmat);
    0
}

/// Original: `imodMatRotateVector` (`imat.c:259`).
///
/// Applies a rotation by `angle` (in degrees) about the vector `v` to the
/// matrix in `mat`.
pub fn imod_mat_rotate_vector(mat: &mut Imat, mut angle: f64, v: &Ipoint) -> i32 {
    if mat.dim == 2 {
        return -1;
    }

    if v.x == 0.0 && v.y == 0.0 && v.z == 0.0 {
        return -1;
    }

    angle *= 0.017453293;
    let cosa = angle.cos();
    let sina = angle.sin();
    let omca = 1.0 - cosa;

    if sina - sina != 0. {
        return 0;
    }

    let mut aval: f64 = ((v.x * v.x) + (v.y * v.y) + (v.z * v.z)) as f64;
    if aval == 0.0 {
        return 0;
    }
    aval = aval.sqrt();
    let x = v.x as f64 / aval;
    let y = v.y as f64 / aval;
    let z = v.z as f64 / aval;

    let Some(mut rmat) = imod_mat_new(mat.dim) else {
        return -1;
    };
    let Some(mut omat) = imod_mat_new(mat.dim) else {
        imod_mat_delete(&mut rmat);
        return -1;
    };

    rmat.data[0] = (x * x * omca + cosa) as f32;
    rmat.data[1] = (y * x * omca + (sina * z)) as f32;
    rmat.data[2] = (z * x * omca - (sina * y)) as f32;

    rmat.data[4] = (x * y * omca - (sina * z)) as f32;
    rmat.data[5] = (y * y * omca + cosa) as f32;
    rmat.data[6] = (z * y * omca + (sina * x)) as f32;

    rmat.data[8] = (x * z * omca + (sina * y)) as f32;
    rmat.data[9] = (y * z * omca - (sina * x)) as f32;
    rmat.data[10] = (z * z * omca + cosa) as f32;
    imod_mat_mult(mat, &rmat, &mut omat);
    imod_mat_copy(&omat, mat);
    imod_mat_delete(&mut omat);
    imod_mat_delete(&mut rmat);
    0
}

/// Original: `imodMatFindVector` (`imat.c:322`).
///
/// Given a rotation matrix `mat`, finds a single rotation axis described by
/// vector `v` and the amount of rotation `angle` about that axis, in degrees.
pub fn imod_mat_find_vector(mat: &Imat, angle: &mut f64, v: &mut Ipoint) -> i32 {
    let xsin = 0.5 * (mat.data[6] - mat.data[9]) as f64;
    let ysin = 0.5 * (mat.data[8] - mat.data[2]) as f64;
    let zsin = 0.5 * (mat.data[1] - mat.data[4]) as f64;

    let sina = (xsin * xsin + ysin * ysin + zsin * zsin).sqrt();
    let cosa = 0.5 * (mat.data[0] as f64 + mat.data[5] as f64 + mat.data[10] as f64 - 1.0);
    *angle = sina.atan2(cosa);

    if *angle < 1.0e-8 {
        v.x = 1.0;
        v.y = 0.0;
        v.z = 0.0;
    } else if sina < 1.0e-8 {
        v.x = ((mat.data[0] as f64 - cosa) / (1.0 - cosa)).sqrt() as f32;
        v.y = ((mat.data[5] as f64 - cosa) / (1.0 - cosa)).sqrt() as f32;
        v.z = ((mat.data[10] as f64 - cosa) / (1.0 - cosa)).sqrt() as f32;
        if mat.data[1] < 0. {
            v.y *= -1.;
        }
        if mat.data[2] < 0. {
            v.z *= -1.;
        }
    } else {
        v.x = (xsin / sina) as f32;
        v.y = (ysin / sina) as f32;
        v.z = (zsin / sina) as f32;
    }

    *angle /= 0.017453293;
    0
}

/// Original: `imodMatGetNatAngles` (`imat.c:361`).
///
/// Given a 3D rotation matrix in `mat`, finds the angles of rotation about the
/// three axes, in the order Z, Y, X, and returns them in `x`, `y`, and `z`.
/// Returns 1 if the determinant of the matrix is not near zero.
pub fn imod_mat_get_nat_angles(mat: &Imat, x: &mut f64, y: &mut f64, z: &mut f64) -> i32 {
    match matrix_to_angles(&mat.data, 4) {
        Err(()) => 1,
        Ok((ax, ay, az)) => {
            *x = ax;
            *y = ay;
            *z = az;
            imod_mat_unique_angles(x, y, z);
            0
        }
    }
}

/// Original: `imodMatUniqueAngles` (`imat.c:375`).
///
/// Converts the three angles `x`, `y`, and `z` for rotations about the X, Y,
/// and Z axes into a unique set of angles, with `x` between +/-90 and `y` and
/// `z` between +/-180.
pub fn imod_mat_unique_angles(x: &mut f64, y: &mut f64, z: &mut f64) {
    while *x > 180. {
        *x -= 360.;
    }
    while *x <= -180. {
        *x += 360.;
    }
    while *y > 180. {
        *y -= 360.;
    }
    while *y <= -180. {
        *y += 360.;
    }
    while *z > 180. {
        *z -= 360.;
    }
    while *z <= -180. {
        *z += 360.;
    }
    if (*x).abs() > 90. {
        *x += 180. * if *x > 0. { -1. } else { 1. };
        *y = if *y >= 0. { 1. } else { -1. } * 180. - *y;
        *z += 180. * if *z > 0. { -1. } else { 1. };
    }
}

/// Original: `imodMatUniqueRotationPt` (`imat.c:401`).
///
/// Calls `imod_mat_unique_angles` with the X, Y, and Z angles in the three
/// members of `pt`.
pub fn imod_mat_unique_rotation_pt(pt: &mut Ipoint) {
    let mut x = pt.x as f64;
    let mut y = pt.y as f64;
    let mut z = pt.z as f64;
    imod_mat_unique_angles(&mut x, &mut y, &mut z);
    pt.x = x as f32;
    pt.y = y as f32;
    pt.z = z as f32;
}

/// Original: `imodMatTransform2D` (`imat.c:416`).
///
/// Applies the 2D transformation in matrix `mat` to the point `pt` and returns
/// the transformed position in `rpt`.
pub fn imod_mat_transform2d(mat: &Imat, pt: &Ipoint, rpt: &mut Ipoint) {
    rpt.x = (mat.data[0] * pt.x) + (mat.data[3] * pt.y) + mat.data[6];
    rpt.y = (mat.data[1] * pt.x) + (mat.data[4] * pt.y) + mat.data[7];
    rpt.z = pt.z;
}

/// Original: `imodMatTransform3D` (`imat.c:427`).
///
/// Applies the 3D transformation in matrix `mat` to the point `pt` and returns
/// the transformed position in `rpt`.
pub fn imod_mat_transform3d(mat: &Imat, pt: &Ipoint, rpt: &mut Ipoint) {
    rpt.x = (mat.data[0] * pt.x) + (mat.data[4] * pt.y) + (mat.data[8] * pt.z) + mat.data[12];
    rpt.y = (mat.data[1] * pt.x) + (mat.data[5] * pt.y) + (mat.data[9] * pt.z) + mat.data[13];
    rpt.z = (mat.data[2] * pt.x) + (mat.data[6] * pt.y) + (mat.data[10] * pt.z) + mat.data[14];
}

/// Original: `imodMatTransform` (`imat.c:441`).
///
/// Applies the transformation in matrix `mat` to the point `pt` and returns
/// the transformed position in `rpt`.  `mat` can be 2D or 3D.
pub fn imod_mat_transform(mat: &Imat, pt: &Ipoint, rpt: &mut Ipoint) {
    if mat.dim == 2 {
        imod_mat_transform2d(mat, pt, rpt);
        return;
    }
    imod_mat_transform3d(mat, pt, rpt);
}

/// Original: `imodMatInverse` (`imat.c:453`).
///
/// Returns the inverse of the matrix `mat`, or `None` for memory error.
pub fn imod_mat_inverse(mat: &Imat) -> Option<Imat> {
    let mut imat = imod_mat_new(mat.dim)?;

    let mdata = &mat.data;
    let mval;
    if mat.dim == 2 {
        mval = mdata[8] * ((mdata[0] * mdata[4]) - (mdata[1] * mdata[3]));

        let idata = &mut imat.data;
        idata[0] = mdata[4] * mdata[8];
        idata[1] = -mdata[1] * mdata[8];
        idata[2] = 0.0;
        idata[3] = -mdata[3] * mdata[8];
        idata[4] = mdata[0] * mdata[8];
        idata[5] = 0.0;
        idata[6] = (mdata[3] * mdata[7]) - (mdata[4] * mdata[6]);
        idata[7] = (mdata[1] * mdata[6]) - (mdata[0] * mdata[7]);
        idata[8] = (mdata[0] * mdata[4]) - (mdata[1] * mdata[3]);

        for i in 0..9usize {
            idata[i] /= mval;
        }
    } else {
        /* 3-D inverse */
        mval = mdata[0] * mdata[5] * mdata[10]
            + mdata[1] * mdata[6] * mdata[8]
            + mdata[4] * mdata[9] * mdata[2]
            - mdata[8] * mdata[5] * mdata[2]
            - mdata[1] * mdata[4] * mdata[10]
            - mdata[0] * mdata[6] * mdata[9];

        let idata = &mut imat.data;
        idata[0] = (mdata[5] * mdata[10] - mdata[6] * mdata[9]) / mval;
        idata[1] = (mdata[2] * mdata[9] - mdata[1] * mdata[10]) / mval;
        idata[2] = (mdata[1] * mdata[6] - mdata[2] * mdata[5]) / mval;
        idata[4] = (mdata[6] * mdata[8] - mdata[4] * mdata[10]) / mval;
        idata[5] = (mdata[0] * mdata[10] - mdata[2] * mdata[8]) / mval;
        idata[6] = (mdata[2] * mdata[4] - mdata[6] * mdata[0]) / mval;
        idata[8] = (mdata[4] * mdata[9] - mdata[5] * mdata[8]) / mval;
        idata[9] = (mdata[1] * mdata[8] - mdata[0] * mdata[9]) / mval;
        idata[10] = (mdata[5] * mdata[0] - mdata[1] * mdata[4]) / mval;
        idata[12] = -(idata[0] * mdata[12] + idata[4] * mdata[13] + idata[8] * mdata[14]);
        idata[13] = -(idata[1] * mdata[12] + idata[5] * mdata[13] + idata[9] * mdata[14]);
        idata[14] = -(idata[2] * mdata[12] + idata[6] * mdata[13] + idata[10] * mdata[14]);
    }
    Some(imat)
}

/* 11/17/05: removed a translation of amat_to_rotmagstr.. This is in midas. */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_matrices_are_identity_with_source_sizes() {
        let m2 = imod_mat_new(2).unwrap();
        assert_eq!(m2.size, 9);
        assert_eq!(m2.data, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]);
        let m3 = imod_mat_new(3).unwrap();
        assert_eq!(m3.size, 16);
        assert_eq!(
            m3.data,
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.
            ]
        );
        assert!(imod_mat_new(4).is_none());
    }

    /// `imodMatMult` walks `n` 0..16 with `j` restarting after index 3 and `i`
    /// advancing by 4 (`imat.c:138-146`); multiplying identity by identity must
    /// reproduce identity, and the column-major product must match the explicit
    /// reference product.
    #[test]
    fn mult_matches_source_index_walk() {
        let ident = imod_mat_new(3).unwrap();
        let mut out = imod_mat_new(3).unwrap();
        imod_mat_mult(&ident, &ident, &mut out);
        assert_eq!(out.data, ident.data);

        let mut a = imod_mat_new(3).unwrap();
        for (i, v) in a.data.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let mut b = imod_mat_new(3).unwrap();
        for (i, v) in b.data.iter_mut().enumerate() {
            *v = (16 - i) as f32;
        }
        let mut c = imod_mat_new(3).unwrap();
        imod_mat_mult(&b, &a, &mut c);
        // Reference: out[n] = sum_k m1[4k + n%4] * m2[4*(n/4) + k]
        let mut reference = vec![0.0f32; 16];
        for n in 0..16usize {
            let mut sum = 0.0f32;
            for k in 0..4usize {
                sum += a.data[4 * k + n % 4] * b.data[4 * (n / 4) + k];
            }
            reference[n] = sum;
        }
        assert_eq!(c.data, reference);
    }

    /// `imodMatTrans` writes into elements 12..14 of a 3D matrix and 6..7 of a
    /// 2D one (`imat.c:154-166`).
    #[test]
    fn trans_and_scale_land_in_source_slots() {
        let mut mat = imod_mat_new(3).unwrap();
        imod_mat_trans(
            &mut mat,
            &Ipoint {
                x: 1.,
                y: 2.,
                z: 3.,
            },
        );
        assert_eq!((mat.data[12], mat.data[13], mat.data[14]), (1., 2., 3.));
        assert_eq!(
            imod_mat_scale(
                &mut mat,
                &Ipoint {
                    x: 2.,
                    y: 4.,
                    z: 8.
                }
            ),
            0
        );
        // Scaling is applied after the translation, so the translation scales too.
        assert_eq!((mat.data[0], mat.data[5], mat.data[10]), (2., 4., 8.));
        assert_eq!((mat.data[12], mat.data[13], mat.data[14]), (2., 8., 24.));

        let mut mat2 = imod_mat_new(2).unwrap();
        imod_mat_trans(
            &mut mat2,
            &Ipoint {
                x: 5.,
                y: 6.,
                z: 7.,
            },
        );
        assert_eq!((mat2.data[6], mat2.data[7]), (5., 6.));
    }

    /// The 90-degree rotations of `imodMatRot` (`imat.c:199-256`) applied to a
    /// unit point must agree with the hand-rolled sequential rotations that the
    /// C uses elsewhere.
    #[test]
    fn rot_about_each_axis_matches_hand_rotation() {
        for (axis, expected) in [
            (
                B3D_X,
                Ipoint {
                    x: 1.,
                    y: -3.,
                    z: 2.,
                },
            ),
            (
                B3D_Y,
                Ipoint {
                    x: 3.,
                    y: 2.,
                    z: -1.,
                },
            ),
            (
                B3D_Z,
                Ipoint {
                    x: -2.,
                    y: 1.,
                    z: 3.,
                },
            ),
        ] {
            let mut mat = imod_mat_new(3).unwrap();
            assert_eq!(imod_mat_rot(&mut mat, 90., axis), 0);
            let mut out = Ipoint::default();
            imod_mat_transform(
                &mat,
                &Ipoint {
                    x: 1.,
                    y: 2.,
                    z: 3.,
                },
                &mut out,
            );
            assert!(
                (out.x - expected.x).abs() < 1e-5
                    && (out.y - expected.y).abs() < 1e-5
                    && (out.z - expected.z).abs() < 1e-5,
                "axis {axis}: got {out:?} want {expected:?}"
            );
        }
        let mut mat = imod_mat_new(3).unwrap();
        assert_eq!(imod_mat_rot(&mut mat, 90., 3), -1);
    }

    /// `imodMatInverse` (`imat.c:453-513`) must undo a compound transform.
    #[test]
    fn inverse_round_trips_a_compound_transform() {
        let mut mat = imod_mat_new(3).unwrap();
        imod_mat_trans(
            &mut mat,
            &Ipoint {
                x: 3.,
                y: -2.,
                z: 7.,
            },
        );
        imod_mat_scale(
            &mut mat,
            &Ipoint {
                x: 2.,
                y: 0.5,
                z: 4.,
            },
        );
        imod_mat_rot(&mut mat, 30., B3D_Z);
        let inv = imod_mat_inverse(&mat).unwrap();
        let pt = Ipoint {
            x: 11.,
            y: -5.,
            z: 2.5,
        };
        let mut fwd = Ipoint::default();
        let mut back = Ipoint::default();
        imod_mat_transform(&mat, &pt, &mut fwd);
        imod_mat_transform(&inv, &fwd, &mut back);
        assert!((back.x - pt.x).abs() < 1e-3, "{back:?}");
        assert!((back.y - pt.y).abs() < 1e-3, "{back:?}");
        assert!((back.z - pt.z).abs() < 1e-3, "{back:?}");
    }

    /// `imodMatUniqueAngles` (`imat.c:375-399`) wraps into range and, when |x|
    /// exceeds 90, folds x and z by 180 and complements y.
    #[test]
    fn unique_angles_folds_out_of_range_x() {
        let (mut x, mut y, mut z) = (200.0f64, 400.0f64, -500.0f64);
        imod_mat_unique_angles(&mut x, &mut y, &mut z);
        // 200 -> -160, |x| > 90 so x folds by +180, y is complemented and z folds.
        assert!((x - 20.).abs() < 1e-9, "{x}");
        assert!((y - 140.).abs() < 1e-9, "{y}");
        assert!((z - 40.).abs() < 1e-9, "{z}");

        let (mut x, mut y, mut z) = (10.0f64, 20.0f64, 30.0f64);
        imod_mat_unique_angles(&mut x, &mut y, &mut z);
        assert_eq!((x, y, z), (10., 20., 30.));
    }

    /// `imodMatFindVector` (`imat.c:322-359`) recovers the axis and angle of a
    /// single-axis rotation.
    #[test]
    fn find_vector_recovers_z_rotation() {
        let mut mat = imod_mat_new(3).unwrap();
        imod_mat_rot(&mut mat, 37., B3D_Z);
        let mut angle = 0.0f64;
        let mut v = Ipoint::default();
        assert_eq!(imod_mat_find_vector(&mat, &mut angle, &mut v), 0);
        assert!((angle - 37.).abs() < 1e-3, "{angle}");
        assert!(
            v.x.abs() < 1e-5 && v.y.abs() < 1e-5 && (v.z - 1.).abs() < 1e-5,
            "{v:?}"
        );
    }

    /// `imodMatRotateVector` (`imat.c:259-320`) about Z must agree with
    /// `imodMatRot` about `b3dZ`.
    #[test]
    fn rotate_vector_about_z_matches_axis_rotation() {
        let mut a = imod_mat_new(3).unwrap();
        imod_mat_rot(&mut a, 25., B3D_Z);
        let mut b = imod_mat_new(3).unwrap();
        assert_eq!(
            imod_mat_rotate_vector(
                &mut b,
                25.,
                &Ipoint {
                    x: 0.,
                    y: 0.,
                    z: 1.
                }
            ),
            0
        );
        for i in 0..16 {
            assert!((a.data[i] - b.data[i]).abs() < 1e-6, "slot {i}");
        }
        assert_eq!(imod_mat_rotate_vector(&mut b, 25., &Ipoint::default()), -1);
    }
}
