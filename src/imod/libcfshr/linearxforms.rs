//! Translation of `IMOD/libcfshr/linearxforms.c`.
#![allow(dead_code)]
use crate::imod::libcfshr::b3dutil::{CArg, c_format};
use std::cell::Cell;
use std::io::{BufRead, Write};

/// Matches `xfUnit`.
pub fn xf_unit(matrix: &mut [f32], value: f32, rows: usize) {
    matrix[..3 * rows].fill(0.);
    matrix[0] = value;
    matrix[1 + rows] = value;
    if rows > 2 {
        matrix[2 * (1 + rows)] = 1.;
    }
}
/// Matches `xfunit`.
pub fn xfunit(matrix: &mut [f32], value: f32) {
    xf_unit(matrix, value, 2)
}
/// Matches `xfCopy`.
pub fn xf_copy(from: &[f32], from_rows: usize, to: &mut [f32], to_rows: usize) {
    if to_rows > 2 {
        xf_unit(to, 1., to_rows);
    }
    for column in 0..3 {
        for row in 0..2 {
            to[row + column * to_rows] = from[row + column * from_rows];
        }
    }
}
/// Matches `xfcopy`.
pub fn xfcopy(from: &[f32], to: &mut [f32]) {
    xf_copy(from, 2, to, 2)
}
/// Matches `xfMult`.
pub fn xf_mult(first: &[f32], second: &[f32], product: &mut [f32], rows: usize) {
    let mut temp = [0.; 9];
    let x = 2 * rows;
    let y = x + 1;
    temp[0] = second[0] * first[0] + second[rows] * first[1];
    temp[1] = second[1] * first[0] + second[rows + 1] * first[1];
    temp[rows] = second[0] * first[rows] + second[rows] * first[rows + 1];
    temp[rows + 1] = second[1] * first[rows] + second[rows + 1] * first[rows + 1];
    temp[x] = second[0] * first[x] + second[rows] * first[y] + second[x];
    temp[y] = second[1] * first[x] + second[rows + 1] * first[y] + second[y];
    if rows > 2 {
        temp[2] = 0.;
        temp[5] = 0.;
        temp[8] = 1.;
    }
    product[..3 * rows].copy_from_slice(&temp[..3 * rows]);
}
/// Matches `xfmult`.
pub fn xfmult(first: &[f32], second: &[f32], product: &mut [f32]) {
    xf_mult(first, second, product, 2)
}
/// Matches `xfInvert`.
pub fn xf_invert(matrix: &[f32], inverse: &mut [f32], rows: usize) {
    let mut temp = [0.; 9];
    let determinant = matrix[0] * matrix[rows + 1] - matrix[rows] * matrix[1];
    let x = 2 * rows;
    let y = x + 1;
    temp[0] = matrix[rows + 1] / determinant;
    temp[rows] = -matrix[rows] / determinant;
    temp[1] = -matrix[1] / determinant;
    temp[rows + 1] = matrix[0] / determinant;
    temp[x] = -(temp[0] * matrix[x] + temp[rows] * matrix[y]);
    temp[y] = -(temp[1] * matrix[x] + temp[rows + 1] * matrix[y]);
    if rows > 2 {
        temp[2] = 0.;
        temp[5] = 0.;
        temp[8] = 1.;
    }
    inverse[..3 * rows].copy_from_slice(&temp[..3 * rows]);
}
/// Matches `xfinvert`.
pub fn xfinvert(matrix: &[f32], inverse: &mut [f32]) {
    xf_invert(matrix, inverse, 2)
}
/// Matches `xfApply`.
pub fn xf_apply(
    matrix: &[f32],
    x_center: f32,
    y_center: f32,
    x: f32,
    y: f32,
    rows: usize,
) -> (f32, f32) {
    let xa = x - x_center;
    let ya = y - y_center;
    (
        matrix[0] * xa + matrix[rows] * ya + matrix[2 * rows] + x_center,
        matrix[1] * xa + matrix[rows + 1] * ya + matrix[2 * rows + 1] + y_center,
    )
}
/// Matches `xfapply`.
pub fn xfapply(matrix: &[f32], x_center: f32, y_center: f32, x: f32, y: f32) -> (f32, f32) {
    xf_apply(matrix, x_center, y_center, x, y, 2)
}
/// Matches `anglesToMatrix`.
pub fn angles_to_matrix(angles: &[f32; 3], matrix: &mut [f32], rows: usize) {
    let (a, b, g) = (
        angles[0] as f64 * 0.0174532921,
        angles[1] as f64 * 0.0174532921,
        angles[2] as f64 * 0.0174532921,
    );
    let (ca, cb, cg, sa, sb, sg) = (a.cos(), b.cos(), g.cos(), a.sin(), b.sin(), g.sin());
    matrix[0] = (cb * cg) as f32;
    matrix[rows] = (-cb * sg) as f32;
    matrix[2 * rows] = sb as f32;
    matrix[1] = (sa * sb * cg + ca * sg) as f32;
    matrix[1 + rows] = (-sa * sb * sg + ca * cg) as f32;
    matrix[1 + 2 * rows] = (-sa * cb) as f32;
    matrix[2] = (-ca * sb * cg + sa * sg) as f32;
    matrix[2 + rows] = (ca * sb * sg + sa * cg) as f32;
    matrix[2 + 2 * rows] = (ca * cb) as f32;
}
/// Matches `icalc_matrix`.
pub fn icalc_matrix(angles: &[f32; 3], matrix: &mut [f32]) {
    angles_to_matrix(angles, matrix, 3)
}
thread_local! {
    /// `linearxforms.c:226`: `double sDet;` is a file-scope global, not a local,
    /// and `icalc_angles` prints it after `matrixToAngles` has failed.  Keeping
    /// it a local would lose the value that failure path reports.  A
    /// thread-local `Cell` holds it without `static mut`: it is written and read
    /// inside a single `icalc_angles` call, so the narrower scope changes
    /// nothing about when it is set or what is printed.
    pub static S_DET: Cell<f64> = const { Cell::new(0.0) };
}
/// Matches `matrixToAngles`.
pub fn matrix_to_angles(matrix: &[f32], rows: usize) -> Result<(f64, f64, f64), ()> {
    let (r11, r12, r13, r21, r22, r23, r31, r32, r33) = (
        matrix[0] as f64,
        matrix[rows] as f64,
        matrix[2 * rows] as f64,
        matrix[1] as f64,
        matrix[1 + rows] as f64,
        matrix[1 + 2 * rows] as f64,
        matrix[2] as f64,
        matrix[2 + rows] as f64,
        matrix[2 + 2 * rows] as f64,
    );
    S_DET.set(
        r11 * r22 * r33 - r11 * r23 * r32 + r12 * r23 * r31 - r12 * r21 * r33 + r13 * r21 * r32
            - r13 * r22 * r31,
    );
    S_DET.set(S_DET.get() - 1.0);
    if S_DET.get() > 0.01 || S_DET.get() < -0.01 {
        return Err(());
    }
    let (a, b, g) = if (r13 - 1.).abs() < 1e-7 || (r13 + 1.).abs() < 1e-7 {
        (0., r13.asin(), r21.atan2(r22))
    } else if r13.abs() <= 1e-7 {
        ((-r23).atan2(r33), 0., (-r12).atan2(r11))
    } else {
        let a = (-r23).atan2(r33);
        let g = (-r12).atan2(r11);
        let cb = if g.cos().abs() > 0.01 {
            r11 / g.cos()
        } else {
            -r12 / g.sin()
        };
        (a, r13.atan2(cb), g)
    };
    Ok((a / 0.017453292, b / 0.017453292, g / 0.017453292))
}
/// Matches `icalc_angles`.  `linearxforms.c:302` returns `void`: on a matrix
/// that is not a pure rotation it leaves `angles` untouched and prints the five
/// lines the old Fortran routine printed, to stdout.  That printing is part of
/// the contract — a caller has no other way to learn the call failed.
pub fn icalc_angles(angles: &mut [f32; 3], matrix: &[f32]) {
    if let Ok((x, y, z)) = matrix_to_angles(matrix, 3) {
        angles[0] = x as f32;
        angles[1] = y as f32;
        angles[2] = z as f32;
        return;
    }

    // And here is what the old fortran routine did:
    let mut out = std::io::stdout();
    let _ = out.write_all(
        c_format(
            "icalc_angles - matrix %10.6f %10.6f %10.6f\n",
            &[
                CArg::Dbl(matrix[0] as f64),
                CArg::Dbl(matrix[3] as f64),
                CArg::Dbl(matrix[6] as f64),
            ],
        )
        .as_bytes(),
    );
    let _ = out.write_all(
        c_format(
            "                      %10.6f %10.6f %10.6f\n",
            &[
                CArg::Dbl(matrix[1] as f64),
                CArg::Dbl(matrix[4] as f64),
                CArg::Dbl(matrix[7] as f64),
            ],
        )
        .as_bytes(),
    );
    let _ = out.write_all(
        c_format(
            "                      %10.6f %10.6f %10.6f\n",
            &[
                CArg::Dbl(matrix[2] as f64),
                CArg::Dbl(matrix[5] as f64),
                CArg::Dbl(matrix[8] as f64),
            ],
        )
        .as_bytes(),
    );
    let _ = out.write_all(c_format("determinant - %f\n", &[CArg::Dbl(S_DET.get())]).as_bytes());
    let _ = out
        .write_all(c_format("ERROR: icalc_angles - Not a pure rotation matrix\n", &[]).as_bytes());
    let _ = out.flush();
}

/// Matches `invertMatrix`.
pub fn invert_matrix(matrix: &[f32; 9], inverse: &mut [f32; 9]) {
    let d = matrix[0] * matrix[4] * matrix[8]
        + matrix[1] * matrix[5] * matrix[6]
        + matrix[3] * matrix[7] * matrix[2]
        - matrix[6] * matrix[4] * matrix[2]
        - matrix[1] * matrix[3] * matrix[8]
        - matrix[0] * matrix[5] * matrix[7];
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / d;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / d;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / d;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / d;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / d;
    inverse[5] = (matrix[2] * matrix[3] - matrix[5] * matrix[0]) / d;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / d;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / d;
    inverse[8] = (matrix[4] * matrix[0] - matrix[1] * matrix[3]) / d;
}
/// Matches `inv_matrix`.
pub fn inv_matrix(matrix: &[f32; 9], inverse: &mut [f32; 9]) {
    invert_matrix(matrix, inverse)
}
/// Matches `readOneXform`.
pub fn read_one_xform<R: BufRead>(reader: &mut R, xf: &mut [f32; 6]) -> i32 {
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Err(_) => 2,
        Ok(0) => 1,
        Ok(_) => {
            let v = line
                .split_whitespace()
                .map(str::parse::<f32>)
                .collect::<Result<Vec<_>, _>>();
            match v {
                Ok(v) if v.len() >= 6 => {
                    xf.copy_from_slice(&[v[0], v[2], v[1], v[3], v[4], v[5]]);
                    0
                }
                _ => 3,
            }
        }
    }
}
/// Matches `readAllXforms`.
pub fn read_all_xforms<R: BufRead>(
    reader: &mut R,
    xforms: &mut [[f32; 6]],
    number_read: &mut usize,
) -> i32 {
    for index in 0..xforms.len() {
        let ret = read_one_xform(reader, &mut xforms[index]);
        if ret < 2 {
            *number_read = index + 1;
        }
        if ret != 0 {
            return if ret == 1 { 0 } else { ret };
        }
    }
    0
}
/// Matches `exitFromXFReadError`.
pub fn exit_from_xf_read_error(error: i32, description: &str) -> Result<(), String> {
    if error == 0 {
        Ok(())
    } else if error == 3 {
        Err(format!(
            "Reading {description} transform file: fewer than 6 values on a line"
        ))
    } else {
        Err(format!("Reading {description} transform file"))
    }
}
/// Matches `writeXform`.
pub fn write_xform<W: Write>(writer: &mut W, xf: &[f32; 6]) -> i32 {
    match writeln!(
        writer,
        " {:11.7} {:11.7} {:11.7} {:11.7} {:11.3} {:11.3}",
        xf[0], xf[2], xf[1], xf[3], xf[4], xf[5]
    ) {
        Ok(_) => 0,
        Err(_) => 1,
    }
}
