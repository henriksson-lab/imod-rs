use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

/// 2D affine transform.
///
/// Represents the operation:
///   xp = a11 * x + a12 * y + dx
///   yp = a21 * x + a22 * y + dy
///
/// In IMOD's .xf file format, one line contains:
///   a11 a12 a21 a22 dx dy
/// (Fortran column-major with 2 rows: f(1,1) f(1,2) f(2,1) f(2,2) f(1,3) f(2,3))
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearTransform {
    pub a11: f32,
    pub a12: f32,
    pub a21: f32,
    pub a22: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Default for LinearTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl LinearTransform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
            dx: 0.0,
            dy: 0.0,
        }
    }

    /// Create a pure rotation (in degrees).
    pub fn rotation(angle_deg: f32) -> Self {
        let rad = angle_deg.to_radians();
        let c = rad.cos();
        let s = rad.sin();
        Self {
            a11: c,
            a12: -s,
            a21: s,
            a22: c,
            dx: 0.0,
            dy: 0.0,
        }
    }

    /// Create a pure translation.
    pub fn translation(dx: f32, dy: f32) -> Self {
        Self {
            a11: 1.0,
            a12: 0.0,
            a21: 0.0,
            a22: 1.0,
            dx,
            dy,
        }
    }

    /// Create a uniform scale.
    pub fn scale(s: f32) -> Self {
        Self {
            a11: s,
            a12: 0.0,
            a21: 0.0,
            a22: s,
            dx: 0.0,
            dy: 0.0,
        }
    }

    /// Multiply self (applied first) by other (applied second).
    /// Equivalent to IMOD's xfMult(self, other, result).
    pub fn then(&self, other: &Self) -> Self {
        Self {
            a11: other.a11 * self.a11 + other.a12 * self.a21,
            a21: other.a21 * self.a11 + other.a22 * self.a21,
            a12: other.a11 * self.a12 + other.a12 * self.a22,
            a22: other.a21 * self.a12 + other.a22 * self.a22,
            dx: other.a11 * self.dx + other.a12 * self.dy + other.dx,
            dy: other.a21 * self.dx + other.a22 * self.dy + other.dy,
        }
    }

    /// Compute the inverse transform.
    pub fn inverse(&self) -> Self {
        let det = self.a11 * self.a22 - self.a12 * self.a21;
        let inv_a11 = self.a22 / det;
        let inv_a12 = -self.a12 / det;
        let inv_a21 = -self.a21 / det;
        let inv_a22 = self.a11 / det;
        Self {
            a11: inv_a11,
            a12: inv_a12,
            a21: inv_a21,
            a22: inv_a22,
            dx: -(inv_a11 * self.dx + inv_a12 * self.dy),
            dy: -(inv_a21 * self.dx + inv_a22 * self.dy),
        }
    }

    /// Apply transform to a point, with center of transformation at (xcen, ycen).
    /// Matches IMOD's xfApply.
    pub fn apply(&self, xcen: f32, ycen: f32, x: f32, y: f32) -> (f32, f32) {
        let xadj = x - xcen;
        let yadj = y - ycen;
        let xp = self.a11 * xadj + self.a12 * yadj + self.dx + xcen;
        let yp = self.a21 * xadj + self.a22 * yadj + self.dy + ycen;
        (xp, yp)
    }

    /// Apply transform to a point (no center offset).
    pub fn apply_raw(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.a11 * x + self.a12 * y + self.dx,
            self.a21 * x + self.a22 * y + self.dy,
        )
    }

    /// Determinant of the linear part.
    pub fn determinant(&self) -> f32 {
        self.a11 * self.a22 - self.a12 * self.a21
    }

    /// Extract the rotation angle in degrees (from the linear part).
    pub fn rotation_angle(&self) -> f32 {
        self.a21.atan2(self.a11).to_degrees()
    }

    /// Extract the scale (geometric mean of singular values approximation).
    pub fn scale_factor(&self) -> f32 {
        self.determinant().abs().sqrt()
    }
}

/// Read transforms from an .xf or .xg file.
/// Format: one line per transform, 6 values: a11 a12 a21 a22 dx dy
pub fn read_xf_file(path: impl AsRef<Path>) -> io::Result<Vec<LinearTransform>> {
    let file = std::fs::File::open(path.as_ref())?;
    let reader = io::BufReader::new(file);
    let mut transforms = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let vals: Vec<f32> = trimmed
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() >= 6 {
            transforms.push(LinearTransform {
                a11: vals[0],
                a12: vals[1],
                a21: vals[2],
                a22: vals[3],
                dx: vals[4],
                dy: vals[5],
            });
        }
    }

    Ok(transforms)
}

/// Write transforms to an .xf or .xg file.
/// Format matches IMOD: 4f12.7 2f12.3
pub fn write_xf_file(
    path: impl AsRef<Path>,
    transforms: &[LinearTransform],
) -> io::Result<()> {
    let mut file = std::fs::File::create(path.as_ref())?;
    for xf in transforms {
        writeln!(
            file,
            "{:12.7}{:12.7}{:12.7}{:12.7}{:12.3}{:12.3}",
            xf.a11, xf.a12, xf.a21, xf.a22, xf.dx, xf.dy
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 3D rotation matrices from/to Euler angles
// ---------------------------------------------------------------------------

/// A 3x3 rotation matrix stored in column-major order (matching IMOD convention).
///
/// Element layout: `data[col * 3 + row]`, so `data[0]` = r11, `data[1]` = r21,
/// `data[2]` = r31, `data[3]` = r12, etc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationMatrix3 {
    pub data: [f64; 9],
}

impl Default for RotationMatrix3 {
    fn default() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl RotationMatrix3 {
    /// Access element at (row, col), zero-indexed.
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.data[col * 3 + row]
    }

    /// Set element at (row, col), zero-indexed.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[col * 3 + row] = val;
    }
}

/// Given rotation angles (in degrees) about the X, Y, and Z axes, compute the 3D
/// rotation matrix.
///
/// Translated from IMOD `anglesToMatrix` in `linearxforms.c`.
///
/// Conventions:
///   - `[R]` premultiplies a column vector of coordinates.
///   - Rotations are applied in the order Z first, X last, i.e. `[R] = [X][Y][Z]`.
///   - Rotations are right-handed: a positive angle rotates counter-clockwise when
///     looking down the respective axis.
///
/// `angles` = `(x_deg, y_deg, z_deg)`.
pub fn angles_to_matrix(angles: (f64, f64, f64)) -> RotationMatrix3 {
    const CNV: f64 = 0.017_453_292_1;
    let alpha = angles.0 * CNV;
    let beta = angles.1 * CNV;
    let gamma = angles.2 * CNV;
    let ca = alpha.cos();
    let cb = beta.cos();
    let cg = gamma.cos();
    let sa = alpha.sin();
    let sb = beta.sin();
    let sg = gamma.sin();

    let mut m = RotationMatrix3::default();
    // row 0
    m.set(0, 0, cb * cg);
    m.set(0, 1, -cb * sg);
    m.set(0, 2, sb);
    // row 1
    m.set(1, 0, sa * sb * cg + ca * sg);
    m.set(1, 1, -sa * sb * sg + ca * cg);
    m.set(1, 2, -sa * cb);
    // row 2
    m.set(2, 0, -ca * sb * cg + sa * sg);
    m.set(2, 1, ca * sb * sg + sa * cg);
    m.set(2, 2, ca * cb);
    m
}

/// Decompose a 3D rotation matrix into Euler angles (in degrees) about X, Y, and Z axes.
///
/// Translated from IMOD `matrixToAngles` in `linearxforms.c`.
///
/// Returns `Ok((x_deg, y_deg, z_deg))` on success, or `Err(det_minus_one)` if the
/// determinant of the matrix is not near 1.0.
///
/// The conventions of [`angles_to_matrix`] are followed: rotations are applied
/// Z first, X last.
pub fn matrix_to_angles(matrix: &RotationMatrix3) -> Result<(f64, f64, f64), f64> {
    const CNV: f64 = 0.017_453_292;
    const CRIT: f64 = 0.01;
    const SMALL: f64 = 0.000_000_1;

    let r11 = matrix.at(0, 0);
    let r12 = matrix.at(0, 1);
    let r13 = matrix.at(0, 2);
    let r21 = matrix.at(1, 0);
    let r22 = matrix.at(1, 1);
    let r23 = matrix.at(1, 2);
    let r31 = matrix.at(2, 0);
    let r32 = matrix.at(2, 1);
    let r33 = matrix.at(2, 2);

    // Check determinant
    let det = r11 * r22 * r33 - r11 * r23 * r32 + r12 * r23 * r31 - r12 * r21 * r33
        + r13 * r21 * r32 - r13 * r22 * r31;
    let det_err = det - 1.0;
    if det_err > CRIT || det_err < -CRIT {
        return Err(det_err);
    }

    let (alpha, beta, gamma);

    let test1 = (r13 - 1.0).abs();
    let test2 = (r13 + 1.0).abs();
    if test1 < SMALL || test2 < SMALL {
        beta = r13.asin();
        gamma = r21.atan2(r22);
        alpha = 0.0;
    } else if r13.abs() <= SMALL {
        beta = 0.0;
        gamma = (-r12).atan2(r11);
        alpha = (-r23).atan2(r33);
    } else {
        alpha = (-r23).atan2(r33);
        gamma = (-r12).atan2(r11);
        let cosg = gamma.cos();
        let sing = gamma.sin();
        let cosb = if cosg > CRIT || cosg < -CRIT {
            r11 / cosg
        } else {
            -r12 / sing
        };
        beta = r13.atan2(cosb);
    }

    Ok((alpha / CNV, beta / CNV, gamma / CNV))
}

// ---------------------------------------------------------------------------
// Find 2D linear transform from point correspondences
// ---------------------------------------------------------------------------

/// Result of [`find_transform`] when deviation information is requested.
#[derive(Debug, Clone)]
pub struct TransformFitResult {
    /// The computed transform.
    pub xf: LinearTransform,
    /// Mean deviation between transformed source points and target points.
    pub dev_avg: f64,
    /// Standard deviation of the deviations.
    pub dev_sd: f64,
    /// Maximum deviation.
    pub dev_max: f64,
    /// Index (0-based) of the point with maximum deviation.
    pub max_dev_point: usize,
}

/// Mode for [`find_transform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    /// Solve for a general 2D linear transform (6 DOF).
    General,
    /// Solve for translation only (2 DOF).
    TranslationOnly,
    /// Solve for rotation and translation (3 DOF).
    RotationTranslation,
    /// Solve for rotation, translation, and uniform magnification (4 DOF).
    RotationTranslationMag,
}

/// Find a 2D linear transform from corresponding point pairs.
///
/// Translated from IMOD `findTransform` in `findtransform.c`.
///
/// `source` and `target` are slices of (x, y) points. Coordinates should be
/// centered (i.e., `xcen`/`ycen` already subtracted). The center values are only
/// needed if you want to reconstruct uncentered coordinates later.
///
/// For `TransformMode::General`, a least-squares solution is computed via the
/// normal equations (matching IMOD's `multRegress` path).
///
/// Returns `None` if the system is singular or there are fewer points than needed.
pub fn find_transform(
    source: &[(f64, f64)],
    target: &[(f64, f64)],
    mode: TransformMode,
) -> Option<TransformFitResult> {
    let n = source.len();
    if n == 0 || n != target.len() {
        return None;
    }

    let xf = match mode {
        TransformMode::TranslationOnly => {
            // Mean difference
            let mut dx = 0.0_f64;
            let mut dy = 0.0_f64;
            for i in 0..n {
                dx += target[i].0 - source[i].0;
                dy += target[i].1 - source[i].1;
            }
            dx /= n as f64;
            dy /= n as f64;
            LinearTransform {
                a11: 1.0,
                a12: 0.0,
                a21: 0.0,
                a22: 1.0,
                dx: dx as f32,
                dy: dy as f32,
            }
        }
        TransformMode::RotationTranslation | TransformMode::RotationTranslationMag => {
            // Compute means and sums of squares / cross-products of deviations
            if n < 2 {
                return None;
            }
            let mut mx0 = 0.0_f64;
            let mut my0 = 0.0_f64;
            let mut mx1 = 0.0_f64;
            let mut my1 = 0.0_f64;
            for i in 0..n {
                mx0 += source[i].0;
                my0 += source[i].1;
                mx1 += target[i].0;
                my1 += target[i].1;
            }
            let nf = n as f64;
            mx0 /= nf;
            my0 /= nf;
            mx1 /= nf;
            my1 /= nf;

            // Sums of deviations cross-products:
            // ss_xx = sum((sx - mx0)*(tx - mx1)), etc.
            // Using the indexing from the C code with statMatrices, the relevant
            // cross-product sums are:
            //   ssDev[icolX-1][0] ~ sum(dx_src * dx_tgt)  i.e. ss(target_x, source_x)
            //   ssDev[icolX-1][1] ~ sum(dy_src * dx_tgt)  i.e. ss(target_x, source_y)
            //   ssDev[icolY-1][0] ~ sum(dx_src * dy_tgt)  i.e. ss(target_y, source_x)
            //   ssDev[icolY-1][1] ~ sum(dy_src * dy_tgt)  i.e. ss(target_y, source_y)
            //   ssDev[0][0]       ~ sum(dx_src^2)
            //   ssDev[1][1]       ~ sum(dy_src^2)
            let mut ss_src_xx = 0.0_f64;
            let mut ss_src_yy = 0.0_f64;
            let mut ss_tx_sx = 0.0_f64;
            let mut ss_tx_sy = 0.0_f64;
            let mut ss_ty_sx = 0.0_f64;
            let mut ss_ty_sy = 0.0_f64;

            for i in 0..n {
                let dsx = source[i].0 - mx0;
                let dsy = source[i].1 - my0;
                let dtx = target[i].0 - mx1;
                let dty = target[i].1 - my1;
                ss_src_xx += dsx * dsx;
                ss_src_yy += dsy * dsy;
                ss_tx_sx += dtx * dsx;
                ss_tx_sy += dtx * dsy;
                ss_ty_sx += dty * dsx;
                ss_ty_sy += dty * dsy;
            }

            // theta = atan(-(ss_tx_sy - ss_ty_sx) / (ss_tx_sx + ss_ty_sy))
            let denom = ss_tx_sx + ss_ty_sy;
            let numer = -(ss_tx_sy - ss_ty_sx);
            let theta = numer.atan2(denom);
            let mut sin_theta = theta.sin();
            let mut cos_theta = theta.cos();

            if mode == TransformMode::RotationTranslationMag {
                let gmag = (ss_tx_sx + ss_ty_sy) * cos_theta
                    - (ss_tx_sy - ss_ty_sx) * sin_theta;
                let src_ss = ss_src_xx + ss_src_yy;
                if src_ss.abs() < 1e-30 {
                    return None;
                }
                let gmag = gmag / src_ss;
                sin_theta *= gmag;
                cos_theta *= gmag;
            }

            let dx = mx1 - mx0 * cos_theta + my0 * sin_theta;
            let dy = my1 - mx0 * sin_theta - my0 * cos_theta;
            LinearTransform {
                a11: cos_theta as f32,
                a12: -sin_theta as f32,
                a21: sin_theta as f32,
                a22: cos_theta as f32,
                dx: dx as f32,
                dy: dy as f32,
            }
        }
        TransformMode::General => {
            // Least-squares via normal equations: solve for a11,a12 and a21,a22
            // separately (two regressions, each with 2 predictors).
            // For each target coordinate (tx or ty):
            //   tx = b0*sx + b1*sy + c  (where sx,sy are source coords)
            if n < 3 {
                return None;
            }
            let nf = n as f64;
            // Means
            let mut mx = 0.0_f64;
            let mut my = 0.0_f64;
            let mut mtx = 0.0_f64;
            let mut mty = 0.0_f64;
            for i in 0..n {
                mx += source[i].0;
                my += source[i].1;
                mtx += target[i].0;
                mty += target[i].1;
            }
            mx /= nf;
            my /= nf;
            mtx /= nf;
            mty /= nf;

            // Sums of squares and cross products (deviations from mean)
            let mut sxx = 0.0_f64;
            let mut syy = 0.0_f64;
            let mut sxy = 0.0_f64;
            let mut stx_x = 0.0_f64;
            let mut stx_y = 0.0_f64;
            let mut sty_x = 0.0_f64;
            let mut sty_y = 0.0_f64;
            for i in 0..n {
                let dx = source[i].0 - mx;
                let dy = source[i].1 - my;
                let dtx = target[i].0 - mtx;
                let dty = target[i].1 - mty;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
                stx_x += dtx * dx;
                stx_y += dtx * dy;
                sty_x += dty * dx;
                sty_y += dty * dy;
            }

            // Solve 2x2 system for each dependent variable:
            //   [sxx  sxy] [b0]   [stx_x]      [sty_x]
            //   [sxy  syy] [b1] = [stx_y]  or  [sty_y]
            let det = sxx * syy - sxy * sxy;
            if det.abs() < 1e-30 {
                return None;
            }
            let inv_det = 1.0 / det;

            let a11 = (syy * stx_x - sxy * stx_y) * inv_det;
            let a12 = (sxx * stx_y - sxy * stx_x) * inv_det;
            let a21 = (syy * sty_x - sxy * sty_y) * inv_det;
            let a22 = (sxx * sty_y - sxy * sty_x) * inv_det;
            let cx = mtx - a11 * mx - a12 * my;
            let cy = mty - a21 * mx - a22 * my;

            LinearTransform {
                a11: a11 as f32,
                a12: a12 as f32,
                a21: a21 as f32,
                a22: a22 as f32,
                dx: cx as f32,
                dy: cy as f32,
            }
        }
    };

    // Compute deviation statistics
    let mut dev_sum = 0.0_f64;
    let mut dev_sum_sq = 0.0_f64;
    let mut dev_max = -1.0_f64;
    let mut max_idx = 0usize;
    for i in 0..n {
        let (xx, yy) = xf.apply_raw(source[i].0 as f32, source[i].1 as f32);
        let xdev = target[i].0 - xx as f64;
        let ydev = target[i].1 - yy as f64;
        let dev = (xdev * xdev + ydev * ydev).sqrt();
        dev_sum += dev;
        dev_sum_sq += dev * dev;
        if dev > dev_max {
            dev_max = dev;
            max_idx = i;
        }
    }
    let nf = n as f64;
    let dev_avg = dev_sum / nf;
    let dev_sd = if n > 1 {
        let var = (dev_sum_sq - dev_sum * dev_sum / nf) / (nf - 1.0);
        if var > 0.0 { var.sqrt() } else { 0.0 }
    } else {
        0.0
    };

    Some(TransformFitResult {
        xf,
        dev_avg,
        dev_sd,
        dev_max,
        max_dev_point: max_idx,
    })
}

// ---------------------------------------------------------------------------
// Alternative constructors: read from reader (C-style readOneXform / readAllXforms)
// ---------------------------------------------------------------------------

impl LinearTransform {
    /// Read a single transform from a buffered reader.
    ///
    /// Translated from IMOD `readOneXform` in `linearxforms.c`.
    ///
    /// Reads one line containing 6 whitespace-separated floats in the order
    /// `a11 a12 a21 a22 dx dy`.
    ///
    /// Returns `Ok(Some(xf))` on success, `Ok(None)` at EOF, or `Err` on parse error.
    pub fn read_one<R: BufRead>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            return Ok(None); // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Blank line: try to distinguish EOF-like blank from mid-file blank.
            // IMOD's readOneXform would return EOF in this case via fscanf semantics.
            return Ok(None);
        }
        let vals: Vec<f32> = trimmed
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() < 6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Expected 6 values for transform, got {} on line: {}",
                    vals.len(),
                    trimmed
                ),
            ));
        }
        // File order: a11 a12 a21 a22 dx dy  (same as the .xf format)
        Ok(Some(LinearTransform {
            a11: vals[0],
            a12: vals[1],
            a21: vals[2],
            a22: vals[3],
            dx: vals[4],
            dy: vals[5],
        }))
    }

    /// Read all transforms from a buffered reader until EOF.
    ///
    /// Translated from IMOD `readAllXforms` in `linearxforms.c`.
    ///
    /// Optionally limit the number read with `max_read` (pass `None` for unlimited).
    pub fn read_all<R: BufRead>(reader: &mut R, max_read: Option<usize>) -> io::Result<Vec<Self>> {
        let mut xforms = Vec::new();
        loop {
            if let Some(max) = max_read {
                if xforms.len() >= max {
                    break;
                }
            }
            match Self::read_one(reader)? {
                Some(xf) => xforms.push(xf),
                None => break,
            }
        }
        Ok(xforms)
    }

    /// Read all transforms from a file path. Convenience wrapper around [`Self::read_all`].
    ///
    /// This is equivalent to `readAllXforms` called on a freshly opened file.
    pub fn read_all_from_file(path: impl AsRef<Path>) -> io::Result<Vec<Self>> {
        let file = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);
        Self::read_all(&mut reader, None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests_new {
    use super::*;

    #[test]
    fn angles_roundtrip() {
        let angles = (10.0_f64, 20.0, 30.0);
        let m = angles_to_matrix(angles);
        let (x, y, z) = matrix_to_angles(&m).unwrap();
        assert!((x - angles.0).abs() < 1e-4, "x: {} vs {}", x, angles.0);
        assert!((y - angles.1).abs() < 1e-4, "y: {} vs {}", y, angles.1);
        assert!((z - angles.2).abs() < 1e-4, "z: {} vs {}", z, angles.2);
    }

    #[test]
    fn angles_identity() {
        let m = angles_to_matrix((0.0, 0.0, 0.0));
        let id = RotationMatrix3::default();
        for i in 0..9 {
            assert!((m.data[i] - id.data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn matrix_to_angles_bad_det() {
        let mut m = RotationMatrix3::default();
        m.set(0, 0, 2.0); // break the determinant
        assert!(matrix_to_angles(&m).is_err());
    }

    #[test]
    fn find_transform_translation() {
        let src = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let tgt = vec![(5.0, 3.0), (6.0, 3.0), (5.0, 4.0)];
        let r = find_transform(&src, &tgt, TransformMode::TranslationOnly).unwrap();
        assert!((r.xf.dx - 5.0).abs() < 1e-5);
        assert!((r.xf.dy - 3.0).abs() < 1e-5);
    }

    #[test]
    fn find_transform_general_identity() {
        let src = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let tgt = src.clone();
        let r = find_transform(&src, &tgt, TransformMode::General).unwrap();
        assert!((r.xf.a11 - 1.0).abs() < 1e-5);
        assert!((r.xf.a22 - 1.0).abs() < 1e-5);
        assert!(r.xf.a12.abs() < 1e-5);
        assert!(r.xf.a21.abs() < 1e-5);
        assert!(r.dev_max < 1e-5);
    }

    #[test]
    fn read_one_from_string() {
        let data = b"   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n";
        let mut reader = io::BufReader::new(&data[..]);
        let xf = LinearTransform::read_one(&mut reader).unwrap().unwrap();
        assert!((xf.a11 - 1.0).abs() < 1e-6);
        assert!((xf.a22 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn read_all_from_string() {
        let data = b"1 0 0 1 5 3\n0.5 -0.5 0.5 0.5 1 2\n";
        let mut reader = io::BufReader::new(&data[..]);
        let xfs = LinearTransform::read_all(&mut reader, None).unwrap();
        assert_eq!(xfs.len(), 2);
        assert!((xfs[0].dx - 5.0).abs() < 1e-6);
        assert!((xfs[1].a11 - 0.5).abs() < 1e-6);
    }
}
