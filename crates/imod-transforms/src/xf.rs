use std::io::{self, BufRead, Write};
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
