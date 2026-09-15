//! Translation of `IMOD/librgctf/matrix.{h,cpp}`.

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// `RotationMatrix`, the source's fixed 3-by-3 single-precision matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationMatrix {
    pub m: [[f32; 3]; 3],
}

impl Default for RotationMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl RotationMatrix {
    /// C++ `RotationMatrix::RotationMatrix`.
    pub fn new() -> Self {
        Self { m: [[0.0; 3]; 3] }
    }

    /// C++ `RotationMatrix::ReturnTransposed`.
    pub fn transposed(self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    /// C++ `RotationMatrix::SetToIdentity`.
    pub fn set_to_identity(&mut self) {
        self.m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    /// C++ `RotationMatrix::SetToRotation`.
    pub fn set_to_rotation(&mut self, input_x: f32, input_y: f32, input_z: f32) {
        let radians = std::f32::consts::PI / 180.0;
        let (sin_x, cos_x) = (input_x * radians).sin_cos();
        let (sin_y, cos_y) = (input_y * radians).sin_cos();
        let (sin_z, cos_z) = (input_z * radians).sin_cos();
        let sin_x_sin_y = sin_x * sin_y;
        let cos_x_sin_y = cos_x * sin_y;
        self.m[0][0] = cos_y * cos_z;
        self.m[1][0] = cos_y * sin_z;
        self.m[2][0] = -sin_y;
        self.m[0][1] = sin_x_sin_y * cos_z - cos_x * sin_z;
        self.m[1][1] = sin_x_sin_y * sin_z + cos_x * cos_z;
        self.m[2][1] = sin_x * cos_y;
        self.m[0][2] = cos_x_sin_y * cos_z + sin_x * sin_z;
        self.m[1][2] = cos_x_sin_y * sin_z - sin_x * cos_z;
        self.m[2][2] = cos_x * cos_y;
    }

    /// C++ `RotationMatrix::SetToConstant`.
    pub fn set_to_constant(&mut self, constant: f32) {
        self.m = [[constant; 3]; 3];
    }

    /// C++ `RotationMatrix::SetToValues`.
    pub fn set_to_values(
        &mut self,
        m00: f32,
        m10: f32,
        m20: f32,
        m01: f32,
        m11: f32,
        m21: f32,
        m02: f32,
        m12: f32,
        m22: f32,
    ) {
        self.m = [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]];
    }

    /// C++ inline `RotationMatrix::RotateCoords`.
    pub fn rotate_coords(&self, input_x: f32, input_y: f32, input_z: f32) -> (f32, f32, f32) {
        (
            self.m[0][0] * input_x + self.m[0][1] * input_y + self.m[0][2] * input_z,
            self.m[1][0] * input_x + self.m[1][1] * input_y + self.m[1][2] * input_z,
            self.m[2][0] * input_x + self.m[2][1] * input_y + self.m[2][2] * input_z,
        )
    }

    /// C++ inline `RotationMatrix::RotateCoords2D`.
    pub fn rotate_coords_2d(&self, input_x: f32, input_y: f32) -> (f32, f32) {
        (
            self.m[0][0] * input_x + self.m[0][1] * input_y,
            self.m[1][0] * input_x + self.m[1][1] * input_y,
        )
    }
}

impl Add for RotationMatrix {
    type Output = Self;

    /// C++ `RotationMatrix::operator +`.
    fn add(self, other: Self) -> Self {
        Self {
            m: [
                [
                    self.m[0][0] + other.m[0][0],
                    self.m[0][1] + other.m[0][1],
                    self.m[0][2] + other.m[0][2],
                ],
                [
                    self.m[1][0] + other.m[1][0],
                    self.m[1][1] + other.m[1][1],
                    self.m[1][2] + other.m[1][2],
                ],
                [
                    self.m[2][0] + other.m[2][0],
                    self.m[2][1] + other.m[2][1],
                    self.m[2][2] + other.m[2][2],
                ],
            ],
        }
    }
}

impl Sub for RotationMatrix {
    type Output = Self;

    /// C++ `RotationMatrix::operator -`.
    fn sub(self, other: Self) -> Self {
        Self {
            m: [
                [
                    self.m[0][0] - other.m[0][0],
                    self.m[0][1] - other.m[0][1],
                    self.m[0][2] - other.m[0][2],
                ],
                [
                    self.m[1][0] - other.m[1][0],
                    self.m[1][1] - other.m[1][1],
                    self.m[1][2] - other.m[1][2],
                ],
                [
                    self.m[2][0] - other.m[2][0],
                    self.m[2][1] - other.m[2][1],
                    self.m[2][2] - other.m[2][2],
                ],
            ],
        }
    }
}

impl Mul for RotationMatrix {
    type Output = Self;

    /// C++ `RotationMatrix::operator *`.
    fn mul(self, other: Self) -> Self {
        Self {
            m: [
                [
                    self.m[0][0] * other.m[0][0]
                        + self.m[0][1] * other.m[1][0]
                        + self.m[0][2] * other.m[2][0],
                    self.m[0][0] * other.m[0][1]
                        + self.m[0][1] * other.m[1][1]
                        + self.m[0][2] * other.m[2][1],
                    self.m[0][0] * other.m[0][2]
                        + self.m[0][1] * other.m[1][2]
                        + self.m[0][2] * other.m[2][2],
                ],
                [
                    self.m[1][0] * other.m[0][0]
                        + self.m[1][1] * other.m[1][0]
                        + self.m[1][2] * other.m[2][0],
                    self.m[1][0] * other.m[0][1]
                        + self.m[1][1] * other.m[1][1]
                        + self.m[1][2] * other.m[2][1],
                    self.m[1][0] * other.m[0][2]
                        + self.m[1][1] * other.m[1][2]
                        + self.m[1][2] * other.m[2][2],
                ],
                [
                    self.m[2][0] * other.m[0][0]
                        + self.m[2][1] * other.m[1][0]
                        + self.m[2][2] * other.m[2][0],
                    self.m[2][0] * other.m[0][1]
                        + self.m[2][1] * other.m[1][1]
                        + self.m[2][2] * other.m[2][1],
                    self.m[2][0] * other.m[0][2]
                        + self.m[2][1] * other.m[1][2]
                        + self.m[2][2] * other.m[2][2],
                ],
            ],
        }
    }
}

impl AddAssign for RotationMatrix {
    /// C++ `RotationMatrix::operator +=`.
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl SubAssign for RotationMatrix {
    /// C++ `RotationMatrix::operator -=`.
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl MulAssign for RotationMatrix {
    /// C++ `RotationMatrix::operator *=`.
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

#[cfg(test)]
mod tests {
    use super::RotationMatrix;

    #[test]
    fn source_matrix_operations_preserve_coordinate_order() {
        let mut matrix = RotationMatrix::new();
        matrix.set_to_values(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0);
        assert_eq!(matrix.rotate_coords(1.0, 2.0, 3.0), (14.0, 32.0, 50.0));
        assert_eq!(
            matrix.transposed().m,
            [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]
        );

        matrix.set_to_rotation(0.0, 0.0, 90.0);
        let (x, y, z) = matrix.rotate_coords(1.0, 0.0, 0.0);
        assert!(x.abs() < 0.000_001);
        assert!((y - 1.0).abs() < 0.000_001);
        assert!(z.abs() < 0.000_001);
    }
}
