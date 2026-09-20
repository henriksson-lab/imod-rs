//! Translation of `IMOD/librgctf/matrix.{h,cpp}`.

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// `RotationMatrix`, the source's fixed 3-by-3 single-precision matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationMatrix {
    pub m: [[f32; 3]; 3],
}

/// `RotationMatrix::RotationMatrix` (`matrix.cpp:41`), retained for source
/// call sites that construct an explicitly zeroed matrix.
pub fn rotation_matrix() -> RotationMatrix {
    RotationMatrix::new()
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

    /// C++ `RotationMatrix::SetToConstant`.
    pub fn set_to_constant(&mut self, constant: f32) {
        self.m = [[constant; 3]; 3];
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
