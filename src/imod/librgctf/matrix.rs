//! Translation of `IMOD/librgctf/matrix.{h,cpp}`.

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// C++ `#define AL_PI 3.14159265358979323846` (`matrix.cpp:6`).
const AL_PI: f64 = 3.14159265358979323846;

/// C++ `#define FLOATSINCOS(x, s, c)` (`matrix.cpp:14`).
///
/// On x86-64 Linux the macro expands to the x87 `fsincos` instruction, which
/// evaluates both functions in 80-bit extended precision from the double
/// argument `(x) * AL_PI / 128.0` and then stores each result into a `float`.
/// `f64::sin_cos` is the closest reachable equivalent: the argument and the
/// rounding to `float` are the same, only the internal precision of the
/// transcendental differs.  `SetToRotation` is the only caller and nothing in
/// `ctffind` reaches it.
fn floatsincos(x: f32) -> (f32, f32) {
    let angle = f64::from(x) * AL_PI / 128.0;
    (angle.sin() as f32, angle.cos() as f32)
}

/// `RotationMatrix`, the source's fixed 3-by-3 single-precision matrix
/// (`matrix.h:3`).
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
    /// C++ `RotationMatrix::RotationMatrix` (`matrix.cpp:41`).
    pub fn new() -> Self {
        let mut matrix = Self { m: [[0.0; 3]; 3] };
        matrix.set_to_constant(0.0);
        matrix
    }

    /// C++ `RotationMatrix::ReturnTransposed` (`matrix.cpp:178`).
    pub fn return_transposed(&self) -> Self {
        let mut temp_matrix = Self::new();

        temp_matrix.m[0][0] = self.m[0][0];
        temp_matrix.m[0][1] = self.m[1][0];
        temp_matrix.m[0][2] = self.m[2][0];
        temp_matrix.m[1][0] = self.m[0][1];
        temp_matrix.m[1][1] = self.m[1][1];
        temp_matrix.m[1][2] = self.m[2][1];
        temp_matrix.m[2][0] = self.m[0][2];
        temp_matrix.m[2][1] = self.m[1][2];
        temp_matrix.m[2][2] = self.m[2][2];

        temp_matrix
    }

    /// C++ `RotationMatrix::SetToIdentity` (`matrix.cpp:195`).
    pub fn set_to_identity(&mut self) {
        self.m[0][0] = 1.0;
        self.m[1][0] = 0.0;
        self.m[2][0] = 0.0;
        self.m[0][1] = 0.0;
        self.m[1][1] = 1.0;
        self.m[2][1] = 0.0;
        self.m[0][2] = 0.0;
        self.m[1][2] = 0.0;
        self.m[2][2] = 1.0;
    }

    /// C++ `RotationMatrix::SetToRotation` (`matrix.cpp:208`).
    pub fn set_to_rotation(&mut self, input_x: f32, input_y: f32, input_z: f32) {
        let x = ((256.0 / 360.0) * f64::from(input_x)) as f32;
        let y = ((256.0 / 360.0) * f64::from(input_y)) as f32;
        let z = ((256.0 / 360.0) * f64::from(input_z)) as f32;

        // MAKE_ROTATION_f(x, y, z) (`matrix.cpp:16`)
        let (sin_x, cos_x) = floatsincos(x);
        let (sin_y, cos_y) = floatsincos(y);
        let (sin_z, cos_z) = floatsincos(z);

        let sinx_siny = sin_x * sin_y;
        let cosx_siny = cos_x * sin_y;

        self.m[0][0] = cos_y * cos_z;
        self.m[1][0] = cos_y * sin_z;
        self.m[2][0] = -sin_y;
        self.m[0][1] = (sinx_siny * cos_z) - (cos_x * sin_z);
        self.m[1][1] = (sinx_siny * sin_z) + (cos_x * cos_z);
        self.m[2][1] = sin_x * cos_y;
        self.m[0][2] = (cosx_siny * cos_z) + (sin_x * sin_z);
        self.m[1][2] = (cosx_siny * sin_z) - (sin_x * cos_z);
        self.m[2][2] = cos_x * cos_y;
    }

    /// C++ `RotationMatrix::SetToConstant` (`matrix.cpp:241`).
    pub fn set_to_constant(&mut self, constant: f32) {
        self.m[0][0] = constant;
        self.m[1][0] = constant;
        self.m[2][0] = constant;
        self.m[0][1] = constant;
        self.m[1][1] = constant;
        self.m[2][1] = constant;
        self.m[0][2] = constant;
        self.m[1][2] = constant;
        self.m[2][2] = constant;
    }

    /// C++ `RotationMatrix::SetToValues` (`matrix.cpp:254`).
    #[allow(clippy::too_many_arguments)]
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
        self.m[0][0] = m00;
        self.m[1][0] = m10;
        self.m[2][0] = m20;
        self.m[0][1] = m01;
        self.m[1][1] = m11;
        self.m[2][1] = m21;
        self.m[0][2] = m02;
        self.m[1][2] = m12;
        self.m[2][2] = m22;
    }

    /// C++ inline `RotationMatrix::RotateCoords` (`matrix.h:26`).
    #[allow(clippy::too_many_arguments)]
    pub fn rotate_coords(
        &self,
        input_x_coord: f32,
        input_y_coord: f32,
        input_z_coord: f32,
        output_x_coord: &mut f32,
        output_y_coord: &mut f32,
        output_z_coord: &mut f32,
    ) {
        *output_x_coord = self.m[0][0] * input_x_coord
            + self.m[0][1] * input_y_coord
            + self.m[0][2] * input_z_coord;
        *output_y_coord = self.m[1][0] * input_x_coord
            + self.m[1][1] * input_y_coord
            + self.m[1][2] * input_z_coord;
        *output_z_coord = self.m[2][0] * input_x_coord
            + self.m[2][1] * input_y_coord
            + self.m[2][2] * input_z_coord;
    }

    /// C++ inline `RotationMatrix::RotateCoords2D` (`matrix.h:32`).
    pub fn rotate_coords_2d(
        &self,
        input_x_coord: f32,
        input_y_coord: f32,
        output_x_coord: &mut f32,
        output_y_coord: &mut f32,
    ) {
        *output_x_coord = self.m[0][0] * input_x_coord + self.m[0][1] * input_y_coord;
        *output_y_coord = self.m[1][0] * input_x_coord + self.m[1][1] * input_y_coord;
    }
}

impl Add for RotationMatrix {
    type Output = Self;

    /// C++ `RotationMatrix::operator +` (`matrix.cpp:46`).
    fn add(self, other: Self) -> Self {
        let mut temp_matrix = Self::new();

        temp_matrix.m[0][0] = self.m[0][0] + other.m[0][0];
        temp_matrix.m[0][1] = self.m[0][1] + other.m[0][1];
        temp_matrix.m[0][2] = self.m[0][2] + other.m[0][2];
        temp_matrix.m[1][0] = self.m[1][0] + other.m[1][0];
        temp_matrix.m[1][1] = self.m[1][1] + other.m[1][1];
        temp_matrix.m[1][2] = self.m[1][2] + other.m[1][2];
        temp_matrix.m[2][0] = self.m[2][0] + other.m[2][0];
        temp_matrix.m[2][1] = self.m[2][1] + other.m[2][1];
        temp_matrix.m[2][2] = self.m[2][2] + other.m[2][2];

        temp_matrix
    }
}

impl Sub for RotationMatrix {
    type Output = Self;

    /// C++ `RotationMatrix::operator -` (`matrix.cpp:63`).
    fn sub(self, other: Self) -> Self {
        let mut temp_matrix = Self::new();

        temp_matrix.m[0][0] = self.m[0][0] - other.m[0][0];
        temp_matrix.m[0][1] = self.m[0][1] - other.m[0][1];
        temp_matrix.m[0][2] = self.m[0][2] - other.m[0][2];
        temp_matrix.m[1][0] = self.m[1][0] - other.m[1][0];
        temp_matrix.m[1][1] = self.m[1][1] - other.m[1][1];
        temp_matrix.m[1][2] = self.m[1][2] - other.m[1][2];
        temp_matrix.m[2][0] = self.m[2][0] - other.m[2][0];
        temp_matrix.m[2][1] = self.m[2][1] - other.m[2][1];
        temp_matrix.m[2][2] = self.m[2][2] - other.m[2][2];

        temp_matrix
    }
}

impl Mul for RotationMatrix {
    type Output = Self;

    /// C++ `RotationMatrix::operator *` (`matrix.cpp:80`).
    fn mul(self, other: Self) -> Self {
        let mut temp_matrix = Self::new();

        temp_matrix.m[0][0] = self.m[0][0] * other.m[0][0]
            + self.m[0][1] * other.m[1][0]
            + self.m[0][2] * other.m[2][0];
        temp_matrix.m[0][1] = self.m[0][0] * other.m[0][1]
            + self.m[0][1] * other.m[1][1]
            + self.m[0][2] * other.m[2][1];
        temp_matrix.m[0][2] = self.m[0][0] * other.m[0][2]
            + self.m[0][1] * other.m[1][2]
            + self.m[0][2] * other.m[2][2];
        temp_matrix.m[1][0] = self.m[1][0] * other.m[0][0]
            + self.m[1][1] * other.m[1][0]
            + self.m[1][2] * other.m[2][0];
        temp_matrix.m[1][1] = self.m[1][0] * other.m[0][1]
            + self.m[1][1] * other.m[1][1]
            + self.m[1][2] * other.m[2][1];
        temp_matrix.m[1][2] = self.m[1][0] * other.m[0][2]
            + self.m[1][1] * other.m[1][2]
            + self.m[1][2] * other.m[2][2];
        temp_matrix.m[2][0] = self.m[2][0] * other.m[0][0]
            + self.m[2][1] * other.m[1][0]
            + self.m[2][2] * other.m[2][0];
        temp_matrix.m[2][1] = self.m[2][0] * other.m[0][1]
            + self.m[2][1] * other.m[1][1]
            + self.m[2][2] * other.m[2][1];
        temp_matrix.m[2][2] = self.m[2][0] * other.m[0][2]
            + self.m[2][1] * other.m[1][2]
            + self.m[2][2] * other.m[2][2];

        temp_matrix
    }
}

impl AddAssign for RotationMatrix {
    /// C++ `RotationMatrix::operator +=` (`matrix.cpp:128`).
    fn add_assign(&mut self, other: Self) {
        self.m[0][0] += other.m[0][0];
        self.m[0][1] += other.m[0][1];
        self.m[0][2] += other.m[0][2];
        self.m[1][0] += other.m[1][0];
        self.m[1][1] += other.m[1][1];
        self.m[1][2] += other.m[1][2];
        self.m[2][0] += other.m[2][0];
        self.m[2][1] += other.m[2][1];
        self.m[2][2] += other.m[2][2];
    }
}

impl SubAssign for RotationMatrix {
    /// C++ `RotationMatrix::operator -=` (`matrix.cpp:148`).
    fn sub_assign(&mut self, other: Self) {
        self.m[0][0] -= other.m[0][0];
        self.m[0][1] -= other.m[0][1];
        self.m[0][2] -= other.m[0][2];
        self.m[1][0] -= other.m[1][0];
        self.m[1][1] -= other.m[1][1];
        self.m[1][2] -= other.m[1][2];
        self.m[2][0] -= other.m[2][0];
        self.m[2][1] -= other.m[2][1];
        self.m[2][2] -= other.m[2][2];
    }
}

impl MulAssign for RotationMatrix {
    /// C++ `RotationMatrix::operator *=` (`matrix.cpp:168`).
    fn mul_assign(&mut self, other: Self) {
        let temp_matrix = *self * other;
        *self = temp_matrix;
    }
}
