//! Translation of `IMOD/librgctf/angles_and_shifts.{h,cpp}`.

use super::matrix::RotationMatrix;

/// Particle Euler angles, shifts, and their corresponding rotation matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct AnglesAndShifts {
    pub euler_matrix: RotationMatrix,
    euler_phi: f32,
    euler_theta: f32,
    euler_psi: f32,
    shift_x: f32,
    shift_y: f32,
}

impl Default for AnglesAndShifts {
    fn default() -> Self {
        Self::new()
    }
}

impl AnglesAndShifts {
    /// C++ `AnglesAndShifts::AnglesAndShifts()`.
    pub fn new() -> Self {
        let mut angles = Self {
            euler_matrix: RotationMatrix::new(),
            euler_phi: 0.0,
            euler_theta: 0.0,
            euler_psi: 0.0,
            shift_x: 0.0,
            shift_y: 0.0,
        };
        angles.euler_matrix.set_to_identity();
        angles
    }

    /// C++ `AnglesAndShifts::Init`.
    pub fn init(
        &mut self,
        wanted_euler_phi_in_degrees: f32,
        wanted_euler_theta_in_degrees: f32,
        wanted_euler_psi_in_degrees: f32,
        wanted_shift_x: f32,
        wanted_shift_y: f32,
    ) {
        self.shift_x = wanted_shift_x;
        self.shift_y = wanted_shift_y;
        self.generate_euler_matrices(
            wanted_euler_phi_in_degrees,
            wanted_euler_theta_in_degrees,
            wanted_euler_psi_in_degrees,
        );
    }

    /// C++ `AnglesAndShifts::GenerateEulerMatrices`.
    pub fn generate_euler_matrices(&mut self, phi: f32, theta: f32, psi: f32) {
        self.euler_phi = phi;
        self.euler_theta = theta;
        self.euler_psi = psi;
        let radians = std::f32::consts::PI / 180.0;
        let (sin_phi, cos_phi) = (phi * radians).sin_cos();
        let (sin_theta, cos_theta) = (theta * radians).sin_cos();
        let (sin_psi, cos_psi) = (psi * radians).sin_cos();
        self.euler_matrix.m[0][0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi;
        self.euler_matrix.m[1][0] = sin_phi * cos_theta * cos_psi + cos_phi * sin_psi;
        self.euler_matrix.m[2][0] = -sin_theta * cos_psi;
        self.euler_matrix.m[0][1] = -cos_phi * cos_theta * sin_psi - sin_phi * cos_psi;
        self.euler_matrix.m[1][1] = -sin_phi * cos_theta * sin_psi + cos_phi * cos_psi;
        self.euler_matrix.m[2][1] = sin_theta * sin_psi;
        self.euler_matrix.m[0][2] = sin_theta * cos_phi;
        self.euler_matrix.m[1][2] = sin_theta * sin_phi;
        self.euler_matrix.m[2][2] = cos_theta;
    }

    /// C++ inline `AnglesAndShifts::ReturnShiftX`.
    pub fn shift_x(&self) -> f32 {
        self.shift_x
    }

    /// C++ inline `AnglesAndShifts::ReturnShiftY`.
    pub fn shift_y(&self) -> f32 {
        self.shift_y
    }
}
