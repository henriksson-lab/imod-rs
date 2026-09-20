//! Translation of `IMOD/librgctf/angles_and_shifts.{h,cpp}`.

use super::functions::deg_2_rad;
use super::matrix::RotationMatrix;

/// C++ `AnglesAndShifts` (`angles_and_shifts.h:3`): particle Euler angles and
/// shifts, with their rotation matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// C++ `AnglesAndShifts::AnglesAndShifts()` (`angles_and_shifts.cpp:3`).
    pub fn new() -> Self {
        let mut this = Self {
            euler_matrix: RotationMatrix::new(),
            euler_phi: 0.0,
            euler_theta: 0.0,
            euler_psi: 0.0,
            shift_x: 0.0,
            shift_y: 0.0,
        };
        this.euler_phi = 0.0;
        this.euler_theta = 0.0;
        this.euler_psi = 0.0;
        this.shift_x = 0.0;
        this.shift_y = 0.0;
        this.euler_matrix.set_to_identity();
        this
    }

    /// C++ `AnglesAndShifts::AnglesAndShifts(float, float, float, float, float)`
    /// (`angles_and_shifts.cpp:13`).
    pub fn with_angles(
        wanted_euler_phi: f32,
        wanted_euler_theta: f32,
        wanted_euler_psi: f32,
        wanted_shift_x: f32,
        wanted_shift_y: f32,
    ) -> Self {
        let mut this = Self {
            euler_matrix: RotationMatrix::new(),
            euler_phi: 0.0,
            euler_theta: 0.0,
            euler_psi: 0.0,
            shift_x: 0.0,
            shift_y: 0.0,
        };
        this.init(
            wanted_euler_phi,
            wanted_euler_theta,
            wanted_euler_psi,
            wanted_shift_x,
            wanted_shift_y,
        );
        this
    }

    /// C++ `AnglesAndShifts::Init` (`angles_and_shifts.cpp:18`).
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

    /// C++ `AnglesAndShifts::GenerateEulerMatrices` (`angles_and_shifts.cpp:25`).
    pub fn generate_euler_matrices(
        &mut self,
        wanted_euler_phi_in_degrees: f32,
        wanted_euler_theta_in_degrees: f32,
        wanted_euler_psi_in_degrees: f32,
    ) {
        self.euler_phi = wanted_euler_phi_in_degrees;
        self.euler_theta = wanted_euler_theta_in_degrees;
        self.euler_psi = wanted_euler_psi_in_degrees;
        let cos_phi = deg_2_rad(self.euler_phi).cos();
        let sin_phi = deg_2_rad(self.euler_phi).sin();
        let cos_theta = deg_2_rad(self.euler_theta).cos();
        let sin_theta = deg_2_rad(self.euler_theta).sin();
        let cos_psi = deg_2_rad(self.euler_psi).cos();
        let sin_psi = deg_2_rad(self.euler_psi).sin();
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

    /// C++ `AnglesAndShifts::GenerateRotationMatrix2D` (`angles_and_shifts.cpp:54`).
    pub fn generate_rotation_matrix_2d(&mut self, wanted_rotation_angle_in_degrees: f32) {
        self.euler_psi = wanted_rotation_angle_in_degrees;
        let cos_psi = deg_2_rad(self.euler_psi).cos();
        let sin_psi = deg_2_rad(self.euler_psi).sin();
        self.euler_matrix.m[0][0] = cos_psi;
        self.euler_matrix.m[1][0] = sin_psi;
        self.euler_matrix.m[2][0] = 0.0;
        self.euler_matrix.m[0][1] = -sin_psi;
        self.euler_matrix.m[1][1] = cos_psi;
        self.euler_matrix.m[2][1] = 0.0;
        self.euler_matrix.m[0][2] = 0.0;
        self.euler_matrix.m[1][2] = 0.0;
        self.euler_matrix.m[2][2] = 1.0;
    }

    /// C++ inline `AnglesAndShifts::ReturnPhiAngle` (`angles_and_shifts.h:17`).
    pub fn return_phi_angle(&self) -> f32 {
        self.euler_phi
    }

    /// C++ inline `AnglesAndShifts::ReturnThetaAngle` (`angles_and_shifts.h:18`).
    pub fn return_theta_angle(&self) -> f32 {
        self.euler_theta
    }

    /// C++ inline `AnglesAndShifts::ReturnPsiAngle` (`angles_and_shifts.h:19`).
    pub fn return_psi_angle(&self) -> f32 {
        self.euler_psi
    }

    /// C++ inline `AnglesAndShifts::ReturnShiftX` (`angles_and_shifts.h:20`).
    pub fn return_shift_x(&self) -> f32 {
        self.shift_x
    }

    /// C++ inline `AnglesAndShifts::ReturnShiftY` (`angles_and_shifts.h:21`).
    pub fn return_shift_y(&self) -> f32 {
        self.shift_y
    }
}
