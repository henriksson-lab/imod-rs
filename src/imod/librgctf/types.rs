//! Value types from `IMOD/librgctf/core_headers.h`.

use rustfft::num_complex::Complex32;

/// C++ header constant `I`; named for Rust's conventional constant style.
pub const IMAGINARY_UNIT: Complex32 = Complex32::new(0.0, 1.0);

/// C++ `Peak`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Peak {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub value: f32,
    pub physical_address_within_image: i64,
}

/// C++ `Kernel2D`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Kernel2d {
    pub pixel_index: [i32; 4],
    pub pixel_weight: [f32; 4],
}

/// C++ `CurvePoint`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CurvePoint {
    pub index_m: i32,
    pub index_n: i32,
    pub value_m: f32,
    pub value_n: f32,
}

#[cfg(test)]
mod tests {
    use super::{CurvePoint, IMAGINARY_UNIT, Kernel2d, Peak};

    #[test]
    fn core_header_value_types_have_owned_zero_defaults() {
        assert_eq!(Peak::default().physical_address_within_image, 0);
        assert_eq!(Kernel2d::default().pixel_index, [0; 4]);
        assert_eq!(CurvePoint::default().value_m, 0.0);
        assert_eq!(
            IMAGINARY_UNIT,
            rustfft::num_complex::Complex32::new(0.0, 1.0)
        );
    }
}
