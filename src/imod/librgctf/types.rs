//! Value types from `IMOD/librgctf/core_headers.h`.

use std::ops::{Add, Mul, MulAssign, Sub};

/// C++ `Peak` (`core_headers.h:1`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Peak {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub value: f32,
    pub physical_address_within_image: i64,
}

/// C++ `Kernel2D` (`core_headers.h:9`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Kernel2d {
    pub pixel_index: [i32; 4],
    pub pixel_weight: [f32; 4],
}

/// C++ `CurvePoint` (`core_headers.h:14`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CurvePoint {
    pub index_m: i32,
    pub index_n: i32,
    pub value_m: f32,
    pub value_n: f32,
}

/// `std::complex<float>`, as the library uses it.
///
/// The `Image` class aliases its complex array onto the same allocation as its
/// real array, so this type must be exactly two consecutive `float`s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// `std::abs(std::complex<float>)`, the modulus.
    pub fn abs(self) -> f32 {
        f32::hypot(self.re, self.im)
    }
}

/// C++ `const std::complex<float> I(0.0,1.0)` (`core_headers.h:31`).
pub const I: Complex = Complex { re: 0.0, im: 1.0 };

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl Mul<f32> for Complex {
    type Output = Self;
    fn mul(self, other: f32) -> Self {
        Self::new(self.re * other, self.im * other)
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl MulAssign<f32> for Complex {
    fn mul_assign(&mut self, other: f32) {
        *self = *self * other;
    }
}
