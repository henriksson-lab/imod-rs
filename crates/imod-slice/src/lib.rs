//! 2D image slice type with pixel operations, filtering, and interpolation.
//!
//! The central type is [`Slice`], a 2-D image whose pixels are stored as `f32`.
//! This crate also provides pixel-wise arithmetic ([`add`], [`subtract`],
//! [`multiply`], [`scale`], [`clamp`], [`threshold`], [`invert`]),
//! convolution ([`convolve_3x3`]), and cubic/bilinear/nearest-neighbor
//! interpolation for affine image transforms ([`cubinterp`]).

mod cubinterp;
mod ops;
mod slice;

pub use cubinterp::*;
pub use ops::*;
pub use slice::*;

#[cfg(test)]
mod tests;
