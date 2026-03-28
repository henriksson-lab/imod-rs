//! 2D/3D geometric transforms and file I/O for .xf, .xg, and .tlt files.
//!
//! This crate provides [`LinearTransform`] for 2-D affine transforms (rotation,
//! scaling, translation) with reading and writing in the IMOD `.xf` / `.xg`
//! file formats, plus utilities for tilt-angle files (`.tlt`, `.rawtlt`).

mod xf;
mod tilt;

pub use xf::*;
pub use tilt::*;

#[cfg(test)]
mod tests;
