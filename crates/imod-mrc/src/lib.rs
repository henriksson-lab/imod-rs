//! MRC image file format reading and writing.
//!
//! This crate implements the MRC 2014 file format as used in cryo-EM and electron
//! tomography. The MRC format stores 2-D images and 3-D volumes with a fixed
//! 1024-byte header, an optional extended header (SERI, FEI, etc.), and
//! contiguous pixel data in one of several data modes (byte, short, float,
//! complex, RGB, and others).
//!
//! Key types:
//! - [`MrcHeader`] -- the 1024-byte on-disk header with dimensions, pixel size,
//!   statistics, labels, and more.
//! - [`MrcReader`] -- opens an MRC file and reads slices as `f32` data.
//! - [`MrcWriter`] -- creates a new MRC file and writes slices sequentially.
//! - [`SectionMetadata`] -- per-section metadata parsed from extended headers.

mod header;
mod read;
mod write;
mod exthead;

pub use header::MrcHeader;
pub use read::MrcReader;
pub use write::MrcWriter;
pub use exthead::*;

#[cfg(test)]
mod tests;
