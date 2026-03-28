//! Multi-format image I/O with backend trait for MRC, TIFF, and other formats.
//!
//! Defines the [`ImageFile`] trait that all image backends implement, along with
//! [`ImageInfo`] for dimensions and metadata, and [`ImageFormat`] for detecting
//! the file type from its extension. Concrete backends:
//! - [`MrcImageFile`] -- reads MRC/ST/ALI/REC files via `imod-mrc`.
//! - [`TiffImageFile`] -- reads grayscale TIFF files (8/16/32-bit, multi-page).

mod format;
mod mrc_backend;
mod tiff_backend;

pub use format::*;
pub use mrc_backend::*;
pub use tiff_backend::*;

#[cfg(test)]
mod tests;
