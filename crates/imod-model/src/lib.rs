//! IMOD model file format reading and writing (binary chunk-based format).
//!
//! The IMOD model format is a big-endian, chunk-based binary format identified
//! by the magic bytes `IMOD` + `V1.2`. A model contains objects, each holding
//! contours (ordered lists of 3-D points) and optionally triangle meshes, plus
//! auxiliary chunks for materials, clipping planes, views, and slicer angles.
//!
//! Key types:
//! - [`ImodModel`] -- the top-level model with objects, views, and metadata.
//! - [`ImodObject`] -- a named object with color, contours, and meshes.
//! - [`ImodContour`] -- an ordered list of [`Point3f`](imod_core::Point3f) points.
//! - [`ImodMesh`] -- a triangle mesh with vertices and index-based draw commands.
//!
//! Use [`read_model`] and [`write_model`] for file I/O.

mod types;
mod read;
mod write;

pub use types::*;
pub use read::read_model;
pub use write::write_model;

#[cfg(test)]
mod tests;
