//! Mesh generation: contour skinning, marching cubes isosurface extraction, simplification.
//!
//! This crate provides two mesh-generation algorithms:
//! - **Marching cubes** ([`marching_cubes`]) -- extracts an [`IsosurfaceMesh`]
//!   from a 3-D scalar volume at a given iso-value threshold.
//! - **Contour skinning** ([`skin_contours`]) -- builds a [`ContourMesh`]
//!   triangle strip connecting two [`Contour2d`] outlines on adjacent Z sections.

mod marching_cubes;
mod skin;

pub use marching_cubes::*;
pub use skin::*;

#[cfg(test)]
mod tests;
