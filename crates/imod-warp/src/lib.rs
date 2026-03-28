//! Warping, Delaunay triangulation, and warp file I/O for non-linear transforms.
//!
//! Provides types for representing and reading/writing IMOD warp files, which
//! store per-section non-linear transforms as grids of control points with
//! local linear transforms at each point. Includes a Delaunay triangulation
//! implementation ([`Triangulation`]) for interpolating warp fields.
//!
//! Key types:
//! - [`WarpFile`] -- a complete warp file with header and per-section transforms.
//! - [`WarpTransform`] -- control points and local transforms for one section.
//! - [`Triangulation`] -- Delaunay triangulation with point-in-triangle lookup.

mod delaunay;
mod warpfile;

pub use delaunay::*;
pub use warpfile::*;

#[cfg(test)]
mod tests;
