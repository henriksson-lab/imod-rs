//! Translation boundary for `IMOD/libmesh`.
//!
//! One module per source file, as source mirroring requires:
//! `mkmesh.c`, `objprep.c`, `remesh.c`, `skeletonize.c` and `skinobj.c`.
//! `IMOD/include/mkmesh.h` is the paired header and is merged into the
//! modules that define what it declares.

pub mod mkmesh;
pub mod objprep;
pub mod remesh;
pub mod skeletonize;
pub mod skinobj;
