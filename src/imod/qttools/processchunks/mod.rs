//! Rust translations of `IMOD/qttools/processchunks`.

pub const CHUNK_SYNC: i32 = -1;
pub const CHUNK_NOT_DONE: i32 = 0;
pub const CHUNK_ASSIGNED: i32 = 1;
pub const CHUNK_DONE: i32 = 2;
pub const CHUNK_TO_SKIP: i32 = 3;

pub mod comfilejobs;
pub mod machinehandler;
pub mod processchunks;
pub mod processhandler;
