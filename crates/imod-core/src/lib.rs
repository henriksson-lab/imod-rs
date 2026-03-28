//! Core types and error handling for the IMOD Rust rewrite.
//!
//! This crate provides shared foundational types used across all IMOD library
//! crates, including the unified error type [`ImodError`], MRC data mode
//! definitions ([`MrcMode`]), extended header types, pixel unit enumerations,
//! and basic geometry primitives such as [`Point3f`].

mod error;
mod types;

pub use error::ImodError;
pub use types::*;
