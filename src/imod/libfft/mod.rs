//! Translation of IMOD's `libfft` source family.
//!
//! The public routines retain the original in-place float layouts.  Names are
//! systematically mapped from C camel case to Rust snake case.
#![allow(clippy::missing_safety_doc, dead_code)]

pub mod cmplft;
pub mod diprp;
pub mod hermft;
pub mod mdftkd;
pub mod odfft;
pub mod realft;
pub mod rustfft_backend;
pub mod srfp;
pub mod thrdfft;
pub mod todfft;

pub use cmplft::cmplft;
pub use diprp::diprp;
pub use hermft::hermft;
pub use mdftkd::{mdftkd, r2cftk, r3cftk, r4cftk, r5cftk, r8cftk, rpcftk};
pub use odfft::{cleanup_fft_plans, nice_fft_limit, odfft, odfft_c, using_fftw};
pub use realft::realft;
pub use srfp::srfp;
pub use thrdfft::thrdfft;
pub use todfft::{fft_add_time, fft_start_timer, parallel_todfft, todfft, todfft_c};
