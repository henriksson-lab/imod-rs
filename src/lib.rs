#![allow(dead_code)]
// A float literal in this crate is copied from the vendored source verbatim,
// so clippy's "approximate value of `PI`" is always the wrong suggestion here:
// `filtxcorr.c:2080` is `float pi = 3.141593;`, whose bit pattern `0x40490fdc`
// is one ulp above `core::f32::consts::PI`.  Substituting the exact constant
// perturbed every FFT phase factor -- see the trap list in `CLAUDE.md`.
#![allow(clippy::approx_constant)]

pub mod imod;
