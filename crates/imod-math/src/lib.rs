//! Numerical mathematics library: statistics, regression, optimization, linear algebra.
//!
//! Provides routines translated from IMOD's C math libraries:
//! - **Statistics** -- mean, standard deviation, robust statistics (median, MAD),
//!   percentile computation, and outlier removal.
//! - **Regression** -- multiple linear regression, robust regression with Tukey
//!   bisquare weighting, and polynomial fitting.
//! - **Optimization** -- Nelder-Mead simplex minimizer ([`amoeba`]).
//! - **Linear algebra** -- Gauss-Jordan elimination ([`gaussj`]).
//! - **Geometric fitting** -- circle, sphere, and ellipse fitting.
//! - **Parsing** -- comma/dash-separated integer list parsing ([`parse_list`]).

mod stats;
pub mod parselist;
pub mod amoeba;
pub mod gaussj;
pub mod circlefit;
pub mod regression;

pub use stats::*;
pub use parselist::parse_list;
pub use amoeba::{amoeba, amoeba_init, dual_amoeba, AmoebaResult, DualAmoebaResult};
pub use gaussj::{gaussj, gaussj_det, GaussjError, GaussjResult};
pub use circlefit::{
    circle_through_3pts, fit_sphere, fit_sphere_wgt, fit_centered_ellipse,
    CircleResult, SphereFitResult, EllipseFitResult,
};
pub use regression::*;

#[cfg(test)]
mod tests;
