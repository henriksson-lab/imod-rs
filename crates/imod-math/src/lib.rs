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
