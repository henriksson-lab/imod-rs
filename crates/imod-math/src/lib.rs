mod stats;
pub mod parselist;
pub mod amoeba;

pub use stats::*;
pub use parselist::parse_list;
pub use amoeba::{amoeba, amoeba_init, dual_amoeba, AmoebaResult, DualAmoebaResult};

#[cfg(test)]
mod tests;
