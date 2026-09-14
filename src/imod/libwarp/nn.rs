//! Translation scaffolding for `IMOD/libwarp/nn.h`.
#![allow(dead_code)]

use std::cell::Cell;

/// Nearest-neighbour interpolation rule, translated from C `NN_RULE`
/// (`nn.h:26`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnRule {
    Sibson = 0,
    NonSibsonian = 1,
}

/// A three-dimensional interpolation point, translated from C `point`
/// (`nn.h:31`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

thread_local! {
    /// C global `nn_verbose` (`nn.h:314`, defined in `nncommon.c`).
    ///
    /// A `static mut` cannot be read without `unsafe`, so the three `nn`
    /// globals are thread-local cells: read with `NN_VERBOSE.get()`, write
    /// with `NN_VERBOSE.set(v)`.
    pub static NN_VERBOSE: Cell<i32> = const { Cell::new(0) };
    /// C global `nn_rule` (`nn.h:321`, defined in `nncommon.c`).
    pub static NN_RULE: Cell<NnRule> = const { Cell::new(NnRule::Sibson) };
    /// C global `nn_test_vertice` (`nn.h:330`, defined in `nncommon.c`).
    pub static NN_TEST_VERTICE: Cell<i32> = const { Cell::new(-1) };
}

/// C global `nn_version` (`version.h`), declared `extern char* nn_version` by
/// `nn.h:325`. Nothing in the tree writes it, so it is a plain string here.
pub static NN_VERSION: &str = "1.82";
