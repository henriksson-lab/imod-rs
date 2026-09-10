//! Translation scaffolding for `IMOD/libwarp/nn.h`.
#![allow(dead_code)]

/// C `NN_RULE` (`nn.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NnRule {
    Sibson = 0,
    NonSibsonian = 1,
}

/// C `point` (`nn.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// C global `nn_verbose` (`nncommon.c`).
pub static mut NN_VERBOSE: i32 = 0;
/// C global `nn_test_vertice` (`nncommon.c`).
pub static mut NN_TEST_VERTICE: i32 = -1;
/// C global `nn_rule` (`nncommon.c`).
pub static mut NN_RULE: NnRule = NnRule::Sibson;
/// C global `nn_version` (`version.h`).
pub static mut NN_VERSION: *const core::ffi::c_char = c"1.82".as_ptr();
