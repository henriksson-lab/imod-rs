//! Value types from `IMOD/librgctf/core_headers.h`.

/// C++ `Peak`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Peak {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub value: f32,
    pub physical_address_within_image: i64,
}

/// C++ `Kernel2D`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Kernel2d {
    pub pixel_index: [i32; 4],
    pub pixel_weight: [f32; 4],
}

/// C++ `CurvePoint`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CurvePoint {
    pub index_m: i32,
    pub index_n: i32,
    pub value_m: f32,
    pub value_n: f32,
}
