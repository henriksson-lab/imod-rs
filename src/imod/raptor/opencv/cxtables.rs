//! Lookup tables from `IMOD/raptor/opencv/cxtables.cpp`.
//!
//! The original translation unit contains data only.  The const expressions
//! preserve every table entry while retaining the relationship between an
//! index and the value it represents.

/// `icvDepthToType`: maps an IPL depth encoding to its OpenCV depth code.
pub const ICV_DEPTH_TO_TYPE: [i8; 18] = [
    -1, -1, 0, 1, 2, 3, -1, -1, 5, 4, -1, -1, -1, -1, -1, -1, 6, -1,
];

/// `icv8x32fTab`, indexed by an unsigned byte plus 128.
pub const ICV8_X32_F_TAB: [f32; 384] = {
    let mut table = [0.0; 384];
    let mut index = 0;
    while index < table.len() {
        table[index] = index as f32 - 128.0;
        index += 1;
    }
    table
};

/// `icv8x16uSqrTab`, indexed by a signed byte value plus 255.
pub const ICV8_X16_U_SQR_TAB: [u16; 511] = {
    let mut table = [0; 511];
    let mut index = 0;
    while index < table.len() {
        let value = index as i32 - 255;
        table[index] = (value * value) as u16;
        index += 1;
    }
    table
};

/// `icvSaturate8u`, indexed by an integer in `-256..=512` plus 256.
pub const ICV_SATURATE_8_U: [u8; 769] = {
    let mut table = [0; 769];
    let mut index = 0;
    while index < table.len() {
        let value = index as i32 - 256;
        table[index] = if value < 0 {
            0
        } else if value > 255 {
            255
        } else {
            value as u8
        };
        index += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::{ICV_DEPTH_TO_TYPE, ICV_SATURATE_8_U, ICV8_X16_U_SQR_TAB, ICV8_X32_F_TAB};

    #[test]
    fn cxtables_match_the_original_indexing_contracts() {
        assert_eq!(
            ICV_DEPTH_TO_TYPE,
            [
                -1, -1, 0, 1, 2, 3, -1, -1, 5, 4, -1, -1, -1, -1, -1, -1, 6, -1
            ]
        );
        assert_eq!(
            (ICV8_X32_F_TAB[0], ICV8_X32_F_TAB[128], ICV8_X32_F_TAB[383]),
            (-128.0, 0.0, 255.0)
        );
        assert_eq!(
            (
                ICV8_X16_U_SQR_TAB[0],
                ICV8_X16_U_SQR_TAB[255],
                ICV8_X16_U_SQR_TAB[510]
            ),
            (65025, 0, 65025)
        );
        assert_eq!(
            (
                ICV_SATURATE_8_U[0],
                ICV_SATURATE_8_U[256],
                ICV_SATURATE_8_U[511],
                ICV_SATURATE_8_U[768]
            ),
            (0, 0, 255, 255)
        );
    }
}
