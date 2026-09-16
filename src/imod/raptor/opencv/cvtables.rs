//! Lookup tables from `IMOD/raptor/opencv/cvtables.cpp`.
//!
//! This translation unit has no executable functions.  Its three tables are
//! formed in const blocks so their values remain identical to the source
//! arrays without a second, manually maintained copy of the sequences.

/// `icv8x32fTab_cv`, indexed by a value in `-256..=511` plus 256.
pub const ICV8_X32_F_TAB_CV: [f32; 768] = {
    let mut table = [0.0; 768];
    let mut index = 0;
    while index < table.len() {
        table[index] = index as f32 - 256.0;
        index += 1;
    }
    table
};

/// `icv8x32fSqrTab`, indexed by a byte value plus 128.
pub const ICV8_X32_F_SQR_TAB: [f32; 384] = {
    let mut table = [0.0; 384];
    let mut index = 0;
    while index < table.len() {
        let value = index as i32 - 128;
        table[index] = (value * value) as f32;
        index += 1;
    }
    table
};

/// `icvSaturate8u_cv`, indexed by an integer in `-256..=512` plus 256.
pub const ICV_SATURATE_8_U_CV: [u8; 769] = {
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
    use super::{ICV_SATURATE_8_U_CV, ICV8_X32_F_SQR_TAB, ICV8_X32_F_TAB_CV};

    #[test]
    fn cvtables_match_the_original_indexing_contracts() {
        assert_eq!(
            (
                ICV8_X32_F_TAB_CV[0],
                ICV8_X32_F_TAB_CV[256],
                ICV8_X32_F_TAB_CV[767]
            ),
            (-256.0, 0.0, 511.0)
        );
        assert_eq!(
            (
                ICV8_X32_F_SQR_TAB[0],
                ICV8_X32_F_SQR_TAB[128],
                ICV8_X32_F_SQR_TAB[383]
            ),
            (16384.0, 0.0, 65025.0)
        );
        assert_eq!(
            (
                ICV_SATURATE_8_U_CV[0],
                ICV_SATURATE_8_U_CV[256],
                ICV_SATURATE_8_U_CV[511],
                ICV_SATURATE_8_U_CV[768]
            ),
            (0, 0, 255, 255)
        );
    }
}
