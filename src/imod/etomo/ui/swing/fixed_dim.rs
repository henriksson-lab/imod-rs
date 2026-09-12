//! `IMOD/Etomo/src/etomo/ui/swing/FixedDim.java`.
#![allow(dead_code, non_upper_case_globals)]

use super::panel::Dimension;

/// Java `FixedDim`; Java's mutable `Dimension` constants are represented as
/// constructors so a caller receives its own mutable value.
pub struct FixedDim;
impl FixedDim {
    pub const INLINE_SQUARE_SIZE: Dimension = Dimension {
        width: 22,
        height: 22,
    };
    pub const TINY_SQUARE_SIZE: Dimension = Dimension {
        width: 13,
        height: 13,
    };
    pub const x0_y0: Dimension = Dimension {
        width: 0,
        height: 0,
    };
    pub const x1_y0: Dimension = Dimension {
        width: 1,
        height: 0,
    };
    pub const x2_y0: Dimension = Dimension {
        width: 2,
        height: 0,
    };
    pub const x3_y0: Dimension = Dimension {
        width: 3,
        height: 0,
    };
    pub const x5_y0: Dimension = Dimension {
        width: 5,
        height: 0,
    };
    pub const x7_y0: Dimension = Dimension {
        width: 7,
        height: 0,
    };
    pub const x9_y0: Dimension = Dimension {
        width: 9,
        height: 0,
    };
    pub const x10_y0: Dimension = Dimension {
        width: 10,
        height: 0,
    };
    pub const x12_y0: Dimension = Dimension {
        width: 12,
        height: 0,
    };
    pub const x15_y0: Dimension = Dimension {
        width: 15,
        height: 0,
    };
    pub const x20_y0: Dimension = Dimension {
        width: 20,
        height: 0,
    };
    pub const x25_y0: Dimension = Dimension {
        width: 25,
        height: 0,
    };
    pub const x30_y0: Dimension = Dimension {
        width: 30,
        height: 0,
    };
    pub const x40_y0: Dimension = Dimension {
        width: 40,
        height: 0,
    };
    pub const x45_y0: Dimension = Dimension {
        width: 45,
        height: 0,
    };
    pub const x50_y0: Dimension = Dimension {
        width: 50,
        height: 0,
    };
    pub const x70_y0: Dimension = Dimension {
        width: 70,
        height: 0,
    };
    pub const x96_y0: Dimension = Dimension {
        width: 96,
        height: 0,
    };
    pub const x119_y0: Dimension = Dimension {
        width: 119,
        height: 0,
    };
    pub const x120_y0: Dimension = Dimension {
        width: 120,
        height: 0,
    };
    pub const x130_y0: Dimension = Dimension {
        width: 130,
        height: 0,
    };
    pub const x142_y0: Dimension = Dimension {
        width: 142,
        height: 0,
    };
    pub const x143_y0: Dimension = Dimension {
        width: 143,
        height: 0,
    };
    pub const x150_y0: Dimension = Dimension {
        width: 150,
        height: 0,
    };
    pub const x167_y0: Dimension = Dimension {
        width: 167,
        height: 0,
    };
    pub const x175_y0: Dimension = Dimension {
        width: 175,
        height: 0,
    };
    pub const x179_y0: Dimension = Dimension {
        width: 179,
        height: 0,
    };
    pub const x181_y0: Dimension = Dimension {
        width: 181,
        height: 0,
    };
    pub const x197_y0: Dimension = Dimension {
        width: 197,
        height: 0,
    };
    pub const x200_y0: Dimension = Dimension {
        width: 200,
        height: 0,
    };
    pub const x264_y0: Dimension = Dimension {
        width: 264,
        height: 0,
    };
    pub const x272_y0: Dimension = Dimension {
        width: 272,
        height: 0,
    };
    pub const x0_y1: Dimension = Dimension {
        width: 0,
        height: 1,
    };
    pub const x0_y2: Dimension = Dimension {
        width: 0,
        height: 2,
    };
    pub const x0_y3: Dimension = Dimension {
        width: 0,
        height: 3,
    };
    pub const x0_y5: Dimension = Dimension {
        width: 0,
        height: 5,
    };
    pub const x0_y6: Dimension = Dimension {
        width: 0,
        height: 6,
    };
    pub const x0_y7: Dimension = Dimension {
        width: 0,
        height: 7,
    };
    pub const x0_y9: Dimension = Dimension {
        width: 0,
        height: 9,
    };
    pub const x0_y10: Dimension = Dimension {
        width: 0,
        height: 10,
    };
    pub const x0_y13: Dimension = Dimension {
        width: 0,
        height: 13,
    };
    pub const x0_y15: Dimension = Dimension {
        width: 0,
        height: 15,
    };
    pub const x0_y20: Dimension = Dimension {
        width: 0,
        height: 20,
    };
    pub const x0_y23: Dimension = Dimension {
        width: 0,
        height: 23,
    };
    pub const x0_y30: Dimension = Dimension {
        width: 0,
        height: 30,
    };
    pub const x0_y40: Dimension = Dimension {
        width: 0,
        height: 40,
    };
    pub const x0_y200: Dimension = Dimension {
        width: 0,
        height: 200,
    };
    pub const folder_button: Dimension = Self::INLINE_SQUARE_SIZE;
    pub const file_chooser: Dimension = Dimension {
        width: 400,
        height: 400,
    };
    pub const frame_border: Dimension = Dimension {
        width: 10,
        height: 48,
    };
    pub const separator: Dimension = Dimension {
        width: 100,
        height: 1,
    };
    pub const process_panel: Dimension = Dimension {
        width: 80,
        height: 130,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_dimensions_are_retained() {
        assert_eq!(FixedDim::x143_y0.width, 143);
        assert_eq!(FixedDim::folder_button, FixedDim::INLINE_SQUARE_SIZE);
    }
}
