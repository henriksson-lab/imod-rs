//! `IMOD/Etomo/src/etomo/logic/TrimvolReorientation.java`.
//!
//! The batchruntomo `reorient` directive value.  Java's typesafe-enum pattern is
//! mirrored as a `Copy` struct with associated constants.

use crate::imod::etomo::r#type::meta_data::MetaData;

/// Java `rcsid`.
pub const RCSID: &str = "$Id:$";

/// Java final `TrimvolReorientation`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrimvolReorientation {
    /// Java public final field `value`.
    pub value: i32,
}

impl TrimvolReorientation {
    /// Java `NONE = new TrimvolReorientation(0)`.
    pub const NONE: TrimvolReorientation = TrimvolReorientation::new(0);
    /// Java `FLIP = new TrimvolReorientation(1)`.
    pub const FLIP: TrimvolReorientation = TrimvolReorientation::new(1);
    /// Java `ROTATE = new TrimvolReorientation(2)`.
    pub const ROTATE: TrimvolReorientation = TrimvolReorientation::new(2);

    /// Java `DEFAULT = ROTATE`.
    pub const DEFAULT: TrimvolReorientation = TrimvolReorientation::ROTATE;

    /// Java private `TrimvolReorientation(int)`.
    const fn new(value: i32) -> TrimvolReorientation {
        TrimvolReorientation { value }
    }

    /// Java static `toDirectiveValue(MetaData)`.
    pub fn to_directive_value(meta_data: &MetaData) -> i32 {
        if meta_data.is_post_trimvol_swap_yz() {
            return Self::FLIP.value;
        }
        if meta_data.is_post_trimvol_rotate_x() {
            return Self::ROTATE.value;
        }
        Self::NONE.value
    }
}
