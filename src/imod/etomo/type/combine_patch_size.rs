//! `IMOD/Etomo/src/etomo/type/CombinePatchSize.java`.
//!
//! Loading `setupcombine -info` is an application/process boundary.  The
//! parsed fixed-size table is supplied by that boundary with
//! `set_patch_size_array`; all identity, option, and XYZ behavior from the
//! Java typesafe enum is retained here.
#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};

pub const EMPTY_ELEMENT: i32 = -1;
pub const X_INDEX: usize = 0;
pub const Y_INDEX: usize = 1;
pub const Z_INDEX: usize = 2;

/// Java `CombinePatchSize` singleton identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum CombinePatchSize {
    Small,
    Medium,
    Large,
    ExtraLarge,
    Custom,
}

/// The values Java `loadXYZ` mutates in its five singleton instances.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchSizeArray {
    pub small: [i32; 3],
    pub medium: [i32; 3],
    pub large: [i32; 3],
    pub extra_large: [i32; 3],
    pub loaded: bool,
}

impl Default for PatchSizeArray {
    fn default() -> Self {
        Self {
            small: [EMPTY_ELEMENT; 3],
            medium: [EMPTY_ELEMENT; 3],
            large: [EMPTY_ELEMENT; 3],
            extra_large: [EMPTY_ELEMENT; 3],
            loaded: false,
        }
    }
}

static PATCH_SIZE_ARRAY: OnceLock<Mutex<PatchSizeArray>> = OnceLock::new();

impl CombinePatchSize {
    /// Java `getInstance(String)` after `SetupCombine.getInfoOnPatchSizes` has
    /// populated the process boundary table.
    pub fn get_instance(input: Option<&str>) -> Option<Self> {
        let input = input?.trim();
        if input.is_empty() {
            return None;
        }
        if input.eq_ignore_ascii_case("S") || input.eq_ignore_ascii_case("Small") {
            return Some(Self::Small);
        }
        if input.eq_ignore_ascii_case("M") || input.eq_ignore_ascii_case("Medium") {
            return Some(Self::Medium);
        }
        if input.eq_ignore_ascii_case("L") || input.eq_ignore_ascii_case("Large") {
            return Some(Self::Large);
        }
        if input.eq_ignore_ascii_case("E") || input.eq_ignore_ascii_case("Extra large") {
            return Some(Self::ExtraLarge);
        }
        if input.eq_ignore_ascii_case("Custom") || !input.contains(',') {
            return Some(Self::Custom);
        }
        let xyz = input.split(',').collect::<Vec<_>>();
        Self::get_instance_xyz_strings(Some(&xyz))
    }

    /// Java `getInstance(String[])`.
    pub fn get_instance_xyz_strings(xyz: Option<&[&str]>) -> Option<Self> {
        let xyz = xyz?;
        for patch_size in [Self::Small, Self::Medium, Self::Large, Self::ExtraLarge] {
            if patch_size.equals_strings(Some(xyz)) {
                return Some(patch_size);
            }
        }
        Some(Self::Custom)
    }

    /// Java `getInstance(int[])`.
    pub fn get_instance_xyz_ints(xyz: Option<&[i32]>) -> Option<Self> {
        let xyz = xyz?;
        for patch_size in [Self::Small, Self::Medium, Self::Large, Self::ExtraLarge] {
            if patch_size.equals_ints(Some(xyz)) {
                return Some(patch_size);
            }
        }
        Some(Self::Custom)
    }

    /// Java `equals(String[])`.
    pub fn equals_strings(self, xyz: Option<&[&str]>) -> bool {
        let array = PATCH_SIZE_ARRAY
            .get_or_init(|| Mutex::new(PatchSizeArray::default()))
            .lock()
            .unwrap();
        let values = match self {
            Self::Small => array.small,
            Self::Medium => array.medium,
            Self::Large => array.large,
            Self::ExtraLarge => array.extra_large,
            Self::Custom => [EMPTY_ELEMENT; 3],
        };
        for index in 0..values.len() {
            match xyz.and_then(|xyz| xyz.get(index)).copied() {
                None => {
                    if values[index] != EMPTY_ELEMENT {
                        return false;
                    }
                }
                Some(value) if value.trim().is_empty() => {
                    if values[index] != EMPTY_ELEMENT {
                        return false;
                    }
                }
                Some(value) => match value.trim().parse::<i32>() {
                    Ok(value) if values[index] != value => return false,
                    Ok(_) => {}
                    Err(_) => return false,
                },
            }
        }
        true
    }

    /// Java `equals(int[])`.
    pub fn equals_ints(self, xyz: Option<&[i32]>) -> bool {
        let array = PATCH_SIZE_ARRAY
            .get_or_init(|| Mutex::new(PatchSizeArray::default()))
            .lock()
            .unwrap();
        let values = match self {
            Self::Small => array.small,
            Self::Medium => array.medium,
            Self::Large => array.large,
            Self::ExtraLarge => array.extra_large,
            Self::Custom => [EMPTY_ELEMENT; 3],
        };
        for index in 0..values.len() {
            match xyz.and_then(|xyz| xyz.get(index)).copied() {
                None => {
                    if values[index] == EMPTY_ELEMENT {
                        return false;
                    }
                }
                Some(value) if values[index] != value => return false,
                Some(_) => {}
            }
        }
        true
    }

    /// Java `getXYZLen`.
    pub fn get_xyz_len(self) -> usize {
        3
    }

    /// Java `getXYZ(int)`.
    pub fn get_xyz(self, index: usize) -> i32 {
        let array = PATCH_SIZE_ARRAY
            .get_or_init(|| Mutex::new(PatchSizeArray::default()))
            .lock()
            .unwrap();
        match self {
            Self::Small => array.small[index],
            Self::Medium => array.medium[index],
            Self::Large => array.large[index],
            Self::ExtraLarge => array.extra_large[index],
            Self::Custom => EMPTY_ELEMENT,
        }
    }

    /// Java private `loadXYZ`, represented by its process-result handoff.
    pub fn set_patch_size_array(array: PatchSizeArray) {
        *PATCH_SIZE_ARRAY
            .get_or_init(|| Mutex::new(PatchSizeArray::default()))
            .lock()
            .unwrap() = array;
    }

    /// Java `getOption`.
    pub fn get_option(self) -> &'static str {
        match self {
            Self::Small => "S",
            Self::Medium => "M",
            Self::Large => "L",
            Self::ExtraLarge => "E",
            Self::Custom => "Custom",
        }
    }

    /// Java `isDefault`.
    pub fn is_default(self) -> bool {
        self == Self::Medium
    }

    /// Java `getValue` always returns null.
    pub fn get_value(self) -> Option<()> {
        let _ = self;
        None
    }

    /// Java `getLabel`.
    pub fn get_label(self) -> &'static str {
        match self {
            Self::Small => "Small",
            Self::Medium => "Medium",
            Self::Large => "Large",
            Self::ExtraLarge => "Extra large",
            Self::Custom => "Custom",
        }
    }
}

impl std::fmt::Display for CombinePatchSize {
    /// Java `toString`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.get_option())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_sizes_are_selected_from_loaded_xyz() {
        CombinePatchSize::set_patch_size_array(PatchSizeArray {
            medium: [64, 64, 32],
            loaded: true,
            ..Default::default()
        });
        assert_eq!(
            CombinePatchSize::get_instance_xyz_ints(Some(&[64, 64, 32])),
            Some(CombinePatchSize::Medium)
        );
        assert_eq!(
            CombinePatchSize::get_instance(Some("88, 88, 44")),
            Some(CombinePatchSize::Custom)
        );
    }
}
