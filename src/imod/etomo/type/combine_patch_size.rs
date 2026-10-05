//! `IMOD/Etomo/src/etomo/type/CombinePatchSize.java`.
//!
//! The five Java singletons are the enum's variants; each one's mutable `xyz`
//! array lives in [`XYZ`], and the Java static `PATCH_SIZE_ARRAY` (null until
//! `setupcombine -info` has been read) in [`PATCH_SIZE_ARRAY`].
#![allow(dead_code)]

use std::sync::Mutex;

use regex::Regex;

use crate::imod::etomo::comscript::setup_combine::SetupCombine;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::util::utilities;

const VALUE_INDEX: usize = 0;
const XYZ_INDEX: usize = 1;
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

/// Java static `PATCH_SIZE_ARRAY` (`None` is Java's null).
static PATCH_SIZE_ARRAY: Mutex<Option<Vec<CombinePatchSize>>> = Mutex::new(None);

/// Each singleton's `private final int[] xyz`, indexed by variant.
static XYZ: Mutex<[[i32; 3]; 5]> = Mutex::new([[EMPTY_ELEMENT; 3]; 5]);

/// The monitor of Java's `synchronized (SMALL)` in `loadXYZ`.
static SMALL_MONITOR: Mutex<()> = Mutex::new(());

fn is_debug() -> bool {
    etomo_director::ARGUMENTS.lock().unwrap().is_debug()
}

impl CombinePatchSize {
    /// The singleton's `label` field.
    fn label(self) -> &'static str {
        match self {
            Self::Small => "Small",
            Self::Medium => "Medium",
            Self::Large => "Large",
            Self::ExtraLarge => "Extra large",
            Self::Custom => "Custom",
        }
    }

    /// The singleton's `value` field (`CUSTOM`'s value is its label).
    fn value(self) -> &'static str {
        match self {
            Self::Small => "S",
            Self::Medium => "M",
            Self::Large => "L",
            Self::ExtraLarge => "E",
            Self::Custom => "Custom",
        }
    }

    /// The singleton's `xyz` array.
    fn xyz(self) -> [i32; 3] {
        XYZ.lock().unwrap()[self as usize]
    }

    /// Java `getInstance(String)`.  Returns null if input is empty.  Returns
    /// the instance described by input, if input is a character string.
    /// Returns the instance that matches input, if input is a standard xyz
    /// patch size.  Otherwise returns the CUSTOM instance.
    pub fn get_instance(input: Option<&str>) -> Option<Self> {
        let input = input?.trim();
        if input.is_empty() {
            return None;
        }
        for instance in [Self::Small, Self::Medium, Self::Large, Self::ExtraLarge] {
            if input.eq_ignore_ascii_case(instance.value())
                || input.eq_ignore_ascii_case(instance.label())
            {
                return Some(instance);
            }
        }
        if input.eq_ignore_ascii_case(Self::Custom.value())
            || input.eq_ignore_ascii_case(Self::Custom.label())
            || !input.contains(',')
        {
            return Some(Self::Custom);
        }
        // Should be x,y,z. See if it matches one of the fixed instances.
        let xyz_array = utilities::java_lang_string_split(input, &Regex::new(",").unwrap());
        let xyz_array: Vec<&str> = xyz_array.iter().map(String::as_str).collect();
        Self::load_xyz();
        let patch_size_array = PATCH_SIZE_ARRAY.lock().unwrap().clone();
        if let Some(patch_size_array) = patch_size_array {
            for instance in patch_size_array {
                if instance.equals_strings(Some(&xyz_array)) {
                    return Some(instance);
                }
            }
        }
        Some(Self::Custom)
    }

    /// Java `getInstance(String[])`.
    pub fn get_instance_xyz_strings(xyz: Option<&[&str]>) -> Option<Self> {
        let xyz = xyz?;
        // See if xyz match one of the fixed instances.
        Self::load_xyz();
        let patch_size_array = PATCH_SIZE_ARRAY.lock().unwrap().clone();
        if let Some(patch_size_array) = patch_size_array {
            for instance in patch_size_array {
                if instance.equals_strings(Some(xyz)) {
                    return Some(instance);
                }
            }
        }
        Some(Self::Custom)
    }

    /// Java `getInstance(int[])`.
    pub fn get_instance_xyz_ints(xyz: Option<&[i32]>) -> Option<Self> {
        let xyz = xyz?;
        // See if xyz match one of the fixed instances.
        Self::load_xyz();
        let patch_size_array = PATCH_SIZE_ARRAY.lock().unwrap().clone();
        if let Some(patch_size_array) = patch_size_array {
            for instance in patch_size_array {
                if instance.equals_ints(Some(xyz)) {
                    return Some(instance);
                }
            }
        }
        Some(Self::Custom)
    }

    /// Java private `getFixedInstance(String)`.  Does not return the CUSTOM
    /// instance.  Returns the instance described by input.  Otherwise returns
    /// null.
    fn get_fixed_instance(input: Option<&str>) -> Option<Self> {
        let input = input?;
        [Self::Small, Self::Medium, Self::Large, Self::ExtraLarge]
            .into_iter()
            .find(|instance| {
                input.eq_ignore_ascii_case(instance.value())
                    || input.eq_ignore_ascii_case(instance.label())
            })
    }

    /// Java `equals(String[])`.  Returns true if xyzArray equals xyz.  Empty
    /// elements are equal.
    pub fn equals_strings(self, xyz_array: Option<&[&str]>) -> bool {
        Self::load_xyz();
        let xyz = self.xyz();
        for i in 0..xyz.len() {
            match xyz_array.and_then(|xyz_array| xyz_array.get(i)).copied() {
                None => {
                    if xyz[i] != EMPTY_ELEMENT {
                        return false;
                    }
                }
                Some(element) => {
                    // `xyzArray[i].matches("\\s*")`
                    let blank = element.chars().all(char::is_whitespace);
                    if xyz[i] != EMPTY_ELEMENT || !blank {
                        // `Integer.valueOf(xyzArray[i])`
                        match element.parse::<i32>() {
                            Ok(value) => {
                                if xyz[i] != value {
                                    return false;
                                }
                            }
                            Err(_) => {
                                eprintln!(
                                    "java.lang.NumberFormatException: For input string: \"{element}\""
                                );
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Java `equals(int[])`.
    pub fn equals_ints(self, xyz_array: Option<&[i32]>) -> bool {
        Self::load_xyz();
        let xyz = self.xyz();
        for i in 0..xyz.len() {
            // Empty elements are equal
            match xyz_array.and_then(|xyz_array| xyz_array.get(i)).copied() {
                None => {
                    if xyz[i] == EMPTY_ELEMENT {
                        return false;
                    }
                }
                Some(value) => {
                    if xyz[i] != value {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Java `getXYZLen`.
    pub fn get_xyz_len(self) -> usize {
        self.xyz().len()
    }

    /// Java `getXYZ(int)`.
    pub fn get_xyz(self, index: usize) -> i32 {
        Self::load_xyz();
        self.xyz()[index]
    }

    /// Java private static `loadXYZ`.  Creates and loads PATCH_SIZE_ARRAY, and
    /// sets xyz for each fixed instance.  For missing xyz elements, preserves
    /// -1.
    fn load_xyz() {
        // Load PATCH_SIZE_ARRAY once.
        if PATCH_SIZE_ARRAY.lock().unwrap().is_some() {
            return;
        }
        if is_debug() {
            eprintln!("waiting for sync");
        }
        let _synchronized = SMALL_MONITOR.lock().unwrap();
        if PATCH_SIZE_ARRAY.lock().unwrap().is_some() {
            if is_debug() {
                eprintln!("PATCH_SIZE_ARRAY was created");
            }
            return;
        }
        // Run setupcombine -info
        let Some(output) = SetupCombine::get_info_on_patch_sizes() else {
            if is_debug() {
                eprintln!("output is null");
            }
            return;
        };
        // Get the instances and the patch sizes and place them in PATCH_SIZE_ARRAY
        let mut patch_size_array = Vec::new();
        if is_debug() {
            eprintln!("output.length:{}", output.len());
        }
        let colon = Regex::new(r"\s*:\s*").unwrap();
        let white_space = Regex::new(r"\s+").unwrap();
        for (i, line) in output.iter().enumerate() {
            // Example of output:
            // S: 64 64 32
            let array = utilities::java_lang_string_split(line, &colon);
            if array.len() <= VALUE_INDEX {
                if is_debug() {
                    eprintln!("unused output[{i}]:{line}");
                }
                continue;
            }
            let combine_patch_size = Self::get_fixed_instance(Some(&array[VALUE_INDEX]));
            if is_debug() {
                eprintln!(
                    "combinePatchSize:{}",
                    combine_patch_size.map_or("null".to_string(), |size| size.to_string())
                );
            }
            let Some(combine_patch_size) = combine_patch_size else {
                if is_debug() {
                    eprintln!("unused output[{i}]:{line}");
                }
                continue;
            };
            if array.len() <= XYZ_INDEX {
                if is_debug() {
                    eprintln!("unused output[{i}]:{line}");
                }
                continue;
            }
            // set the xyz member variable
            let xyz_array = utilities::java_lang_string_split(&array[XYZ_INDEX], &white_space);
            patch_size_array.push(combine_patch_size);
            let len = xyz_array.len().min(3);
            for (j, element) in xyz_array.iter().take(len).enumerate() {
                // Preserve -1 for missing xyz values
                match element.parse::<i32>() {
                    Ok(value) => XYZ.lock().unwrap()[combine_patch_size as usize][j] = value,
                    Err(_) => {
                        if is_debug() {
                            eprintln!("bad xyzArray[{j}]:{element}");
                        }
                        eprintln!(
                            "java.lang.NumberFormatException: For input string: \"{element}\""
                        );
                    }
                }
            }
        }
        *PATCH_SIZE_ARRAY.lock().unwrap() = Some(patch_size_array);
    }

    /// Java `getOption`.
    pub fn get_option(self) -> &'static str {
        self.value()
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
        self.label()
    }
}

/// Java `CombinePatchSize implements EnumeratedType`.
impl crate::imod::etomo::r#type::enumerated_type::EnumeratedType for CombinePatchSize {
    fn is_default(&self) -> bool {
        CombinePatchSize::is_default(*self)
    }
    /// Java returns null; the trait's value is non-nullable, so an empty (null-valued)
    /// number stands in for it.
    fn get_value(&self) -> crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber {
        crate::imod::etomo::r#type::const_etomo_number::ConstEtomoNumber::new()
    }
    fn get_label(&self) -> Option<String> {
        Some(CombinePatchSize::get_label(*self).to_owned())
    }
}

impl std::fmt::Display for CombinePatchSize {
    /// Java `toString`.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.get_option())
    }
}
