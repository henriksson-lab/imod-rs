//! `IMOD/Etomo/src/etomo/comscript/ConstPatchcrawl3DParam.java`.
//!
//! A Java class with package-private state (not an interface), so a struct here;
//! `Patchcrawl3DParam` holds it as its `base` through `Deref`/`DerefMut`.
//!
//! `invertYLimits` is behind a `Mutex`: `Patchcrawl3DParam.updateComScriptCommand`
//! sets it while writing the script, and that method takes `&self`.

use std::sync::Mutex;

use super::fortran_input_string::FortranInputString;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java package-private `REFERENCE_FILE_KEY`.
pub const REFERENCE_FILE_KEY: &str = "ReferenceFile";
/// Java package-private `FILE_TO_ALIGN_KEY`.
pub const FILE_TO_ALIGN_KEY: &str = "FileToAlign";
/// Java package-private `OUTPUT_FILE_KEY`.
pub const OUTPUT_FILE_KEY: &str = "OutputFile";
/// Java package-private `B_SOURCE_TRANSFORM_KEY`.
pub const B_SOURCE_TRANSFORM_KEY: &str = "BSourceTransform";
/// Java package-private `B_SOURCE_OR_SIZE_XYZ_KEY`.
pub const B_SOURCE_OR_SIZE_XYZ_KEY: &str = "BSourceOrSizeXYZ";
/// Java package-private `REGION_MODEL_KEY`.
pub const REGION_MODEL_KEY: &str = "RegionModel";
/// Java `KERNEL_SIGMA_KEY`.
pub const KERNEL_SIGMA_KEY: &str = "KernelSigma";
/// Java `INITIAL_SHIFT_XYZ_KEY`.
pub const INITIAL_SHIFT_XYZ_KEY: &str = "InitialShiftXYZ";

/// Java package-private `X_INDEX`.
pub const X_INDEX: i32 = 0;
/// Java package-private `Y_INDEX`.
pub const Y_INDEX: i32 = 1;
/// Java package-private `Z_INDEX`.
pub const Z_INDEX: i32 = 2;

/// Java `ConstPatchcrawl3DParam`.
#[derive(Debug)]
pub struct ConstPatchcrawl3DParam {
    /// Java `patchSizeXYZ`.
    pub(crate) patch_size_xyz: FortranInputString,
    /// Java `numberOfPatchesXYZ`.
    pub(crate) number_of_patches_xyz: FortranInputString,
    /// Java `xMinAndMax`.
    pub(crate) x_min_and_max: FortranInputString,
    /// Java `yMinAndMax`.
    pub(crate) y_min_and_max: FortranInputString,
    /// Java `zMinAndMax`.
    pub(crate) z_min_and_max: FortranInputString,
    /// Java `bSourceBorderXLoHi`.
    pub(crate) b_source_border_x_lo_hi: FortranInputString,
    /// Java `bSourceBorderYZLoHi`.
    pub(crate) b_source_border_yz_lo_hi: FortranInputString,
    /// Java `initialShiftXYZ`.
    pub(crate) initial_shift_xyz: FortranInputString,
    /// Java `kernelSigma`.
    pub(crate) kernel_sigma: ScriptParameter,
    /// Java `invertYLimits`.
    pub(crate) invert_y_limits: Mutex<EtomoBoolean2>,
    /// Java `referenceFile`.
    pub(crate) reference_file: Option<String>,
    /// Java `fileToAlign`.
    pub(crate) file_to_align: Option<String>,
    /// Java `outputFile`.
    pub(crate) output_file: Option<String>,
    /// Java `bSourceTransform`.
    pub(crate) b_source_transform: Option<String>,
    /// Java `bSourceOrSizeXYZ`.
    pub(crate) b_source_or_size_xyz: Option<String>,
    /// Java `regionModel`.
    pub(crate) region_model: Option<String>,
}

impl ConstPatchcrawl3DParam {
    /// Java package-private `ConstPatchcrawl3DParam()`.
    pub(crate) fn new() -> ConstPatchcrawl3DParam {
        let mut instance = ConstPatchcrawl3DParam {
            patch_size_xyz: FortranInputString::new_with_key(Some("PatchSizeXYZ"), 3),
            number_of_patches_xyz: FortranInputString::new_with_key(Some("NumberOfPatchesXYZ"), 3),
            x_min_and_max: FortranInputString::new_with_key(Some("XMinAndMax"), 2),
            y_min_and_max: FortranInputString::new_with_key(Some("YMinAndMax"), 2),
            z_min_and_max: FortranInputString::new_with_key(Some("ZMinAndMax"), 2),
            b_source_border_x_lo_hi: FortranInputString::new_with_key(
                Some("BSourceBorderXLoHi"),
                2,
            ),
            b_source_border_yz_lo_hi: FortranInputString::new_with_key(
                Some("BSourceBorderYZLoHi"),
                2,
            ),
            initial_shift_xyz: FortranInputString::new_with_key(Some(INITIAL_SHIFT_XYZ_KEY), 3),
            kernel_sigma: ScriptParameter::new_with_type_and_name(Type::Double, KERNEL_SIGMA_KEY),
            invert_y_limits: Mutex::new(EtomoBoolean2::new_with_name("InvertYLimits")),
            reference_file: None,
            file_to_align: None,
            output_file: None,
            b_source_transform: None,
            b_source_or_size_xyz: None,
            region_model: None,
        };
        instance.patch_size_xyz.set_integer_type(true);
        instance.number_of_patches_xyz.set_integer_type(true);
        instance.x_min_and_max.set_integer_type(true);
        instance.y_min_and_max.set_integer_type(true);
        instance.z_min_and_max.set_integer_type(true);
        instance.b_source_border_x_lo_hi.set_integer_type(true);
        instance.b_source_border_yz_lo_hi.set_integer_type(true);
        instance.kernel_sigma.set_display_value_int(1);
        instance.reset();
        instance
    }

    /// Java package-private `reset`.
    pub(crate) fn reset(&mut self) {
        self.patch_size_xyz.reset();
        self.number_of_patches_xyz.reset();
        self.x_min_and_max.reset();
        self.y_min_and_max.reset();
        self.z_min_and_max.reset();
        self.reference_file = None;
        self.file_to_align = None;
        self.output_file = None;
        self.b_source_transform = None;
        self.b_source_or_size_xyz = None;
        self.b_source_border_x_lo_hi.reset();
        self.region_model = None;
        self.initial_shift_xyz.set_default(); // optional parameter
        self.kernel_sigma.reset();
        self.invert_y_limits.get_mut().unwrap().reset();
    }

    /// Java `isUseBoundaryModel`.
    pub fn is_use_boundary_model(&self) -> bool {
        self.region_model.is_some()
    }

    /// Java `getXPatchSize`.
    pub fn get_x_patch_size(&self) -> i32 {
        self.patch_size_xyz.get_int(X_INDEX)
    }

    /// Java `getYPatchSize`.
    pub fn get_y_patch_size(&self) -> i32 {
        self.patch_size_xyz.get_int(Y_INDEX)
    }

    /// Java `getZPatchSize`.
    pub fn get_z_patch_size(&self) -> i32 {
        self.patch_size_xyz.get_int(Z_INDEX)
    }

    /// Java `getNX`.
    pub fn get_nx(&self) -> i32 {
        self.number_of_patches_xyz.get_int(X_INDEX)
    }

    /// Java `getNY`.
    pub fn get_ny(&self) -> i32 {
        self.number_of_patches_xyz.get_int(Y_INDEX)
    }

    /// Java `getNZ`.
    pub fn get_nz(&self) -> i32 {
        self.number_of_patches_xyz.get_int(Z_INDEX)
    }

    /// Java `getXLow`.
    pub fn get_x_low(&self) -> i32 {
        self.x_min_and_max.get_int(0)
    }

    /// Java `getXHigh`.
    pub fn get_x_high(&self) -> i32 {
        self.x_min_and_max.get_int(1)
    }

    /// Java `getYLow`.
    pub fn get_y_low(&self) -> i32 {
        self.y_min_and_max.get_int(0)
    }

    /// Java `getYHigh`.
    pub fn get_y_high(&self) -> i32 {
        self.y_min_and_max.get_int(1)
    }

    /// Java `getZLow`.
    pub fn get_z_low(&self) -> i32 {
        self.z_min_and_max.get_int(0)
    }

    /// Java `getZHigh`.
    pub fn get_z_high(&self) -> i32 {
        self.z_min_and_max.get_int(1)
    }

    /// Java `getInitialShiftX`.
    pub fn get_initial_shift_x(&self) -> String {
        self.initial_shift_xyz.to_string_index(X_INDEX)
    }

    /// Java `getInitialShiftY`.
    pub fn get_initial_shift_y(&self) -> String {
        self.initial_shift_xyz.to_string_index(Y_INDEX)
    }

    /// Java `getInitialShiftZ`.
    pub fn get_initial_shift_z(&self) -> String {
        self.initial_shift_xyz.to_string_index(Z_INDEX)
    }

    /// Java `isKernelSigmaActive`.
    pub fn is_kernel_sigma_active(&self) -> bool {
        self.kernel_sigma.is_active()
    }

    /// Java `getKernelSigma`.
    pub fn get_kernel_sigma(&self) -> &ConstEtomoNumber {
        &self.kernel_sigma.base.base
    }
}
