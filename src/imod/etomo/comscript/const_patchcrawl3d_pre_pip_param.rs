//! `IMOD/Etomo/src/etomo/comscript/ConstPatchcrawl3DPrePIPParam.java`.
//!
//! A package-private Java class with state, so a struct here;
//! `Patchcrawl3DPrePIPParam` holds it as its `base` through `Deref`/`DerefMut`.

use super::fortran_input_string::FortranInputString;

/// Java `ConstPatchcrawl3DPrePIPParam`.
#[derive(Clone, Debug)]
pub struct ConstPatchcrawl3DPrePIPParam {
    /// Java `xPatchSize`.
    pub(crate) x_patch_size: i32,
    /// Java `yPatchSize`.
    pub(crate) y_patch_size: i32,
    /// Java `zPatchSize`.
    pub(crate) z_patch_size: i32,
    /// Java `nX`.
    pub(crate) n_x: i32,
    /// Java `nY`.
    pub(crate) n_y: i32,
    /// Java `nZ`.
    pub(crate) n_z: i32,
    /// Java `xLow`.
    pub(crate) x_low: i32,
    /// Java `xHigh`.
    pub(crate) x_high: i32,
    /// Java `yLow`.
    pub(crate) y_low: i32,
    /// Java `yHigh`.
    pub(crate) y_high: i32,
    /// Java `zLow`.
    pub(crate) z_low: i32,
    /// Java `zHigh`.
    pub(crate) z_high: i32,
    /// Java `maxShift`.
    pub(crate) max_shift: i32,
    /// Java `fileA`.
    pub(crate) file_a: Option<String>,
    /// Java `fileB`.
    pub(crate) file_b: Option<String>,
    /// Java `outputFile`.
    pub(crate) output_file: Option<String>,
    /// Java `transformFile`.
    pub(crate) transform_file: Option<String>,
    /// Java `originalFileB`.
    pub(crate) original_file_b: Option<String>,
    /// Java `borders`.
    pub(crate) borders: FortranInputString,
    /// Java `boundaryModel`.
    pub(crate) boundary_model: Option<String>,
}

impl ConstPatchcrawl3DPrePIPParam {
    /// Java package-private `ConstPatchcrawl3DPrePIPParam()`.
    pub(crate) fn new() -> ConstPatchcrawl3DPrePIPParam {
        let mut instance = ConstPatchcrawl3DPrePIPParam {
            x_patch_size: 0,
            y_patch_size: 0,
            z_patch_size: 0,
            n_x: 0,
            n_y: 0,
            n_z: 0,
            x_low: 0,
            x_high: 0,
            y_low: 0,
            y_high: 0,
            z_low: 0,
            z_high: 0,
            max_shift: 0,
            file_a: Some(String::new()),
            file_b: Some(String::new()),
            output_file: Some(String::new()),
            transform_file: Some(String::new()),
            original_file_b: Some(String::new()),
            borders: FortranInputString::new(4),
            boundary_model: Some(String::new()),
        };
        instance.reset();
        instance
    }

    /// Java package-private `getFileA`.
    pub(crate) fn get_file_a(&self) -> Option<&str> {
        self.file_a.as_deref()
    }

    /// Java package-private `getFileB`.
    pub(crate) fn get_file_b(&self) -> Option<&str> {
        self.file_b.as_deref()
    }

    /// Java package-private `getNX`.
    pub(crate) fn get_nx(&self) -> i32 {
        self.n_x
    }

    /// Java package-private `getNY`.
    pub(crate) fn get_ny(&self) -> i32 {
        self.n_y
    }

    /// Java package-private `getNZ`.
    pub(crate) fn get_nz(&self) -> i32 {
        self.n_z
    }

    /// Java package-private `getOriginalFileB`.
    pub(crate) fn get_original_file_b(&self) -> Option<&str> {
        self.original_file_b.as_deref()
    }

    /// Java package-private `getTransformFile`.
    pub(crate) fn get_transform_file(&self) -> Option<&str> {
        self.transform_file.as_deref()
    }

    /// Java package-private `getBorders`.
    pub(crate) fn get_borders(&self) -> String {
        self.borders.to_string()
    }

    /// Java package-private `getBordersFortranInputString`.
    pub(crate) fn get_borders_fortran_input_string(&self) -> &FortranInputString {
        &self.borders
    }

    /// Java package-private `getXHigh`.
    pub(crate) fn get_x_high(&self) -> i32 {
        self.x_high
    }

    /// Java package-private `getXLow`.
    pub(crate) fn get_x_low(&self) -> i32 {
        self.x_low
    }

    /// Java package-private `getXPatchSize`.
    pub(crate) fn get_x_patch_size(&self) -> i32 {
        self.x_patch_size
    }

    /// Java package-private `getYHigh`.
    pub(crate) fn get_y_high(&self) -> i32 {
        self.y_high
    }

    /// Java package-private `getYLow`.
    pub(crate) fn get_y_low(&self) -> i32 {
        self.y_low
    }

    /// Java package-private `getYPatchSize`.
    pub(crate) fn get_y_patch_size(&self) -> i32 {
        self.y_patch_size
    }

    /// Java package-private `getZHigh`.
    pub(crate) fn get_z_high(&self) -> i32 {
        self.z_high
    }

    /// Java package-private `getZLow`.
    pub(crate) fn get_z_low(&self) -> i32 {
        self.z_low
    }

    /// Java package-private `getZPatchSize`.
    pub(crate) fn get_z_patch_size(&self) -> i32 {
        self.z_patch_size
    }

    /// Java package-private `getMaxShift`.
    pub(crate) fn get_max_shift(&self) -> i32 {
        self.max_shift
    }

    /// Java package-private `getBoundaryModel`.
    pub(crate) fn get_boundary_model(&self) -> Option<&str> {
        self.boundary_model.as_deref()
    }

    /// Java package-private `isUseBoundaryModel`.  A null model (a
    /// NullPointerException in Java) is not in use.
    pub(crate) fn is_use_boundary_model(&self) -> bool {
        self.boundary_model
            .as_deref()
            .is_some_and(|model| model != "")
    }

    /// Java package-private `getOutputFile`.
    pub(crate) fn get_output_file(&self) -> Option<&str> {
        self.output_file.as_deref()
    }

    /// Java package-private `reset`.
    pub(crate) fn reset(&mut self) {
        self.x_patch_size = 0;
        self.y_patch_size = 0;
        self.z_patch_size = 0;
        self.n_x = 0;
        self.n_y = 0;
        self.n_z = 0;
        self.x_low = 0;
        self.x_high = 0;
        self.y_low = 0;
        self.y_high = 0;
        self.z_low = 0;
        self.z_high = 0;
        self.max_shift = 0;
        self.file_a = Some(String::new());
        self.file_b = Some(String::new());
        self.output_file = Some(String::new());
        self.transform_file = Some(String::new());
        self.original_file_b = Some(String::new());
        self.borders = FortranInputString::new(4);
        let int_flag = [true, true, true, true];
        self.borders.set_integer_type_array(&int_flag);
        self.boundary_model = Some(String::new());
    }
}
