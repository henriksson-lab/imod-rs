//! `IMOD/Etomo/src/etomo/comscript/ConstTomopitchParam.java`.

use super::param_utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java `COMMAND`.
pub const COMMAND: &str = "tomopitch";
/// Java `MODEL_FILE`.
pub const MODEL_FILE: &str = "ModelFile";
/// Java `SPACING_IN_Y`.
pub const SPACING_IN_Y: &str = "SpacingInY";
/// Java `SCALE_FACTOR`.
pub const SCALE_FACTOR: &str = "ScaleFactor";
/// Java `PARAMETER_FILE`.
pub const PARAMETER_FILE: &str = "ParameterFile";

/// Java class `ConstTomopitchParam`.  `TomopitchParam` extends it and holds it as its
/// `base`, reaching the package-private fields directly as the Java subclass does.
pub struct ConstTomopitchParam {
    pub(crate) manager: &'static ApplicationManager,
    pub(crate) axis_id: AxisID,
    /// Java `Vector modelFiles`.  An element is null when the script gives the keyword
    /// with no value.
    pub(crate) model_files: Vec<Option<String>>,
    pub(crate) extra_thickness: ScriptParameter,
    pub(crate) spacing_in_y: f64,
    pub(crate) scale_factor: f64,
    /// Java `String parameterFile`.
    pub(crate) parameter_file: Option<String>,
    pub(crate) angle_offset_old: ScriptParameter,
    pub(crate) z_shift_old: ScriptParameter,
    pub(crate) x_axis_tilt_old: ScriptParameter,
    pub(crate) no_x_axis_tilt: EtomoBoolean2,
}

impl ConstTomopitchParam {
    /// Java `ConstTomopitchParam(ApplicationManager, AxisID)`.
    pub fn new(manager: &'static ApplicationManager, axis_id: AxisID) -> ConstTomopitchParam {
        let mut param = ConstTomopitchParam {
            manager,
            axis_id,
            model_files: Vec::new(),
            extra_thickness: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "ExtraThickness",
            ),
            spacing_in_y: f64::NAN,
            scale_factor: f64::NAN,
            parameter_file: Some(String::new()),
            angle_offset_old: ScriptParameter::new_with_type_and_name(
                Type::Double,
                "AngleOffsetOld",
            ),
            z_shift_old: ScriptParameter::new_with_type_and_name(Type::Double, "ZShiftOld"),
            x_axis_tilt_old: ScriptParameter::new_with_type_and_name(Type::Double, "XAxisTiltOld"),
            no_x_axis_tilt: EtomoBoolean2::new_with_name("NoXAxisTilt"),
        };
        param.reset();
        param
    }

    /// Java package-private `reset`.
    pub(crate) fn reset(&mut self) {
        self.model_files = Vec::new();
        self.spacing_in_y = f64::NAN;
        self.scale_factor = f64::NAN;
        self.parameter_file = Some(String::new());
        self.angle_offset_old.reset();
        self.z_shift_old.reset();
        self.x_axis_tilt_old.reset();
    }

    /// Java `getModelFilesSize`.
    pub fn get_model_files_size(&self) -> i32 {
        self.model_files.len() as i32
    }

    /// Java `getModelFile(int)`.  Java's `Vector.get` throws for an index out of range;
    /// here that is `None`, as is a null element.
    pub fn get_model_file(&self, index: i32) -> Option<&str> {
        if index < 0 {
            return None;
        }
        self.model_files
            .get(index as usize)
            .and_then(|model_file| model_file.as_deref())
    }

    /// Java `getExtraThicknessString`.
    pub fn get_extra_thickness_string(&self) -> String {
        self.extra_thickness.to_string()
    }

    /// Java `isNoXAxisTilt`.
    pub fn is_no_x_axis_tilt(&self) -> bool {
        self.no_x_axis_tilt.is()
    }

    /// Java `isExtraThicknessNull`.
    pub fn is_extra_thickness_null(&self) -> bool {
        self.extra_thickness.is_null()
    }

    /// Java `getSpacingInYString`.
    pub fn get_spacing_in_y_string(&self) -> String {
        param_utilities::value_of_double(self.spacing_in_y)
    }

    /// Java `getScaleFactorString`.
    pub fn get_scale_factor_string(&self) -> String {
        param_utilities::value_of_double(self.scale_factor)
    }

    /// Java `getParameterFile`.
    pub fn get_parameter_file(&self) -> Option<&str> {
        self.parameter_file.as_deref()
    }
}

/// Java `toString`.
impl std::fmt::Display for ConstTomopitchParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut string = String::from("\n");
        string.push_str(&format!("ModelFilesSize:{}\n", self.get_model_files_size()));
        for i in 0..self.get_model_files_size() {
            string.push_str(&format!(
                "{MODEL_FILE}:{}\n",
                self.get_model_file(i).unwrap_or("null")
            ));
        }
        string.push_str(&format!(
            "{}:{}\n",
            self.extra_thickness.get_name(),
            self.get_extra_thickness_string()
        ));
        string.push_str(&format!(
            "{SCALE_FACTOR}:{}\n",
            self.get_scale_factor_string()
        ));
        string.push_str(&format!(
            "{SPACING_IN_Y}:{}\n",
            self.get_spacing_in_y_string()
        ));
        string.push_str(&format!(
            "{PARAMETER_FILE}:{}\n",
            self.get_parameter_file().unwrap_or("null")
        ));
        f.write_str(&string)
    }
}
