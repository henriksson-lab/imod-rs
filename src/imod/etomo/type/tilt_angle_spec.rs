//! `IMOD/Etomo/src/etomo/type/TiltAngleSpec.java`.
//!
//! How the tilt angles of an axis are specified: extracted from the stack, a range
//! (start and step), a `.rawtlt` file, or an explicit list.
//!
//! Java `Properties` is the deterministic `BTreeMap<String, String>` used throughout
//! the translation (`etomo/storage/storable.rs`).
//!
//! **`prepend == ""`.**  `store(Properties, String)` and `load(Properties, String)` test
//! `prepend == ""`, a reference comparison that is true for the interned literal `""`
//! (which is what `store(Properties)`/`load(Properties)` pass) and false for a computed
//! empty string.  Every caller in the source passes either the literal or a non-empty
//! group, so the test is translated as `prepend.is_empty()`.

use std::collections::BTreeMap;

use super::const_etomo_number::{Type, java_lang_double_to_string, java_lang_double_value_of};
use super::script_parameter::ScriptParameter;
use super::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::comscript::bad_com_script_exception::BadComScriptException;
use crate::imod::etomo::comscript::com_script_command::ComScriptCommand;
use crate::imod::etomo::comscript::command_param::ParseComScriptError;
use crate::imod::etomo::comscript::fortran_input_string::FortranInputString;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";
/// Java `STORE_KEY`.
pub const STORE_KEY: &str = "TiltAngle";
/// Java `TYPE_STORE_KEY`.
pub const TYPE_STORE_KEY: &str = "Type";
/// Java `RANGE_MIN_STORE_KEY`.
pub const RANGE_MIN_STORE_KEY: &str = "RangeMin";
/// Java `TILT_ANGLES_STORE_KEY`.
pub const TILT_ANGLES_STORE_KEY: &str = "TiltAngles";

/// Java `TiltAngleSpec`.
#[derive(Clone, Debug)]
pub struct TiltAngleSpec {
    /// Java package-private field `type`.  Java `TiltAngleType.fromString` can put null
    /// here; see `load`.
    pub(crate) r#type: TiltAngleType,
    /// Java package-private field `rangeMin`.
    pub(crate) range_min: f64,
    /// Java package-private field `rangeStep`.
    pub(crate) range_step: f64,
    /// Java package-private field `tiltAngles[]`; null is `None`.
    pub(crate) tilt_angles: Option<Vec<f64>>,
    /// Java package-private field `tiltAngleFilename`.
    pub(crate) tilt_angle_filename: String,
    /// Java field `rangeMinKey`.
    range_min_key: Option<String>,
    /// Java field `rangeStepKey`.
    range_step_key: Option<String>,
    /// Java field `tiltAngleFilenameKey`.
    tilt_angle_filename_key: Option<String>,
    /// Java field `tiltAnglesKey`.
    tilt_angles_key: Option<String>,
    /// Java field `rangeMinShortKey`.
    range_min_short_key: Option<String>,
    /// Java field `rangeStepShortKey`.
    range_step_short_key: Option<String>,
    /// Java field `tiltAngleFilenameShortKey`.
    tilt_angle_filename_short_key: Option<String>,
    /// Java field `tiltAnglesShortKey`.
    tilt_angles_short_key: Option<String>,
}

impl TiltAngleSpec {
    /// The Java object allocation the constructors share: every field at its declared
    /// initial value (the keys null; the rest is overwritten by `reset()`).
    fn allocate() -> TiltAngleSpec {
        TiltAngleSpec {
            r#type: TiltAngleType::Extract,
            range_min: 0.0,
            range_step: 0.0,
            tilt_angles: None,
            tilt_angle_filename: String::new(),
            range_min_key: None,
            range_step_key: None,
            tilt_angle_filename_key: None,
            tilt_angles_key: None,
            range_min_short_key: None,
            range_step_short_key: None,
            tilt_angle_filename_short_key: None,
            tilt_angles_short_key: None,
        }
    }

    /// Java `TiltAngleSpec()`.
    pub fn new() -> TiltAngleSpec {
        let mut instance = TiltAngleSpec::allocate();
        instance.reset();
        instance
    }

    /// Java `TiltAngleSpec(TiltAngleSpec)`.  Copies the four long keys; the short keys
    /// are not copied, as in the source.
    pub fn new_from_instance(src: &TiltAngleSpec) -> TiltAngleSpec {
        let mut instance = TiltAngleSpec::allocate();
        instance.set(Some(src));
        instance.range_min_key = src.range_min_key.clone();
        instance.range_step_key = src.range_step_key.clone();
        instance.tilt_angle_filename_key = src.tilt_angle_filename_key.clone();
        instance.tilt_angles_key = src.tilt_angles_key.clone();
        instance
    }

    /// Java `setRangeMinKey`.
    pub fn set_range_min_key(
        &mut self,
        range_min_key: Option<&str>,
        range_min_short_key: Option<&str>,
    ) {
        self.range_min_key = range_min_key.map(|s| s.to_string());
        self.range_min_short_key = range_min_short_key.map(|s| s.to_string());
    }

    /// Java `setRangeStepKey`.
    pub fn set_range_step_key(
        &mut self,
        range_step_key: Option<&str>,
        range_step_short_key: Option<&str>,
    ) {
        self.range_step_key = range_step_key.map(|s| s.to_string());
        self.range_step_short_key = range_step_short_key.map(|s| s.to_string());
    }

    /// Java `setTiltAngleFilenameKey`.
    pub fn set_tilt_angle_filename_key(
        &mut self,
        tilt_angle_filename_key: Option<&str>,
        tilt_angle_filename_short_key: Option<&str>,
    ) {
        self.tilt_angle_filename_key = tilt_angle_filename_key.map(|s| s.to_string());
        self.tilt_angle_filename_short_key = tilt_angle_filename_short_key.map(|s| s.to_string());
    }

    /// Java `setTiltAnglesKey`.
    pub fn set_tilt_angles_key(
        &mut self,
        tilt_angles_key: Option<&str>,
        tilt_angles_short_key: Option<&str>,
    ) {
        self.tilt_angles_key = tilt_angles_key.map(|s| s.to_string());
        self.tilt_angles_short_key = tilt_angles_short_key.map(|s| s.to_string());
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.r#type = TiltAngleType::Extract;
        self.range_min = -60.0;
        self.range_step = 1.0;
        self.tilt_angles = None;
        self.tilt_angle_filename = String::new();
    }

    /// Java `set(TiltAngleSpec)`.  Does not copy `tiltAngles`, as in the source.
    pub fn set(&mut self, src: Option<&TiltAngleSpec>) {
        self.reset();
        if let Some(src) = src {
            self.r#type = src.get_type();
            self.range_min = src.get_range_min();
            self.range_step = src.get_range_step();
            self.tilt_angle_filename = src.get_tilt_angle_filename();
        }
    }

    /// Java `setType`.  The source's per-type branches are empty ("TODO what sort of
    /// clean do we need to do upon setting the type").
    pub fn set_type(&mut self, r#type: TiltAngleType) {
        self.r#type = r#type;
        if r#type == TiltAngleType::File {}
        if r#type == TiltAngleType::Range {}
        if r#type == TiltAngleType::List {}
        if r#type == TiltAngleType::Extract {}
    }

    /// Java `getType`.
    pub fn get_type(&self) -> TiltAngleType {
        self.r#type
    }

    /// Java `setRangeMin(double)`.
    pub fn set_range_min_double(&mut self, range_min: f64) {
        self.range_min = range_min;
    }

    /// Java `setRangeMin(String)`.  `Double.parseDouble` throws
    /// `NumberFormatException` (unchecked) for a malformed string; the error is returned
    /// here and the field is left unchanged, as it is when the Java throw unwinds.
    pub fn set_range_min_string(&mut self, range_min: &str) -> Result<(), String> {
        self.range_min = java_lang_double_value_of(range_min)?;
        Ok(())
    }

    /// Java `getRangeMin`.
    pub fn get_range_min(&self) -> f64 {
        self.range_min
    }

    /// Java `setRangeStep(double)`.
    pub fn set_range_step_double(&mut self, range_step: f64) {
        self.range_step = range_step;
    }

    /// Java `setRangeStep(String)`.  See `set_range_min_string`.
    pub fn set_range_step_string(&mut self, range_step: &str) -> Result<(), String> {
        self.range_step = java_lang_double_value_of(range_step)?;
        Ok(())
    }

    /// Java `getRangeStep`.
    pub fn get_range_step(&self) -> f64 {
        self.range_step
    }

    /// Java `setTiltAngleFilename`.
    pub fn set_tilt_angle_filename(&mut self, tilt_angle_filename: &str) {
        //
        // NOTE validation, does it need to exits, format?
        //
        self.tilt_angle_filename = tilt_angle_filename.to_string();
    }

    /// Java `getTiltAngleFilename`.  Return the filename containing the tilt angles.
    pub fn get_tilt_angle_filename(&self) -> String {
        self.tilt_angle_filename.clone()
    }

    /// Java `getTiltAngles`.  Return the appropriate tilt angle representation
    /// depending upon the state of type attribute.
    ///
    /// Upstream bug fixed in translation (TiltAngleSpec.java:237-243): for `LIST` the
    /// source indexes `tiltAngles[tiltAngles.length - 1]`, which throws
    /// `NullPointerException` when no list was loaded and
    /// `ArrayIndexOutOfBoundsException` for an empty one.  Both now return "", the same
    /// representation the method gives for a type with nothing to show.
    pub fn get_tilt_angles(&self) -> String {
        if self.r#type == TiltAngleType::File {
            return self.tilt_angle_filename.clone();
        }
        if self.r#type == TiltAngleType::Range {
            return format!(
                "{},{}",
                java_lang_double_to_string(self.range_min),
                java_lang_double_to_string(self.range_step)
            );
        }
        if self.r#type == TiltAngleType::List {
            let tilt_angles = match &self.tilt_angles {
                Some(tilt_angles) if !tilt_angles.is_empty() => tilt_angles,
                _ => return String::new(),
            };
            let mut list = String::new();
            for i in 0..tilt_angles.len() - 1 {
                list.push_str(&format!("{},", java_lang_double_to_string(tilt_angles[i])));
            }
            list.push_str(&java_lang_double_to_string(
                tilt_angles[tilt_angles.len() - 1],
            ));
            return list;
        }
        String::new()
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.  Insert the objects attributes into the
    /// properties object.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let group = if prepend.is_empty() {
            STORE_KEY.to_string()
        } else {
            format!("{}.{}", prepend, STORE_KEY)
        };
        props.insert(
            format!("{}.{}", group, TYPE_STORE_KEY),
            self.r#type.to_string(),
        );
        props.insert(
            format!("{}.{}", group, RANGE_MIN_STORE_KEY),
            java_lang_double_to_string(self.range_min),
        );
        props.insert(
            format!("{}.RangeStep", group),
            java_lang_double_to_string(self.range_step),
        );
        props.insert(
            format!("{}.TiltAngleFilename", group),
            self.tilt_angle_filename.clone(),
        );
        if let Some(tilt_angles) = &self.tilt_angles {
            if !tilt_angles.is_empty() {
                let list = FortranInputString::get_instance_from_doubles(Some(tilt_angles));
                list.store(props, Some(&format!("{}.{}", group, TILT_ANGLES_STORE_KEY)));
            }
        }
    }

    /// Java `load(Properties)`.
    pub fn load(&mut self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.  Get the objects attributes from the properties
    /// object.
    ///
    /// Upstream bugs fixed in translation:
    /// - TiltAngleSpec.java:290-291: `TiltAngleType.fromString` returns null for an
    ///   unrecognised `Type` value, and the next `store` then throws
    ///   `NullPointerException` on `type.toString()`.  An unrecognised value now loads
    ///   as the property's own default, `Extract`.
    /// - TiltAngleSpec.java:292-293: `Double.parseDouble` on a malformed `RangeMin` or
    ///   `RangeStep` throws `NumberFormatException` out of the whole `.edf` load.  A
    ///   malformed value now loads as the property's default ("-90", "1").
    pub fn load_with_prepend(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        let group = if prepend.is_empty() {
            STORE_KEY.to_string()
        } else {
            format!("{}.{}", prepend, STORE_KEY)
        };
        self.r#type = TiltAngleType::from_string(
            props
                .get(&format!("{}.{}", group, TYPE_STORE_KEY))
                .map(|s| s.as_str())
                .unwrap_or("Extract"),
        )
        .unwrap_or(TiltAngleType::Extract);
        self.range_min = java_lang_double_value_of(
            props
                .get(&format!("{}.RangeMin", group))
                .map(|s| s.as_str())
                .unwrap_or("-90"),
        )
        .unwrap_or(-90.0);
        self.range_step = java_lang_double_value_of(
            props
                .get(&format!("{}.RangeStep", group))
                .map(|s| s.as_str())
                .unwrap_or("1"),
        )
        .unwrap_or(1.0);
        self.tilt_angle_filename = props
            .get(&format!("{}.TiltAngleFilename", group))
            .cloned()
            .unwrap_or_default();
        match FortranInputString::get_instance(
            props,
            &format!("{}.{}", group, TILT_ANGLES_STORE_KEY),
        ) {
            Ok(list) => {
                if list.size() > 0 {
                    self.tilt_angles = Some(list.get_double());
                }
            }
            Err(e) => {
                e.print_stack_trace();
            }
        }
    }

    /// Java `parse(ComScriptCommand)`.
    ///
    /// The source throws `IllegalStateException("Missing rangeStep key.")` when the
    /// caller never configured the range step keys; that is a programming error in the
    /// caller (an unchecked exception no source caller catches) and stays a panic.
    /// A key the caller never set is Java null; the `ScriptParameter` built from it is
    /// given an empty name here, which, like the null name, matches no keyword.
    pub fn parse(&mut self, script_command: &ComScriptCommand) -> Result<(), ParseComScriptError> {
        // Get rangeMin
        let mut range_min = ScriptParameter::new_with_short_name(
            Type::Double,
            self.range_min_key.as_deref().unwrap_or(""),
            self.range_min_short_key.as_deref().unwrap_or(""),
        );
        range_min
            .parse(script_command)
            .map_err(ParseComScriptError::InvalidParameter)?;
        if !range_min.is_null() {
            self.r#type = TiltAngleType::Range;
            self.range_min = range_min.get_double();
            // Get rangeStep
            let mut range_step = ScriptParameter::new_with_short_name(
                Type::Double,
                self.range_step_key.as_deref().unwrap_or(""),
                self.range_step_short_key.as_deref().unwrap_or(""),
            );
            range_step
                .parse(script_command)
                .map_err(ParseComScriptError::InvalidParameter)?;
            if !range_step.is_null() {
                self.range_step = range_step.get_double();
            } else if self.range_step_key.is_none() && self.range_min_short_key.is_none() {
                panic!("Missing rangeStep key.");
            } else {
                return Err(ParseComScriptError::InvalidParameter(
                    crate::imod::etomo::comscript::invalid_parameter_exception::InvalidParameterException::new(
                        Some("First tilt angle is set, but increment is missing."),
                    ),
                ));
            }
        }
        // Get tiltAngleFilename
        let mut tilt_angle_filename = script_command
            .get_value(self.tilt_angle_filename_key.as_deref())
            .map_err(ParseComScriptError::InvalidParameter)?;
        let blank = match &tilt_angle_filename {
            None => true,
            Some(value) => super::const_etomo_number::java_lang_string_matches_whitespace(value),
        };
        if blank && self.tilt_angle_filename_short_key.is_some() {
            tilt_angle_filename = script_command
                .get_value(self.tilt_angle_filename_short_key.as_deref())
                .map_err(ParseComScriptError::InvalidParameter)?;
        }
        match tilt_angle_filename {
            Some(value)
                if !super::const_etomo_number::java_lang_string_matches_whitespace(&value) =>
            {
                self.tilt_angle_filename = value;
                self.r#type = TiltAngleType::File;
            }
            _ => {
                self.tilt_angle_filename = String::new();
            }
        }
        // Get tiltAngles (Successive entries accumulate)
        if let Some(tilt_angles_key) = &self.tilt_angles_key {
            if !super::const_etomo_number::java_lang_string_matches_whitespace(tilt_angles_key) {
                // TODO(unit): needs etomo/comscript/FortranInputStringList.java -
                // `list = new FortranInputStringList(tiltAnglesKey); list.parse(scriptCommand);
                // tiltAngles = list.getDouble();` (consolidating successive entries).
            }
        }
        Ok(())
    }

    /// Java `updateComScript(ComScriptCommand)`.
    pub fn update_com_script(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        let _ = &script_command;
        if self.r#type == TiltAngleType::Range {
            // TODO(unit): needs etomo/comscript/ParamUtilities.java -
            // `ParamUtilities.updateScriptParameter(scriptCommand, rangeMinKey, rangeMin)` and
            // `ParamUtilities.updateScriptParameter(scriptCommand, rangeStepKey, rangeStep)`.
        } else if self.r#type == TiltAngleType::File {
            // TODO(unit): needs etomo/comscript/ParamUtilities.java -
            // `ParamUtilities.updateScriptParameter(scriptCommand, tiltAngleFilenameKey,
            // tiltAngleFilename)`.
        } else if self.r#type == TiltAngleType::List {
            if let Some(tilt_angles) = &self.tilt_angles {
                if !tilt_angles.is_empty() {
                    let list = FortranInputString::get_instance_from_doubles(Some(tilt_angles));
                    let _ = list;
                    // TODO(unit): needs etomo/comscript/ParamUtilities.java -
                    // `ParamUtilities.updateScriptParameter(scriptCommand, tiltAnglesKey, list)`.
                }
            }
        } else {
            return Err(BadComScriptException::new(&format!(
                "Type {}, cannot be updated in ComScriptCommand",
                self.r#type
            )));
        }
        Ok(())
    }
}

impl Default for TiltAngleSpec {
    fn default() -> TiltAngleSpec {
        TiltAngleSpec::new()
    }
}

/// Java `toString`.  The source concatenates the `double[]` itself, which prints the
/// array's identity (`[D@1b6d3586`) - not reproducible; the list is shown as
/// `java.util.Arrays.toString` would print it instead, and a null array as "null".
impl std::fmt::Display for TiltAngleSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tilt_angles = match &self.tilt_angles {
            None => "null".to_string(),
            Some(tilt_angles) => format!(
                "[{}]",
                tilt_angles
                    .iter()
                    .map(|a| java_lang_double_to_string(*a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        write!(
            f,
            "[type={},rangeMin={},rangeStep={},tiltAngles={},tiltAngleFilename={}]",
            self.r#type,
            java_lang_double_to_string(self.range_min),
            java_lang_double_to_string(self.range_step),
            tilt_angles,
            self.tilt_angle_filename
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_matches_java_default_layout() {
        // Java reference (MetaData default store): Setup.AxisA.TiltAngle.RangeMin=-60.0,
        // RangeStep=1.0, TiltAngleFilename=, Type=Extract.
        let spec = TiltAngleSpec::new();
        let mut props = BTreeMap::new();
        spec.store_with_prepend(&mut props, "Setup.AxisA");
        assert_eq!(
            props.get("Setup.AxisA.TiltAngle.RangeMin").unwrap(),
            "-60.0"
        );
        assert_eq!(props.get("Setup.AxisA.TiltAngle.RangeStep").unwrap(), "1.0");
        assert_eq!(
            props
                .get("Setup.AxisA.TiltAngle.TiltAngleFilename")
                .unwrap(),
            ""
        );
        assert_eq!(props.get("Setup.AxisA.TiltAngle.Type").unwrap(), "Extract");
        assert_eq!(props.len(), 4);
    }

    #[test]
    fn load_defaults_and_round_trip() {
        let mut spec = TiltAngleSpec::new();
        spec.load_with_prepend(&BTreeMap::new(), "Setup.AxisB");
        // Missing properties load the source's load defaults, not reset()'s.
        assert_eq!(spec.get_range_min(), -90.0);
        assert_eq!(spec.get_range_step(), 1.0);
        assert_eq!(spec.get_type(), TiltAngleType::Extract);

        let mut src = TiltAngleSpec::new();
        src.set_type(TiltAngleType::Range);
        src.set_range_min_double(-54.0);
        src.set_range_step_double(3.0);
        let mut props = BTreeMap::new();
        src.store(&mut props);
        let mut dst = TiltAngleSpec::new();
        dst.load(&props);
        assert_eq!(dst.get_tilt_angles(), "-54.0,3.0");
        let mut props2 = BTreeMap::new();
        dst.store(&mut props2);
        assert_eq!(props, props2);
    }

    #[test]
    fn bad_values_load_defaults_instead_of_throwing() {
        let mut props = BTreeMap::new();
        props.insert("TiltAngle.Type".to_string(), "Bogus".to_string());
        props.insert("TiltAngle.RangeMin".to_string(), "abc".to_string());
        let mut spec = TiltAngleSpec::new();
        spec.load(&props);
        assert_eq!(spec.get_type(), TiltAngleType::Extract);
        assert_eq!(spec.get_range_min(), -90.0);
    }

    #[test]
    fn list_without_angles_is_empty() {
        let mut spec = TiltAngleSpec::new();
        spec.set_type(TiltAngleType::List);
        assert_eq!(spec.get_tilt_angles(), "");
        spec.tilt_angles = Some(vec![-1.5, 0.0, 2.0]);
        assert_eq!(spec.get_tilt_angles(), "-1.5,0.0,2.0");
    }
}
