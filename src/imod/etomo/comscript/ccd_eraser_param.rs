//! `IMOD/Etomo/src/etomo/comscript/CCDEraserParam.java`.
//!
//! Java `CCDEraserParam extends ConstCCDEraserParam implements Command, CommandParam`.
//! The superclass state is the `base` field, reached through `Deref`/`DerefMut`.
//!
//! **Null strings.**  Java's `String` fields can be null: `annulusWidth` has no
//! initialiser, and `ComScriptCommand.getValue` returns null for a keyword present
//! with no value.  The source then calls `.equals("")` on them, which throws
//! NullPointerException (CCDEraserParam.java:118, :185-331; uncaught in
//! `updateComScriptCommand`).  Fixed in translation: a null value is treated as the
//! empty string by those tests, so its keyword is deleted (BUGS.md).

use std::path::PathBuf;
use std::sync::Arc;

use super::bad_com_script_exception::BadComScriptException;
use super::com_script_command::ComScriptCommand;
use super::command::Command;
use super::command_details::CommandDetails;
use super::command_mode::CommandMode;
use super::command_param::{CommandParam, ParseComScriptError};
use super::const_ccd_eraser_param::*;
use super::utilities;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::{
    java_lang_double_to_string, java_lang_double_value_of,
};
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_key::FileKey;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::string_parameter::StringParameter;
use crate::imod::etomo::r#type::view_type::ViewType;
use crate::imod::etomo::ui::swing::ui_harness;

/// Java `COMMAND_NAME` (`ProcessName.CCDERASER.toString()`).
pub fn command_name() -> String {
    ProcessName::CCDERASER.to_string()
}

/// Java private static `COMMAND_SIZE` (unused in the source).
#[allow(dead_code)]
const COMMAND_SIZE: i32 = 1;

/// Java nested `CCDEraserParam.Mode implements CommandMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Java `X_RAYS`.
    XRays,
    /// Java `X_RAYS_TRIAL`: no output file in X_RAYS_TRIAL.
    XRaysTrial,
    /// Java `BEADS`.
    Beads,
}

impl std::fmt::Display for Mode {
    /// Java `Object.toString()` (the class declares none): class name and identity
    /// hash; the constant's name stands in for the hash.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Mode::XRays => "X_RAYS",
            Mode::XRaysTrial => "X_RAYS_TRIAL",
            Mode::Beads => "BEADS",
        };
        write!(f, "etomo.comscript.CCDEraserParam$Mode@{name}")
    }
}

impl CommandMode for Mode {}

/// Java `CCDEraserParam`.
pub struct CCDEraserParam {
    /// Java superclass `ConstCCDEraserParam` state.
    pub base: ConstCCDEraserParam,
    line_objects: StringParameter,
    boundary_objects: StringParameter,
    all_section_objects: StringParameter,
    /// Java field `commandArray`; the source never assigns it.
    command_array: Option<Vec<String>>,
    /// Java field `debug` (never read).
    #[allow(dead_code)]
    debug: bool,
    manager: &'static ApplicationManager,
    axis_id: AxisID,
    mode: Option<Mode>,
}

impl std::ops::Deref for CCDEraserParam {
    type Target = ConstCCDEraserParam;

    fn deref(&self) -> &ConstCCDEraserParam {
        &self.base
    }
}

impl std::ops::DerefMut for CCDEraserParam {
    fn deref_mut(&mut self) -> &mut ConstCCDEraserParam {
        &mut self.base
    }
}

impl CCDEraserParam {
    /// Java `CCDEraserParam(ApplicationManager, AxisID, CommandMode)`.  The source's
    /// mode is only ever compared against this class's own `Mode` constants.
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        mode: Option<Mode>,
    ) -> CCDEraserParam {
        CCDEraserParam {
            base: ConstCCDEraserParam::new(),
            line_objects: StringParameter::new(LINE_OBJECTS_KEY),
            boundary_objects: StringParameter::new(BOUNDARY_OBJECTS_KEY),
            all_section_objects: StringParameter::new(ALL_SECTION_OBJECTS_KEY),
            command_array: None,
            debug: false,
            manager,
            axis_id,
            mode,
        }
    }

    /// Java `validate()`.
    pub fn validate(&self) -> bool {
        if self.base.better_radius.is_null() {
            ui_harness::post_message_dialog(
                Some(self.manager as &'static dyn BaseManager),
                "Empty Better Radius value.  Please enter a value.".to_string(),
                "Entry Error".to_string(),
                None,
            );
            return false;
        }
        true
    }

    /// Java package-private `convertOuterRadius()`.  Converts outerRadius to
    /// annulusWidth.  Does nothing if annulusWidth has a value.  Ccderaser handles
    /// empty annulusWidth and maximumRadius.  Conversion from outerRadius to
    /// annulusWidth is annulusWidth = outerRadius - maximumWidth.
    ///
    /// `Err` carries the NumberFormatException message of `Double.parseDouble`.
    pub(crate) fn convert_outer_radius(&mut self) -> Result<(), String> {
        // if annulusWidth already set then return
        // (null strings are empty here; see the module docs)
        if self.base.annulus_width.as_deref().unwrap_or("") != ""
            || self.base.outer_radius.as_deref().unwrap_or("") == ""
            || self.base.maximum_radius.as_deref().unwrap_or("") == ""
        {
            return Ok(());
        }
        let outer_radius = java_lang_double_value_of(self.base.outer_radius.as_deref().unwrap())?;
        let maximum_radius =
            java_lang_double_value_of(self.base.maximum_radius.as_deref().unwrap())?;
        self.base.annulus_width = Some(java_lang_double_to_string(outer_radius - maximum_radius));
        Ok(())
    }

    /// Java `setInputFile(String)`.
    pub fn set_input_file(&mut self, input_file: Option<&str>) {
        self.base.input_file = input_file.map(|s| s.to_string());
    }

    /// Java `setOutputFile(String)` (deprecated 3/15/2019).
    pub fn set_output_file_string(&mut self, input: Option<&str>) {
        self.base.output_file = input.map(|s| s.to_string());
        self.base.output_file_type = None;
    }

    /// Java `setOutputFile(FileType)`.
    pub fn set_output_file_type(&mut self, file_type: &Arc<FileType>) {
        self.base.output_file = file_type.get_file_name(
            Some(self.manager as &'static dyn BaseManager),
            Some(self.axis_id),
        );
        self.base.output_file_type = Some(Arc::clone(file_type));
    }

    /// Java `setModelFile(String)`.
    pub fn set_model_file(&mut self, model_file: Option<&str>) {
        self.base.model_file = model_file.map(|s| s.to_string());
    }

    /// Java `setGlobalReplacementList(String)`.
    pub fn set_global_replacement_list(&mut self, input: Option<&str>) {
        self.base.global_replacement_list = input.map(|s| s.to_string());
    }

    /// Java `setLocalReplacementList(String)`.
    pub fn set_local_replacement_list(&mut self, input: Option<&str>) {
        self.base.local_replacement_list = input.map(|s| s.to_string());
    }

    /// Java `setBoundaryReplacementList(String)`.
    pub fn set_boundary_replacement_list(&mut self, input: Option<&str>) {
        self.base.boundary_replacement_list = input.map(|s| s.to_string());
    }

    /// Java `setBorderPixels(String)`.
    pub fn set_border_pixels(&mut self, input: Option<&str>) {
        self.base.border_pixels = input.map(|s| s.to_string());
    }

    /// Java `setPolynomialOrder(String)`.
    pub fn set_polynomial_order(&mut self, input: Option<&str>) {
        self.base.polynomial_order = input.map(|s| s.to_string());
    }

    /// Java `setIncludeAdjacentPoints(boolean)`.
    pub fn set_include_adjacent_points(&mut self, include_adjacent_points: bool) {
        self.base.include_adjacent_points = include_adjacent_points;
    }

    /// Java `setDiffCriterion(String)`.
    pub fn set_diff_criterion(&mut self, string: Option<&str>) {
        self.base.diff_criterion = string.map(|s| s.to_string());
    }

    /// Java `setEdgeExclusion(String)`.
    pub fn set_edge_exclusion(&mut self, string: Option<&str>) {
        self.base.edge_exclusion = string.map(|s| s.to_string());
    }

    /// Java `setFindPeaks(boolean)`.
    pub fn set_find_peaks(&mut self, b: bool) {
        self.base.find_peaks = b;
    }

    /// Java `setGrowCriterion(String)`.
    pub fn set_grow_criterion(&mut self, string: Option<&str>) {
        self.base.grow_criterion = string.map(|s| s.to_string());
    }

    /// Java `setGiantCriterion(String)`.
    pub fn set_giant_criterion(&mut self, string: Option<&str>) {
        self.base.giant_criterion = string.map(|s| s.to_string());
    }

    /// Java `setBigDiffCriterion(String)`.
    pub fn set_big_diff_criterion(&mut self, string: Option<&str>) {
        self.base.big_diff_criterion = string.map(|s| s.to_string());
    }

    /// Java `setExtraLargeRadius(String)`.
    pub fn set_extra_large_radius(&mut self, string: Option<&str>) {
        self.base.extra_large_radius = string.map(|s| s.to_string());
    }

    /// Java `setMaximumRadius(String)`.
    pub fn set_maximum_radius(&mut self, string: Option<&str>) {
        self.base.maximum_radius = string.map(|s| s.to_string());
    }

    /// Java `setAnnulusWidth(String)`.
    pub fn set_annulus_width(&mut self, string: Option<&str>) {
        self.base.annulus_width = string.map(|s| s.to_string());
    }

    /// Java `setPeakCriterion(String)`.
    pub fn set_peak_criterion(&mut self, string: Option<&str>) {
        self.base.peak_criterion = string.map(|s| s.to_string());
    }

    /// Java `setPointModel(String)`.
    pub fn set_point_model(&mut self, string: Option<&str>) {
        self.base.point_model = string.map(|s| s.to_string());
    }

    /// Java `setTrialMode(boolean)`.
    pub fn set_trial_mode(&mut self, b: bool) {
        self.base.trial_mode = b;
    }

    /// Java `setXyScanSize(String)`.
    pub fn set_xy_scan_size(&mut self, string: Option<&str>) {
        self.base.xy_scan_size = string.map(|s| s.to_string());
    }

    /// Java `setScanCriterion(String)`.
    pub fn set_scan_criterion(&mut self, string: Option<&str>) {
        self.base.scan_criterion = string.map(|s| s.to_string());
    }

    /// Java `getLineObjects()`.
    pub fn get_line_objects(&self) -> String {
        self.line_objects.to_string()
    }

    /// Java `getBoundaryObjects()`.
    pub fn get_boundary_objects(&self) -> String {
        self.boundary_objects.to_string()
    }

    /// Java `getAllSectionObjects()`.
    pub fn get_all_section_objects(&self) -> String {
        self.all_section_objects.to_string()
    }

    /// Java `setBetterRadius(double)`.
    pub fn set_better_radius(&mut self, input: f64) {
        self.base.better_radius.set_double(input);
    }

    /// Java `setExpandCircleIterations(Object)`: `input.toString()`.
    pub fn set_expand_circle_iterations(&mut self, input: &dyn std::fmt::Display) {
        self.base.expand_circle_iterations = Some(input.to_string());
    }

    /// Java `resetExpandCircleIterations()`.
    pub fn reset_expand_circle_iterations(&mut self) {
        self.base.expand_circle_iterations = Some(String::new());
    }
}

impl CommandParam for CCDEraserParam {
    /// Java `parseComScriptCommand(ComScriptCommand)`.  Get the parameters from the
    /// ComScriptCommand containing the ccderaser command and parameters.
    fn parse_com_script_command(
        &mut self,
        script_command: &ComScriptCommand,
    ) -> Result<(), ParseComScriptError> {
        // Check to be sure that it is a ccderaser command
        if script_command.get_command() != Some("ccderaser") {
            return Err(BadComScriptException::new("Not a ccderaser command").into());
        }

        let input_args = script_command.get_input_arguments();
        // Java reads `cmdLineArgs` and never uses it.
        let _cmd_line_args = script_command.get_command_line_args();
        if script_command.is_keyword_value_pairs() {
            let b = &mut self.base;
            b.find_peaks = script_command.has_keyword(Some(FIND_PEAKS_KEY))?;
            b.peak_criterion = script_command.get_value(Some(PEAK_CRITERION_KEY))?;
            b.diff_criterion = script_command.get_value(Some(DIFF_CRITERION_KEY))?;
            b.grow_criterion = script_command.get_value(Some(GROW_CRITERION_KEY))?;
            b.scan_criterion = script_command.get_value(Some(SCAN_CRITERION_KEY))?;
            b.maximum_radius = script_command.get_value(Some(MAXIMUM_RADIUS_KEY))?;
            b.annulus_width = script_command.get_value(Some(ANNULUS_WIDTH_KEY))?;
            b.expand_circle_iterations =
                script_command.get_value(Some(EXPAND_CIRCLE_ITERATIONS_KEY))?;
            b.better_radius.parse(script_command)?;
            b.xy_scan_size = script_command.get_value(Some(X_Y_SCAN_SIZE_KEY))?;
            b.edge_exclusion = script_command.get_value(Some(EDGE_EXCLUSION_WIDTH_KEY))?;
            b.point_model = script_command.get_value(Some("PointModel"))?;
            b.trial_mode = script_command.has_keyword(Some(TRIAL_MODE_KEY))?;

            b.input_file = script_command.get_value(Some(INPUT_FILE_KEY))?;
            b.output_file = script_command.get_value(Some(OUTPUT_FILE_KEY))?;
            b.output_file_type = None;
            b.model_file = script_command.get_value(Some("ModelFile"))?;
            b.global_replacement_list = script_command.get_value(Some(ALL_SECTION_OBJECTS_KEY))?;
            b.local_replacement_list = script_command.get_value(Some(LINE_OBJECTS_KEY))?;
            b.boundary_replacement_list = script_command.get_value(Some(BOUNDARY_OBJECTS_KEY))?;
            b.border_pixels = script_command.get_value(Some(BORDER_SIZE_KEY))?;
            b.polynomial_order = script_command.get_value(Some(POLYNOMIAL_ORDER_KEY))?;
            b.include_adjacent_points = !script_command.has_keyword(Some("ExcludeAdjacent"))?;
            b.giant_criterion = script_command.get_value(Some(GIANT_CRITERION_KEY))?;
            b.big_diff_criterion = script_command.get_value(Some(BIG_DIFF_CRITERION_KEY))?;
            b.extra_large_radius = script_command.get_value(Some(EXTRA_LARGE_RADIUS_KEY))?;
            // handle out-of-date parameters
            b.outer_radius = script_command.get_value(Some("OuterRadius"))?;
            // CCDEraserParam.java:118: a null outerRadius is a NullPointerException
            // (caught by ComScriptUtil as a parse failure).  Fixed: null is empty.
            if self.base.outer_radius.as_deref().unwrap_or("") != "" {
                // `Double.parseDouble` NumberFormatException, caught by the source's
                // caller as a parse failure.
                self.convert_outer_radius()
                    .map_err(ParseComScriptError::NumberFormat)?;
            }
            self.line_objects.parse(script_command)?;
            self.boundary_objects.parse(script_command)?;
            self.all_section_objects.parse(script_command)?;
        } else {
            // `inputArgs[k].getArgument()`: too few arguments is an
            // ArrayIndexOutOfBoundsException, which the source's caller
            // (ComScriptUtil.initialize) catches as a parse failure; returned as an
            // error here.
            let mut args: Vec<Option<String>> = Vec::with_capacity(8);
            for k in 0..8 {
                match input_args.get(k) {
                    Some(arg) => args.push(arg.borrow().get_argument().map(|s| s.to_string())),
                    None => {
                        return Err(BadComScriptException::new(&format!(
                            "java.lang.ArrayIndexOutOfBoundsException: {k}"
                        ))
                        .into());
                    }
                }
            }
            let b = &mut self.base;
            b.input_file = args[0].clone();
            b.output_file = args[1].clone();
            b.output_file_type = None;
            b.model_file = args[2].clone();
            b.global_replacement_list = args[3].clone();
            b.local_replacement_list = args[4].clone();
            b.border_pixels = args[5].clone();
            b.polynomial_order = args[6].clone();
            // `matches("\\s*1\\s*")`; a null argument (NullPointerException in Java)
            // does not match here.
            b.include_adjacent_points = args[7].as_deref().is_some_and(|arg| {
                arg.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r'))
                    == "1"
            });

            // Turn on the automatic mode with the defaults from the new com script
            b.find_peaks = true;
            let junk =
                crate::imod::etomo::util::utilities::remove_extension(b.input_file.as_deref());
            let dataset_name = junk.unwrap_or_else(|| "null".to_string());
            let ext = crate::imod::etomo::util::utilities::get_extension(b.input_file.as_deref());
            b.output_file = Some(format!(
                "{}_fixed{}",
                dataset_name,
                match &ext {
                    Some(ext) => format!("{}{}", extension::EXTENSION_DIVIDER, ext),
                    None => String::new(),
                }
            ));
            b.output_file_type = Some(Arc::clone(&file_type::CLASS.fixed_xrays_stack));
            b.peak_criterion = Some("10.0".to_string());
            b.diff_criterion = Some("8.0".to_string());
            b.grow_criterion = Some("4.0".to_string());
            b.edge_exclusion = Some("4".to_string());
            b.point_model = Some(format!("{dataset_name}_peak.mod"));
            b.maximum_radius = Some("2.1".to_string());
            b.annulus_width = Some("2.0".to_string());
            b.xy_scan_size = Some("100".to_string());
            b.scan_criterion = Some("3.0".to_string());
        }
        Ok(())
    }

    /// Java `updateComScriptCommand(ComScriptCommand)`.
    fn update_com_script_command(
        &self,
        script_command: &mut ComScriptCommand,
    ) -> Result<(), BadComScriptException> {
        // Check to be sure that it is a ccderaser xommand
        if script_command.get_command() != Some("ccderaser") {
            return Err(BadComScriptException::new("Not a ccderaser command"));
        }
        // Switch to keyword/value pairs
        script_command.use_keyword_value();

        let b = &self.base;
        script_command.set_value(Some(INPUT_FILE_KEY), b.input_file.as_deref());

        // Each `!x.equals("")` below: null counts as empty (see the module docs).
        if b.output_file.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(OUTPUT_FILE_KEY), b.output_file.as_deref());
        } else {
            script_command.delete_key(Some(OUTPUT_FILE_KEY));
        }

        if b.find_peaks {
            script_command.set_value(Some(FIND_PEAKS_KEY), Some(""));
        } else {
            script_command.delete_key(Some(FIND_PEAKS_KEY));
        }

        if b.peak_criterion.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(PEAK_CRITERION_KEY), b.peak_criterion.as_deref());
        } else {
            script_command.delete_key(Some(PEAK_CRITERION_KEY));
        }
        if b.diff_criterion.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(DIFF_CRITERION_KEY), b.diff_criterion.as_deref());
        } else {
            script_command.delete_key(Some(DIFF_CRITERION_KEY));
        }
        if b.grow_criterion.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(GROW_CRITERION_KEY), b.grow_criterion.as_deref());
        } else {
            script_command.delete_key(Some(GROW_CRITERION_KEY));
        }
        if b.scan_criterion.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(SCAN_CRITERION_KEY), b.scan_criterion.as_deref());
        } else {
            script_command.delete_key(Some(SCAN_CRITERION_KEY));
        }
        if b.maximum_radius.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(MAXIMUM_RADIUS_KEY), b.maximum_radius.as_deref());
        } else {
            script_command.delete_key(Some(MAXIMUM_RADIUS_KEY));
        }
        if b.annulus_width.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(ANNULUS_WIDTH_KEY), b.annulus_width.as_deref());
        } else {
            script_command.delete_key(Some(ANNULUS_WIDTH_KEY));
        }
        if b.expand_circle_iterations.as_deref().unwrap_or("") != "" {
            script_command.set_value(
                Some(EXPAND_CIRCLE_ITERATIONS_KEY),
                b.expand_circle_iterations.as_deref(),
            );
        } else {
            script_command.delete_key(Some(EXPAND_CIRCLE_ITERATIONS_KEY));
        }
        b.better_radius.update_com_script(script_command);
        if b.xy_scan_size.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(X_Y_SCAN_SIZE_KEY), b.xy_scan_size.as_deref());
        } else {
            script_command.delete_key(Some(X_Y_SCAN_SIZE_KEY));
        }
        if b.edge_exclusion.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(EDGE_EXCLUSION_WIDTH_KEY), b.edge_exclusion.as_deref());
        } else {
            script_command.delete_key(Some(EDGE_EXCLUSION_WIDTH_KEY));
        }
        if b.point_model.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some("PointModel"), b.point_model.as_deref());
        } else {
            script_command.delete_key(Some("PointModel"));
        }
        if b.model_file.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some("ModelFile"), b.model_file.as_deref());
        } else {
            script_command.delete_key(Some("ModelFile"));
        }
        if b.global_replacement_list.as_deref().unwrap_or("") != "" {
            script_command.set_value(
                Some(ALL_SECTION_OBJECTS_KEY),
                b.global_replacement_list.as_deref(),
            );
        } else {
            script_command.delete_key(Some(ALL_SECTION_OBJECTS_KEY));
        }
        if b.local_replacement_list.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(LINE_OBJECTS_KEY), b.local_replacement_list.as_deref());
        } else {
            script_command.delete_key(Some(LINE_OBJECTS_KEY));
        }
        if b.boundary_replacement_list.as_deref().unwrap_or("") != "" {
            script_command.set_value(
                Some(BOUNDARY_OBJECTS_KEY),
                b.boundary_replacement_list.as_deref(),
            );
        } else {
            script_command.delete_key(Some(BOUNDARY_OBJECTS_KEY));
        }
        if b.border_pixels.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(BORDER_SIZE_KEY), b.border_pixels.as_deref());
        } else {
            script_command.delete_key(Some(BORDER_SIZE_KEY));
        }
        if b.polynomial_order.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(POLYNOMIAL_ORDER_KEY), b.polynomial_order.as_deref());
        } else {
            script_command.delete_key(Some(POLYNOMIAL_ORDER_KEY));
        }

        if b.include_adjacent_points {
            script_command.delete_key(Some("ExcludeAdjacent"));
        } else {
            script_command.set_value(Some("ExcludeAdjacent"), Some(""));
        }

        if b.trial_mode {
            script_command.set_value(Some(TRIAL_MODE_KEY), Some(""));
        } else {
            script_command.delete_key(Some(TRIAL_MODE_KEY));
        }
        if b.giant_criterion.as_deref().unwrap_or("") != "" {
            script_command.set_value(Some(GIANT_CRITERION_KEY), b.giant_criterion.as_deref());
        } else {
            script_command.delete_key(Some(GIANT_CRITERION_KEY));
        }
        if b.big_diff_criterion.as_deref().unwrap_or("") != "" {
            script_command.set_value(
                Some(BIG_DIFF_CRITERION_KEY),
                b.big_diff_criterion.as_deref(),
            );
        } else {
            script_command.delete_key(Some(BIG_DIFF_CRITERION_KEY));
        }
        if b.extra_large_radius.as_deref().unwrap_or("") != "" {
            script_command.set_value(
                Some(EXTRA_LARGE_RADIUS_KEY),
                b.extra_large_radius.as_deref(),
            );
        } else {
            script_command.delete_key(Some(EXTRA_LARGE_RADIUS_KEY));
        }
        // remove out-of-date parameters
        if b.outer_radius.as_deref().unwrap_or("") != "" {
            script_command.delete_key(Some("OuterRadius"));
        }

        // Always add these when erasing X-rays
        if self.manager.get_const_meta_data().get_view_type() == ViewType::Montage
            && (self.mode == Some(Mode::XRays) || self.mode == Some(Mode::XRaysTrial))
        {
            let dataset_name = self
                .manager
                .get_base_meta_data()
                .and_then(|meta_data| meta_data.get_dataset_name())
                .unwrap_or_else(|| "null".to_string());
            script_command.set_value(
                Some("PieceListFile"),
                Some(&format!(
                    "{}{}.pl",
                    dataset_name,
                    self.axis_id.get_extension()
                )),
            );
            script_command.set_value(
                Some("OverlapsForModel"),
                Some(&format!(
                    "{},{}",
                    utilities::MONTAGE_SEPARATION,
                    utilities::MONTAGE_SEPARATION
                )),
            );
        }
        Ok(())
    }

    /// Java `initializeDefaults()`.
    fn initialize_defaults(&mut self) {}
}

impl Command for CCDEraserParam {
    /// Java `getCommandArray()`.  Creates the command, if it doesn't exist, and
    /// returns command array.
    fn get_command_array(&self) -> Option<Vec<String>> {
        if self.command_array.is_none() {
            if self.mode == Some(Mode::Beads) {
                return Some(vec![ProcessName::GOLD_ERASER.get_comscript(self.axis_id)]);
            } else {
                return Some(vec![ProcessName::ERASER.get_comscript(self.axis_id)]);
            }
        }
        self.command_array.clone()
    }

    /// Java `getAxisID()`.
    fn get_axis_id(&self) -> AxisID {
        self.axis_id
    }

    /// Java `getCommand()`.
    fn get_command(&self) -> Option<String> {
        if self.mode == Some(Mode::Beads) {
            return Some(ProcessName::GOLD_ERASER.get_comscript(self.axis_id));
        }
        Some(ProcessName::ERASER.get_comscript(self.axis_id))
    }

    /// Java `getCommandInputFile()`.
    fn get_command_input_file(&self) -> Option<PathBuf> {
        // `new File(manager.getPropertyUserDir(), input_file)`; a null child is a
        // NullPointerException in Java and `None` here.
        let name = self.base.input_file.as_deref()?;
        match self.manager.get_property_user_dir() {
            Some(dir) => Some(PathBuf::from(dir).join(name)),
            None => Some(PathBuf::from(name)),
        }
    }

    /// Java `getCommandLine()`.
    fn get_command_line(&self) -> Option<String> {
        self.get_command()
    }

    /// Java `getCommandMode()`.
    fn get_command_mode(&self) -> Option<&dyn CommandMode> {
        self.mode.as_ref().map(|mode| mode as &dyn CommandMode)
    }

    /// Java `getCommandName()`.
    fn get_command_name(&self) -> Option<String> {
        if self.mode == Some(Mode::Beads) {
            return Some(ProcessName::GOLD_ERASER.to_string());
        }
        Some(ProcessName::ERASER.to_string())
    }

    /// Java `getCommandOutputFile()`.
    fn get_command_output_file(&self) -> Option<PathBuf> {
        // `new File(manager.getPropertyUserDir(), output_file)`; a null child is a
        // NullPointerException in Java and `None` here.
        let name = self.base.output_file.as_deref()?;
        match self.manager.get_property_user_dir() {
            Some(dir) => Some(PathBuf::from(dir).join(name)),
            None => Some(PathBuf::from(name)),
        }
    }

    /// Java `getOutputImageFileType()` (deprecated 3/15/2019).
    fn get_output_image_file_type(&self) -> Option<Arc<FileType>> {
        if let Some(output_file_type) = &self.base.output_file_type {
            return Some(Arc::clone(output_file_type));
        }
        FileType::get_instance_from_manager(
            Some(self.manager as &'static dyn BaseManager),
            self.axis_id,
            true,
            true,
            self.base.output_file.as_deref(),
        )
    }

    /// Java `getOutputImageFileKey()`: the `FileType` is its own `FileKey`.
    fn get_output_image_file_key(&self) -> Option<FileKey> {
        self.base
            .output_file_type
            .as_ref()
            .map(|file_type| (**file_type).clone())
    }

    /// Java `getOutputImageFileType2()` (deprecated 3/15/2019).
    fn get_output_image_file_type2(&self) -> Option<Arc<FileType>> {
        None
    }

    /// Java `getOutputImageFileKey2()`.
    fn get_output_image_file_key2(&self) -> Option<FileKey> {
        None
    }

    /// Java `getProcessName()`.
    fn get_process_name(&self) -> Option<ProcessName> {
        if self.mode == Some(Mode::Beads) {
            return Some(ProcessName::GOLD_ERASER);
        }
        Some(ProcessName::ERASER)
    }

    /// Java `getSubcommandDetails()`.
    fn get_subcommand_details(&self) -> Option<&dyn CommandDetails> {
        None
    }

    /// Java `getSubcommandProcessName()`.
    fn get_subcommand_process_name(&self) -> Option<String> {
        None
    }

    /// Java `isMessageReporter()`.
    fn is_message_reporter(&self) -> bool {
        false
    }
}
