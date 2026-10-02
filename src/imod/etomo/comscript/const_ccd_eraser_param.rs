//! `IMOD/Etomo/src/etomo/comscript/ConstCCDEraserParam.java`.
//!
//! Const part of the parameters for the ccderaser program.  Java's
//! `ConstCCDEraserParam` is a *class*, the superclass of `CCDEraserParam` holding the
//! package-private fields; it is translated as a struct that `CCDEraserParam` holds as
//! its `base` field and reaches through `Deref`/`DerefMut`, as
//! `type/script_parameter.rs` does for its superclass.
//!
//! The Java `String` fields can be null (an uninitialised `annulusWidth`, or
//! `ComScriptCommand.getValue` of a keyword present with no value), so they are
//! `Option<String>`.

use std::sync::Arc;

use crate::imod::etomo::r#type::const_etomo_number::{Type, java_lang_integer_parse_int};
use crate::imod::etomo::r#type::file_type::FileType;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

pub const ANNULUS_WIDTH_KEY: &str = "AnnulusWidth";
pub const INPUT_FILE_KEY: &str = "InputFile";
pub const OUTPUT_FILE_KEY: &str = "OutputFile";
pub const FIND_PEAKS_KEY: &str = "FindPeaks";
pub const PEAK_CRITERION_KEY: &str = "PeakCriterion";
pub const DIFF_CRITERION_KEY: &str = "DiffCriterion";
pub const GROW_CRITERION_KEY: &str = "GrowCriterion";
pub const SCAN_CRITERION_KEY: &str = "ScanCriterion";
pub const MAXIMUM_RADIUS_KEY: &str = "MaximumRadius";
pub const EXPAND_CIRCLE_ITERATIONS_KEY: &str = "ExpandCircleIterations";
pub const X_Y_SCAN_SIZE_KEY: &str = "XYScanSize";
pub const EDGE_EXCLUSION_WIDTH_KEY: &str = "EdgeExclusionWidth";
pub const LINE_OBJECTS_KEY: &str = "LineObjects";
pub const ALL_SECTION_OBJECTS_KEY: &str = "AllSectionObjects";
pub const BORDER_SIZE_KEY: &str = "BorderSize";
pub const POLYNOMIAL_ORDER_KEY: &str = "PolynomialOrder";
pub const TRIAL_MODE_KEY: &str = "TrialMode";
pub const BOUNDARY_OBJECTS_KEY: &str = "BoundaryObjects";
pub const MODEL_FILE_KEY: &str = "ModelFile";
pub const GIANT_CRITERION_KEY: &str = "GiantCriterion";
pub const BIG_DIFF_CRITERION_KEY: &str = "BigDiffCriterion";
pub const EXTRA_LARGE_RADIUS_KEY: &str = "ExtraLargeRadius";

/// Java `ConstCCDEraserParam`.
pub struct ConstCCDEraserParam {
    pub(crate) better_radius: ScriptParameter,
    pub(crate) input_file: Option<String>,
    pub(crate) output_file: Option<String>,
    /// Set to null when outputFile assigned to an unknown file.
    pub(crate) output_file_type: Option<Arc<FileType>>,
    pub(crate) find_peaks: bool,
    pub(crate) peak_criterion: Option<String>,
    pub(crate) diff_criterion: Option<String>,
    pub(crate) grow_criterion: Option<String>,
    pub(crate) scan_criterion: Option<String>,
    pub(crate) edge_exclusion: Option<String>,
    pub(crate) maximum_radius: Option<String>,
    pub(crate) expand_circle_iterations: Option<String>,
    /// Declared without an initialiser in the source, so it starts null.
    pub(crate) annulus_width: Option<String>,
    pub(crate) xy_scan_size: Option<String>,
    pub(crate) point_model: Option<String>,
    pub(crate) trial_mode: bool,
    pub(crate) model_file: Option<String>,
    pub(crate) global_replacement_list: Option<String>,
    pub(crate) local_replacement_list: Option<String>,
    pub(crate) boundary_replacement_list: Option<String>,
    pub(crate) border_pixels: Option<String>,
    pub(crate) polynomial_order: Option<String>,
    pub(crate) include_adjacent_points: bool,
    pub(crate) giant_criterion: Option<String>,
    pub(crate) big_diff_criterion: Option<String>,
    pub(crate) extra_large_radius: Option<String>,
    /// Out of date parameter, replaced by annulusWidth.
    pub(crate) outer_radius: Option<String>,
}

impl ConstCCDEraserParam {
    /// The Java field initialisers (the class has only the implicit constructor).
    pub(crate) fn new() -> ConstCCDEraserParam {
        let empty = || Some(String::new());
        ConstCCDEraserParam {
            better_radius: ScriptParameter::new_with_type_and_name(Type::Double, "BetterRadius"),
            input_file: empty(),
            output_file: empty(),
            output_file_type: None,
            find_peaks: false,
            peak_criterion: empty(),
            diff_criterion: empty(),
            grow_criterion: empty(),
            scan_criterion: empty(),
            edge_exclusion: empty(),
            maximum_radius: empty(),
            expand_circle_iterations: empty(),
            annulus_width: None,
            xy_scan_size: empty(),
            point_model: empty(),
            trial_mode: false,
            model_file: empty(),
            global_replacement_list: empty(),
            local_replacement_list: empty(),
            boundary_replacement_list: empty(),
            border_pixels: empty(),
            polynomial_order: empty(),
            include_adjacent_points: true,
            giant_criterion: empty(),
            big_diff_criterion: empty(),
            extra_large_radius: empty(),
            outer_radius: empty(),
        }
    }

    /// Java `isValid()`.
    pub fn is_valid(&self) -> bool {
        let mut valid = true;

        // Check to see if any of the integer parameters do not parse as integers
        // or there range is in appropriate
        // (`Integer.parseInt(null)` throws NumberFormatException, caught here.)
        match self
            .border_pixels
            .as_deref()
            .ok_or_else(|| "null".to_string())
            .and_then(java_lang_integer_parse_int)
        {
            Ok(int_border_pixels) => {
                if int_border_pixels < 0 {
                    valid = false;
                }
            }
            Err(_) => {
                valid = false;
            }
        }
        valid
    }

    /// Java `getInputFile()`.
    pub fn get_input_file(&self) -> Option<String> {
        self.input_file.clone()
    }

    /// Java `getOutputFile()`.
    pub fn get_output_file(&self) -> Option<String> {
        self.output_file.clone()
    }

    /// Java `getModelFile()`.
    pub fn get_model_file(&self) -> Option<String> {
        self.model_file.clone()
    }

    /// Java `getGlobalReplacementList()`.
    pub fn get_global_replacement_list(&self) -> Option<String> {
        self.global_replacement_list.clone()
    }

    /// Java `getlocalReplacementList()` (sic).
    pub fn getlocal_replacement_list(&self) -> Option<String> {
        self.local_replacement_list.clone()
    }

    /// Java `getBoundaryReplacementList()`.
    pub fn get_boundary_replacement_list(&self) -> Option<String> {
        self.boundary_replacement_list.clone()
    }

    /// Java `getBorderPixels()`.
    pub fn get_border_pixels(&self) -> Option<String> {
        self.border_pixels.clone()
    }

    /// Java `getPolynomialOrder()`.
    pub fn get_polynomial_order(&self) -> Option<String> {
        self.polynomial_order.clone()
    }

    /// Java `getIncludeAdjacentPoints()`.
    pub fn get_include_adjacent_points(&self) -> bool {
        self.include_adjacent_points
    }

    /// Java `getDiffCriterion()`.
    pub fn get_diff_criterion(&self) -> Option<String> {
        self.diff_criterion.clone()
    }

    /// Java `getEdgeExclusion()`.
    pub fn get_edge_exclusion(&self) -> Option<String> {
        self.edge_exclusion.clone()
    }

    /// Java `isFindPeaks()`.
    pub fn is_find_peaks(&self) -> bool {
        self.find_peaks
    }

    /// Java `getGrowCriterion()`.
    pub fn get_grow_criterion(&self) -> Option<String> {
        self.grow_criterion.clone()
    }

    /// Java `getGiantCriterion()`.
    pub fn get_giant_criterion(&self) -> Option<String> {
        self.giant_criterion.clone()
    }

    /// Java `getBigDiffCriterion()`.
    pub fn get_big_diff_criterion(&self) -> Option<String> {
        self.big_diff_criterion.clone()
    }

    /// Java `getExtraLargeRadius()`.
    pub fn get_extra_large_radius(&self) -> Option<String> {
        self.extra_large_radius.clone()
    }

    /// Java `getMaximumRadius()`.
    pub fn get_maximum_radius(&self) -> Option<String> {
        self.maximum_radius.clone()
    }

    /// Java `isExpandCircleIterationsSet()`.  The source compares references
    /// (`expandCircleIterations != ""`); every "unset" assignment in the source is the
    /// interned `""` literal (or `ComScriptCommand.getValue`'s absent-keyword `""`), so
    /// this is "not the empty string", and null counts as set, as it does in Java.
    pub fn is_expand_circle_iterations_set(&self) -> bool {
        self.expand_circle_iterations.as_deref() != Some("")
    }

    /// Java `isBetterRadiusSet()`.
    pub fn is_better_radius_set(&self) -> bool {
        !self.better_radius.is_null()
    }

    /// Java `getBetterRadius()`.
    pub fn get_better_radius(&self) -> f64 {
        self.better_radius.get_double()
    }

    /// Java `getExpandCircleIterations()`.
    pub fn get_expand_circle_iterations(&self) -> Option<String> {
        self.expand_circle_iterations.clone()
    }

    /// Java `getAnnulusWidth()`.
    pub fn get_annulus_width(&self) -> Option<String> {
        self.annulus_width.clone()
    }

    /// Java `getPeakCriterion()`.
    pub fn get_peak_criterion(&self) -> Option<String> {
        self.peak_criterion.clone()
    }

    /// Java `getPointModel()`.
    pub fn get_point_model(&self) -> Option<String> {
        self.point_model.clone()
    }

    /// Java `isTrialMode()`.
    pub fn is_trial_mode(&self) -> bool {
        self.trial_mode
    }

    /// Java `getXyScanSize()`.
    pub fn get_xy_scan_size(&self) -> Option<String> {
        self.xy_scan_size.clone()
    }

    /// Java `getScanCriterion()`.
    pub fn get_scan_criterion(&self) -> Option<String> {
        self.scan_criterion.clone()
    }
}
