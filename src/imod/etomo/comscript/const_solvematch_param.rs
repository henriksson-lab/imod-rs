//! `IMOD/Etomo/src/etomo/comscript/ConstSolvematchParam.java`.
//!
//! Java's `ConstSolvematchParam` is a class with package-private state, not an
//! interface, so it is a struct here; `SolvematchParam` holds it as its `base`
//! and reaches it through `Deref`/`DerefMut`.
//!
//! Implementation note: this is not derived from either ConstSolvematchmodParam
//! ConstSolvematchshiftParam because that old functionality should be able to be
//! absorbed by this class.

use super::fortran_input_string::FortranInputString;
use super::string_list::StringList;
use crate::imod::etomo::r#type::const_etomo_number::{ConstEtomoNumber, Type};
use crate::imod::etomo::r#type::fiducial_match::FiducialMatch;
use crate::imod::etomo::r#type::script_parameter::ScriptParameter;

/// Java `OUTPUT_FILE`.
pub const OUTPUT_FILE: &str = "OutputFile";
// The A and B in the keywords are misleading and only correct if we are
// matching from A to B. Their association swaps if the matching direction is
// from B to A. That is why static strings are called TO and FROM, they
// should make the code in this class easier to read.
/// Java `TO_FIDUCIAL_FILE`.
pub const TO_FIDUCIAL_FILE: &str = "AFiducialFile";
/// Java `FROM_FIDUCIAL_FILE`.
pub const FROM_FIDUCIAL_FILE: &str = "BFiducialFile";
/// Java `TO_CORRESPONDENCE_LIST`.
pub const TO_CORRESPONDENCE_LIST: &str = "ACorrespondenceList";
/// Java `FROM_CORRESPONDENCE_LIST`.
pub const FROM_CORRESPONDENCE_LIST: &str = "BCorrespondenceList";
/// Java `SCALE_FACTORS`.
pub const SCALE_FACTORS: &str = "ScaleFactors";
/// Java `XAXIS_TILTS`.
pub const XAXIS_TILTS: &str = "XAxisTilts";
/// Java `MAXIMUM_RESIDUAL`.
pub const MAXIMUM_RESIDUAL: &str = "MaximumResidual";
/// Java `TO_TOMOGRAM_OR_SIZE_XYZ`.
pub const TO_TOMOGRAM_OR_SIZE_XYZ: &str = "ATomogramOrSizeXYZ";
/// Java `FROM_TOMOGRAM_OR_SIZE_XYZ`.
pub const FROM_TOMOGRAM_OR_SIZE_XYZ: &str = "BTomogramOrSizeXYZ";
/// Java `SURFACE_OR_USE_MODELS`.
pub const SURFACE_OR_USE_MODELS: &str = "SurfacesOrUseModels";
/// Java `TO_MATCHING_MODEL`.
pub const TO_MATCHING_MODEL: &str = "AMatchingModel";
/// Java `FROM_MATCHING_MODEL`.
pub const FROM_MATCHING_MODEL: &str = "BMatchingModel";
/// Java `MATCHING_A_TO_B`.
pub const MATCHING_A_TO_B: &str = "MatchingAtoB";
/// Java `TRANSFER_COORDINATE_FILE`.
pub const TRANSFER_COORDINATE_FILE: &str = "TransferCoordinateFile";
/// Java `A_FIDUCIAL_MODEL`.
pub const A_FIDUCIAL_MODEL: &str = "AFiducialModel";
/// Java `B_FIDUCIAL_MODEL`.
pub const B_FIDUCIAL_MODEL: &str = "BFiducialModel";
/// Java `USE_POINTS`.
pub const USE_POINTS: &str = "UsePoints";
/// Java `CENTER_SHIFT_LIMIT_KEY`.
pub const CENTER_SHIFT_LIMIT_KEY: &str = "CenterShiftLimit";

/// Java `USE_MODEL_ONLY_OPTION`.
pub const USE_MODEL_ONLY_OPTION: i32 = -2;
/// Java `ONE_SIDE_INVERTED_OPTION`.
pub const ONE_SIDE_INVERTED_OPTION: i32 = -1;
/// Java `USE_MODEL_OPTION`.
pub const USE_MODEL_OPTION: i32 = 0;
/// Java `ONE_SIDE_OPTION`.
pub const ONE_SIDE_OPTION: i32 = 1;
/// Java `BOTH_SIDES_OPTION`.
pub const BOTH_SIDES_OPTION: i32 = 2;

/// Java `ConstSolvematchParam`.
#[derive(Clone, Debug)]
pub struct ConstSolvematchParam {
    /// Java `matchBToA`.
    pub(crate) match_b_to_a: bool,
    /// Java `outputFile`.
    pub(crate) output_file: Option<String>,
    /// Java `toFiducialFile`.
    pub(crate) to_fiducial_file: Option<String>,
    /// Java `fromFiducialFile`.
    pub(crate) from_fiducial_file: Option<String>,
    /// Java `toCorrespondenceList`.
    pub(crate) to_correspondence_list: StringList,
    /// Java `fromCorrespondenceList`.
    pub(crate) from_correspondence_list: StringList,
    /// Java `xAxistTilt`.
    pub(crate) x_axist_tilt: FortranInputString,
    /// Java `surfacesOrModel`.
    pub(crate) surfaces_or_model: i32,
    /// Java `maximumResidual`.
    pub(crate) maximum_residual: f64,
    /// Java `toMatchingModel`.
    pub(crate) to_matching_model: Option<String>,
    /// Java `fromMatchingModel`.
    pub(crate) from_matching_model: Option<String>,
    /// Java `toTomogramOrSizeXYZ`.
    pub(crate) to_tomogram_or_size_xyz: Option<String>,
    /// Java `fromTomogramOrSizeXYZ`.
    pub(crate) from_tomogram_or_size_xyz: Option<String>,
    /// Java `scaleFactors`.
    pub(crate) scale_factors: FortranInputString,
    /// Java `transferCoordinateFile`.
    pub(crate) transfer_coordinate_file: Option<String>,
    /// Java `aFiducialModel`.
    pub(crate) a_fiducial_model: Option<String>,
    /// Java `bFiducialModel`.
    pub(crate) b_fiducial_model: Option<String>,
    /// Java `usePoints`.
    pub(crate) use_points: StringList,
    /// Java `centerShiftLimit`.
    pub(crate) center_shift_limit: ScriptParameter,
}

impl ConstSolvematchParam {
    /// Java's implicit `ConstSolvematchParam()`: the field initializers.
    pub(crate) fn new() -> ConstSolvematchParam {
        ConstSolvematchParam {
            match_b_to_a: true,
            output_file: Some(String::new()),
            to_fiducial_file: Some(String::new()),
            from_fiducial_file: Some(String::new()),
            to_correspondence_list: StringList::new_with_n_elements(0),
            from_correspondence_list: StringList::new_with_n_elements(0),
            x_axist_tilt: FortranInputString::new(2),
            surfaces_or_model: i32::MIN,
            maximum_residual: f64::NAN,
            to_matching_model: Some(String::new()),
            from_matching_model: Some(String::new()),
            to_tomogram_or_size_xyz: Some(String::new()),
            from_tomogram_or_size_xyz: Some(String::new()),
            scale_factors: FortranInputString::new(2),
            transfer_coordinate_file: None,
            a_fiducial_model: None,
            b_fiducial_model: None,
            use_points: StringList::new_with_n_elements(0),
            center_shift_limit: ScriptParameter::new_with_type_and_name(
                Type::Double,
                CENTER_SHIFT_LIMIT_KEY,
            ),
        }
    }

    /// Java `getXAxistTilt`.
    pub fn get_x_axist_tilt(&self) -> &FortranInputString {
        &self.x_axist_tilt
    }

    /// Java `getScaleFactors`.
    pub fn get_scale_factors(&self) -> &FortranInputString {
        &self.scale_factors
    }

    /// Java `getFromCorrespondenceList`.
    pub fn get_from_correspondence_list(&self) -> &StringList {
        &self.from_correspondence_list
    }

    /// Java `getFromFiducialFile`.
    pub fn get_from_fiducial_file(&self) -> Option<&str> {
        self.from_fiducial_file.as_deref()
    }

    /// Java `getFromMatchingModel`.
    pub fn get_from_matching_model(&self) -> Option<&str> {
        self.from_matching_model.as_deref()
    }

    /// Java `getFromTomogramOrSizeXYZ`.
    pub fn get_from_tomogram_or_size_xyz(&self) -> Option<&str> {
        self.from_tomogram_or_size_xyz.as_deref()
    }

    /// Java `isMatchBToA`.
    pub fn is_match_b_to_a(&self) -> bool {
        self.match_b_to_a
    }

    /// Java `getMaximumResidual`.
    pub fn get_maximum_residual(&self) -> f64 {
        self.maximum_residual
    }

    /// Java `getCenterShiftLimit`.
    pub fn get_center_shift_limit(&self) -> &ConstEtomoNumber {
        &self.center_shift_limit.base.base
    }

    /// Java `getSurfacesOrModel`.
    pub fn get_surfaces_or_model(&self) -> FiducialMatch {
        match self.surfaces_or_model {
            USE_MODEL_ONLY_OPTION => FiducialMatch::UseModelOnly,
            ONE_SIDE_INVERTED_OPTION => FiducialMatch::OneSideInverted,
            USE_MODEL_OPTION => FiducialMatch::UseModel,
            ONE_SIDE_OPTION => FiducialMatch::OneSide,
            BOTH_SIDES_OPTION => FiducialMatch::BothSides,
            _ => FiducialMatch::NotSet,
        }
    }

    /// Java `getOutputFile`.
    pub fn get_output_file(&self) -> Option<&str> {
        self.output_file.as_deref()
    }

    /// Java `getUsePoints`.
    pub fn get_use_points(&self) -> &StringList {
        &self.use_points
    }

    /// Java `getToCorrespondenceList`.
    pub fn get_to_correspondence_list(&self) -> &StringList {
        &self.to_correspondence_list
    }

    /// Java `getToFiducialFile`.
    pub fn get_to_fiducial_file(&self) -> Option<&str> {
        self.to_fiducial_file.as_deref()
    }

    /// Java `getToMatchingModel`.
    pub fn get_to_matching_model(&self) -> Option<&str> {
        self.to_matching_model.as_deref()
    }

    /// Java `getToTomogramOrSizeXYZ`.
    pub fn get_to_tomogram_or_size_xyz(&self) -> Option<&str> {
        self.to_tomogram_or_size_xyz.as_deref()
    }
}
