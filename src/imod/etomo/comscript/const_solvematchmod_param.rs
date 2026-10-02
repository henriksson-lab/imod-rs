//! `IMOD/Etomo/src/etomo/comscript/ConstSolvematchmodParam.java`.
//!
//! A Java class with package-private state (not an interface), so a struct here;
//! `SolvematchmodParam` holds it as its `base` through `Deref`/`DerefMut`.

use super::fortran_input_string::FortranInputString;
use super::string_list::StringList;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstSolvematchmodParam`.
#[derive(Clone, Debug)]
pub struct ConstSolvematchmodParam {
    /// Java `matchBToA`.
    pub(crate) match_b_to_a: bool,
    /// Java `toFiducialCoordinatesFile`.
    pub(crate) to_fiducial_coordinates_file: Option<String>,
    /// Java `fromFiducialCoordinatesFile`.
    pub(crate) from_fiducial_coordinates_file: Option<String>,
    /// Java `fiducialMatchListA`.
    pub(crate) fiducial_match_list_a: StringList,
    /// Java `fiducialMatchListB`.
    pub(crate) fiducial_match_list_b: StringList,
    /// Java `xAxistTilt`.
    pub(crate) x_axist_tilt: FortranInputString,
    /// Java `residualThreshold`.
    pub(crate) residual_threshold: f64,
    /// Java `nSurfaces`.
    pub(crate) n_surfaces: i32,
    /// Java `toReconstructionFile`.
    pub(crate) to_reconstruction_file: Option<String>,
    /// Java `toMatchingModel`.
    pub(crate) to_matching_model: Option<String>,
    /// Java `fromReconstructionFile`.
    pub(crate) from_reconstruction_file: Option<String>,
    /// Java `fromMatchingModel`.
    pub(crate) from_matching_model: Option<String>,
    /// Java `outputTransformationFile`.
    pub(crate) output_transformation_file: Option<String>,
}

impl ConstSolvematchmodParam {
    /// Java's implicit `ConstSolvematchmodParam()`: the field initializers.
    pub(crate) fn new() -> ConstSolvematchmodParam {
        ConstSolvematchmodParam {
            match_b_to_a: true,
            to_fiducial_coordinates_file: None,
            from_fiducial_coordinates_file: None,
            fiducial_match_list_a: StringList::new_with_n_elements(0),
            fiducial_match_list_b: StringList::new_with_n_elements(0),
            x_axist_tilt: FortranInputString::new(2),
            residual_threshold: 0.0,
            n_surfaces: 0,
            to_reconstruction_file: None,
            to_matching_model: None,
            from_reconstruction_file: None,
            from_matching_model: None,
            output_transformation_file: None,
        }
    }

    /// Java `getFiducialMatchListA`.
    pub fn get_fiducial_match_list_a(&self) -> &StringList {
        &self.fiducial_match_list_a
    }

    /// Java `getFiducialMatchListB`.
    pub fn get_fiducial_match_list_b(&self) -> &StringList {
        &self.fiducial_match_list_b
    }

    /// Java `getFromFiducialCoordinatesFile`.
    pub fn get_from_fiducial_coordinates_file(&self) -> Option<&str> {
        self.from_fiducial_coordinates_file.as_deref()
    }

    /// Java `getFromMatchingModel`.
    pub fn get_from_matching_model(&self) -> Option<&str> {
        self.from_matching_model.as_deref()
    }

    /// Java `getFromReconstructionFile`.
    pub fn get_from_reconstruction_file(&self) -> Option<&str> {
        self.from_reconstruction_file.as_deref()
    }

    /// Java `getNSurfaces`.
    pub fn get_n_surfaces(&self) -> i32 {
        self.n_surfaces
    }

    /// Java `getOutputTransformationFile`.
    pub fn get_output_transformation_file(&self) -> Option<&str> {
        self.output_transformation_file.as_deref()
    }

    /// Java `getResidualThreshold`.
    pub fn get_residual_threshold(&self) -> f64 {
        self.residual_threshold
    }

    /// Java `getToFiducialCoordinatesFile`.
    pub fn get_to_fiducial_coordinates_file(&self) -> Option<&str> {
        self.to_fiducial_coordinates_file.as_deref()
    }

    /// Java `getToMatchingModel`.
    pub fn get_to_matching_model(&self) -> Option<&str> {
        self.to_matching_model.as_deref()
    }

    /// Java `getToReconstructionFile`.
    pub fn get_to_reconstruction_file(&self) -> Option<&str> {
        self.to_reconstruction_file.as_deref()
    }

    /// Java `getXAxistTilt`.
    pub fn get_x_axist_tilt(&self) -> &FortranInputString {
        &self.x_axist_tilt
    }
}
