//! `IMOD/Etomo/src/etomo/comscript/ConstSolvematchshiftParam.java`.
//!
//! A Java class with package-private state (not an interface), so a struct here;
//! `SolvematchshiftParam` holds it as its `base` through `Deref`/`DerefMut`.

use super::fortran_input_string::FortranInputString;
use super::string_list::StringList;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ConstSolvematchshiftParam`.
#[derive(Clone, Debug)]
pub struct ConstSolvematchshiftParam {
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
    /// Java `outputTransformationFile`.
    pub(crate) output_transformation_file: Option<String>,
}

impl ConstSolvematchshiftParam {
    /// Java `ConstSolvematchshiftParam()`.
    pub(crate) fn new() -> ConstSolvematchshiftParam {
        ConstSolvematchshiftParam {
            match_b_to_a: true,
            to_fiducial_coordinates_file: None,
            from_fiducial_coordinates_file: None,
            fiducial_match_list_a: StringList::new_with_n_elements(0),
            fiducial_match_list_b: StringList::new_with_n_elements(0),
            x_axist_tilt: FortranInputString::new(2),
            residual_threshold: 0.0,
            n_surfaces: 0,
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

    /// Java `getXAxistTilt`.
    pub fn get_x_axist_tilt(&self) -> &FortranInputString {
        &self.x_axist_tilt
    }
}
