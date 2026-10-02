//! `IMOD/Etomo/src/etomo/ui/swing/AlignFramesDisplay.java`.
//!
//! Java `public interface AlignFramesDisplay extends ProcessDisplay`, implemented
//! by `AlignFramesPanel`.  The two `getParameters` overloads take the
//! parameter-type suffix (`get_parameters_align_frames_param`,
//! `get_parameters_sort_tilt_frames_param`).  Methods take `&self`: the display
//! is an event-dispatch-thread object (`Rc`) the manager calls back into.

use crate::imod::etomo::comscript::align_frames_param::AlignFramesParam;
use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::comscript::sort_tilt_frames_param::SortTiltFramesParam;
use crate::imod::etomo::local_arguments::LocalArguments;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

use super::process_display::ProcessDisplay;

/// Java `public interface AlignFramesDisplay extends ProcessDisplay`.
pub trait AlignFramesDisplay: ProcessDisplay {
    /// Java `getParameters(AlignFramesParam) throws FortranInputSyntaxException`.
    fn get_parameters_align_frames_param(
        &self,
        param: &mut AlignFramesParam,
    ) -> Result<bool, FortranInputSyntaxException>;

    /// Java `getParameters(SortTiltFramesParam) throws FieldValidationFailedException`.
    fn get_parameters_sort_tilt_frames_param(
        &self,
        param: &mut SortTiltFramesParam,
    ) -> Result<bool, FieldValidationFailedException>;

    /// Java `setParameters(AlignFramesParam)`.
    fn set_parameters(&self, param: &AlignFramesParam);

    /// Java `isListOfInputFilesSelected()`.
    fn is_list_of_input_files_selected(&self) -> bool;

    /// Java `isSelectedFilesSelected()`.
    fn is_selected_files_selected(&self) -> bool;

    /// Java `isAnglesInFilenamesSelected()`.
    fn is_angles_in_filenames_selected(&self) -> bool;

    /// Java `isCorrespondingStackSelected()`.
    fn is_corresponding_stack_selected(&self) -> bool;

    /// Java `isTiltAngleFileSelected()`.
    fn is_tilt_angle_file_selected(&self) -> bool;

    /// Java `isFixedTotalDoseSelected()`.
    fn is_fixed_total_dose_selected(&self) -> bool;

    /// Java `isDoDoseWeightingSelected()`.
    fn is_do_dose_weighting_selected(&self) -> bool;

    /// Java `getTextAreaInputFiles()`.
    fn get_text_area_input_files(&self) -> String;

    /// Java `getRootnameOutputFiles()`.
    fn get_rootname_output_files(&self) -> String;

    /// Java `getOutputImageFileName()`.
    fn get_output_image_file_name(&self) -> String;

    /// Java `setupLocalArguments()`.
    fn setup_local_arguments(&self) -> LocalArguments;

    /// Java `isMetadataFileSelected()`.
    fn is_metadata_file_selected(&self) -> bool;

    /// Java `getNewMdocFileName()`.
    fn get_new_mdoc_file_name(&self) -> String;
}
