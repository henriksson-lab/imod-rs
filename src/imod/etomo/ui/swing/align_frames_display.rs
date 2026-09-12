//! `IMOD/Etomo/src/etomo/ui/swing/AlignFramesDisplay.java`.
//!
//! The Java interface overloads `getParameters` on two unrelated command
//! parameter types.  Rust has no method overloading, so the systematic
//! snake-case translation names those operations after their parameter type.
//! Concrete command-parameter formatting remains owned by their corresponding
//! `comscript` source units.

#![allow(dead_code)]

use crate::imod::etomo::comscript::fortran_input_syntax_exception::FortranInputSyntaxException;
use crate::imod::etomo::local_arguments::LocalArguments;

use super::labeled_text_field::FieldValidationFailedException;
use super::process_display::ProcessDisplay;

/// Java `AlignFramesDisplay extends ProcessDisplay`.
pub trait AlignFramesDisplay: ProcessDisplay {
    /// Java `getParameters(AlignFramesParam)`.
    fn get_align_frames_parameters(
        &self,
        param: &mut Self::AlignFramesParam,
    ) -> Result<bool, FortranInputSyntaxException>;

    /// Java `getParameters(SortTiltFramesParam)`.
    fn get_sort_tilt_frames_parameters(
        &self,
        param: &mut Self::SortTiltFramesParam,
    ) -> Result<bool, FieldValidationFailedException>;

    /// Java `setParameters(AlignFramesParam)`.
    fn set_align_frames_parameters(&mut self, param: &Self::AlignFramesParam);

    /// Java `AlignFramesParam` at this interface boundary.
    type AlignFramesParam;
    /// Java `SortTiltFramesParam` at this interface boundary.
    type SortTiltFramesParam;

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

#[cfg(test)]
mod tests {
    use super::*;

    struct Display;
    impl ProcessDisplay for Display {}
    impl AlignFramesDisplay for Display {
        type AlignFramesParam = String;
        type SortTiltFramesParam = String;

        fn get_align_frames_parameters(
            &self,
            param: &mut String,
        ) -> Result<bool, FortranInputSyntaxException> {
            *param = "alignframes".into();
            Ok(true)
        }
        fn get_sort_tilt_frames_parameters(
            &self,
            param: &mut String,
        ) -> Result<bool, FieldValidationFailedException> {
            *param = "sorttiltframes".into();
            Ok(true)
        }
        fn set_align_frames_parameters(&mut self, _param: &String) {}
        fn is_list_of_input_files_selected(&self) -> bool {
            false
        }
        fn is_selected_files_selected(&self) -> bool {
            false
        }
        fn is_angles_in_filenames_selected(&self) -> bool {
            false
        }
        fn is_corresponding_stack_selected(&self) -> bool {
            false
        }
        fn is_tilt_angle_file_selected(&self) -> bool {
            false
        }
        fn is_fixed_total_dose_selected(&self) -> bool {
            false
        }
        fn is_do_dose_weighting_selected(&self) -> bool {
            false
        }
        fn get_text_area_input_files(&self) -> String {
            String::new()
        }
        fn get_rootname_output_files(&self) -> String {
            String::new()
        }
        fn get_output_image_file_name(&self) -> String {
            String::new()
        }
        fn setup_local_arguments(&self) -> LocalArguments {
            LocalArguments::default()
        }
        fn is_metadata_file_selected(&self) -> bool {
            false
        }
        fn get_new_mdoc_file_name(&self) -> String {
            String::new()
        }
    }

    #[test]
    fn source_overloads_remain_separate_parameter_operations() {
        let display = Display;
        let mut align = String::new();
        let mut sort = String::new();
        assert!(display.get_align_frames_parameters(&mut align).unwrap());
        assert!(display.get_sort_tilt_frames_parameters(&mut sort).unwrap());
        assert_eq!(align, "alignframes");
        assert_eq!(sort, "sorttiltframes");
    }

    #[test]
    fn align_frames_panel_exposes_the_display_contract() {
        use crate::imod::etomo::r#type::axis_id::AxisID;
        use crate::imod::etomo::ui::swing::align_frames_panel::AlignFramesPanel;

        let mut panel = AlignFramesPanel::new(AxisID::Only, ".");
        panel.output_image_file = "frames_ali.mrc".into();
        panel.local_arguments_dir = "frames".into();
        let arguments = AlignFramesDisplay::setup_local_arguments(&panel);
        assert_eq!(
            arguments.arguments.get_raw_image_stack(),
            Some("frames_ali.mrc")
        );
        assert_eq!(
            arguments.arguments.get_dir(),
            Some(std::path::Path::new("frames"))
        );
        let mut param =
            crate::imod::etomo::ui::swing::align_frames_panel::AlignFramesPanelParameters::default(
            );
        assert!(AlignFramesDisplay::get_align_frames_parameters(&panel, &mut param).unwrap());
    }
}
