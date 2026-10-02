//! `IMOD/Etomo/src/etomo/ui/SetupReconInterface.java`.
//!
//! Copyright: Copyright 2012 - 2024 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Shape.**  The interface is used through `Rc<dyn SetupReconInterface>` /
//! `&dyn SetupReconInterface` (SetupReconUIHarness), so every method takes `&self`;
//! implementors keep their mutable state behind interior mutability.  A Java `String`
//! return is `Option<String>` (null), a Java `Integer` is `Option<i32>`, and a method
//! declaring `FieldValidationFailedException` returns `Result<_, it>`.
//!
//! The two Java implementors are `SetupDialog` (ui/swing/setup_dialog.rs, which
//! implements the trait itself) and `DirectiveFileCollection`, a non-UI class with
//! `&mut self` setters and `Option<AxisID>` parameters; it is shared by the harness and
//! the template panel as a [`DirectiveFileCollectionHandle`], and the trait is
//! implemented for that handle at the bottom of this module by forwarding to the
//! collection's own methods.

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::comscript::exclude_views_param::ExcludeViewsParam;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;

/// Rust-only: how a `DirectiveFileCollection` is shared on the event dispatch thread
/// (Java passes the same object between the harness, the template panel and the
/// dialog).
pub type DirectiveFileCollectionHandle = Rc<RefCell<DirectiveFileCollection>>;

/// Java interface `SetupReconInterface`.
pub trait SetupReconInterface {
    /// Java `setBinning(String)`.
    fn set_binning(&self, input: Option<&str>);

    /// Java `setImageRotation(String)`.
    fn set_image_rotation(&self, input: Option<&str>);

    /// Java `setPixelSize(double)`.
    fn set_pixel_size(&self, input: f64);

    /// Java `setHalfFloatModeOutput(Integer)`.
    fn set_half_float_mode_output(&self, input: Option<i32>);

    /// Java `getDataset()` (deprecated 4/8/2019).
    #[deprecated]
    fn get_dataset(&self) -> Option<String>;

    /// Java `getRawImageStack()`.
    fn get_raw_image_stack(&self) -> Option<String>;

    /// Java `isDualAxisSelected()`.
    fn is_dual_axis_selected(&self) -> bool;

    /// Java `getDistortionFile()`.
    fn get_distortion_file(&self) -> Option<String>;

    /// Java `getMagGradientFile()`.
    fn get_mag_gradient_file(&self) -> Option<String>;

    /// Java `validateTiltAngle(AxisID, String)`.
    fn validate_tilt_angle(&self, axis_id: AxisID, error_title: &str) -> bool;

    /// Java `isSingleViewSelected()`.
    fn is_single_view_selected(&self) -> bool;

    /// Java `getBackupDirectory()`.
    fn get_backup_directory(&self) -> Option<String>;

    /// Java `getBinning(boolean) throws FieldValidationFailedException`.
    fn get_binning(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getExcludeList(AxisID, boolean) throws FieldValidationFailedException`.
    fn get_exclude_list(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getTwodir(AxisID, boolean) throws FieldValidationFailedException`.
    fn get_twodir(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `isTwodir(AxisID)`.
    fn is_twodir(&self, axis_id: AxisID) -> bool;

    /// Java `setTwodir(AxisID, double)`.
    fn set_twodir(&self, axis_id: AxisID, input: f64);

    /// Java `getFiducialDiameter(boolean) throws FieldValidationFailedException`.
    fn get_fiducial_diameter(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getImageRotation(AxisID, boolean) throws FieldValidationFailedException`.
    fn get_image_rotation(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getPixelSize(boolean) throws FieldValidationFailedException`.
    fn get_pixel_size(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `getHalfFloatModeOutput()`.
    fn get_half_float_mode_output(&self) -> Option<i32>;

    /// Java `isAdjustedFocusSelected(AxisID)`.
    fn is_adjusted_focus_selected(&self, axis_id: AxisID) -> bool;

    /// Java `isSingleAxisSelected()`.
    fn is_single_axis_selected(&self) -> bool;

    /// Java `isGpuProcessingSelected(String)`.
    fn is_gpu_processing_selected(&self, property_user_dir: Option<&str>) -> bool;

    /// Java `isParallelProcessSelected(String)`.
    fn is_parallel_process_selected(&self, property_user_dir: Option<&str>) -> bool;

    /// Java `getTiltAngleFields(AxisID, TiltAngleSpec, boolean)`.  Fills
    /// `tilt_angle_spec` in place.  `Err` carries the message of an unchecked
    /// `NumberFormatException` the Java lets escape (a malformed starting angle or
    /// increment in the dialog), which `SetupReconUIHarness.getFields` catches.
    fn get_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &mut TiltAngleSpec,
        do_validation: bool,
    ) -> Result<bool, String>;

    /// Java `getDirectiveFileCollection()`.
    fn get_directive_file_collection(&self) -> Option<DirectiveFileCollectionHandle>;

    /// Java `initTiltAngleFields(AxisID, TiltAngleSpec, UserConfiguration)`.
    fn init_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &TiltAngleSpec,
        user_configuration: &UserConfiguration,
    );

    /// Java `getParameters(ExcludeViewsParam, AxisID, boolean, boolean)`.
    fn get_parameters(
        &self,
        param: &mut ExcludeViewsParam,
        axis_id: AxisID,
        dual_axis: bool,
        do_validation: bool,
    ) -> bool;

    /// Java `msgExcludeViewsSucceeded(AxisID, boolean, boolean)`.
    fn msg_exclude_views_succeeded(
        &self,
        axis_id: AxisID,
        process_running: bool,
        process_done: bool,
    );

    /// Java `getDoseSym(AxisID, boolean) throws FieldValidationFailedException`.
    fn get_dose_sym(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException>;

    /// Java `isDoseSym(AxisID)`.
    fn is_dose_sym(&self, axis_id: AxisID) -> bool;

    /// Java `setDoseSym(AxisID, double)`.
    fn set_dose_sym(&self, axis_id: AxisID, input: f64);
}

/// Java `DirectiveFileCollection implements SetupReconInterface`: each method is the
/// collection's own (`storage/directive_file_collection.rs`), borrowed for the one
/// call.  None of them calls back into the harness or the dialog.
impl SetupReconInterface for DirectiveFileCollectionHandle {
    fn set_binning(&self, input: Option<&str>) {
        self.borrow_mut().set_binning(input);
    }

    fn set_image_rotation(&self, input: Option<&str>) {
        self.borrow_mut().set_image_rotation(input);
    }

    fn set_pixel_size(&self, input: f64) {
        self.borrow_mut().set_pixel_size(input);
    }

    fn set_half_float_mode_output(&self, input: Option<i32>) {
        self.borrow_mut().set_half_float_mode_output(input);
    }

    #[allow(deprecated)]
    fn get_dataset(&self) -> Option<String> {
        self.borrow().get_dataset()
    }

    fn get_raw_image_stack(&self) -> Option<String> {
        self.borrow().get_raw_image_stack()
    }

    fn is_dual_axis_selected(&self) -> bool {
        self.borrow().is_dual_axis_selected()
    }

    fn get_distortion_file(&self) -> Option<String> {
        self.borrow().get_distortion_file()
    }

    fn get_mag_gradient_file(&self) -> Option<String> {
        self.borrow().get_mag_gradient_file()
    }

    fn validate_tilt_angle(&self, axis_id: AxisID, error_title: &str) -> bool {
        self.borrow()
            .validate_tilt_angle(Some(axis_id), Some(error_title))
    }

    fn is_single_view_selected(&self) -> bool {
        self.borrow().is_single_view_selected()
    }

    fn get_backup_directory(&self) -> Option<String> {
        self.borrow().get_backup_directory()
    }

    fn get_binning(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.borrow().get_binning_validate(do_validation)
    }

    fn get_exclude_list(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(self
            .borrow()
            .get_exclude_list(Some(axis_id), do_validation))
    }

    fn get_twodir(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(self.borrow().get_twodir(Some(axis_id), do_validation))
    }

    fn is_twodir(&self, axis_id: AxisID) -> bool {
        self.borrow().is_twodir(Some(axis_id))
    }

    fn set_twodir(&self, axis_id: AxisID, input: f64) {
        self.borrow_mut().set_twodir(Some(axis_id), input);
    }

    fn get_fiducial_diameter(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(self.borrow().get_fiducial_diameter(do_validation))
    }

    fn get_image_rotation(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(self
            .borrow()
            .get_image_rotation(Some(axis_id), do_validation))
    }

    fn get_pixel_size(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(self.borrow().get_pixel_size(do_validation))
    }

    fn get_half_float_mode_output(&self) -> Option<i32> {
        self.borrow().get_half_float_mode_output()
    }

    fn is_adjusted_focus_selected(&self, axis_id: AxisID) -> bool {
        self.borrow().is_adjusted_focus_selected(Some(axis_id))
    }

    fn is_single_axis_selected(&self) -> bool {
        self.borrow().is_single_axis_selected()
    }

    fn is_gpu_processing_selected(&self, property_user_dir: Option<&str>) -> bool {
        self.borrow().is_gpu_processing_selected(property_user_dir)
    }

    fn is_parallel_process_selected(&self, property_user_dir: Option<&str>) -> bool {
        self.borrow().is_parallel_process_selected(property_user_dir)
    }

    fn get_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &mut TiltAngleSpec,
        do_validation: bool,
    ) -> Result<bool, String> {
        Ok(self
            .borrow()
            .get_tilt_angle_fields(Some(axis_id), Some(tilt_angle_spec), do_validation))
    }

    /// Java `DirectiveFileCollection.getDirectiveFileCollection()` returns `this`.
    fn get_directive_file_collection(&self) -> Option<DirectiveFileCollectionHandle> {
        Some(self.clone())
    }

    fn init_tilt_angle_fields(
        &self,
        axis_id: AxisID,
        tilt_angle_spec: &TiltAngleSpec,
        user_configuration: &UserConfiguration,
    ) {
        self.borrow_mut()
            .init_tilt_angle_fields(Some(axis_id), tilt_angle_spec, user_configuration);
    }

    fn get_parameters(
        &self,
        param: &mut ExcludeViewsParam,
        axis_id: AxisID,
        dual_axis: bool,
        do_validation: bool,
    ) -> bool {
        self.borrow()
            .get_parameters(param, Some(axis_id), dual_axis, do_validation)
    }

    fn msg_exclude_views_succeeded(
        &self,
        axis_id: AxisID,
        process_running: bool,
        process_done: bool,
    ) {
        self.borrow_mut()
            .msg_exclude_views_succeeded(Some(axis_id), process_running, process_done);
    }

    fn get_dose_sym(
        &self,
        axis_id: AxisID,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        Ok(self.borrow().get_dose_sym(Some(axis_id), do_validation))
    }

    fn is_dose_sym(&self, axis_id: AxisID) -> bool {
        self.borrow().is_dose_sym(Some(axis_id))
    }

    fn set_dose_sym(&self, axis_id: AxisID, input: f64) {
        self.borrow_mut().set_dose_sym(Some(axis_id), input);
    }
}
