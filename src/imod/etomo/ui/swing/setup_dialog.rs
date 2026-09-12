//! `IMOD/Etomo/src/etomo/ui/swing/SetupDialog.java`.
#![allow(dead_code)]
use super::context_menu::{ContextMenu, MouseEvent};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
pub const FIDUCIAL_DIAMETER_LABEL: &str = "Fiducial diameter (nm): ";
pub const AXIS_TYPE_LABEL: &str = "Axis Type";
pub const FRAME_TYPE_LABEL: &str = "Frame Type";
pub const SINGLE_AXIS_LABEL: &str = "Single axis";
pub const MONTAGE_LABEL: &str = "Montage";
pub const SINGLE_FRAME_LABEL: &str = "Single frame";
pub const VIEW_RAW_STACK_LABEL: &str = "View Raw Image Stack";
pub const BACKUP_DIRECTORY_LABEL: &str = "Backup directory: ";
pub const REMOVE_EXCLUDE_VIEW_MSG: &str = "Excluded views have been removed";
/// Direct `DirectiveFileCollection` values read by Java `updateTemplateValues`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SetupDialogDirectiveFileCollection {
    pub dual: Option<bool>,
    pub montage: Option<bool>,
    pub pixel_size: Option<String>,
    pub twodir_a: Option<String>,
    pub twodir_b: Option<String>,
    pub dose_sym_a: Option<String>,
    pub dose_sym_b: Option<String>,
    pub fiducial_diameter: Option<String>,
    pub image_rotation: Option<String>,
    pub half_float_mode_output: Option<i32>,
    pub distortion_file: Option<String>,
    pub binning: Option<String>,
    pub mag_gradient_file: Option<String>,
    pub adjusted_focus_a: Option<bool>,
    pub adjusted_focus_b: Option<bool>,
    pub remove_excluded_views: Option<bool>,
    pub delete_old_files: Option<bool>,
}
/// Java private static `BackupDirectoryActionListener`.
pub struct BackupDirectoryActionListener;
impl BackupDirectoryActionListener {
    pub fn action_performed<E: SetupDialogExpert>(
        dialog: &mut SetupDialog<E>,
        selected_file: Option<String>,
    ) {
        dialog.backup_directory_action(selected_file);
    }
}
/// Java private static `DistortionFileActionListener`.
pub struct DistortionFileActionListener;
impl DistortionFileActionListener {
    pub fn action_performed<E: SetupDialogExpert>(
        dialog: &mut SetupDialog<E>,
        selected_file: Option<String>,
    ) {
        dialog.distortion_file_action(selected_file);
    }
}
/// Java private static `MagGradientFileActionListener`.
pub struct MagGradientFileActionListener;
impl MagGradientFileActionListener {
    pub fn action_performed<E: SetupDialogExpert>(
        dialog: &mut SetupDialog<E>,
        selected_file: Option<String>,
    ) {
        dialog.mag_gradient_file_action(selected_file);
    }
}
/// Java private static `ViewRawStackAActionListener`.
pub struct ViewRawStackAActionListener;
impl ViewRawStackAActionListener {
    pub fn action_performed<E: SetupDialogExpert>(
        dialog: &SetupDialog<E>,
    ) -> Option<(String, AxisID)> {
        dialog.view_raw_stack_a()
    }
}
/// Java private static `ViewRawStackBActionListener`.
pub struct ViewRawStackBActionListener;
impl ViewRawStackBActionListener {
    pub fn action_performed<E: SetupDialogExpert>(
        dialog: &SetupDialog<E>,
    ) -> Option<(String, AxisID)> {
        dialog.view_raw_stack_b()
    }
}
/// Java private static `SetupDialogActionListener`; template action execution is a
/// direct `SetupDialogExpert` boundary and the command remains unchanged.
pub struct SetupDialogActionListener;
impl SetupDialogActionListener {
    pub fn action_performed(action_command: String) -> String {
        action_command
    }
}
/// Canonical Java `SetupDialogExpert` boundary.
pub use super::setup_dialog_expert::SetupDialogExpert;
/// Java final `SetupDialog`: Swing widgets/filesystem/template collaborators remain boundaries.
pub struct SetupDialog<E: SetupDialogExpert> {
    pub expert: E,
    pub raw_image_stack: String,
    pub backup_directory: String,
    pub single_axis: bool,
    pub dual_axis: bool,
    pub single_view: bool,
    pub montage: bool,
    pub pixel_size: String,
    pub fiducial_diameter: String,
    pub image_rotation: String,
    pub distortion_file: String,
    pub binning: String,
    pub mag_gradient_file: String,
    pub parallel_process: bool,
    pub gpu_processing: bool,
    pub gpu_processing_enabled: bool,
    pub exclude_list_a: String,
    pub exclude_list_b: String,
    pub adjusted_focus_a: bool,
    pub adjusted_focus_b: bool,
    pub remove_excluded_views: bool,
    pub delete_old_files: bool,
    pub remove_exclude_views_msg_a: String,
    pub remove_exclude_views_msg_b: String,
    pub exclude_views_succeeded_a: bool,
    pub exclude_views_succeeded_b: bool,
    pub displayed: bool,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub calibration_available: bool,
    pub root_panel_y_axis: bool,
    pub raw_image_stack_files_only: bool,
    pub raw_image_stack_absolute_path: bool,
    pub raw_image_stack_display_limit: usize,
    pub dataset_panel_created: bool,
    pub data_type_panel_created: bool,
    pub per_axis_info_panel_created: bool,
    pub progress_panel_visible: bool,
    pub advanced_button_present: bool,
    pub exit_buttons_added: bool,
    pub listeners_added: bool,
    pub axis_b_panel_visible: bool,
    pub half_float_mode_output_enabled: bool,
    pub half_float_mode_output: bool,
    pub half_float_mode_output_if_float: bool,
    pub half_float_mode_output_active_visible: bool,
    pub view_raw_stack_a_command: String,
    pub view_raw_stack_b_command: String,
    pub context_menu_event: Option<MouseEvent>,
    pub execute_enabled: bool,
    pub cancel_enabled: bool,
    pub twodir_a_selected: bool,
    pub twodir_b_selected: bool,
    pub dose_sym_a_selected: bool,
    pub dose_sym_b_selected: bool,
    pub twodir_a: String,
    pub twodir_b: String,
    pub dose_sym_a: String,
    pub dose_sym_b: String,
    pub exclude_list_a_enabled: bool,
    pub exclude_list_b_enabled: bool,
    pub adjusted_focus_a_enabled: bool,
    pub adjusted_focus_b_enabled: bool,
    pub view_raw_stack_a_enabled: bool,
    pub view_raw_stack_b_enabled: bool,
    pub twodir_a_enabled: bool,
    pub twodir_b_enabled: bool,
    pub dose_sym_a_enabled: bool,
    pub dose_sym_b_enabled: bool,
    pub distortion_file_visible: bool,
    pub binning_visible: bool,
    pub mag_gradient_info_visible: bool,
    pub raw_image_stack_field_tooltip: String,
    pub raw_image_stack_button_tooltip: String,
    pub backup_directory_field_tooltip: String,
    pub backup_directory_button_tooltip: String,
    pub scan_header_tooltip: String,
    pub directive_file_collection: Option<SetupDialogDirectiveFileCollection>,
    pub tooltip_assignments: Vec<(String, String)>,
}
impl<E: SetupDialogExpert> SetupDialog<E> {
    /// Java private `SetupDialog(SetupDialogExpert, ApplicationManager, AxisID,
    /// DialogType, boolean, ValidationSet, AxisProgressPanel)`.  Manager, template,
    /// tilt-angle, progress, and Swing component construction remain their direct
    /// source boundaries; all construction decisions made by this unit are retained.
    pub fn new_with_construction(
        expert: E,
        axis_id: AxisID,
        dialog_type: DialogType,
        calibration_available: bool,
        half_float_mode_output_enabled: bool,
    ) -> Self {
        let mut dialog = Self::new(expert);
        dialog.axis_id = axis_id;
        dialog.dialog_type = dialog_type;
        dialog.calibration_available = calibration_available;
        dialog.half_float_mode_output_enabled = half_float_mode_output_enabled;
        dialog.root_panel_y_axis = true;
        dialog.raw_image_stack_files_only = true;
        dialog.raw_image_stack_absolute_path = true;
        dialog.raw_image_stack_display_limit = 50;
        dialog.view_raw_stack_a_command =
            format!("{}{}", VIEW_RAW_STACK_LABEL, AxisID::First.get_extension());
        dialog.view_raw_stack_b_command =
            format!("{}{}", VIEW_RAW_STACK_LABEL, AxisID::Second.get_extension());
        dialog.create_dataset_panel();
        dialog.create_data_type_panel();
        dialog.create_per_axis_info_panel();
        dialog.advanced_button_present = !calibration_available;
        dialog.exit_buttons_added = true;
        dialog.update_display(false, false);
        dialog
    }

    /// Java static `getInstance`.
    pub fn get_instance(
        expert: E,
        axis_id: AxisID,
        dialog_type: DialogType,
        calibration_available: bool,
        half_float_mode_output_enabled: bool,
    ) -> Self {
        let mut instance = Self::new_with_construction(
            expert,
            axis_id,
            dialog_type,
            calibration_available,
            half_float_mode_output_enabled,
        );
        instance.add_listeners();
        instance
    }

    pub fn new(expert: E) -> Self {
        Self {
            expert,
            raw_image_stack: String::new(),
            backup_directory: String::new(),
            single_axis: true,
            dual_axis: false,
            single_view: true,
            montage: false,
            pixel_size: String::new(),
            fiducial_diameter: String::new(),
            image_rotation: String::new(),
            distortion_file: String::new(),
            binning: String::new(),
            mag_gradient_file: String::new(),
            parallel_process: false,
            gpu_processing: false,
            gpu_processing_enabled: false,
            exclude_list_a: String::new(),
            exclude_list_b: String::new(),
            adjusted_focus_a: false,
            adjusted_focus_b: false,
            remove_excluded_views: false,
            delete_old_files: false,
            remove_exclude_views_msg_a: String::new(),
            remove_exclude_views_msg_b: String::new(),
            exclude_views_succeeded_a: false,
            exclude_views_succeeded_b: false,
            displayed: true,
            axis_id: AxisID::First,
            dialog_type: DialogType::SetupRecon,
            calibration_available: false,
            root_panel_y_axis: false,
            raw_image_stack_files_only: false,
            raw_image_stack_absolute_path: false,
            raw_image_stack_display_limit: 0,
            dataset_panel_created: false,
            data_type_panel_created: false,
            per_axis_info_panel_created: false,
            progress_panel_visible: false,
            advanced_button_present: true,
            exit_buttons_added: false,
            listeners_added: false,
            axis_b_panel_visible: true,
            half_float_mode_output_enabled: false,
            half_float_mode_output: false,
            half_float_mode_output_if_float: false,
            half_float_mode_output_active_visible: false,
            view_raw_stack_a_command: String::new(),
            view_raw_stack_b_command: String::new(),
            context_menu_event: None,
            execute_enabled: true,
            cancel_enabled: true,
            twodir_a_selected: false,
            twodir_b_selected: false,
            dose_sym_a_selected: false,
            dose_sym_b_selected: false,
            twodir_a: String::new(),
            twodir_b: String::new(),
            dose_sym_a: String::new(),
            dose_sym_b: String::new(),
            exclude_list_a_enabled: true,
            exclude_list_b_enabled: true,
            adjusted_focus_a_enabled: false,
            adjusted_focus_b_enabled: false,
            view_raw_stack_a_enabled: true,
            view_raw_stack_b_enabled: true,
            twodir_a_enabled: false,
            twodir_b_enabled: false,
            dose_sym_a_enabled: false,
            dose_sym_b_enabled: false,
            distortion_file_visible: false,
            binning_visible: false,
            mag_gradient_info_visible: false,
            raw_image_stack_field_tooltip: String::new(),
            raw_image_stack_button_tooltip: String::new(),
            backup_directory_field_tooltip: String::new(),
            backup_directory_button_tooltip: String::new(),
            scan_header_tooltip: String::new(),
            directive_file_collection: None,
            tooltip_assignments: Vec::new(),
        }
    }

    /// Java private `createDatasetPanel`.
    pub fn create_dataset_panel(&mut self) {
        self.dataset_panel_created = true;
    }

    /// Java private `createDataTypePanel`.
    pub fn create_data_type_panel(&mut self) {
        self.data_type_panel_created = true;
        self.half_float_mode_output_active_visible = false;
    }

    /// Java private `createPerAxisInfoPanel`.
    pub fn create_per_axis_info_panel(&mut self) {
        self.per_axis_info_panel_created = true;
        self.adjusted_focus_a = false;
        self.adjusted_focus_b = false;
        self.update_display(false, false);
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.listeners_added = true;
    }

    /// Java `popUpContextMenu(MouseEvent)`.
    pub fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        self.context_menu_event = Some(mouse_event);
    }

    /// Java `updateDisplay(boolean, boolean)` subset owned by the constructor-created
    /// controls; concrete tilt-angle and process-panel painting remain boundaries.
    pub fn update_display(&mut self, process_running: bool, process_done: bool) {
        self.half_float_mode_output_active_visible = self.half_float_mode_output_enabled
            && (self.half_float_mode_output || self.half_float_mode_output_if_float);
        self.delete_old_files = self.remove_excluded_views && !process_running;
        let enabled = if process_running {
            false
        } else {
            process_done || self.displayed
        };
        self.execute_enabled = enabled;
        self.cancel_enabled = enabled;
        self.adjusted_focus_a_enabled = self.montage;
        self.adjusted_focus_b_enabled = self.dual_axis && self.montage;
        self.axis_b_panel_visible = self.dual_axis;
        self.twodir_a_enabled = self.twodir_a_selected;
        self.dose_sym_a_enabled = self.twodir_a_selected;
        if self.dual_axis {
            self.twodir_b_enabled = self.twodir_b_selected;
            self.dose_sym_b_enabled = self.twodir_b_selected;
        }
    }
    pub fn get_raw_image_stack(&self) -> &str {
        &self.raw_image_stack
    }
    /// Java private `isRemoveExcludedViews`.
    pub fn is_remove_excluded_views(&self) -> bool {
        self.remove_excluded_views
    }
    pub fn get_dataset_name(&self) -> String {
        let n = self.raw_image_stack.trim().rsplit('/').next().unwrap_or("");
        let n = n.rsplit_once('.').map_or(n, |(x, _)| x);
        n.strip_suffix(&AxisID::First.get_extension())
            .or_else(|| n.strip_suffix(&AxisID::Second.get_extension()))
            .unwrap_or(n)
            .into()
    }
    /// Java package-private `getDirectory`.
    pub fn get_directory(&self) -> Option<String> {
        self.raw_image_stack
            .rsplit_once('/')
            .map(|(directory, _)| directory.into())
    }
    pub fn done(&mut self) {
        let d = self.raw_image_stack.rsplit_once('/').map(|(d, _)| d);
        self.expert
            .done_setup_dialog(self.remove_excluded_views, d, self.dual_axis);
        self.displayed = false
    }
    pub fn msg_setup_recon_failed(&mut self) {
        self.expert.setup_recon_failed();
        self.update_display(false, true);
    }
    pub fn set_raw_image_stack(&mut self, input: &str) {
        self.raw_image_stack = input.into()
    }
    pub fn set_parallel_process(&mut self, input: bool) {
        self.parallel_process = input
    }
    pub fn set_gpu_processing_enabled(&mut self, input: bool) {
        self.gpu_processing_enabled = input
    }
    pub fn set_gpu_processing(&mut self, input: bool) {
        self.gpu_processing = input
    }
    pub fn set_binning(&mut self, input: impl Into<String>) {
        self.binning = input.into();
    }
    /// Java `setHalfFloatModeOutput(Integer)`; CopyTomoComs values are 1 and 2.
    pub fn set_half_float_mode_output(&mut self, input: Option<i32>) {
        self.half_float_mode_output = input == Some(1);
        self.half_float_mode_output_if_float = input == Some(2);
        self.update_display(false, false);
    }

    /// Java `showProgressPanel`.
    pub fn show_progress_panel(&mut self) {
        self.progress_panel_visible = true;
        self.update_display(true, false);
    }

    /// Java `buttonExecuteAction`; dataset validation is the direct `DatasetTool`
    /// boundary and is supplied by the caller.
    pub fn button_execute_action(&self, dataset_name_valid: bool) -> bool {
        dataset_name_valid
    }

    /// Java `actionPerformed(ActionEvent)` for the two mutually-exclusive half-float
    /// controls. `loadHeader` stays at the `SetupDialogExpert` boundary.
    pub fn action_performed(&mut self, action_command: Option<&str>) {
        if action_command == Some("halfFloatModeOutput")
            && self.half_float_mode_output
            && self.half_float_mode_output_if_float
        {
            self.half_float_mode_output_if_float = false;
        } else if action_command == Some("halfFloatModeOutputIfFloat")
            && self.half_float_mode_output_if_float
        {
            self.half_float_mode_output = false;
        }
        self.update_display(false, false);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.  The
    /// returned source call arguments cross to the still-native 3dmod launcher.
    pub fn action(&self, action_command: &str) -> Option<(String, AxisID)> {
        let raw_image_stack = self.raw_image_stack.trim();
        if raw_image_stack.is_empty() {
            return None;
        }
        let extension = raw_image_stack
            .rsplit_once('.')
            .map(|(_, extension)| extension)?;
        if action_command == self.view_raw_stack_a_command {
            Some((format!(".{extension}"), AxisID::First))
        } else if action_command == self.view_raw_stack_b_command {
            Some((format!(".{extension}"), AxisID::Second))
        } else {
            None
        }
    }

    pub fn focus_gained(&self) {}
    pub fn focus_lost(&mut self) {
        self.control_event(false, false);
    }

    /// Java `controlEvent`; filesystem discovery of existing excludeviews info files
    /// remains a direct file-system boundary represented by its two results.
    pub fn control_event(
        &mut self,
        exclude_views_info_a_exists: bool,
        exclude_views_info_b_exists: bool,
    ) {
        self.update_display(false, false);
        if self.raw_image_stack.is_empty() {
            self.remove_exclude_views_msg_a.clear();
            self.remove_exclude_views_msg_b.clear();
            return;
        }
        self.remove_exclude_views_msg_a = if exclude_views_info_a_exists {
            REMOVE_EXCLUDE_VIEW_MSG.into()
        } else {
            String::new()
        };
        self.remove_exclude_views_msg_b = if self.dual_axis && exclude_views_info_b_exists {
            REMOVE_EXCLUDE_VIEW_MSG.into()
        } else {
            String::new()
        };
        self.update_display(false, false);
    }

    /// Java `getParameters(ExcludeViewsParam, AxisID, boolean, boolean)` direct
    /// parameter values.
    pub fn get_parameters_exclude_views(&self, axis_id: AxisID) -> (String, bool, bool) {
        let stack_name = self.get_dataset_name();
        (
            format!("{}{}", stack_name, axis_id.get_extension()),
            self.montage,
            self.delete_old_files && self.remove_excluded_views,
        )
    }

    /// Java `checkpoint` boundary marker for all source controls.
    pub fn checkpoint(&self) {}

    pub fn get_directive_file_collection(&self) -> Option<&SetupDialogDirectiveFileCollection> {
        self.directive_file_collection.as_ref()
    }
    /// Java `setParameters(UserConfiguration)`.
    pub fn set_parameters_user_configuration(&mut self, remove_excluded_views: bool) {
        self.remove_excluded_views = remove_excluded_views;
        self.update_display(false, false);
    }
    /// Java `updateTemplateValues` with every directive value copied in source order.
    pub fn update_template_values(&mut self, directives: SetupDialogDirectiveFileCollection) {
        if let Some(dual) = directives.dual {
            self.set_dual_axis(dual);
            if !dual {
                self.set_single_axis(true);
            }
        }
        if let Some(montage) = directives.montage {
            self.set_montage(montage);
            if !montage {
                self.set_single_view(true);
            }
        }
        if let Some(pixel_size) = &directives.pixel_size {
            self.pixel_size = pixel_size.clone();
        }
        let twodir_a = directives.twodir_a.clone();
        if let Some(value) = &twodir_a {
            self.set_twodir(AxisID::First, value);
        }
        if let Some(value) = &directives.twodir_b {
            self.set_twodir(AxisID::Second, value);
        } else if let Some(value) = &twodir_a {
            self.set_twodir(AxisID::Second, value);
        }
        let dose_sym_a = directives.dose_sym_a.clone();
        if let Some(_value) = &dose_sym_a {
            // Java's source assigns twodirA rather than doseSymA to tfDoseSym.
            self.set_dose_sym(AxisID::First, twodir_a.clone().unwrap_or_default());
        }
        if let Some(value) = &directives.dose_sym_b {
            self.set_dose_sym(AxisID::Second, value);
        } else if let Some(value) = &dose_sym_a {
            self.set_dose_sym(AxisID::Second, value);
        }
        if let Some(value) = &directives.fiducial_diameter {
            self.fiducial_diameter = value.clone();
        }
        if let Some(value) = &directives.image_rotation {
            self.image_rotation = value.clone();
        }
        match directives.half_float_mode_output {
            Some(1) => self.half_float_mode_output = true,
            Some(2) => self.half_float_mode_output_if_float = true,
            _ => {}
        }
        if let Some(value) = &directives.distortion_file {
            self.distortion_file = value.clone();
        }
        if let Some(value) = &directives.binning {
            self.binning = value.clone();
        }
        if let Some(value) = &directives.mag_gradient_file {
            self.mag_gradient_file = value.clone();
        }
        if let Some(value) = directives.adjusted_focus_a {
            self.adjusted_focus_a = value;
        }
        if let Some(value) = directives.adjusted_focus_b {
            self.adjusted_focus_b = value;
        }
        if let Some(value) = directives.remove_excluded_views {
            self.remove_excluded_views = value;
        }
        if let Some(value) = directives.delete_old_files {
            self.delete_old_files = value;
        }
        self.directive_file_collection = Some(directives);
        self.update_display(false, false);
    }
    pub fn view_raw_stack_a(&self) -> Option<(String, AxisID)> {
        self.action(&self.view_raw_stack_a_command)
    }
    pub fn view_raw_stack_b(&self) -> Option<(String, AxisID)> {
        self.action(&self.view_raw_stack_b_command)
    }
    pub fn set_backup_directory(&mut self, input: impl Into<String>) {
        self.backup_directory = input.into();
    }
    pub fn set_distortion_file(&mut self, input: impl Into<String>) {
        self.distortion_file = input.into();
    }
    pub fn set_mag_gradient_file(&mut self, input: impl Into<String>) {
        self.mag_gradient_file = input.into();
    }
    pub fn set_adjusted_focus(&mut self, axis_id: AxisID, input: bool) {
        if axis_id == AxisID::Second {
            self.adjusted_focus_b = input;
        } else {
            self.adjusted_focus_a = input;
        }
    }
    pub fn set_axis_type_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("axisType".into(), tooltip.into()));
    }
    pub fn set_distortion_file_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("distortionFile".into(), tooltip.into()));
    }
    pub fn set_view_type_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("viewType".into(), tooltip.into()));
    }
    pub fn set_pixel_size_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("pixelSize".into(), tooltip.into()));
    }
    pub fn set_fiducial_diameter_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("fiducialDiameter".into(), tooltip.into()));
    }
    pub fn set_image_rotation_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("imageRotation".into(), tooltip.into()));
    }
    pub fn set_half_float_mode_output_tooltip(
        &mut self,
        tooltip: impl Into<String>,
        if_float_tooltip: impl Into<String>,
    ) {
        self.tooltip_assignments
            .push(("halfFloatModeOutput".into(), tooltip.into()));
        self.tooltip_assignments
            .push(("halfFloatModeOutputIfFloat".into(), if_float_tooltip.into()));
    }
    pub fn set_binning_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("binning".into(), tooltip.into()));
    }
    pub fn set_view_raw_stack_tooltip(&mut self, tooltip: impl Into<String>) {
        let tooltip = tooltip.into();
        self.tooltip_assignments
            .push(("viewRawStackA".into(), tooltip.clone()));
        self.tooltip_assignments
            .push(("viewRawStackB".into(), tooltip));
    }
    pub fn set_adjusted_focus_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("adjustedFocus".into(), tooltip.into()));
    }
    pub fn set_exclude_list_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("excludeList".into(), tooltip.into()));
    }
    pub fn set_twodir_tooltip(&mut self) {
        self.tooltip_assignments.push((
            "twodir".into(),
            "Tilt series was bidirectional or dose-symmetric from the given starting angle".into(),
        ));
    }
    pub fn set_execute_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("execute".into(), tooltip.into()));
    }
    pub fn set_parallel_process_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("parallelProcess".into(), tooltip.into()));
    }
    pub fn set_gpu_processing_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("gpuProcessing".into(), tooltip.into()));
    }
    pub fn set_mag_gradient_file_tooltip(&mut self, tooltip: impl Into<String>) {
        self.tooltip_assignments
            .push(("magGradientFile".into(), tooltip.into()));
    }
    pub fn set_tooltips(&mut self) {
        self.tooltip_assignments.push((
            "removeExcludedViews".into(),
            "Make a new stack with excluded views removed before running copytomocoms".into(),
        ));
        self.tooltip_assignments.push(("deleteOldFiles".into(), "delete original file and keep excluded views, which can be used to restore the original file.".into()));
    }

    pub fn is_single_axis_selected(&self) -> bool {
        self.single_axis
    }
    pub fn is_single_view_selected(&self) -> bool {
        self.single_view
    }
    pub fn is_dual_axis_selected(&self) -> bool {
        self.dual_axis
    }
    pub fn get_axis_type(&self) -> AxisType {
        if self.dual_axis {
            AxisType::DualAxis
        } else {
            AxisType::SingleAxis
        }
    }
    pub fn set_single_axis(&mut self, input: bool) {
        self.single_axis = input;
        if input {
            self.dual_axis = false;
        }
    }
    pub fn set_dual_axis(&mut self, input: bool) {
        self.dual_axis = input;
        if input {
            self.single_axis = false;
        }
        self.update_display(false, false);
    }
    pub fn set_single_view(&mut self, input: bool) {
        self.single_view = input;
        if input {
            self.montage = false;
        }
        self.update_display(false, false);
    }
    pub fn set_montage(&mut self, input: bool) {
        self.montage = input;
        if input {
            self.single_view = false;
        }
        self.update_display(false, false);
    }
    pub fn set_pixel_size(&mut self, input: f64) {
        self.pixel_size = input.to_string();
    }
    pub fn set_fiducial_diameter(&mut self, input: f64) {
        self.fiducial_diameter = input.to_string();
    }
    pub fn set_image_rotation(&mut self, input: impl ToString) {
        self.image_rotation = input.to_string();
    }
    pub fn get_binning(&self, _do_validation: bool) -> String {
        self.binning.clone()
    }
    pub fn get_exclude_list(&self, axis_id: AxisID, _do_validation: bool) -> String {
        if axis_id == AxisID::Second {
            self.exclude_list_b.clone()
        } else {
            self.exclude_list_a.clone()
        }
    }
    pub fn set_exclude_list(&mut self, axis_id: AxisID, input: impl Into<String>) {
        if axis_id == AxisID::Second {
            self.exclude_list_b = input.into();
        } else {
            self.exclude_list_a = input.into();
        }
    }
    pub fn get_twodir(&self, axis_id: AxisID, _do_validation: bool) -> String {
        if axis_id == AxisID::Second {
            self.twodir_b.clone()
        } else {
            self.twodir_a.clone()
        }
    }
    pub fn is_twodir(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            self.twodir_b_selected && !self.dose_sym_b_selected
        } else {
            self.twodir_a_selected && !self.dose_sym_a_selected
        }
    }
    pub fn set_twodir(&mut self, axis_id: AxisID, input: impl Into<String>) {
        if axis_id == AxisID::Second {
            self.twodir_b_selected = true;
            self.dose_sym_b_selected = false;
            self.twodir_b = input.into();
        } else {
            self.twodir_a_selected = true;
            self.dose_sym_a_selected = false;
            self.twodir_a = input.into();
        }
        self.update_display(false, false);
    }
    /// Java overload `setTwodir(AxisID, boolean)`.
    pub fn set_twodir_selected(&mut self, axis_id: AxisID, selected: bool) {
        if axis_id == AxisID::Second {
            self.twodir_b_selected = selected;
            self.dose_sym_b_selected = false;
        } else {
            self.twodir_a_selected = selected;
            self.dose_sym_a_selected = false;
        }
        self.update_display(false, false);
    }
    /// Java overload `setTwodir(AxisID, double)`.
    pub fn set_twodir_double(&mut self, axis_id: AxisID, input: f64) {
        self.set_twodir(axis_id, input.to_string());
    }
    pub fn set_dose_sym(&mut self, axis_id: AxisID, input: impl Into<String>) {
        if axis_id == AxisID::Second {
            self.twodir_b_selected = true;
            self.dose_sym_b_selected = true;
            self.dose_sym_b = input.into();
        } else {
            self.twodir_a_selected = true;
            self.dose_sym_a_selected = true;
            self.dose_sym_a = input.into();
        }
        self.update_display(false, false);
    }
    /// Java overload `setDoseSym(AxisID, boolean)`.
    pub fn set_dose_sym_selected(&mut self, axis_id: AxisID, selected: bool) {
        if axis_id == AxisID::Second {
            self.twodir_b_selected = selected;
            self.dose_sym_b_selected = selected;
        } else {
            self.twodir_a_selected = selected;
            self.dose_sym_a_selected = selected;
        }
        self.update_display(false, false);
    }
    /// Java overload `setDoseSym(AxisID, double)`.
    pub fn set_dose_sym_double(&mut self, axis_id: AxisID, input: f64) {
        self.set_dose_sym(axis_id, input.to_string());
    }
    pub fn set_exclude_list_enabled(&mut self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.exclude_list_b_enabled = enable;
        } else {
            self.exclude_list_a_enabled = enable;
        }
    }
    pub fn set_twodir_enabled(&mut self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.twodir_b_enabled = enable;
        } else {
            self.twodir_a_enabled = enable;
        }
    }
    pub fn set_dose_sym_enabled(&mut self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.dose_sym_b_enabled = enable;
        } else {
            self.dose_sym_a_enabled = enable;
        }
    }
    pub fn set_view_raw_stack_enabled(&mut self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.view_raw_stack_b_enabled = enable;
        } else {
            self.view_raw_stack_a_enabled = enable;
        }
    }
    /// Java `initTiltAngleFields`; the tilt-angle panel call stays at its direct boundary.
    pub fn init_tilt_angle_fields(&self, _axis_id: AxisID) {}
    /// Java deprecated `getDataset`.
    pub fn get_dataset(&self) -> &str {
        self.get_raw_image_stack()
    }
    /// Java `getTiltAngleFields`; direct tilt-angle-panel result.
    pub fn get_tilt_angle_fields(&self, panel_result: bool) -> bool {
        panel_result
    }
    pub fn get_views_to_skip(&self, axis_id: AxisID, _do_validation: bool) -> Option<String> {
        Some(self.get_exclude_list(axis_id, false))
    }
    pub fn get_backup_directory(&self) -> String {
        self.backup_directory.clone()
    }
    pub fn get_distortion_file(&self) -> String {
        self.distortion_file.clone()
    }
    pub fn get_mag_gradient_file(&self) -> String {
        self.mag_gradient_file.clone()
    }
    pub fn is_parallel_process_selected(&self, _property_user_dir: &str) -> bool {
        self.parallel_process
    }
    pub fn is_gpu_processing_selected(&self, _property_user_dir: &str) -> bool {
        self.gpu_processing
    }
    pub fn is_adjusted_focus_selected(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            self.adjusted_focus_b
        } else {
            self.adjusted_focus_a
        }
    }
    pub fn get_pixel_size(&self, _do_validation: bool) -> String {
        self.pixel_size.clone()
    }
    pub fn get_half_float_mode_output(&self) -> Option<i32> {
        if self.half_float_mode_output {
            Some(1)
        } else if self.half_float_mode_output_if_float {
            Some(2)
        } else {
            None
        }
    }
    pub fn get_fiducial_diameter(&self, _do_validation: bool) -> String {
        self.fiducial_diameter.clone()
    }
    /// Java `validateTiltAngle`; result from the direct tilt-angle boundary.
    pub fn validate_tilt_angle(&self, panel_result: bool) -> bool {
        panel_result
    }
    pub fn get_image_rotation(&self, _axis_id: AxisID, _do_validation: bool) -> String {
        self.image_rotation.clone()
    }

    pub fn equals_single_axis_action_command(&self, action_command: &str) -> bool {
        action_command == SINGLE_AXIS_LABEL
    }
    pub fn equals_dual_axis_action_command(&self, action_command: &str) -> bool {
        action_command == "Dual axis"
    }
    pub fn equals_single_view_action_command(&self, action_command: &str) -> bool {
        action_command == SINGLE_FRAME_LABEL
    }
    pub fn set_adjusted_focus_enabled(&mut self, axis_id: AxisID, enable: bool) {
        if axis_id == AxisID::Second {
            self.adjusted_focus_b_enabled = enable;
        } else {
            self.adjusted_focus_a_enabled = enable;
        }
    }
    pub fn equals_montage_action_command(&self, action_command: &str) -> bool {
        action_command == MONTAGE_LABEL
    }
    pub fn equals_scan_header_action_command(&self, action_command: &str) -> bool {
        action_command == "Scan Header"
    }
    /// Java `equalsTemplateActionCommand`; template-panel dispatch is retained at
    /// its canonical panel boundary, so its comparison result crosses this method.
    pub fn equals_template_action_command(&self, template_panel_matches: bool) -> bool {
        template_panel_matches
    }
    /// Java `expand(GlobalExpandButton)`.
    pub fn expand_global(&mut self, expanded: bool) {
        self.update_advanced(expanded);
    }
    pub fn expand(&self) {}
    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&mut self, advanced: bool) {
        if self.calibration_available {
            return;
        }
        self.distortion_file_visible = advanced;
        self.binning_visible = advanced;
        self.mag_gradient_info_visible = advanced;
    }
    /// Java `setMagGradientInfoVisible`; the Java method intentionally has no body.
    pub fn set_mag_gradient_info_visible(&mut self, _visible: bool) {}
    /// Java `getFile`; native chooser execution is an explicit GUI boundary. The
    /// chosen path is passed back as `selected_file`.
    pub fn get_file(&self, selected_file: Option<String>) -> Option<String> {
        selected_file
    }
    pub fn backup_directory_action(&mut self, selected_file: Option<String>) {
        if let Some(file) = self.get_file(selected_file) {
            self.backup_directory = file;
        }
    }
    pub fn distortion_file_action(&mut self, selected_file: Option<String>) {
        if let Some(file) = self.get_file(selected_file) {
            self.distortion_file = file;
        }
    }
    pub fn mag_gradient_file_action(&mut self, selected_file: Option<String>) {
        if let Some(file) = self.get_file(selected_file) {
            self.mag_gradient_file = file;
        }
    }
    pub fn set_raw_image_stack_tooltip(
        &mut self,
        field_tooltip: impl Into<String>,
        button_tooltip: impl Into<String>,
    ) {
        self.raw_image_stack_field_tooltip = field_tooltip.into();
        self.raw_image_stack_button_tooltip = button_tooltip.into();
    }
    pub fn set_backup_directory_tooltip(
        &mut self,
        field_tooltip: impl Into<String>,
        button_tooltip: impl Into<String>,
    ) {
        self.backup_directory_field_tooltip = field_tooltip.into();
        self.backup_directory_button_tooltip = button_tooltip.into();
    }
    pub fn set_scan_header_tooltip(&mut self, tooltip: impl Into<String>) {
        self.scan_header_tooltip = tooltip.into();
    }
    /// Java `getDoseSym(AxisID, boolean)`.
    pub fn get_dose_sym(&self, axis_id: AxisID, _do_validation: bool) -> String {
        if axis_id == AxisID::Second {
            self.dose_sym_b.clone()
        } else {
            self.dose_sym_a.clone()
        }
    }
    pub fn is_dose_sym(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            self.twodir_b_selected && self.dose_sym_b_selected
        } else {
            self.twodir_a_selected && self.dose_sym_a_selected
        }
    }
    pub fn is_remove_exclude_views_msg(&self, axis: AxisID) -> bool {
        if axis == AxisID::Second {
            !self.remove_exclude_views_msg_b.is_empty()
        } else {
            !self.remove_exclude_views_msg_a.is_empty()
        }
    }
    pub fn msg_exclude_views_succeeded(&mut self, axis: AxisID) {
        if axis == AxisID::Second {
            self.exclude_views_succeeded_b = true;
            self.exclude_list_b.clear();
            self.remove_exclude_views_msg_b = "Excluded views have been removed".into()
        } else {
            self.exclude_views_succeeded_a = true;
            self.exclude_list_a.clear();
            self.remove_exclude_views_msg_a = "Excluded views have been removed".into()
        }
    }
}

impl<E: SetupDialogExpert> ContextMenu for SetupDialog<E> {
    fn pop_up_context_menu(&mut self, mouse_event: MouseEvent) {
        SetupDialog::pop_up_context_menu(self, mouse_event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct E;
    impl SetupDialogExpert for E {
        fn done_setup_dialog(&mut self, _: bool, _: Option<&str>, _: bool) {}
        fn setup_recon_failed(&mut self) {}
    }
    #[test]
    fn derives_dataset_root() {
        let mut d = SetupDialog::new(E);
        d.set_raw_image_stack("/tmp/dataa.mrc");
        assert_eq!(d.get_dataset_name(), "data");
    }

    #[test]
    fn get_instance_retains_constructor_panel_and_listener_sequence() {
        let dialog =
            SetupDialog::get_instance(E, AxisID::First, DialogType::SetupRecon, true, false);
        assert!(dialog.root_panel_y_axis);
        assert!(dialog.dataset_panel_created);
        assert!(dialog.data_type_panel_created);
        assert!(dialog.per_axis_info_panel_created);
        assert!(dialog.listeners_added);
        assert!(!dialog.advanced_button_present);
        assert_eq!(dialog.raw_image_stack_display_limit, 50);
    }

    #[test]
    fn construction_assigns_axis_specific_raw_stack_commands() {
        let dialog = SetupDialog::new_with_construction(
            E,
            AxisID::First,
            DialogType::SetupRecon,
            false,
            true,
        );
        assert_eq!(dialog.view_raw_stack_a_command, "View Raw Image Stacka");
        assert_eq!(dialog.view_raw_stack_b_command, "View Raw Image Stackb");
        assert!(dialog.advanced_button_present);
    }

    #[test]
    fn action_routes_raw_stack_extension_to_matching_axis_button() {
        let mut dialog =
            SetupDialog::get_instance(E, AxisID::First, DialogType::SetupRecon, false, true);
        dialog.set_raw_image_stack("/data/seriesa.mrc");
        assert_eq!(
            dialog.action(&dialog.view_raw_stack_a_command),
            Some((".mrc".into(), AxisID::First))
        );
        assert_eq!(dialog.action("other"), None);
    }

    #[test]
    fn half_float_controls_are_exclusive_and_display_state_tracks_selection() {
        let mut dialog =
            SetupDialog::get_instance(E, AxisID::First, DialogType::SetupRecon, false, true);
        dialog.half_float_mode_output = true;
        dialog.half_float_mode_output_if_float = true;
        dialog.action_performed(Some("halfFloatModeOutput"));
        assert!(!dialog.half_float_mode_output_if_float);
        assert!(dialog.half_float_mode_output_active_visible);
        assert_eq!(dialog.get_half_float_mode_output(), Some(1));
    }

    #[test]
    fn twodir_and_dose_symmetric_state_are_axis_specific() {
        let mut dialog = SetupDialog::new(E);
        dialog.set_twodir(AxisID::First, "-60");
        dialog.set_dose_sym(AxisID::Second, "-45");
        assert!(dialog.is_twodir(AxisID::First));
        assert!(!dialog.is_twodir(AxisID::Second));
        assert_eq!(dialog.get_twodir(AxisID::First, true), "-60");
        assert_eq!(dialog.dose_sym_b, "-45");
    }

    #[test]
    fn advanced_state_tracks_exact_source_fields_except_with_calibration() {
        let mut dialog = SetupDialog::new_with_construction(
            E,
            AxisID::First,
            DialogType::SetupRecon,
            false,
            true,
        );
        dialog.update_advanced(true);
        assert!(dialog.distortion_file_visible);
        assert!(dialog.binning_visible);
        assert!(dialog.mag_gradient_info_visible);
        let mut calibration = SetupDialog::new_with_construction(
            E,
            AxisID::First,
            DialogType::SetupRecon,
            true,
            true,
        );
        calibration.update_advanced(true);
        assert!(!calibration.distortion_file_visible);
    }

    #[test]
    fn file_actions_and_tooltips_only_retain_selected_boundary_values() {
        let mut dialog = SetupDialog::new(E);
        dialog.backup_directory_action(Some("/data/backup".into()));
        dialog.distortion_file_action(None);
        dialog.mag_gradient_file_action(Some("/data/mag.grad".into()));
        dialog.set_raw_image_stack_tooltip("field", "button");
        assert_eq!(dialog.backup_directory, "/data/backup");
        assert!(dialog.distortion_file.is_empty());
        assert_eq!(dialog.mag_gradient_file, "/data/mag.grad");
        assert_eq!(dialog.raw_image_stack_button_tooltip, "button");
    }

    #[test]
    fn template_values_copy_axis_a_twodir_to_missing_axis_b_and_preserve_source_dose_bug() {
        let mut dialog = SetupDialog::new(E);
        dialog.update_template_values(SetupDialogDirectiveFileCollection {
            dual: Some(true),
            montage: Some(true),
            twodir_a: Some("-60".into()),
            dose_sym_a: Some("-50".into()),
            pixel_size: Some("1.2".into()),
            ..Default::default()
        });
        assert!(dialog.dual_axis);
        assert!(dialog.montage);
        assert_eq!(dialog.twodir_b, "-60");
        // The original assigns twodirA to tfDoseSym rather than doseSymA.
        assert_eq!(dialog.dose_sym_a, "-60");
        assert_eq!(dialog.dose_sym_b, "-50");
        assert_eq!(dialog.pixel_size, "1.2");
    }

    #[test]
    fn user_configuration_and_tooltips_drive_source_visible_values() {
        let mut dialog = SetupDialog::new(E);
        dialog.set_parameters_user_configuration(true);
        dialog.set_axis_type_tooltip("axis");
        dialog.set_view_raw_stack_tooltip("view");
        dialog.set_tooltips();
        assert!(dialog.remove_excluded_views);
        assert_eq!(
            dialog.tooltip_assignments[0],
            ("axisType".into(), "axis".into())
        );
        assert!(
            dialog
                .tooltip_assignments
                .iter()
                .any(|(name, _)| name == "removeExcludedViews")
        );
    }

    #[test]
    fn overload_state_and_axis_enablement_do_not_change_source_selections() {
        let mut dialog = SetupDialog::new(E);
        dialog.set_twodir_double(AxisID::First, -55.0);
        dialog.set_dose_sym_selected(AxisID::Second, true);
        dialog.set_twodir_enabled(AxisID::First, false);
        dialog.set_view_raw_stack_enabled(AxisID::Second, false);
        assert_eq!(dialog.get_twodir(AxisID::First, true), "-55");
        assert!(dialog.is_twodir(AxisID::First));
        assert!(dialog.is_dose_sym(AxisID::Second));
        assert!(!dialog.twodir_a_enabled);
        assert!(!dialog.view_raw_stack_b_enabled);
    }

    #[test]
    fn dataset_and_directory_accessors_match_source_contracts() {
        let mut dialog = SetupDialog::new(E);
        dialog.set_raw_image_stack("/work/dataset.mrc");
        assert_eq!(dialog.get_dataset(), "/work/dataset.mrc");
        assert_eq!(dialog.get_directory().as_deref(), Some("/work"));
        dialog.set_exclude_list(AxisID::Second, "1,3");
        assert_eq!(
            dialog.get_views_to_skip(AxisID::Second, true).as_deref(),
            Some("1,3")
        );
        assert!(dialog.get_tilt_angle_fields(true));
        assert!(!dialog.validate_tilt_angle(false));
    }

    #[test]
    fn source_listener_adapters_dispatch_their_exact_dialog_methods() {
        let mut dialog =
            SetupDialog::get_instance(E, AxisID::First, DialogType::SetupRecon, false, true);
        BackupDirectoryActionListener::action_performed(&mut dialog, Some("/backup".into()));
        dialog.set_raw_image_stack("/images/stack.mrc");
        assert_eq!(dialog.backup_directory, "/backup");
        assert_eq!(
            ViewRawStackAActionListener::action_performed(&dialog),
            Some((".mrc".into(), AxisID::First))
        );
        dialog.set_dual_axis(true);
        assert_eq!(dialog.get_axis_type(), AxisType::DualAxis);
        assert_eq!(
            SetupDialogActionListener::action_performed("scan".into()),
            "scan"
        );
    }
}
