//! `IMOD/Etomo/src/etomo/ui/swing/AnisotropicDiffusionDialog.java`.
//!
//! Swing construction, file choosing, and the concrete `ParallelManager` are
//! presentation/application boundaries.  The dialog's field ownership,
//! parameter transfer, validation order, and action dispatch are kept here.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use super::process_interface::ProcessInterface;
use super::{
    check_box::CheckBox,
    filter_full_volume_panel::{self, FilterFullVolumePanel},
    labeled_text_field::LabeledTextField,
    multi_line_button::MultiLineButton,
    spinner::Spinner,
    tilt_panel::{Deferred3dmodButton, MouseEvent},
};
use crate::imod::etomo::r#type::{
    axis_id::AxisID, dialog_type::DialogType, processing_method::ProcessingMethod,
};
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;

pub const CLEANUP_LABEL: &str = filter_full_volume_panel::CLEANUP_LABEL;
pub const FILTER_FULL_VOLUME_LABEL: &str = filter_full_volume_panel::FILTER_FULL_VOLUME_LABEL;
pub const MEMORY_PER_CHUNK_DEFAULT: i32 = filter_full_volume_panel::MEMORY_PER_CHUNK_DEFAULT;
pub const MEMORY_PER_CHUNK_LABEL: &str = filter_full_volume_panel::MEMORY_PER_CHUNK_LABEL;
pub const TEST_VOLUME_NAME: &str = "test.input";
pub const K_VALUE_LIST_LABEL: &str = "List of K values: ";
pub const ITERATION_LIST_LABEL: &str = "List of iterations: ";
pub const K_VALUE_LABEL: &str = "K value: ";
pub const ITERATION_LABEL: &str = "Iterations: ";
pub const DIALOG_TYPE: DialogType = DialogType::AnisotropicDiffusion;

/// Java `FileTextField` source-owned state at the unported file chooser boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FileTextField {
    pub prompt: String,
    pub file: Option<PathBuf>,
    pub button_enabled: bool,
    pub action_listener_count: usize,
}
impl FileTextField {
    pub fn get_partial_path_instance(prompt: &str) -> Self {
        Self {
            prompt: prompt.into(),
            button_enabled: true,
            ..Self::default()
        }
    }
    pub fn get_file_name(&self) -> String {
        self.file
            .as_ref()
            .and_then(|f| f.file_name())
            .map(|x| x.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
    pub fn get_file_absolute_path(&self) -> String {
        self.file
            .as_ref()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
    pub fn set_text(&mut self, value: &str) {
        self.file = (!value.is_empty()).then(|| PathBuf::from(value));
    }
    pub fn set_file(&mut self, value: Option<&Path>) {
        self.file = value.map(Path::to_path_buf);
    }
    pub fn is_empty(&self) -> bool {
        self.file.is_none()
    }
    pub fn set_button_enabled(&mut self, value: bool) {
        self.button_enabled = value;
    }
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
}

/// Exact source-visible layout order; component painting remains a GUI boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnisotropicDiffusionDialogLayout {
    pub root_axis: &'static str,
    pub root_border: String,
    pub root_component_order: Vec<&'static str>,
    pub first_column_order: Vec<&'static str>,
    pub second_column_order: Vec<&'static str>,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
}

pub trait AnisotropicDiffusionDialogMetaData: filter_full_volume_panel::ParallelMetaData {
    fn set_root_name(&mut self, value: String);
    fn set_volume(&mut self, value: String);
    fn volume(&self) -> String;
    fn set_load_with_flipping(&mut self, value: bool);
    fn load_with_flipping(&self) -> bool;
    fn set_test_k_value_list(&mut self, value: String);
    fn test_k_value_list(&self) -> String;
    fn set_test_iteration(&mut self, value: i32);
    fn test_iteration(&self) -> i32;
    fn set_test_k_value(&mut self, value: String);
    fn test_k_value(&self) -> String;
    fn set_test_iteration_list(&mut self, value: String);
    fn test_iteration_list(&self) -> String;
}
pub trait AnisotropicDiffusionDialogRubberband {
    fn get_parameters_meta_data<M: AnisotropicDiffusionDialogMetaData>(&self, meta: &mut M);
    fn set_parameters_meta_data<M: AnisotropicDiffusionDialogMetaData>(&mut self, meta: &M);
    fn get_parameters_trimvol<T: AnisotropicDiffusionDialogTrimvolParam>(
        &self,
        param: &mut T,
        validate: bool,
    ) -> bool;
}
pub trait AnisotropicDiffusionDialogTrimvolParam {
    fn set_flipped_volume(&mut self, value: bool);
    fn set_swap_yz(&mut self, value: bool);
    fn set_rotate_x(&mut self, value: bool);
    fn set_convert_to_bytes(&mut self, value: bool);
    fn set_input_file_name(&mut self, value: String);
    fn set_output_file_name(&mut self, value: String);
    fn set_format_of_output_file(&mut self, value: String);
}
pub trait AnisotropicDiffusionDialogParam:
    filter_full_volume_panel::AnisotropicDiffusionParam
{
    fn set_subdir_name(&mut self, value: Option<String>);
    fn set_k_value_list(&mut self, value: String) -> Option<String>;
    fn set_iteration_list(&mut self, value: String) -> bool;
    fn set_format(&mut self, value: String);
    fn set_input_file_name(&mut self, value: String);
}
pub trait AnisotropicDiffusionDialogChunksetupParam:
    filter_full_volume_panel::ChunksetupParam
{
    fn set_command_file(&mut self, value: String);
    fn set_subdir_name(&mut self, value: Option<String>);
    fn set_input_file(&mut self, value: String);
    fn set_output_file(&mut self, value: String);
}
/// Java `ProcesschunksParam.setSubdirName` used by overloaded
/// `getParameters(ParallelParam)`.
pub trait AnisotropicDiffusionDialogParallelParam {
    fn set_subdir_name(&mut self, value: Option<String>);
}
/// Concrete `ParallelManager`, dataset validation, chooser, and UIHarness calls.
pub trait AnisotropicDiffusionDialogManager:
    filter_full_volume_panel::FilterFullVolumePanelManager
{
    fn make_subdir(&mut self, name: &str) -> bool;
    fn delete_subdir(&mut self, name: &str) -> bool;
    fn property_user_dir(&self) -> PathBuf;
    fn image_output_format(&self) -> String;
    fn open_message_dialog(&mut self, message: &str, title: &str);
    fn trim_volume(&mut self);
    fn anisotropic_diffusion_varying_k(&mut self, subdir: &str, method: ProcessingMethod);
    fn anisotropic_diffusion_varying_iteration(&mut self, subdir: &str);
    fn imod(&mut self, key: &str, file: PathBuf, options: Option<()>, flip: bool);
    fn imod_varying_k_value(&mut self, subdir: &str, test: &str, flip: bool);
    fn imod_varying_iteration(&mut self, subdir: &str, test: &str, flip: bool);
    fn set_new_param_file(&mut self, file: &Path);
    fn validate_dataset_name(&mut self, file: &Path) -> bool;
}

pub struct AnisotropicDiffusionDialog<R> {
    pub root_panel: AnisotropicDiffusionDialogLayout,
    pub btn_view_full_volume: MultiLineButton,
    pub ftf_volume: FileTextField,
    pub btn_extract_test_volume: MultiLineButton,
    pub btn_view_test_volume: MultiLineButton,
    pub cb_load_with_flipping: CheckBox,
    pub ltf_test_k_value_list: LabeledTextField,
    pub sp_test_iteration: Spinner,
    pub btn_run_varying_k: MultiLineButton,
    pub btn_view_varying_k: MultiLineButton,
    pub ltf_test_k_value: LabeledTextField,
    pub ltf_test_iteration_list: LabeledTextField,
    pub btn_run_varying_iteration: MultiLineButton,
    pub btn_view_varying_iteration: MultiLineButton,
    pub filter_full_volume_panel: FilterFullVolumePanel,
    pub pnl_test_volume_rubberband: R,
    pub test_volume_name: String,
    pub subdir_name: Option<String>,
    pub debug: bool,
    pub mediator_registered: bool,
}
impl<R> AnisotropicDiffusionDialog<R> {
    pub fn new<M: AnisotropicDiffusionDialogManager>(
        manager: &M,
        rubberband: R,
        test_volume_name: String,
    ) -> Self {
        let mut dialog = Self {
            root_panel: AnisotropicDiffusionDialogLayout::default(),
            btn_view_full_volume: MultiLineButton::new_with_label(Some("View Full Volume")),
            ftf_volume: FileTextField::get_partial_path_instance("Pick a volume"),
            btn_extract_test_volume: MultiLineButton::new_with_label(Some("Extract Test Volume")),
            btn_view_test_volume: MultiLineButton::new_with_label(Some("View Test Volume")),
            cb_load_with_flipping: CheckBox::new_with_text("Load with flipping"),
            ltf_test_k_value_list: LabeledTextField::new(
                FieldType::FloatingPointArray,
                K_VALUE_LIST_LABEL,
            ),
            sp_test_iteration: Spinner::get_labeled_instance(ITERATION_LABEL, 10, 1, 200, 1),
            btn_run_varying_k: MultiLineButton::new_with_label(Some("Run with Different K Values")),
            btn_view_varying_k: MultiLineButton::new_with_label(Some(
                "View Different K Values Test Results",
            )),
            ltf_test_k_value: LabeledTextField::new(FieldType::FloatingPoint, K_VALUE_LABEL),
            ltf_test_iteration_list: LabeledTextField::new(
                FieldType::IntegerList,
                ITERATION_LIST_LABEL,
            ),
            btn_run_varying_iteration: MultiLineButton::new_with_label(Some(
                "Run with Different Iterations",
            )),
            btn_view_varying_iteration: MultiLineButton::new_with_label(Some(
                "View Different Iteration Test Results",
            )),
            filter_full_volume_panel: FilterFullVolumePanel::get_instance(manager, DIALOG_TYPE),
            pnl_test_volume_rubberband: rubberband,
            test_volume_name,
            subdir_name: None,
            debug: false,
            mediator_registered: false,
        };
        dialog.create_panel();
        dialog.set_tool_tip_text();
        dialog.mediator_registered = true;
        dialog
    }
    pub fn get_instance<M: AnisotropicDiffusionDialogManager>(
        manager: &M,
        rubberband: R,
        test_volume_name: String,
    ) -> Self {
        let mut value = Self::new(manager, rubberband, test_volume_name);
        value.add_listeners();
        value
    }
    pub fn set_tool_tip_text(&mut self) {
        self.cb_load_with_flipping.set_tool_tip_text(Some("Load volumes into 3dmod with flipping of Y and Z; use this for a tomogram that has not been flipped or rotated in post-processing."));
        self.btn_view_full_volume
            .set_tool_tip_text(Some("View the full volume in 3dmod."));
        self.btn_extract_test_volume.set_tool_tip_text(Some(
            "Cut out a test volume from the indicated coordinate range.",
        ));
        self.btn_view_test_volume
            .set_tool_tip_text(Some("View the test volume in 3dmod."));
        self.ltf_test_k_value_list.set_tool_tip_text(Some("Set of K threshold values to try on the test volume with the given number of iterations."));
        self.sp_test_iteration
            .set_tool_tip_text(Some("Number of iterations to run for each K value."));
        self.root_panel.tooltip_initialized = true;
    }
    pub fn create_panel(&mut self) {
        self.ltf_test_k_value_list.set_required(true);
        self.ltf_test_iteration_list.set_required(true);
        self.ltf_test_k_value.set_required(true);
        self.ltf_test_k_value_list.set_text_preferred_width(10);
        self.ltf_test_iteration_list.set_text_preferred_width(10);
        self.root_panel.root_axis = "X_AXIS";
        self.root_panel.root_border = "Anisotropic Diffusion".into();
        self.root_panel.root_component_order = vec!["first", "second"];
        self.root_panel.first_column_order =
            vec!["volume", "load-with-flipping", "extract-test-volume"];
        self.root_panel.second_column_order =
            vec!["varying-k", "varying-iterations", "filter-full-volume"];
    }
    pub fn pop_up_context_menu(&self, _mouse_event: MouseEvent) {}
    pub fn set_debug(&mut self, input: bool) {
        self.debug = input;
    }
    pub fn get_processing_method(&self) -> ProcessingMethod {
        ProcessingMethod::PpCpu
    }
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }
    pub fn lock_processing_method(&mut self, _lock: bool) {}
    pub fn add_listeners(&mut self) {
        self.ftf_volume.add_action_listener();
        for button in [
            &mut self.btn_view_full_volume,
            &mut self.btn_extract_test_volume,
            &mut self.btn_view_test_volume,
            &mut self.btn_run_varying_k,
            &mut self.btn_view_varying_k,
            &mut self.btn_run_varying_iteration,
            &mut self.btn_view_varying_iteration,
        ] {
            button.add_action_listener();
        }
        self.root_panel.listener_count = 8;
    }
    pub fn get_dialog_type(&self) -> DialogType {
        DIALOG_TYPE
    }
    /// Java overloaded `getParameters(ParallelParam)`.
    pub fn get_parameters_parallel<P: AnisotropicDiffusionDialogParallelParam>(
        &self,
        param: &mut P,
    ) {
        param.set_subdir_name(self.subdir_name.clone());
    }
    pub fn get_container(&self) -> &AnisotropicDiffusionDialogLayout {
        &self.root_panel
    }
    pub fn get_memory_per_chunk(&self) -> i32 {
        self.filter_full_volume_panel.get_memory_per_chunk()
    }
    pub fn init_subdir<M: AnisotropicDiffusionDialogManager>(&mut self, manager: &mut M) -> bool {
        if self.ftf_volume.is_empty() {
            manager.open_message_dialog(
                "Please choose a volume before running this function.",
                "Entry Error",
            );
            return false;
        }
        if self.subdir_name.is_none() {
            let name = format!("naddir.{}", self.ftf_volume.get_file_name());
            if !manager.make_subdir(&name) {
                return false;
            }
            self.subdir_name = Some(name);
        }
        true
    }
    pub fn get_subdirectory<M: AnisotropicDiffusionDialogManager>(
        &mut self,
        manager: &mut M,
    ) -> Option<String> {
        self.init_subdir(manager)
            .then(|| self.subdir_name.clone())
            .flatten()
    }
    pub fn clean_up<M: AnisotropicDiffusionDialogManager>(&mut self, manager: &mut M) {
        if self
            .subdir_name
            .as_deref()
            .is_some_and(|name| manager.delete_subdir(name))
        {
            self.subdir_name = None;
        }
    }
    pub fn get_volume(&self) -> String {
        self.ftf_volume.get_file_absolute_path()
    }
    pub fn is_load_with_flipping(&self) -> bool {
        self.cb_load_with_flipping.is_selected()
    }
    pub fn set_method(&mut self, _processing_method: ProcessingMethod) {}
    pub fn is_use_gpu(&self) -> bool {
        false
    }
    pub fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}
    pub fn set_use_queue_check_box(&mut self, _use_queue_checkbox: Option<CheckBox>) {}
    pub fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    pub fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
    pub fn update_gpu(&mut self, _disable_gpu: bool) {}
}

impl<R> QueueTableListener for AnisotropicDiffusionDialog<R> {
    fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        Self::queue_table_event_action(self, event);
    }
}

impl<R> ProcessInterface for AnisotropicDiffusionDialog<R> {
    type QueueCheckBox = CheckBox;
    fn update_gpu(&mut self, disable_gpu: bool) {
        Self::update_gpu(self, disable_gpu);
    }
    fn get_processing_method(&self) -> ProcessingMethod {
        Self::get_processing_method(self)
    }
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        Self::get_secondary_processing_method(self)
    }
    fn lock_processing_method(&mut self, lock: bool) {
        Self::lock_processing_method(self, lock);
    }
    fn set_method(&mut self, method: ProcessingMethod) {
        Self::set_method(self, method);
    }
    fn is_use_gpu(&self) -> bool {
        Self::is_use_gpu(self)
    }
    fn set_use_queue_check_box(&mut self, check_box: Option<CheckBox>) {
        Self::set_use_queue_check_box(self, check_box);
    }
    fn add_queue_table_listener(&mut self, listener: &mut dyn QueueTableListener) {
        Self::add_queue_table_listener(self, listener);
    }
    fn remove_queue_table_listener(&mut self, listener: &mut dyn QueueTableListener) {
        Self::remove_queue_table_listener(self, listener);
    }
}
impl<R: AnisotropicDiffusionDialogRubberband> AnisotropicDiffusionDialog<R> {
    pub fn get_initial_parameters<M: AnisotropicDiffusionDialogMetaData>(&self, meta: &mut M) {
        meta.set_root_name(self.ftf_volume.get_file_name());
        meta.set_volume(self.ftf_volume.get_file_absolute_path());
    }
    pub fn get_parameters_meta_data<M: AnisotropicDiffusionDialogMetaData>(&self, meta: &mut M) {
        meta.set_load_with_flipping(self.cb_load_with_flipping.is_selected());
        self.pnl_test_volume_rubberband
            .get_parameters_meta_data(meta);
        meta.set_test_k_value_list(self.ltf_test_k_value_list.get_text());
        meta.set_test_iteration(self.sp_test_iteration.get_value());
        meta.set_test_k_value(self.ltf_test_k_value.get_text());
        meta.set_test_iteration_list(self.ltf_test_iteration_list.get_text());
        self.filter_full_volume_panel.get_parameters_meta_data(meta);
    }
    /// Java `getParametersForTrimvol(ParallelMetaData)`.
    pub fn get_parameters_for_trimvol_meta_data<M: AnisotropicDiffusionDialogMetaData>(
        &self,
        meta: &mut M,
    ) {
        self.pnl_test_volume_rubberband
            .get_parameters_meta_data(meta);
    }
    pub fn set_parameters<
        M: AnisotropicDiffusionDialogMetaData + AnisotropicDiffusionDialogManager,
    >(
        &mut self,
        manager: &mut M,
        meta: &M,
    ) {
        self.ftf_volume.set_button_enabled(false);
        self.ftf_volume.set_text(&meta.volume());
        self.cb_load_with_flipping
            .set_selected(meta.load_with_flipping());
        self.pnl_test_volume_rubberband
            .set_parameters_meta_data(meta);
        self.ltf_test_k_value_list
            .set_text(&meta.test_k_value_list());
        self.sp_test_iteration.set_value(meta.test_iteration());
        self.ltf_test_k_value.set_text(&meta.test_k_value());
        self.ltf_test_iteration_list
            .set_text(&meta.test_iteration_list());
        self.filter_full_volume_panel.set_parameters_meta_data(meta);
        self.init_subdir(manager);
    }
    pub fn get_parameters_for_trimvol<T: AnisotropicDiffusionDialogTrimvolParam>(
        &self,
        param: &mut T,
        validate: bool,
        format: String,
    ) -> bool {
        if !self
            .pnl_test_volume_rubberband
            .get_parameters_trimvol(param, validate)
        {
            return false;
        }
        param.set_flipped_volume(self.cb_load_with_flipping.is_selected());
        param.set_swap_yz(false);
        param.set_rotate_x(false);
        param.set_convert_to_bytes(false);
        param.set_input_file_name(self.ftf_volume.get_file_name());
        param.set_output_file_name(
            self.subdir_name
                .as_ref()
                .map(|d| {
                    Path::new(d)
                        .join(&self.test_volume_name)
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default(),
        );
        param.set_format_of_output_file(format);
        true
    }
    pub fn get_parameters_for_varying_k<
        M: AnisotropicDiffusionDialogManager,
        P: AnisotropicDiffusionDialogParam,
    >(
        &self,
        manager: &mut M,
        param: &mut P,
        validate: bool,
    ) -> bool {
        let value = match self.ltf_test_k_value_list.get_text_validated(validate) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if let Some(error) = param.set_k_value_list(value) {
            manager.open_message_dialog(&format!("{K_VALUE_LIST_LABEL}{error}"), "Entry Error");
            return false;
        }
        let Some(subdir) = self.subdir_name.as_ref() else {
            return false;
        };
        if !manager
            .property_user_dir()
            .join(subdir)
            .join(&self.test_volume_name)
            .exists()
        {
            manager.open_message_dialog(
                "Test volume has not been created.  Please extract test volume.",
                "Entry Error",
            );
            return false;
        }
        param.set_iteration(self.sp_test_iteration.get_value());
        param.set_subdir_name(Some(subdir.clone()));
        param.set_input_file_name(self.test_volume_name.clone());
        true
    }
    pub fn get_parameters_anisotropic_diffusion<P: AnisotropicDiffusionDialogParam>(
        &self,
        param: &mut P,
        validate: bool,
    ) -> bool {
        param.set_subdir_name(self.subdir_name.clone());
        self.filter_full_volume_panel
            .get_parameters_anisotropic_diffusion(param, validate)
    }
    pub fn get_parameters_chunksetup<P: AnisotropicDiffusionDialogChunksetupParam>(
        &self,
        param: &mut P,
        output: String,
    ) {
        self.filter_full_volume_panel
            .get_parameters_chunksetup(param);
        param.set_command_file("nad_eed_3d.com".into());
        param.set_subdir_name(self.subdir_name.clone());
        param.set_input_file(self.ftf_volume.get_file_name());
        param.set_output_file(output);
    }
    pub fn get_parameters_for_varying_iteration<P: AnisotropicDiffusionDialogParam>(
        &self,
        param: &mut P,
        validate: bool,
        format: String,
    ) -> bool {
        let k = match self.ltf_test_k_value.get_text_validated(validate) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let iterations = match self.ltf_test_iteration_list.get_text_validated(validate) {
            Ok(v) => v,
            Err(_) => return false,
        };
        param.set_k_value(k);
        if !param.set_iteration_list(iterations) {
            return false;
        }
        param.set_format(format);
        param.set_subdir_name(self.subdir_name.clone());
        param.set_input_file_name(self.test_volume_name.clone());
        true
    }
    pub fn action<M: AnisotropicDiffusionDialogManager>(&mut self, manager: &mut M, command: &str) {
        if self.btn_extract_test_volume.get_action_command() == Some(command) {
            if self.init_subdir(manager) {
                manager.trim_volume();
            }
        } else if self.btn_run_varying_k.get_action_command() == Some(command) {
            if self.init_subdir(manager) {
                manager.anisotropic_diffusion_varying_k(
                    self.subdir_name.as_deref().unwrap(),
                    self.get_processing_method(),
                );
            }
        } else if self.btn_run_varying_iteration.get_action_command() == Some(command) {
            if self.init_subdir(manager) {
                manager
                    .anisotropic_diffusion_varying_iteration(self.subdir_name.as_deref().unwrap());
            }
        } else if self.btn_view_full_volume.get_action_command() == Some(command) {
            if let Some(file) = self.ftf_volume.file.clone() {
                manager.imod(
                    "Volume",
                    file,
                    None,
                    self.cb_load_with_flipping.is_selected(),
                );
            }
        } else if self.btn_view_test_volume.get_action_command() == Some(command) {
            if let Some(subdir) = &self.subdir_name {
                manager.imod(
                    "TestVolume",
                    Path::new(subdir).join(&self.test_volume_name),
                    None,
                    self.cb_load_with_flipping.is_selected(),
                );
            }
        } else if self.btn_view_varying_k.get_action_command() == Some(command) {
            if let Some(subdir) = &self.subdir_name {
                manager.imod_varying_k_value(
                    subdir,
                    &self.test_volume_name,
                    self.cb_load_with_flipping.is_selected(),
                );
            }
        } else if self.btn_view_varying_iteration.get_action_command() == Some(command) {
            if let Some(subdir) = &self.subdir_name {
                manager.imod_varying_iteration(
                    subdir,
                    &self.test_volume_name,
                    self.cb_load_with_flipping.is_selected(),
                );
            }
        }
    }
    pub fn open_volume<M: AnisotropicDiffusionDialogManager>(
        &mut self,
        manager: &mut M,
        volume: Option<PathBuf>,
    ) {
        let Some(volume) = volume else {
            return;
        };
        if volume.is_dir() || !volume.exists() {
            manager.open_message_dialog("Please choose a volume", "Entry Error");
            return;
        }
        if !manager.validate_dataset_name(&volume) {
            return;
        }
        self.ftf_volume.set_file(Some(&volume));
        manager.set_new_param_file(&volume);
        if !self.init_subdir(manager) {
            self.ftf_volume.set_file(None);
            return;
        }
        self.ftf_volume.set_button_enabled(false);
    }
}

/// Java private nested `ADDActionListener` adapter.
pub struct AddActionListener;
impl AddActionListener {
    pub fn action_performed<
        M: AnisotropicDiffusionDialogManager,
        R: AnisotropicDiffusionDialogRubberband,
    >(
        dialog: &mut AnisotropicDiffusionDialog<R>,
        manager: &mut M,
        command: &str,
    ) {
        dialog.action(manager, command);
    }
}
/// Java private nested `VolumeActionListener` adapter.
pub struct VolumeActionListener;
impl VolumeActionListener {
    pub fn action_performed<
        M: AnisotropicDiffusionDialogManager,
        R: AnisotropicDiffusionDialogRubberband,
    >(
        dialog: &mut AnisotropicDiffusionDialog<R>,
        manager: &mut M,
        volume: Option<PathBuf>,
    ) {
        dialog.open_volume(manager, volume);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Rubberband;
    impl AnisotropicDiffusionDialogRubberband for Rubberband {
        fn get_parameters_meta_data<M: AnisotropicDiffusionDialogMetaData>(&self, _: &mut M) {}
        fn set_parameters_meta_data<M: AnisotropicDiffusionDialogMetaData>(&mut self, _: &M) {}
        fn get_parameters_trimvol<T: AnisotropicDiffusionDialogTrimvolParam>(
            &self,
            _: &mut T,
            _: bool,
        ) -> bool {
            true
        }
    }
    #[derive(Default)]
    struct Manager {
        made: Vec<String>,
        messages: Vec<String>,
    }
    impl filter_full_volume_panel::FilterFullVolumePanelManager for Manager {
        fn chunksetup(
            &mut self,
            _: Option<&Deferred3dmodButton>,
            _: Option<crate::imod::etomo::process::imod_process::Run3dmodMenuOptions>,
            _: DialogType,
            _: ProcessingMethod,
        ) {
        }
        fn imod_anisotropic_diffusion_output(
            &mut self,
            _: Option<crate::imod::etomo::process::imod_process::Run3dmodMenuOptions>,
            _: bool,
        ) {
        }
        fn anisotropic_diffusion_suffix(&self) -> String {
            String::new()
        }
    }
    impl AnisotropicDiffusionDialogManager for Manager {
        fn make_subdir(&mut self, n: &str) -> bool {
            self.made.push(n.into());
            true
        }
        fn delete_subdir(&mut self, _: &str) -> bool {
            true
        }
        fn property_user_dir(&self) -> PathBuf {
            PathBuf::from("/")
        }
        fn image_output_format(&self) -> String {
            "mrc".into()
        }
        fn open_message_dialog(&mut self, m: &str, _: &str) {
            self.messages.push(m.into())
        }
        fn trim_volume(&mut self) {}
        fn anisotropic_diffusion_varying_k(&mut self, _: &str, _: ProcessingMethod) {}
        fn anisotropic_diffusion_varying_iteration(&mut self, _: &str) {}
        fn imod(&mut self, _: &str, _: PathBuf, _: Option<()>, _: bool) {}
        fn imod_varying_k_value(&mut self, _: &str, _: &str, _: bool) {}
        fn imod_varying_iteration(&mut self, _: &str, _: &str, _: bool) {}
        fn set_new_param_file(&mut self, _: &Path) {}
        fn validate_dataset_name(&mut self, _: &Path) -> bool {
            true
        }
    }
    #[test]
    fn constructor_retains_source_layout_and_listeners() {
        let manager = Manager::default();
        let dialog =
            AnisotropicDiffusionDialog::get_instance(&manager, Rubberband, TEST_VOLUME_NAME.into());
        assert_eq!(dialog.root_panel.root_component_order, ["first", "second"]);
        assert_eq!(dialog.root_panel.listener_count, 8);
        assert!(dialog.ltf_test_k_value.required);
    }
    #[test]
    fn empty_volume_prevents_subdirectory_creation() {
        let mut manager = Manager::default();
        let mut dialog =
            AnisotropicDiffusionDialog::get_instance(&manager, Rubberband, TEST_VOLUME_NAME.into());
        assert!(!dialog.init_subdir(&mut manager));
        assert_eq!(manager.messages.len(), 1);
    }
    #[test]
    fn subdirectory_is_derived_once_from_volume_file_name() {
        let mut manager = Manager::default();
        let mut dialog =
            AnisotropicDiffusionDialog::get_instance(&manager, Rubberband, TEST_VOLUME_NAME.into());
        dialog.ftf_volume.set_text("sample.rec");
        assert!(dialog.init_subdir(&mut manager));
        assert_eq!(
            dialog.get_subdirectory(&mut manager).as_deref(),
            Some("naddir.sample.rec")
        );
        assert_eq!(manager.made, ["naddir.sample.rec"]);
    }
}
