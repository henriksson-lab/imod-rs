//! `IMOD/Etomo/src/etomo/ui/swing/ParallelDialog.java`.
//!
//! Swing controls, `ParallelManager`, `ProcessingMethodMediator`, and the
//! file chooser are retained as direct source boundaries.  This module owns
//! all of the Java dialog's state selection, validation order, and action
//! dispatch; it does not replace the parallel-process workflow with a second
//! command-line implementation.
#![allow(dead_code)]

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::check_box::CheckBox;
use super::parallel_panel::QueueTableEvent;
use super::process_interface::ProcessInterface;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use std::path::{Path, PathBuf};

pub const PROCESS_NAME_LABEL: &str = "Process name: ";
pub const USE_GPUS_LABEL: &str = "Use GPUs";
pub const OVERLAP_PIXELS_STEP: i32 = 8;
pub const MEMORY_PER_CHUNK_STEP: i32 = 50;
pub const CHUNK_SETUP_OUTPUT_FILE_LABEL: &str = "Output File: ";

/// Source-visible `BaseScreenState` value accessed by this dialog.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ParallelDialogScreenState {
    pub run_process_button_state: Option<String>,
}

/// Source-visible `ParallelMetaData` fields accessed by this dialog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelDialogMetaData {
    pub root_name: String,
    pub use_gpus: bool,
    pub one_line_command_program: String,
    pub one_line_command_arguments: String,
    pub input_image_file: String,
    pub suffix_for_output_name: String,
    pub format_of_output_file: ImageOutputFormat,
    pub overlap_pixels: i32,
    pub megavoxel_maximum: i32,
}

impl Default for ParallelDialogMetaData {
    fn default() -> Self {
        Self {
            root_name: String::new(),
            use_gpus: false,
            one_line_command_program: String::new(),
            one_line_command_arguments: String::new(),
            input_image_file: String::new(),
            suffix_for_output_name: String::new(),
            format_of_output_file: ImageOutputFormat::Mrc,
            overlap_pixels: OVERLAP_PIXELS_STEP,
            megavoxel_maximum: MEMORY_PER_CHUNK_STEP * 5,
        }
    }
}

/// Source-visible `ChunksetupParam` fields set by this dialog.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunksetupParamState {
    pub one_line_command: Option<(String, String)>,
    pub input_image_file: Option<String>,
    pub suffix_for_output_name: Option<String>,
    pub format_of_output_file: Option<ImageOutputFormat>,
    pub overlap_pixels: Option<i32>,
    pub megavoxel_maximum: Option<i32>,
}

/// Direct calls into Java `ParallelManager`, `ImodManager`, and
/// `ProcessingMethodMediator` recorded at their boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParallelDialogManagerCall {
    Deregister {
        axis_id: AxisID,
    },
    SetMethod {
        method: ProcessingMethod,
    },
    Chunksetup {
        method: ProcessingMethod,
    },
    Processchunks {
        process_name: String,
        method: ProcessingMethod,
    },
    ImodKnownFile {
        output_file: PathBuf,
    },
    ImodUnknownFile {
        startup_window: bool,
    },
    Pack {
        axis_id: AxisID,
    },
}

/// Java `ParallelDialog` source state.  Every primitive Swing field is kept as
/// the value Java reads or writes; actual button/widget construction is the
/// optional native GUI harness boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelDialog {
    pub axis_id: AxisID,
    pub gpu_available: bool,
    pub process_name: String,
    pub use_gpus: bool,
    pub one_line_command_program: String,
    pub one_line_command_arguments: String,
    pub input_image_file: String,
    pub suffix_for_output_name: String,
    pub overlap_pixels: i32,
    pub megavoxel_maximum: i32,
    pub format_of_output_file: ImageOutputFormat,
    pub working_dir: Option<PathBuf>,
    pub chunk_setup_output_file: String,
    pub chunk_setup_output_label: String,
    pub locked: bool,
    pub setup_mode: bool,
    pub non_queue_gpu_checkbox_status: bool,
    pub use_queue_checkbox_present: bool,
    pub use_queue_selected: bool,
    pub chunk_comscript_enabled: bool,
    pub one_line_command_program_editable: bool,
    pub process_name_editable: bool,
    pub use_gpus_enabled: bool,
    pub run_process_file_to_open_known: bool,
    pub run_process_3dmod_file_to_open_known: bool,
    pub chunk_setup_listener_present: bool,
    pub chunk_comscript_listener_present: bool,
    pub run_process_listener_present: bool,
    pub run_process_3dmod_listener_present: bool,
    pub use_gpus_listener_present: bool,
    pub queue_listener_present: bool,
    pub tool_tips_set: bool,
    pub run_process_button_state: Option<String>,
    pub manager_calls: Vec<ParallelDialogManagerCall>,
}

impl ParallelDialog {
    /// Java private constructor, without the native panel construction boundary.
    pub fn new(axis_id: AxisID, gpu_available: bool) -> Self {
        let mut instance = Self {
            axis_id,
            gpu_available,
            process_name: String::new(),
            use_gpus: false,
            one_line_command_program: String::new(),
            one_line_command_arguments: String::new(),
            input_image_file: String::new(),
            suffix_for_output_name: String::new(),
            overlap_pixels: OVERLAP_PIXELS_STEP,
            megavoxel_maximum: MEMORY_PER_CHUNK_STEP * 5,
            format_of_output_file: ImageOutputFormat::Mrc,
            working_dir: None,
            chunk_setup_output_file: String::new(),
            chunk_setup_output_label: CHUNK_SETUP_OUTPUT_FILE_LABEL.into(),
            locked: false,
            setup_mode: true,
            non_queue_gpu_checkbox_status: false,
            use_queue_checkbox_present: false,
            use_queue_selected: false,
            chunk_comscript_enabled: true,
            one_line_command_program_editable: true,
            process_name_editable: true,
            use_gpus_enabled: false,
            run_process_file_to_open_known: false,
            run_process_3dmod_file_to_open_known: false,
            chunk_setup_listener_present: false,
            chunk_comscript_listener_present: false,
            run_process_listener_present: false,
            run_process_3dmod_listener_present: false,
            use_gpus_listener_present: false,
            queue_listener_present: false,
            tool_tips_set: false,
            run_process_button_state: None,
            manager_calls: vec![ParallelDialogManagerCall::SetMethod {
                method: ProcessingMethod::PpCpu,
            }],
        };
        instance.set_tool_tip_text();
        instance.update_display();
        instance
    }

    /// Java `getInstance`, including its post-construction `addListeners` call.
    pub fn get_instance(axis_id: AxisID, gpu_available: bool) -> Self {
        let mut instance = Self::new(axis_id, gpu_available);
        instance.add_listeners();
        instance
    }

    /// Java `getProcessingMethod`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        if self.use_gpus {
            ProcessingMethod::PpGpu
        } else {
            ProcessingMethod::PpCpu
        }
    }

    /// Java `getSecondaryProcessingMethod`; Java returns null.
    pub fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java empty `updateGpu`.
    pub fn update_gpu(&mut self, _disable: bool) {}

    /// Java `lockProcessingMethod`.
    pub fn lock_processing_method(&mut self, lock: bool) {
        self.locked = lock;
        self.update_display();
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.chunk_setup_listener_present = true;
        self.chunk_comscript_listener_present = true;
        self.run_process_listener_present = true;
        self.run_process_3dmod_listener_present = true;
        self.use_gpus_listener_present = true;
    }

    /// Java `getDialogType`.
    pub fn get_dialog_type(&self) -> DialogType {
        DialogType::Parallel
    }

    /// Java `getWorkingDir`.
    pub fn get_working_dir(&self) -> Option<&Path> {
        self.working_dir.as_deref()
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.manager_calls
            .push(ParallelDialogManagerCall::Deregister {
                axis_id: self.axis_id,
            });
    }

    /// Java overloaded `setParameters(BaseScreenState)`.
    pub fn set_parameters_screen_state(&mut self, screen_state: &ParallelDialogScreenState) {
        self.run_process_button_state = screen_state.run_process_button_state.clone();
    }

    /// Java overloaded `setParameters(ParallelMetaData)`.
    pub fn set_parameters_meta_data(&mut self, meta_data: &ParallelDialogMetaData) {
        self.process_name.clone_from(&meta_data.root_name);
        self.use_gpus = meta_data.use_gpus;
        self.one_line_command_program
            .clone_from(&meta_data.one_line_command_program);
        self.one_line_command_arguments
            .clone_from(&meta_data.one_line_command_arguments);
        self.input_image_file
            .clone_from(&meta_data.input_image_file);
        self.suffix_for_output_name
            .clone_from(&meta_data.suffix_for_output_name);
        self.format_of_output_file = meta_data.format_of_output_file;
        self.overlap_pixels = meta_data.overlap_pixels;
        self.megavoxel_maximum = meta_data.megavoxel_maximum;
        self.manager_calls
            .push(ParallelDialogManagerCall::SetMethod {
                method: self.get_processing_method(),
            });
    }

    /// Java overloaded `getParameters(BaseScreenState)`.
    pub fn get_parameters_screen_state(&self, screen_state: &mut ParallelDialogScreenState) {
        screen_state.run_process_button_state = self.run_process_button_state.clone();
    }

    /// Java overloaded `getParameters(ParallelMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &mut ParallelDialogMetaData) {
        meta_data.root_name.clone_from(&self.process_name);
        meta_data.use_gpus = self.use_gpus;
        meta_data
            .one_line_command_program
            .clone_from(&self.one_line_command_program);
        meta_data
            .one_line_command_arguments
            .clone_from(&self.one_line_command_arguments);
        meta_data
            .input_image_file
            .clone_from(&self.input_image_file);
        meta_data
            .suffix_for_output_name
            .clone_from(&self.suffix_for_output_name);
        meta_data.format_of_output_file = self.format_of_output_file;
        meta_data.overlap_pixels = self.overlap_pixels;
        meta_data.megavoxel_maximum = self.megavoxel_maximum;
    }

    /// Java overloaded `getParameters(ChunksetupParam, boolean)`.
    pub fn get_parameters_chunksetup(
        &self,
        param: &mut ChunksetupParamState,
        do_validation: bool,
    ) -> bool {
        if do_validation
            && (self.one_line_command_program.trim().is_empty()
                || self.input_image_file.trim().is_empty()
                || self.suffix_for_output_name.trim().is_empty())
        {
            return false;
        }
        param.one_line_command = Some((
            self.one_line_command_program.clone(),
            self.one_line_command_arguments.clone(),
        ));
        param.input_image_file = Some(self.input_image_file.clone());
        param.suffix_for_output_name = Some(self.suffix_for_output_name.clone());
        param.format_of_output_file = Some(self.format_of_output_file);
        param.overlap_pixels = Some(self.overlap_pixels);
        param.megavoxel_maximum = Some(self.megavoxel_maximum);
        true
    }

    /// Java private `updateDisplay`.
    pub fn update_display(&mut self) {
        self.one_line_command_program_editable = self.setup_mode;
        self.process_name_editable = self.setup_mode;
        self.chunk_comscript_enabled = self.setup_mode;
        self.use_gpus_enabled = self.gpu_available && !self.locked;
        let file_to_open_known = !self.chunk_setup_output_file.is_empty();
        self.run_process_file_to_open_known = file_to_open_known;
        self.run_process_3dmod_file_to_open_known = file_to_open_known;
    }

    /// Java `setSetupMode`.
    pub fn set_setup_mode(&mut self, setup_mode: bool) {
        self.setup_mode = setup_mode;
        self.update_display();
    }

    /// Java `action(ActionEvent)` after extraction of the Swing action command.
    pub fn action_event(&mut self, action_command: Option<&str>) {
        self.action(action_command);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    /// Manager/3dmod calls are intentionally recorded at the native boundary.
    pub fn action(&mut self, action_command: Option<&str>) {
        let Some(action_command) = action_command else {
            return;
        };
        if action_command == "Run Chunksetup" {
            self.manager_calls
                .push(ParallelDialogManagerCall::Chunksetup {
                    method: self.get_processing_method(),
                });
        } else if action_command == "Run Parallel Process" {
            if self.process_name_editable && self.process_name.trim().is_empty() {
                return;
            }
            self.manager_calls
                .push(ParallelDialogManagerCall::Processchunks {
                    process_name: self.process_name.clone(),
                    method: self.get_processing_method(),
                });
        } else if action_command == "Run 3dmod" {
            if self.chunk_setup_output_file.is_empty() {
                self.manager_calls
                    .push(ParallelDialogManagerCall::ImodUnknownFile {
                        startup_window: true,
                    });
            } else {
                self.manager_calls
                    .push(ParallelDialogManagerCall::ImodKnownFile {
                        output_file: PathBuf::from(&self.chunk_setup_output_file),
                    });
            }
        } else if action_command == USE_GPUS_LABEL {
            self.manager_calls
                .push(ParallelDialogManagerCall::SetMethod {
                    method: self.get_processing_method(),
                });
        } else if self.use_queue_checkbox_present && action_command == "Use a cluster" {
            if self.use_queue_selected {
                self.non_queue_gpu_checkbox_status = self.use_gpus;
            } else {
                self.use_gpus = self.non_queue_gpu_checkbox_status;
                self.manager_calls
                    .push(ParallelDialogManagerCall::SetMethod {
                        method: self.get_processing_method(),
                    });
            }
        }
    }

    /// Java `chunkComscriptAction`, after its native file chooser boundary.
    pub fn chunk_comscript_action(&mut self, chunk_comscript: Option<&Path>) {
        let Some(chunk_comscript) = chunk_comscript else {
            return;
        };
        let Some(com_file_name) = chunk_comscript.file_name().and_then(|value| value.to_str())
        else {
            return;
        };
        let Some(index) = com_file_name.rfind("-0") else {
            return;
        };
        self.set_process_name(
            chunk_comscript.parent().map(Path::to_path_buf),
            &com_file_name[..index],
        );
    }

    /// Java `setProcessName`.
    pub fn set_process_name(&mut self, dir: Option<PathBuf>, process_name: &str) {
        self.process_name = process_name.into();
        self.working_dir = dir;
    }

    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.tool_tips_set = true;
    }

    /// Java `setMethod`.
    pub fn set_method(&mut self, processing_method: ProcessingMethod) {
        self.manager_calls
            .push(ParallelDialogManagerCall::SetMethod {
                method: processing_method,
            });
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self, action_command: Option<&str>) {
        self.action_event(action_command);
    }

    /// Java `isUseGpu`.
    pub fn is_use_gpu(&self) -> bool {
        self.use_gpus_enabled && self.use_gpus
    }

    /// Java `setUseQueueCheckBox`.
    pub fn set_use_queue_check_box(&mut self, use_queue_checkbox: Option<CheckBox>) {
        if use_queue_checkbox.is_some() && !self.use_queue_checkbox_present {
            self.queue_listener_present = true;
            self.use_queue_checkbox_present = true;
        }
    }

    /// Native `ButtonComponent.isSelected` state supplied before Java's queue
    /// checkbox listener dispatches `action`.
    pub fn set_use_queue_selected(&mut self, selected: bool) {
        self.use_queue_selected = selected;
    }

    /// Java `setChunkSetupOutputFile`.
    pub fn set_chunk_setup_output_file(&mut self, chunk_setup_output_file: Option<&str>) {
        self.chunk_setup_output_file = chunk_setup_output_file.unwrap_or_default().trim().into();
        self.chunk_setup_output_label = format!(
            "{CHUNK_SETUP_OUTPUT_FILE_LABEL}{}",
            self.chunk_setup_output_file
        );
        self.update_display();
        self.manager_calls.push(ParallelDialogManagerCall::Pack {
            axis_id: self.axis_id,
        });
    }

    /// Java empty `queueTableEventAction`.
    pub fn queue_table_event_action(&mut self, _event: QueueTableEvent) {}

    /// Java empty `addQueueTableListener`.
    pub fn add_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}

    /// Java empty `removeQueueTableListener`.
    pub fn remove_queue_table_listener(&mut self, _listener: &mut dyn QueueTableListener) {}
}

impl QueueTableListener for ParallelDialog {
    fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        Self::queue_table_event_action(self, event);
    }
}

impl ProcessInterface for ParallelDialog {
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

impl AbstractParallelDialog for ParallelDialog {
    /// Java overloaded `getParameters(ParallelParam)`, which is empty.
    fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

    fn get_dialog_type(&self) -> DialogType {
        self.get_dialog_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_uses_source_spinner_defaults_and_registers_cpu_method() {
        let dialog = ParallelDialog::get_instance(AxisID::Only, true);
        assert_eq!(dialog.overlap_pixels, 8);
        assert_eq!(dialog.megavoxel_maximum, 250);
        assert!(dialog.chunk_setup_listener_present);
        assert_eq!(
            dialog.manager_calls,
            vec![ParallelDialogManagerCall::SetMethod {
                method: ProcessingMethod::PpCpu
            }]
        );
    }

    #[test]
    fn output_file_switches_3dmod_action_and_menu_knowledge() {
        let mut dialog = ParallelDialog::new(AxisID::Only, true);
        dialog.action(Some("Run 3dmod"));
        assert!(matches!(
            dialog.manager_calls.last(),
            Some(ParallelDialogManagerCall::ImodUnknownFile {
                startup_window: true
            })
        ));
        dialog.set_chunk_setup_output_file(Some(" out.mrc "));
        assert!(dialog.run_process_3dmod_file_to_open_known);
        dialog.action(Some("Run 3dmod"));
        assert!(matches!(
            dialog.manager_calls.last(),
            Some(ParallelDialogManagerCall::ImodKnownFile { output_file }) if output_file == Path::new("out.mrc")
        ));
    }

    #[test]
    fn chunksetup_validation_follows_required_source_fields() {
        let mut dialog = ParallelDialog::new(AxisID::Only, false);
        let mut parameter = ChunksetupParamState::default();
        assert!(!dialog.get_parameters_chunksetup(&mut parameter, true));
        dialog.one_line_command_program = "tilt".into();
        dialog.input_image_file = "in.mrc".into();
        dialog.suffix_for_output_name = "_out".into();
        assert!(dialog.get_parameters_chunksetup(&mut parameter, true));
        assert_eq!(
            parameter.one_line_command,
            Some(("tilt".into(), String::new()))
        );
    }

    #[test]
    fn chunk_comscript_uses_last_dash_zero_as_java_does() {
        let mut dialog = ParallelDialog::new(AxisID::Only, false);
        dialog.chunk_comscript_action(Some(Path::new("dir/foo-001-sync.com")));
        assert_eq!(dialog.process_name, "foo");
        assert_eq!(dialog.get_working_dir(), Some(Path::new("dir")));
    }
}
