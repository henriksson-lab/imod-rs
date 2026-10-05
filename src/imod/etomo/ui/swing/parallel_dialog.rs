//! `IMOD/Etomo/src/etomo/ui/swing/ParallelDialog.java`.
//!
//! The "Generic Parallel Process" dialog: set up chunk command files for a one-line
//! command with chunksetup, then run any set of chunk command files through
//! processchunks.  An event dispatch thread object (`Rc`, `&self` methods), created
//! by [`ParallelDialog::get_instance`].  `ParallelDialog implements
//! AbstractParallelDialog, ProcessInterface, ActionListener,
//! Run3dmodButtonContainer`; its `actionPerformed` is the closure the buttons are
//! given (holding a weak reference to the dialog).

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::file_text_field2::FileTextField2;
use super::label::Label;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::process_interface::ProcessInterface;
use super::radio_button::RadioButton;
use super::radio_button_interface::EnumeratedTypeRef;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::scaled_image;
use super::simple_button::SimpleButton;
use super::spaced_panel::SpacedPanel;
use super::spinner::Spinner;
use super::swing_component::SwingComponent;
use super::ui_harness;
use super::ui_utilities;
use crate::imod::etomo::base_manager::{self, BaseManager};
use crate::imod::etomo::comscript::chunksetup_param::ChunksetupParam;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_boolean2::EtomoBoolean2;
use crate::imod::etomo::r#type::image_output_format::ImageOutputFormat;
use crate::imod::etomo::r#type::parallel_meta_data::ParallelMetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::util::utilities;

/// Java private static final `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::Parallel;
/// Java private static final `PROCESS_NAME_LABEL`.
const PROCESS_NAME_LABEL: &str = "Process name: ";
/// Java private static final `USE_GPUS_LABEL`.
const USE_GPUS_LABEL: &str = "Use GPUs";
/// Java private static final `OVERLAP_PIXELS_STEP`.
const OVERLAP_PIXELS_STEP: i32 = 8;
/// Java private static final `MEMORY_PER_CHUNK_STEP`.
const MEMORY_PER_CHUNK_STEP: i32 = 50;
/// Java private static final `CHUNK_SETUP_OUTPUT_FILE_LABEL`.
const CHUNK_SETUP_OUTPUT_FILE_LABEL: &str = "Output File: ";

/// Java `public final class ParallelDialog`.
pub struct ParallelDialog {
    /// Java `this` (handed to the mediator and the listeners).
    this: Weak<ParallelDialog>,
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `btnChunkComscript`.
    btn_chunk_comscript: Rc<SimpleButton>,
    /// Java private final `ltfProcessName`.
    ltf_process_name: Rc<LabeledTextField>,
    /// Java private final `btnRunProcess`.
    btn_run_process: Rc<Run3dmodButton>,
    /// Java private final `cbUseGpus`.
    cb_use_gpus: Rc<CheckBox>,
    /// Java private final `nonQueueGpuCheckboxStatus`.
    non_queue_gpu_checkbox_status: RefCell<EtomoBoolean2>,
    /// Java private final `ltfOneLineCommandProgram`.
    ltf_one_line_command_program: Rc<LabeledTextField>,
    /// Java private final `ltfOneLineCommandArguments`.
    ltf_one_line_command_arguments: Rc<LabeledTextField>,
    /// Java private final `ltfSuffixForOutputName`.
    ltf_suffix_for_output_name: Rc<LabeledTextField>,
    /// Java private final `spOverlapPixels`.
    sp_overlap_pixels: Rc<Spinner>,
    /// Java private final `spMegavoxelMaximum`.
    sp_megavoxel_maximum: Rc<Spinner>,
    /// Java private final `btnChunkSetup`.
    btn_chunk_setup: Rc<MultiLineButton>,
    /// Java private final `lChunkSetupOutputFile`.
    l_chunk_setup_output_file: Rc<Label>,
    /// Java private final `bgFormatOfOutputFile`.
    bg_format_of_output_file: Rc<ButtonGroup>,
    /// Java private final `rbFormatOfOutputFileMrc`.
    rb_format_of_output_file_mrc: Rc<RadioButton>,
    /// Java private final `rbFormatOfOutputFileHdf`.
    rb_format_of_output_file_hdf: Rc<RadioButton>,
    /// Java private final `rbFormatOfOutputFileTiff`.
    rb_format_of_output_file_tiff: Rc<RadioButton>,
    /// Java private final `btnRunProcess3dmod`.
    btn_run_process_3dmod: Rc<Run3dmodButton>,

    /// Java private final `manager`.
    manager: &'static ParallelManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `mediator`.
    mediator: Option<Rc<ProcessingMethodMediator>>,
    /// Java private final `gpuAvailable`.
    gpu_available: bool,
    /// Java private final `ftfInputImageFile`.
    ftf_input_image_file: Rc<FileTextField2>,

    /// Java private `useQueueCheckBox`, initially null.
    use_queue_check_box: RefCell<Option<Rc<dyn ButtonComponent>>>,
    /// Java private `workingDir`, initially null.
    working_dir: RefCell<Option<PathBuf>>,
    /// Java private `chunkSetupOutputFile`, initially "".
    chunk_setup_output_file: RefCell<String>,
    /// Java private `locked`, initially false.
    locked: Cell<bool>,
    /// Java private `setupMode`, initially true.
    setup_mode: Cell<bool>,
}

impl ParallelDialog {
    /// Java private `ParallelDialog(ParallelManager, AxisID)`.
    fn new(manager: &'static ParallelManager, axis_id: AxisID) -> Rc<ParallelDialog> {
        Rc::new_cyclic(|this: &Weak<ParallelDialog>| {
            // Field initializers.
            let pnl_root = SpacedPanel::get_instance_void();
            let btn_chunk_comscript =
                SimpleButton::new_scaled_image(Some(if !*utilities::APRIL_FOOLS {
                    &scaled_image::OPEN_FILE
                } else {
                    &scaled_image::OPEN_FILE_FOOL
                }));
            let ltf_process_name = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some(PROCESS_NAME_LABEL),
            );
            let btn_run_process =
                Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                    Some("Run Parallel Process"),
                    Some(DialogType::Parallel),
                );
            let cb_use_gpus = CheckBox::new_string(Some(USE_GPUS_LABEL));
            let ltf_one_line_command_program =
                LabeledTextField::new_field_type_string(FieldType::String, Some("Program: "));
            let ltf_one_line_command_arguments =
                LabeledTextField::new_field_type_string(FieldType::String, Some("Arguments: "));
            let ltf_suffix_for_output_name = LabeledTextField::new_field_type_string(
                FieldType::String,
                Some("Suffix for output file: "),
            );
            let sp_overlap_pixels = Spinner::get_labeled_instance_string_int_int_int_int(
                Some("Min. overlap in pixels: "),
                OVERLAP_PIXELS_STEP,
                OVERLAP_PIXELS_STEP,
                OVERLAP_PIXELS_STEP * 10,
                OVERLAP_PIXELS_STEP,
            );
            let sp_megavoxel_maximum = Spinner::get_labeled_instance_string_int_int_int_int(
                Some("Max. megavoxels per chunk: "),
                MEMORY_PER_CHUNK_STEP * 5,
                MEMORY_PER_CHUNK_STEP,
                22 * MEMORY_PER_CHUNK_STEP,
                MEMORY_PER_CHUNK_STEP,
            );
            let btn_chunk_setup = MultiLineButton::new_string(Some("Run Chunksetup"));
            let l_chunk_setup_output_file = Label::new_string_string(
                Some(CHUNK_SETUP_OUTPUT_FILE_LABEL),
                Some(CHUNK_SETUP_OUTPUT_FILE_LABEL),
            );
            let bg_format_of_output_file = ButtonGroup::new();
            let rb_format_of_output_file_mrc = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(ImageOutputFormat::Mrc),
                Some(&bg_format_of_output_file),
            );
            let rb_format_of_output_file_hdf = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(ImageOutputFormat::Hdf),
                Some(&bg_format_of_output_file),
            );
            let rb_format_of_output_file_tiff = RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(ImageOutputFormat::Tiff),
                Some(&bg_format_of_output_file),
            );
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let btn_run_process_3dmod =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Run 3dmod"),
                    Some(container),
                );
            // Constructor body.
            eprintln!(
                "{}\nDialog: {}",
                utilities::get_date_time_stamp(),
                DialogType::Parallel
            );
            let mediator = manager.get_processing_method_mediator(Some(axis_id));
            let gpu_available = Network::get_total_gpus(
                manager,
                axis_id,
                manager.get_property_user_dir().as_deref(),
            ) >= 1;
            let ftf_input_image_file =
                FileTextField2::get_alt_layout_instance(Some(manager), Some("Input file: "));
            ParallelDialog {
                this: this.clone(),
                pnl_root,
                btn_chunk_comscript,
                ltf_process_name,
                btn_run_process,
                cb_use_gpus,
                non_queue_gpu_checkbox_status: RefCell::new(EtomoBoolean2::new()),
                ltf_one_line_command_program,
                ltf_one_line_command_arguments,
                ltf_suffix_for_output_name,
                sp_overlap_pixels,
                sp_megavoxel_maximum,
                btn_chunk_setup,
                l_chunk_setup_output_file,
                bg_format_of_output_file,
                rb_format_of_output_file_mrc,
                rb_format_of_output_file_hdf,
                rb_format_of_output_file_tiff,
                btn_run_process_3dmod,
                manager,
                axis_id,
                mediator,
                gpu_available,
                ftf_input_image_file,
                use_queue_check_box: RefCell::new(None),
                working_dir: RefCell::new(None),
                chunk_setup_output_file: RefCell::new(String::new()),
                locked: Cell::new(false),
                setup_mode: Cell::new(true),
            }
        })
    }

    /// The rest of the Java constructor body, which needs `this` (`Rc::new_cyclic`
    /// gives no strong reference while the fields are built).
    fn construct(&self) {
        // panels
        let pnl_chunk_setup = JComponent::new_panel();
        let pnl_one_line_command_program = JComponent::new_panel();
        let pnl_output = JComponent::new_panel();
        let pnl_spinners = JComponent::new_panel();
        let pnl_chunk_setup_buttons = JComponent::new_panel();
        let pnl_chunk_setup_output_file = JComponent::new_panel();
        let pnl_format_of_output_file = JComponent::new_panel();
        let pnl_process_name = JComponent::new_panel();
        let pnl_run_process = JComponent::new_panel();
        let pnl_use_gpus = JComponent::new_panel();
        let pnl_run_process_buttons = JComponent::new_panel();
        // init
        self.ltf_one_line_command_program.set_required(true);
        // Swing layout: ltfOneLineCommandProgram.setPreferredWidth(130);
        // ltfOneLineCommandArguments.setColumns(35).
        self.ftf_input_image_file.set_required(true);
        self.ltf_suffix_for_output_name.set_required(true);
        // Swing layout: ltfProcessName.setTextPreferredWidth(125);
        // ltfProcessName.setTextPreferredSize(...).
        self.btn_chunk_comscript.set_name(Some(PROCESS_NAME_LABEL));
        // Swing layout: btnChunkComscript.setPreferredSize(
        // UIUtilities.getScaledFolderButtonDimension()).
        let _ = ui_utilities::get_scaled_folder_button_dimension();
        self.btn_run_process
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_run_process_3dmod.clone() as Rc<dyn Deferred3dmodButton>,
            ));

        // Root
        self.pnl_root
            .set_border(&BeveledBorder::new(Some("Parallel Process")).get_border());
        self.pnl_root.set_box_layout(super::spaced_panel::Y_AXIS);
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y5)).
        self.pnl_root.add_j_panel(&pnl_chunk_setup);
        // Swing layout: pnlRoot.add(Box.createRigidArea(FixedDim.x0_y10)).
        self.pnl_root.add_j_panel(&pnl_run_process);

        // ChunkSetup
        // Swing layout: pnlChunkSetup BoxLayout Y_AXIS.
        pnl_chunk_setup.set_border_title(
            EtchedBorder::new(Some("Set Up Chunks for One-line Command"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_chunk_setup.add(&pnl_one_line_command_program);
        pnl_chunk_setup.add(&self.ltf_one_line_command_arguments.get_component());
        pnl_chunk_setup.add(&SwingComponent::get_component(&*self.ftf_input_image_file));
        pnl_chunk_setup.add(&pnl_output);
        pnl_chunk_setup.add(&pnl_spinners);
        pnl_chunk_setup.add(&pnl_chunk_setup_buttons);
        pnl_chunk_setup.add(&pnl_chunk_setup_output_file);
        // OneLineCommandProgram (BoxLayout X_AXIS, then horizontal glue)
        pnl_one_line_command_program.add(&self.ltf_one_line_command_program.get_component());
        // Output (BoxLayout X_AXIS)
        pnl_output.add(&self.ltf_suffix_for_output_name.get_component());
        pnl_output.add(&pnl_format_of_output_file);
        // FormatOfOutputFile (BoxLayout X_AXIS)
        pnl_format_of_output_file.add(&JComponent::new_label("Output file format: "));
        pnl_format_of_output_file.add(&self.rb_format_of_output_file_mrc.get_component());
        pnl_format_of_output_file.add(&self.rb_format_of_output_file_hdf.get_component());
        pnl_format_of_output_file.add(&self.rb_format_of_output_file_tiff.get_component());
        // Spinners (BoxLayout X_AXIS)
        pnl_spinners.add(&self.sp_overlap_pixels.get_component());
        pnl_spinners.add(&self.sp_megavoxel_maximum.get_component());
        // ChunkSetupButtons (BoxLayout X_AXIS, glue around the button)
        pnl_chunk_setup_buttons.add(&self.btn_chunk_setup.get_component());
        // ChunkSetupOutputFile (BoxLayout X_AXIS)
        pnl_chunk_setup_output_file.add(&self.l_chunk_setup_output_file.get_component());

        // RunProcess (BoxLayout Y_AXIS)
        pnl_run_process.add(&pnl_process_name);
        pnl_run_process.add(&pnl_use_gpus);
        pnl_run_process.add(&pnl_run_process_buttons);
        // ProcessName (BoxLayout X_AXIS)
        pnl_process_name.add(&self.ltf_process_name.get_component());
        pnl_process_name.add(&self.btn_chunk_comscript.get_component());
        // UseGpus (BoxLayout X_AXIS)
        pnl_use_gpus.add(&self.cb_use_gpus.get_component());
        // RunProcessButtons (BoxLayout X_AXIS, glue between the buttons)
        pnl_run_process_buttons.add(&self.btn_run_process.get_component());
        pnl_run_process_buttons.add(&self.btn_run_process_3dmod.get_component());

        self.set_tool_tip_text();
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.register_process_interface(origin.clone());
            mediator
                .set_method_process_interface_processing_method(&origin, ProcessingMethod::PpCpu);
        }
        self.update_display();
    }

    /// Java static `getInstance(ParallelManager, AxisID)`.
    pub fn get_instance(manager: &'static ParallelManager, axis_id: AxisID) -> Rc<ParallelDialog> {
        let instance = ParallelDialog::new(manager, axis_id);
        instance.construct();
        instance.add_listeners();
        instance
    }

    /// Java `actionPerformed(ActionEvent)`, as the listener the buttons are given.
    fn action_listener(&self) -> ActionListener {
        let adaptee = self.this.clone();
        Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action_event(Some(event));
            }
        })
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let listener = self.action_listener();
        self.btn_chunk_setup.add_action_listener(listener.clone());
        // new ChunkComscriptActionListener(this)
        let adaptee = self.this.clone();
        self.btn_chunk_comscript
            .get_component()
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.chunk_comscript_action();
                }
            }));
        let container: Weak<dyn Run3dmodButtonContainer> = self.this.clone();
        self.btn_run_process.set_container(Some(container));
        self.btn_run_process.add_action_listener(listener.clone());
        self.btn_run_process_3dmod
            .add_action_listener(listener.clone());
        self.cb_use_gpus.add_action_listener(Some(listener));
        if let Some(mediator) = &self.mediator {
            mediator.add_gpu_listener(vec![self.cb_use_gpus.clone() as Rc<dyn ButtonComponent>]);
        }
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java `getWorkingDir()`.
    pub fn get_working_dir(&self) -> Option<PathBuf> {
        self.working_dir.borrow().clone()
    }

    /// Java `done()`.
    pub fn done(&self) {
        if let (Some(mediator), Some(this)) = (
            self.manager
                .get_processing_method_mediator(Some(self.axis_id)),
            self.this.upgrade(),
        ) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.deregister_process_interface(&origin);
        }
    }

    /// Java `setParameters(BaseScreenState)`.
    pub fn set_parameters_screen_state(&self, screen_state: &BaseScreenState) {
        self.btn_run_process.set_button_state(
            screen_state.get_button_state(
                self.btn_run_process
                    .create_button_state_key(Some(DIALOG_TYPE))
                    .as_deref(),
            ),
        );
    }

    /// Java `setParameters(ParallelMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &ParallelMetaData) {
        self.ltf_process_name
            .set_text_string(meta_data.get_root_name().as_deref());
        self.cb_use_gpus
            .set_selected_boolean(meta_data.is_use_gpus());
        self.ltf_one_line_command_program
            .set_text_string(meta_data.get_one_line_command_program().as_deref());
        self.ltf_one_line_command_arguments
            .set_text_string(meta_data.get_one_line_command_arguments().as_deref());
        self.ftf_input_image_file
            .set_text_string(meta_data.get_input_image_file().as_deref());
        self.ltf_suffix_for_output_name
            .set_text_string(meta_data.get_suffix_for_output_name().as_deref());
        let format_of_output_file = meta_data.get_format_of_output_file();
        if format_of_output_file == ImageOutputFormat::Mrc {
            self.rb_format_of_output_file_mrc.set_selected_boolean(true);
        } else if format_of_output_file == ImageOutputFormat::Hdf {
            self.rb_format_of_output_file_hdf.set_selected_boolean(true);
        } else if format_of_output_file == ImageOutputFormat::Tiff {
            self.rb_format_of_output_file_tiff
                .set_selected_boolean(true);
        }
        self.sp_overlap_pixels
            .set_value_string(meta_data.get_overlap_pixels().as_deref());
        self.sp_megavoxel_maximum
            .set_value_string(meta_data.get_megavoxel_maximum().as_deref());
        self.set_method(self.get_processing_method());
    }

    /// Java `getParameters(BaseScreenState)`.
    pub fn get_parameters_screen_state(&self, screen_state: &BaseScreenState) {
        screen_state.set_button_state(
            self.btn_run_process.get_button_state_key().as_deref(),
            self.btn_run_process.get_button_state(),
        );
    }

    /// The selected `ImageOutputFormat`: Java `((RadioButton.RadioButtonModel)
    /// bgFormatOfOutputFile.getSelection()).getEnumeratedType()`.
    fn selected_format_of_output_file(&self) -> Option<EnumeratedTypeRef> {
        self.bg_format_of_output_file
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<super::radio_button::RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
    }

    /// Java `getParameters(ParallelMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &ParallelMetaData) {
        meta_data.set_root_name(self.ltf_process_name.get_text_void().as_deref());
        meta_data.set_use_gpus(self.cb_use_gpus.is_selected());
        meta_data.set_one_line_command_program(
            self.ltf_one_line_command_program.get_text_void().as_deref(),
        );
        meta_data.set_one_line_command_arguments(
            self.ltf_one_line_command_arguments
                .get_text_void()
                .as_deref(),
        );
        meta_data.set_input_image_file(self.ftf_input_image_file.get_text_void().as_deref());
        meta_data
            .set_suffix_for_output_name(self.ltf_suffix_for_output_name.get_text_void().as_deref());
        let format = self.selected_format_of_output_file();
        meta_data.set_format_of_output_file(format.as_deref());
        meta_data.set_overlap_pixels(Some(self.sp_overlap_pixels.get_value()));
        meta_data.set_megavoxel_maximum(Some(self.sp_megavoxel_maximum.get_value()));
    }

    /// Java `getParameters(ChunksetupParam, boolean)`.
    pub fn get_parameters_chunksetup_param(
        &self,
        param: &mut ChunksetupParam,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException> {
            let program = self.ltf_one_line_command_program.get_text_boolean(do_validation)?;
            let arguments = self
                .ltf_one_line_command_arguments
                .get_text_boolean(do_validation)?;
            param.set_one_line_command(program.as_deref(), arguments.as_deref());
            let input_image_file = self
                .ftf_input_image_file
                .get_text_boolean_field_displayer(do_validation, None)?;
            param.set_input_image_file(input_image_file.as_deref());
            let suffix = self.ltf_suffix_for_output_name.get_text_boolean(do_validation)?;
            param.set_suffix_for_output_name(suffix.as_deref());
            param.set_format_of_output_file(
                self.selected_format_of_output_file()
                    .and_then(|format| format.downcast_ref::<ImageOutputFormat>().copied()),
            );
            param.set_overlap_pixels(Some(self.sp_overlap_pixels.get_value()));
            param.set_megavoxel_maximum(Some(self.sp_megavoxel_maximum.get_value()));
            Ok(())
        })();
        // catch (FieldValidationFailedException e) { return false; }
        result.is_ok()
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        // Once the dataset file has been created, the process to run can't be changed.
        self.ltf_one_line_command_program
            .set_editable(self.setup_mode.get());
        self.ltf_process_name.set_editable(self.setup_mode.get());
        self.btn_chunk_comscript
            .get_component()
            .set_enabled(self.setup_mode.get());
        // The process method is lock during the parallel process run.
        self.cb_use_gpus
            .set_enabled(self.gpu_available && !self.locked.get());
        // The output file of the parallel process is only known if chunksetup was used to
        // create the chunk com files. Tell the process and 3dmod buttons to adjust their
        // right click menus to handle an unknown file.
        let file_to_open_known = !utilities::is_empty(Some(&self.chunk_setup_output_file.borrow()));
        self.btn_run_process
            .set_file_to_open_known(file_to_open_known);
        self.btn_run_process_3dmod
            .set_file_to_open_known(file_to_open_known);
    }

    /// Java `setSetupMode(boolean)`.
    pub fn set_setup_mode(&self, setup_mode: bool) {
        self.setup_mode.set(setup_mode);
        self.update_display();
    }

    /// Java package-private `action(ActionEvent)`.
    fn action_event(&self, event: Option<&ActionEvent>) {
        if let Some(event) = event {
            self.action(event.get_action_command().unwrap_or(""), None, None);
        }
    }

    /// Java package-private `chunkComscriptAction()`.
    fn chunk_comscript_action(&self) {
        let chunk_comscript =
            base_manager::chunk_comscript_action(Some(self.pnl_root.get_container()));
        if let Some(chunk_comscript) = chunk_comscript {
            let com_file_name =
                utilities::java_io_file_get_name(&chunk_comscript.to_string_lossy());
            // `comFileName.substring(0, comFileName.lastIndexOf("-0"))`; a name without
            // "-0" throws StringIndexOutOfBoundsException, which the source catches and
            // prints.
            match com_file_name.rfind("-0") {
                Some(index) => self.set_process_name(
                    chunk_comscript.parent().map(Path::to_path_buf),
                    &com_file_name[..index],
                ),
                None => eprintln!(
                    "java.lang.StringIndexOutOfBoundsException: begin 0, end -1, length {}",
                    com_file_name.encode_utf16().count()
                ),
            }
        }
    }

    /// Java `setProcessName(File, String)`.
    pub fn set_process_name(&self, dir: Option<PathBuf>, process_name: &str) {
        self.ltf_process_name.set_text_string(Some(process_name));
        *self.working_dir.borrow_mut() = dir;
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        // ChunkSetup
        let tool_tip_text = "Enter command for running a single program to produce the output from the input; omit filenames.";
        self.ltf_one_line_command_program
            .set_tool_tip_text(Some(tool_tip_text));
        self.ltf_one_line_command_arguments
            .set_tool_tip_text(Some(tool_tip_text));
        self.ftf_input_image_file
            .set_tool_tip_text(Some("Name of input image file."));
        self.ltf_suffix_for_output_name.set_tool_tip_text(Some(
            "String to add to the root of the input image file name, before extension, to produce the output image file name.",
        ));
        self.sp_overlap_pixels.set_tool_tip_text(Some(
            "Minimum number of pixels of overlap between the subvolumes.",
        ));
        self.sp_megavoxel_maximum.set_tool_tip_text(Some(
            "Limit each subvolume to the given number of megavoxels; typically memory usage will be 4 times the megavoxels.",
        ));
        self.ltf_process_name.set_tool_tip_text(Some(
            "The process name is based on the name of the first comscript (-001.com or -001-sync.com).",
        ));
        self.btn_chunk_comscript
            .get_component()
            .set_tool_tip_text(Some(
                "Selects the first comscript (-001.com or -001-sync.com).",
            ));
        self.btn_run_process
            .set_tool_tip_text(Some("Runs the process."));
    }

    /// Java `setChunkSetupOutputFile(String)`.
    pub fn set_chunk_setup_output_file(&self, chunk_setup_output_file: Option<&str>) {
        if utilities::is_empty(chunk_setup_output_file) {
            *self.chunk_setup_output_file.borrow_mut() = String::new();
        } else {
            // Java `String.trim()`.
            *self.chunk_setup_output_file.borrow_mut() = chunk_setup_output_file
                .unwrap()
                .trim_matches(|c: char| (c as u32) <= 0x20)
                .to_owned();
        }
        self.l_chunk_setup_output_file
            .get_component()
            .set_text(&format!(
                "{CHUNK_SETUP_OUTPUT_FILE_LABEL}{}",
                self.chunk_setup_output_file.borrow()
            ));
        self.update_display();
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(self.manager))
        });
    }
}

impl Run3dmodButtonContainer for ParallelDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let mut run_3dmod_menu_options = run_3dmod_menu_options;
        // try { ... } catch (FieldValidationFailedException e) {}
        if Some(action_command) == self.btn_chunk_setup.get_action_command().as_deref() {
            self.manager.chunksetup(
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                Some(DIALOG_TYPE),
                Some(self.get_processing_method()),
            );
        } else if Some(action_command) == self.btn_run_process.get_action_command().as_deref() {
            if self.ltf_process_name.is_editable() {
                let Ok(process_name) = self.ltf_process_name.get_text_boolean(true) else {
                    return;
                };
                let working_dir = self.working_dir.borrow().clone();
                // Fixed in translation (ParallelDialog.java:487): with no chunk
                // comscript chosen `workingDir` is null and Java's
                // `DatasetTool.validateDatasetName` throws NullPointerException on
                // the event dispatch thread, which ends the action; it ends here
                // without the exception.  (BUGS.md)
                let Some(working_dir) = working_dir else {
                    eprintln!("java.lang.NullPointerException: no working directory");
                    return;
                };
                if !dataset_tool::validate_dataset_name(
                    self.manager,
                    None,
                    Some(self.axis_id),
                    &working_dir,
                    process_name.as_deref(),
                    DataFileType::Parallel,
                    None,
                    true,
                ) {
                    return;
                }
            }
            let Ok(process_name) = self.ltf_process_name.get_text_boolean(true) else {
                return;
            };
            let run_method = self.mediator.as_ref().map(|mediator| {
                mediator.get_run_method_for_process_interface(self.get_processing_method())
            });
            self.manager.processchunks(
                Some(self.btn_run_process.clone() as ProcessResultDisplayHandle),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                process_name.as_deref(),
                None,
                run_method,
                Some(DIALOG_TYPE),
            );
        } else if Some(action_command) == self.btn_run_process_3dmod.get_action_command().as_deref()
        {
            let chunk_setup_output_file = self.chunk_setup_output_file.borrow().clone();
            if !utilities::is_empty(Some(&chunk_setup_output_file)) {
                self.manager.imod_string_file_run3dmod_menu_options(
                    imod_manager::GENERIC_PARALLEL_PROCESS_OUTPUT_FILE_KEY,
                    Some(Path::new(&chunk_setup_output_file)),
                    run_3dmod_menu_options,
                );
            } else {
                let mut options = run_3dmod_menu_options.take().unwrap_or_default();
                options.set_startup_window(true);
                self.manager.imod_string_run3dmod_menu_options(
                    imod_manager::GENERIC_PARALLEL_PROCESS_OUTPUT_UNKNOWN_FILE_KEY,
                    Some(options),
                );
            }
        } else if self.cb_use_gpus.get_action_command().as_deref() == Some(action_command) {
            self.set_method(self.get_processing_method());
        } else {
            let use_queue_check_box = self.use_queue_check_box.borrow().clone();
            if let Some(use_queue_check_box) = use_queue_check_box
                && use_queue_check_box.get_action_command().as_deref() == Some(action_command)
            {
                if use_queue_check_box.is_selected() {
                    self.non_queue_gpu_checkbox_status
                        .borrow_mut()
                        .set_boolean(self.cb_use_gpus.is_selected());
                } else {
                    self.cb_use_gpus
                        .set_selected_boolean(self.non_queue_gpu_checkbox_status.borrow().is());
                    self.set_method(self.get_processing_method());
                }
            }
        }
    }
}

impl AbstractParallelDialog for ParallelDialog {
    /// Java `getParameters(ParallelParam)`: empty.
    fn get_parameters(&self, _param: &mut dyn ParallelParam) {}

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        DIALOG_TYPE
    }
}

impl QueueTableListener for ParallelDialog {
    /// Java `queueTableEventAction(QueueTableEvent)`: empty.
    fn queue_table_event_action(&self, _event: &QueueTableEvent) {}
}

impl ProcessInterface for ParallelDialog {
    /// Java `updateGpu(boolean)`: empty.
    fn update_gpu(&self, _disable: bool) {}

    /// Java `getProcessingMethod()`.  Get the processing method based on the
    /// dialogs settings.  Dialogs don't need to know if QUEUE is in use in the
    /// parallel panel.
    fn get_processing_method(&self) -> ProcessingMethod {
        if self.cb_use_gpus.is_selected() {
            return ProcessingMethod::PpGpu;
        }
        ProcessingMethod::PpCpu
    }

    /// Java `getSecondaryProcessingMethod()`.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java `lockProcessingMethod(boolean)`.
    fn lock_processing_method(&self, lock: bool) {
        self.locked.set(lock);
        self.update_display();
    }

    /// Java `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(&origin, processing_method);
        }
    }

    /// Java `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        self.cb_use_gpus.is_enabled() && self.cb_use_gpus.is_selected()
    }

    /// Java `setUseQueueCheckBox(ButtonComponent)`.
    fn set_use_queue_check_box(&self, use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {
        if let Some(use_queue_check_box) = use_queue_check_box
            && self.use_queue_check_box.borrow().is_none()
        {
            use_queue_check_box.add_action_listener(self.action_listener());
            *self.use_queue_check_box.borrow_mut() = Some(use_queue_check_box);
        }
    }

    /// Java `addQueueTableListener(QueueTableListener)`: empty.
    fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java `removeQueueTableListener(QueueTableListener)`: empty.
    fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}
}
