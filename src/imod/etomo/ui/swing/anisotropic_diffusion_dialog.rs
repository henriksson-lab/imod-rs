//! `IMOD/Etomo/src/etomo/ui/swing/AnisotropicDiffusionDialog.java`.
//!
//! The "Nonlinear Anisotropic Diffusion" dialog: pick a volume, extract a test
//! volume (trimvol), run nad_eed_3d on it with different K values (processchunks)
//! or different iterations, and filter the full volume (`FilterFullVolumePanel`).
//! An event dispatch thread object (`Rc`, `&self` methods), created by
//! [`AnisotropicDiffusionDialog::get_instance`].

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::abstract_parallel_dialog::AbstractParallelDialog;
use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::file_chooser::{self, FileChooser};
use super::file_text_field::FileTextField;
use super::file_text_field_interface::FileTextFieldInterface;
use super::filter_full_volume_panel::{self, FilterFullVolumePanel};
use super::filter_full_volume_parent::FilterFullVolumeParent;
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::process_interface::ProcessInterface;
use super::rubberband_panel::RubberbandPanel;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::spinner::Spinner;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::anisotropic_diffusion_param::{self, AnisotropicDiffusionParam};
use crate::imod::etomo::comscript::chunksetup_param::ChunksetupParam;
use crate::imod::etomo::comscript::parallel_param::ParallelParam;
use crate::imod::etomo::comscript::processchunks_param::ProcesschunksParam;
use crate::imod::etomo::comscript::trimvol_param::TrimvolParam;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent, MouseEvent, MouseListener};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::processing_method_mediator::ProcessingMethodMediator;
use crate::imod::etomo::storage::tomogram_file_filter::TomogramFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::base_meta_data::BaseMetaData;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::parallel_meta_data::ParallelMetaData;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::queue_table_event::QueueTableEvent;
use crate::imod::etomo::ui::queue_table_listener::QueueTableListener;
use crate::imod::etomo::util::utilities;

/// Java `CLEANUP_LABEL = FilterFullVolumePanel.CLEANUP_LABEL`.
pub const CLEANUP_LABEL: &str = filter_full_volume_panel::CLEANUP_LABEL;
/// Java `FILTER_FULL_VOLUME_LABEL = FilterFullVolumePanel.FILTER_FULL_VOLUME_LABEL`.
pub const FILTER_FULL_VOLUME_LABEL: &str = filter_full_volume_panel::FILTER_FULL_VOLUME_LABEL;
/// Java `MEMORY_PER_CHUNK_DEFAULT = FilterFullVolumePanel.MEMORY_PER_CHUNK_DEFAULT`.
pub const MEMORY_PER_CHUNK_DEFAULT: i32 = filter_full_volume_panel::MEMORY_PER_CHUNK_DEFAULT;
/// Java `MEMORY_PER_CHUNK_LABEL = FilterFullVolumePanel.MEMORY_PER_CHUNK_LABEL`.
pub const MEMORY_PER_CHUNK_LABEL: &str = filter_full_volume_panel::MEMORY_PER_CHUNK_LABEL;
/// Java `TEST_VOLUME_NAME`, deprecated 7/13/2020 (replaced with testVolumeName).
pub const TEST_VOLUME_NAME: &str = "test.input";
/// Java private static final `K_VALUE_LIST_LABEL`.
const K_VALUE_LIST_LABEL: &str = "List of K values: ";
/// Java `ITERATION_LIST_LABEL`.
pub const ITERATION_LIST_LABEL: &str = "List of iterations: ";
/// Java private static final `K_VALUE_LABEL`.
const K_VALUE_LABEL: &str = "K value: ";
/// Java private static final `ITERATION_LABEL`.
const ITERATION_LABEL: &str = "Iterations: ";
/// Java private static final `DIALOG_TYPE`.
const DIALOG_TYPE: DialogType = DialogType::AnisotropicDiffusion;

/// Java `public final class AnisotropicDiffusionDialog implements ContextMenu,
/// AbstractParallelDialog, Run3dmodButtonContainer, FilterFullVolumeParent,
/// ProcessInterface`.
pub struct AnisotropicDiffusionDialog {
    /// Java `this`.
    this: Weak<AnisotropicDiffusionDialog>,
    /// Java private final `rootPanel = SpacedPanel.getInstance()`.
    root_panel: Rc<SpacedPanel>,
    /// Java private final `btnViewFullVolume`.
    btn_view_full_volume: Rc<Run3dmodButton>,
    /// Java private final `ftfVolume`.
    ftf_volume: Rc<FileTextField>,
    /// Java private final `btnExtractTestVolume`.
    btn_extract_test_volume: Rc<MultiLineButton>,
    /// Java private final `btnViewTestVolume`.
    btn_view_test_volume: Rc<Run3dmodButton>,
    /// Java private final `cbLoadWithFlipping`.
    cb_load_with_flipping: Rc<CheckBox>,
    /// Java private final `ltfTestKValueList`.
    ltf_test_k_value_list: Rc<LabeledTextField>,
    /// Java private final `spTestIteration`.
    sp_test_iteration: Rc<Spinner>,
    /// Java private final `btnRunVaryingK`.
    btn_run_varying_k: Rc<Run3dmodButton>,
    /// Java private final `btnViewVaryingK`.
    btn_view_varying_k: Rc<Run3dmodButton>,
    /// Java private final `ltfTestKValue`.
    ltf_test_k_value: Rc<LabeledTextField>,
    /// Java private final `ltfTestIterationList`.
    ltf_test_iteration_list: Rc<LabeledTextField>,
    /// Java private final `btnRunVaryingIteration`.
    btn_run_varying_iteration: Rc<Run3dmodButton>,
    /// Java private final `btnViewVaryingIteration`.
    btn_view_varying_iteration: Rc<Run3dmodButton>,
    /// Java private final `filterFullVolumePanel`.
    filter_full_volume_panel: Rc<FilterFullVolumePanel>,

    /// Java private final `pnlTestVolumeRubberband`.
    pnl_test_volume_rubberband: RefCell<Option<Rc<RubberbandPanel>>>,
    /// Java private final `manager`.
    manager: &'static ParallelManager,
    /// Java private final `mediator`.
    mediator: Option<Rc<ProcessingMethodMediator>>,
    /// Java private final `testVolumeName`.
    test_volume_name: Option<String>,

    /// Java private `subdirName`, initially null.
    subdir_name: RefCell<Option<String>>,
    /// Java private `debug`, initially false.
    debug: Cell<bool>,
}

impl AnisotropicDiffusionDialog {
    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        self.cb_load_with_flipping.set_tool_tip_text(Some(
            "Load volumes into 3dmod with flipping of Y and Z; use this for a tomogram that has not been flipped or rotated in post-processing.",
        ));
        self.btn_view_full_volume
            .set_tool_tip_text(Some("View the full volume in 3dmod."));
        self.btn_extract_test_volume.set_tool_tip_text(Some(
            "Cut out a test volume from the indicated coordinate range.",
        ));
        self.btn_view_test_volume
            .set_tool_tip_text(Some("View the test volume in 3dmod."));
        self.ltf_test_k_value_list.set_tool_tip_text(Some(
            "Set of K threshold values to try on the test volume with the given number of iterations.",
        ));
        self.sp_test_iteration
            .set_tool_tip_text(Some("Number of iterations to run for each K value."));
        self.btn_run_varying_k.set_tool_tip_text(Some(
            "Compute a set of test volumes with the different K threshold values and a fixed number of iterations",
        ));
        self.btn_view_varying_k.set_tool_tip_text(Some(
            "View the volumes computed with different K values in 3dmod.",
        ));
        self.ltf_test_k_value.set_tool_tip_text(Some(
            "Single K threshold value to use with different numbers of iterations.",
        ));
        self.ltf_test_iteration_list.set_tool_tip_text(Some(
            "List of number of iterations to try with the single K value. Comma-separated ranges of numbers are allowed.",
        ));
        self.btn_run_varying_iteration.set_tool_tip_text(Some(
            "Compute a set of test volumes with the different numbers of iterations and a fixed K value",
        ));
        self.btn_view_varying_iteration.set_tool_tip_text(Some(
            "View the volumes computed with different iteration numbers in 3dmod",
        ));
    }

    /// Java package-private `setDebug(boolean)`.
    pub fn set_debug(&self, input: bool) {
        self.debug.set(input);
    }

    /// Java private `AnisotropicDiffusionDialog(ParallelManager)`, field initializers
    /// and the part of the body that does not need `this`.
    fn new(manager: &'static ParallelManager) -> Rc<AnisotropicDiffusionDialog> {
        Rc::new_cyclic(|this: &Weak<AnisotropicDiffusionDialog>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            let root_panel = SpacedPanel::get_instance_void();
            let btn_view_full_volume =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Full Volume"),
                    Some(container.clone()),
                );
            let ftf_volume = FileTextField::get_partial_path_instance("Pick a volume");
            let btn_extract_test_volume = MultiLineButton::new_string(Some("Extract Test Volume"));
            let btn_view_test_volume =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Test Volume"),
                    Some(container.clone()),
                );
            let cb_load_with_flipping = CheckBox::new_string(Some("Load with flipping"));
            let ltf_test_k_value_list = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointArray,
                Some(K_VALUE_LIST_LABEL),
            );
            let sp_test_iteration =
                Spinner::get_labeled_instance_string_int_int_int(Some(ITERATION_LABEL), 10, 1, 200);
            let btn_run_varying_k =
                Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                    Some("Run with Different K Values"),
                    Some(container.clone()),
                );
            let btn_view_varying_k =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Different K Values Test Results"),
                    Some(container.clone()),
                );
            let ltf_test_k_value = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(K_VALUE_LABEL),
            );
            let ltf_test_iteration_list = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some(ITERATION_LIST_LABEL),
            );
            let btn_run_varying_iteration =
                Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                    Some("Run with Different Iterations"),
                    Some(container.clone()),
                );
            let btn_view_varying_iteration =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Different Iteration Test Results"),
                    Some(container),
                );
            // Constructor body.
            eprintln!(
                "{}\nDialog: {}",
                utilities::get_date_time_stamp(),
                DialogType::AnisotropicDiffusion
            );
            let mediator = manager.get_processing_method_mediator(Some(AxisID::Only));
            let parent: Weak<dyn FilterFullVolumeParent> = this.clone();
            let filter_full_volume_panel =
                FilterFullVolumePanel::get_instance(manager, DIALOG_TYPE, parent);
            let test_volume_name = file_type::CLASS
                .nad_test_input
                .get_file_name(Some(manager), Some(AxisID::Only));
            AnisotropicDiffusionDialog {
                this: this.clone(),
                root_panel,
                btn_view_full_volume,
                ftf_volume,
                btn_extract_test_volume,
                btn_view_test_volume,
                cb_load_with_flipping,
                ltf_test_k_value_list,
                sp_test_iteration,
                btn_run_varying_k,
                btn_view_varying_k,
                ltf_test_k_value,
                ltf_test_iteration_list,
                btn_run_varying_iteration,
                btn_view_varying_iteration,
                filter_full_volume_panel,
                pnl_test_volume_rubberband: RefCell::new(None),
                manager,
                mediator,
                test_volume_name,
                subdir_name: RefCell::new(None),
                debug: Cell::new(false),
            }
        })
    }

    /// The rest of the Java constructor body (the layout).
    fn construct(&self) {
        // root
        self.root_panel.set_box_layout(spaced_panel::X_AXIS);
        if *utilities::APRIL_FOOLS {
            self.root_panel
                .set_border(&BeveledBorder::new(Some("Anisotropic Delusion")).get_border());
        } else {
            self.root_panel
                .set_border(&BeveledBorder::new(Some("Anisotropic Diffusion")).get_border());
        }
        // Swing layout: rootPanel.setComponentAlignmentX(Component.CENTER_ALIGNMENT).
        // init
        self.ltf_test_k_value_list.set_required(true);
        self.ltf_test_iteration_list.set_required(true);
        self.ltf_test_k_value.set_required(true);
        // first column
        let pnl_first = SpacedPanel::get_instance_void();
        pnl_first.set_box_layout(spaced_panel::Y_AXIS);
        // volume
        pnl_first.add_container(&self.ftf_volume.get_container());
        let pnl_load_with_flipping = SpacedPanel::get_instance_void();
        pnl_load_with_flipping.set_box_layout(spaced_panel::X_AXIS);
        pnl_load_with_flipping.add_check_box(&self.cb_load_with_flipping);
        pnl_load_with_flipping.add_horizontal_glue();
        pnl_first.add_spaced_panel(&pnl_load_with_flipping);
        // extract
        let pnl_extract = SpacedPanel::get_instance_void();
        pnl_extract.set_box_layout(spaced_panel::Y_AXIS);
        pnl_extract.set_border(&EtchedBorder::new(Some("Extract Test Volume")).get_border());
        // Swing layout: pnlExtract.setComponentAlignmentX(Component.CENTER_ALIGNMENT).
        let pnl_test_volume_rubberband = RubberbandPanel::get_instance_base_manager_string_string_string_string_string_string_string_string_string_run3dmod_button(
            self.manager,
            Some(imod_manager::VOLUME_KEY),
            Some("Test Volume Range:"),
            Some("Get Test Volume Range from 3dmod"),
            Some("Minimum X coordinate on the left side for the test volume range."),
            Some("Maximum X coordinate on the right side for the test volume range."),
            Some("The lower Y coordinate for the test volume range."),
            Some("The upper Y coordinate for the test volume range."),
            Some("The starting slice for the test volume range."),
            Some("The ending slice for the test volume range."),
            Some(self.btn_view_full_volume.clone()),
        );
        pnl_extract.add_container(&pnl_test_volume_rubberband.get_container());
        *self.pnl_test_volume_rubberband.borrow_mut() = Some(pnl_test_volume_rubberband);
        let pnl_extract_buttons = SpacedPanel::get_instance_void();
        pnl_extract_buttons.set_box_layout(spaced_panel::X_AXIS);
        pnl_extract_buttons.add_horizontal_glue();
        pnl_extract_buttons.add_multi_line_button(&self.btn_extract_test_volume);
        pnl_extract_buttons.add_horizontal_glue();
        pnl_extract_buttons.add_multi_line_button(&self.btn_view_test_volume);
        pnl_extract_buttons.add_horizontal_glue();
        pnl_extract.add_spaced_panel(&pnl_extract_buttons);
        pnl_first.add_spaced_panel(&pnl_extract);
        self.root_panel.add_spaced_panel(&pnl_first);
        // second column
        let pnl_second = SpacedPanel::get_instance_void();
        pnl_second.set_box_layout(spaced_panel::Y_AXIS);
        // varying K
        let pnl_varying_k = SpacedPanel::get_instance_void();
        pnl_varying_k.set_box_layout(spaced_panel::Y_AXIS);
        pnl_varying_k
            .set_border(&EtchedBorder::new(Some("Find a K Value for Test Volume")).get_border());
        let pnl_varying_k_fields = SpacedPanel::get_instance_void();
        pnl_varying_k_fields.set_box_layout(spaced_panel::X_AXIS);
        // Swing layout: ltfTestKValueList.setTextPreferredWidth(
        // UIParameters.getInstance().getListWidth()).
        pnl_varying_k_fields.add_labeled_text_field(&self.ltf_test_k_value_list);
        pnl_varying_k_fields.add_spinner(&self.sp_test_iteration);
        pnl_varying_k.add_spaced_panel(&pnl_varying_k_fields);
        let pnl_varying_k_buttons = SpacedPanel::get_instance_void();
        pnl_varying_k_buttons.set_box_layout(spaced_panel::X_AXIS);
        pnl_varying_k_buttons.add_horizontal_glue();
        self.btn_run_varying_k
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_view_varying_k.clone() as Rc<dyn Deferred3dmodButton>
            ));
        pnl_varying_k_buttons.add_multi_line_button(&self.btn_run_varying_k);
        pnl_varying_k_buttons.add_horizontal_glue();
        pnl_varying_k_buttons.add_multi_line_button(&self.btn_view_varying_k);
        pnl_varying_k_buttons.add_horizontal_glue();
        pnl_varying_k.add_spaced_panel(&pnl_varying_k_buttons);
        pnl_second.add_spaced_panel(&pnl_varying_k);
        // varying iterations
        let pnl_varying_iteration = SpacedPanel::get_instance_void();
        pnl_varying_iteration.set_box_layout(spaced_panel::Y_AXIS);
        pnl_varying_iteration.set_border(
            &EtchedBorder::new(Some("Find an Iteration Number for Test Volume")).get_border(),
        );
        let pnl_varying_iteration_fields = SpacedPanel::get_instance_void();
        pnl_varying_iteration_fields.set_box_layout(spaced_panel::X_AXIS);
        pnl_varying_iteration_fields.add_labeled_text_field(&self.ltf_test_k_value);
        // Swing layout: ltfTestIterationList.setTextPreferredWidth(...getListWidth()).
        pnl_varying_iteration_fields.add_labeled_text_field(&self.ltf_test_iteration_list);
        pnl_varying_iteration.add_spaced_panel(&pnl_varying_iteration_fields);
        let pnl_varying_iteration_buttons = SpacedPanel::get_instance_void();
        pnl_varying_iteration_buttons.set_box_layout(spaced_panel::X_AXIS);
        pnl_varying_iteration_buttons.add_horizontal_glue();
        self.btn_run_varying_iteration
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_view_varying_iteration.clone() as Rc<dyn Deferred3dmodButton>,
            ));
        pnl_varying_iteration_buttons.add_multi_line_button(&self.btn_run_varying_iteration);
        pnl_varying_iteration_buttons.add_horizontal_glue();
        pnl_varying_iteration_buttons.add_multi_line_button(&self.btn_view_varying_iteration);
        pnl_varying_iteration_buttons.add_horizontal_glue();
        pnl_varying_iteration.add_spaced_panel(&pnl_varying_iteration_buttons);
        pnl_second.add_spaced_panel(&pnl_varying_iteration);
        pnl_second.add_component(&self.filter_full_volume_panel.get_component());
        self.root_panel.add_spaced_panel(&pnl_second);
        self.set_tool_tip_text();
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.register_process_interface(origin.clone());
            mediator.set_method_process_interface_processing_method(
                &origin,
                self.get_processing_method(),
            );
        }
    }

    /// Java static `getInstance(ParallelManager, AxisID)`.
    pub fn get_instance(
        manager: &'static ParallelManager,
        _axis_id: AxisID,
    ) -> Rc<AnisotropicDiffusionDialog> {
        let instance = AnisotropicDiffusionDialog::new(manager);
        instance.construct();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // new VolumeActionListener(this)
        let adaptee = self.this.clone();
        self.ftf_volume
            .add_action_listener(Rc::new(move |_event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.open_volume();
                }
            }));
        // new ADDActionListener(this)
        let adaptee = self.this.clone();
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            }
        });
        self.btn_view_full_volume
            .add_action_listener(listener.clone());
        self.btn_extract_test_volume
            .add_action_listener(listener.clone());
        self.btn_view_test_volume
            .add_action_listener(listener.clone());
        self.btn_run_varying_k.add_action_listener(listener.clone());
        self.btn_view_varying_k
            .add_action_listener(listener.clone());
        self.btn_run_varying_iteration
            .add_action_listener(listener.clone());
        self.btn_view_varying_iteration
            .add_action_listener(listener);
        let context_menu: Weak<dyn ContextMenu> = self.this.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.root_panel
            .get_container()
            .add_mouse_listener(mouse_adapter);
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.get_container()
    }

    /// The rubberband panel (assigned during construction).
    fn rubberband(&self) -> Rc<RubberbandPanel> {
        self.pnl_test_volume_rubberband
            .borrow()
            .clone()
            .expect("pnlTestVolumeRubberband is assigned by the constructor")
    }

    /// Java `getInitialParameters(ParallelMetaData)`.
    pub fn get_initial_parameters(&self, meta_data: &ParallelMetaData) {
        meta_data.set_root_name(self.ftf_volume.get_file_name().as_deref());
        meta_data.set_volume(self.ftf_volume.get_file_absolute_path().as_deref());
    }

    /// Java `getParameters(ParallelMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &ParallelMetaData) {
        meta_data.set_load_with_flipping(self.cb_load_with_flipping.is_selected());
        self.rubberband()
            .get_parameters_parallel_meta_data(meta_data);
        meta_data.set_test_k_value_list(self.ltf_test_k_value_list.get_text_void().as_deref());
        meta_data.set_test_iteration(Some(self.sp_test_iteration.get_value()));
        meta_data.set_test_k_value(self.ltf_test_k_value.get_text_void().as_deref());
        meta_data.set_test_iteration_list(self.ltf_test_iteration_list.get_text_void().as_deref());
        self.filter_full_volume_panel
            .get_parameters_meta_data(meta_data);
    }

    /// Java `getParametersForTrimvol(ParallelMetaData)`.
    pub fn get_parameters_for_trimvol(&self, meta_data: &ParallelMetaData) {
        self.rubberband().get_parameters_for_trimvol(meta_data);
    }

    /// Java `getMemoryPerChunk()`.
    pub fn get_memory_per_chunk(&self) -> Number {
        self.filter_full_volume_panel.get_memory_per_chunk()
    }

    /// Java `setParameters(ParallelMetaData)`.
    pub fn set_parameters(&self, meta_data: &ParallelMetaData) {
        self.ftf_volume.set_button_enabled(false);
        self.ftf_volume
            .set_text_string(meta_data.get_volume().as_deref());
        self.cb_load_with_flipping
            .set_selected_boolean(meta_data.is_load_with_flipping());
        self.rubberband()
            .set_parameters_parallel_meta_data(meta_data);
        self.ltf_test_k_value_list
            .set_text_string(meta_data.get_test_k_value_list().as_deref());
        self.sp_test_iteration
            .set_value_const_etomo_number(&meta_data.get_test_iteration());
        self.ltf_test_k_value
            .set_text_string(meta_data.get_test_k_value().as_deref());
        self.ltf_test_iteration_list
            .set_text_string(meta_data.get_test_iteration_list().as_deref());
        self.filter_full_volume_panel.set_parameters(meta_data);
        self.init_subdir();
    }

    /// Java `getParameters(TrimvolParam, boolean)`.  Get the parameter values from
    /// the panel.
    pub fn get_parameters_trimvol_param(
        &self,
        param: &mut TrimvolParam,
        do_validation: bool,
    ) -> bool {
        if !self
            .rubberband()
            .get_parameters_trimvol_param_boolean(param, do_validation)
        {
            return false;
        }
        param.set_flipped_volume(self.cb_load_with_flipping.is_selected());
        param.set_swap_yz(false);
        param.set_rotate_x(false);
        param.set_convert_to_bytes(false);
        param.set_input_file_name(self.ftf_volume.get_file_name().as_deref().unwrap_or("null"));
        param.set_output_file_name(&utilities::java_io_file_new(
            self.subdir_name.borrow().as_deref().unwrap_or("null"),
            self.test_volume_name.as_deref().unwrap_or("null"),
        ));
        param.set_format_of_output_file(Some(
            self.manager
                .get_meta_data()
                .base()
                .get_image_output_format(),
        ));
        true
    }

    /// Java `getParametersForVaryingK(AnisotropicDiffusionParam, boolean)`.
    pub fn get_parameters_for_varying_k(
        &self,
        param: &mut AnisotropicDiffusionParam,
        do_validation: bool,
    ) -> bool {
        if self.debug.get() {
            println!(
                "getParametersForVaryingK:ltfTestKValueList.getText()={}",
                self.ltf_test_k_value_list
                    .get_text_void()
                    .unwrap_or_else(|| "null".to_owned())
            );
        }
        let Ok(text) = self.ltf_test_k_value_list.get_text_boolean(do_validation) else {
            return false;
        };
        let error_message = param.set_k_value_list(text.as_deref());
        if let Some(error_message) = error_message {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    &format!("{K_VALUE_LIST_LABEL}{error_message}"),
                    "Entry Error",
                )
            });
            return false;
        }
        param.set_iteration(Some(self.sp_test_iteration.get_value()));
        // Must use an absolute file when testing file existance. Changing the
        // working directory by changing the user.dir property doesn't work for
        // relative file paths. The path looks correct, but File.exists() returns an
        // incorrect result.
        let subdir = match self.manager.get_property_user_dir() {
            Some(property_user_dir) => utilities::java_io_file_new(
                &property_user_dir,
                self.subdir_name.borrow().as_deref().unwrap_or("null"),
            ),
            None => self
                .subdir_name
                .borrow()
                .clone()
                .unwrap_or_else(|| "null".to_owned()),
        };
        if !Path::new(&utilities::java_io_file_new(
            &subdir,
            self.test_volume_name.as_deref().unwrap_or("null"),
        ))
        .exists()
        {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    "Test volume has not been created.  Please extract test volume.",
                    "Entry Error",
                )
            });
            return false;
        }
        param.set_subdir_name(self.subdir_name.borrow().as_deref());
        param.set_input_file_name(self.test_volume_name.as_deref());
        true
    }

    /// Java `getParameters(AnisotropicDiffusionParam, boolean)`.
    pub fn get_parameters_anisotropic_diffusion_param(
        &self,
        param: &mut AnisotropicDiffusionParam,
        do_validation: bool,
    ) -> bool {
        param.set_subdir_name(self.subdir_name.borrow().as_deref());
        self.filter_full_volume_panel
            .get_parameters_anisotropic_diffusion_param(param, do_validation)
    }

    /// Java `getParameters(ChunksetupParam)`.
    pub fn get_parameters_chunksetup_param(&self, param: &mut ChunksetupParam) {
        self.filter_full_volume_panel
            .get_parameters_chunksetup_param(param);
        param.set_command_file(Some(
            &anisotropic_diffusion_param::get_filter_full_file_name(),
        ));
        param.set_subdir_name(self.subdir_name.borrow().as_deref());
        param.set_input_file(self.ftf_volume.get_file_name().as_deref());
        // was: ftfVolume.getFileName() + ".nad";
        // ftfVolume.getFileName() is the root name that is set in metaData.
        param.set_output_file(
            file_type::CLASS
                .anisotropic_diffusion_output
                .get_file_name(Some(self.manager), Some(AxisID::Only))
                .as_deref(),
        );
    }

    /// Java `getParametersForVaryingIteration(AnisotropicDiffusionParam, boolean)`.
    pub fn get_parameters_for_varying_iteration(
        &self,
        param: &mut AnisotropicDiffusionParam,
        do_validation: bool,
    ) -> bool {
        let Ok(k_value) = self.ltf_test_k_value.get_text_boolean(do_validation) else {
            return false;
        };
        param.set_k_value(k_value.as_deref());
        let Ok(iteration_list) = self.ltf_test_iteration_list.get_text_boolean(do_validation)
        else {
            return false;
        };
        if !param.set_iteration_list(iteration_list.as_deref()) {
            return false;
        }
        param.set_format(
            self.manager
                .get_base_meta_data()
                .map(|meta_data| meta_data.base().get_image_output_format()),
        );
        param.set_subdir_name(self.subdir_name.borrow().as_deref());
        param.set_input_file_name(self.test_volume_name.as_deref());
        true
    }

    /// Java `getSubdirectory()`.
    pub fn get_subdirectory(&self) -> Option<String> {
        if !self.init_subdir() {
            return None;
        }
        self.subdir_name.borrow().clone()
    }

    /// Java private `openVolume()`.
    fn open_volume(&self) {
        let chooser = FileChooser::new_base_manager(Some(self.manager));
        // Swing layout: chooser.setPreferredSize(
        // UIParameters.getInstance().getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        chooser.set_file_filter(Some(
            TomogramFileFilter::get_all_image_filename_style_instance(self.manager),
        ));
        let return_val = chooser.show_open_dialog(Some(&self.root_panel.get_container()));
        if return_val != file_chooser::APPROVE_OPTION {
            return;
        }
        let volume: Option<PathBuf> = chooser.get_selected_file();
        let Some(volume) = volume.filter(|volume| !volume.is_dir() && volume.exists()) else {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    "Please choose a volume",
                    "Entry Error",
                )
            });
            return;
        };
        if !dataset_tool::validate_dataset_name_input_file_component(
            self.manager,
            None,
            AxisID::Only,
            Some(&volume),
            DataFileType::Parallel,
            None,
        ) {
            return;
        }
        self.ftf_volume.set_file(Some(volume.clone()));
        self.manager.set_new_param_file_file(&volume);
        if !self.init_subdir() {
            self.ftf_volume.set_file(None);
            return;
        }
        self.ftf_volume.set_button_enabled(false);
    }
}

impl ContextMenu for AnisotropicDiffusionDialog {
    /// Java `popUpContextMenu(MouseEvent)`.  Right mouse button context menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = [
            "Anisotropic Diffusion".to_owned(),
            "3dmod".to_owned(),
            "Processchunks".to_owned(),
            "Chunksetup".to_owned(),
        ];
        let man_page = [
            format!("{}.html", ProcessName::ANISOTROPIC_DIFFUSION),
            "3dmod.html".to_owned(),
            "processchunks.html".to_owned(),
            "chunksetup.html".to_owned(),
        ];
        let log_file_label = ["Anisotropic Diffusion".to_owned()];
        let log_file = [format!("{}.log", ProcessName::ANISOTROPIC_DIFFUSION)];
        // ContextPopup contextPopup =
        let _ = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id_string(
            &self.root_panel.get_container(),
            mouse_event,
            Some("ANISOTROPIC DIFFUSION"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            &log_file_label,
            &log_file,
            self.manager,
            AxisID::Only,
            self.subdir_name.borrow().as_deref(),
        );
    }
}

impl Run3dmodButtonContainer for AnisotropicDiffusionDialog {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let command = Some(command);
        if command == self.btn_extract_test_volume.get_action_command().as_deref() {
            if !self.init_subdir() {
                return;
            }
            self.manager.trim_volume(None);
        } else if command == self.btn_run_varying_k.get_action_command().as_deref() {
            if !self.init_subdir() {
                return;
            }
            let processing_method = self.mediator.as_ref().map(|mediator| {
                mediator.get_run_method_for_process_interface(self.get_processing_method())
            });
            let subdir_name = self.subdir_name.borrow().clone();
            self.manager.anisotropic_diffusion_varying_k(
                subdir_name.as_deref(),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                Some(DIALOG_TYPE),
                processing_method,
            );
        } else if command
            == self
                .btn_run_varying_iteration
                .get_action_command()
                .as_deref()
        {
            if !self.init_subdir() {
                return;
            }
            let subdir_name = self.subdir_name.borrow().clone();
            self.manager.anisotropic_diffusion_varying_iteration(
                subdir_name.as_deref(),
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                Some(DIALOG_TYPE),
            );
        } else if command == self.btn_view_full_volume.get_action_command().as_deref() {
            self.manager.imod_string_file_run3dmod_menu_options_boolean(
                imod_manager::VOLUME_KEY,
                self.ftf_volume.get_file().as_deref(),
                run_3dmod_menu_options,
                self.cb_load_with_flipping.is_selected(),
            );
        } else if command == self.btn_view_test_volume.get_action_command().as_deref() {
            let file = utilities::java_io_file_new(
                self.subdir_name.borrow().as_deref().unwrap_or("null"),
                self.test_volume_name.as_deref().unwrap_or("null"),
            );
            self.manager.imod_string_file_run3dmod_menu_options_boolean(
                imod_manager::TEST_VOLUME_KEY,
                Some(Path::new(&file)),
                run_3dmod_menu_options,
                self.cb_load_with_flipping.is_selected(),
            );
        } else if command == self.btn_view_varying_k.get_action_command().as_deref() {
            let subdir_name = self.subdir_name.borrow().clone();
            self.manager.imod_varying_k_value(
                imod_manager::VARYING_K_TEST_KEY,
                run_3dmod_menu_options,
                subdir_name.as_deref(),
                self.test_volume_name.as_deref(),
                self.cb_load_with_flipping.is_selected(),
            );
        } else if command
            == self
                .btn_view_varying_iteration
                .get_action_command()
                .as_deref()
        {
            let subdir_name = self.subdir_name.borrow().clone();
            self.manager.imod_varying_iteration(
                imod_manager::VARYING_ITERATION_TEST_KEY,
                run_3dmod_menu_options,
                subdir_name.as_deref(),
                self.test_volume_name.as_deref(),
                self.cb_load_with_flipping.is_selected(),
            );
        }
    }
}

impl FilterFullVolumeParent for AnisotropicDiffusionDialog {
    /// Java `cleanUp()`.
    fn clean_up(&self) {
        let subdir_name = self.subdir_name.borrow().clone();
        if let Some(subdir_name) = subdir_name
            && self.manager.delete_subdir(&subdir_name)
        {
            *self.subdir_name.borrow_mut() = None;
        }
    }

    /// Java `getVolume()`.
    fn get_volume(&self) -> Option<String> {
        self.ftf_volume.get_file_absolute_path()
    }

    /// Java `initSubdir()`.  Initialized subdirName if is not already initialized.
    /// Returns false if ftfVolume is empty (subdirName is dependent on ftfVolume).
    // TODO 2206 (the source's own marker)
    fn init_subdir(&self) -> bool {
        if self.ftf_volume.is_empty() {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    "Please choose a volume before running this function.",
                    "Entry Error",
                )
            });
            return false;
        }
        if self.subdir_name.borrow().is_none() {
            let subdir_name = format!(
                "naddir.{}",
                self.ftf_volume
                    .get_file_name()
                    .unwrap_or_else(|| "null".to_owned())
            );
            *self.subdir_name.borrow_mut() = Some(subdir_name.clone());
            if !self.manager.make_subdir(&subdir_name) {
                return false;
            }
        }
        true
    }

    /// Java `isLoadWithFlipping()`.
    fn is_load_with_flipping(&self) -> bool {
        self.cb_load_with_flipping.is_selected()
    }
}

impl AbstractParallelDialog for AnisotropicDiffusionDialog {
    /// Java `getParameters(ParallelParam)`.  (Java casts the param to
    /// `ProcesschunksParam`; every caller passes one.)
    fn get_parameters(&self, param: &mut dyn ParallelParam) {
        if let Some(processchunks_param) =
            (param as &mut dyn std::any::Any).downcast_mut::<ProcesschunksParam>()
        {
            processchunks_param.set_subdir_name(self.subdir_name.borrow().as_deref());
        }
    }

    /// Java `getDialogType()`.
    fn get_dialog_type(&self) -> DialogType {
        DialogType::AnisotropicDiffusion
    }
}

impl QueueTableListener for AnisotropicDiffusionDialog {
    /// Java `queueTableEventAction(QueueTableEvent)`: empty.
    fn queue_table_event_action(&self, _event: &QueueTableEvent) {}
}

impl ProcessInterface for AnisotropicDiffusionDialog {
    /// Java `updateGpu(boolean)`: empty.
    fn update_gpu(&self, _disable_gpu: bool) {}

    /// Java `getProcessingMethod()`.  Dialogs don't need to know if QUEUE is in use
    /// in the parallel panel.
    fn get_processing_method(&self) -> ProcessingMethod {
        ProcessingMethod::PpCpu
    }

    /// Java `getSecondaryProcessingMethod()`.
    fn get_secondary_processing_method(&self) -> Option<ProcessingMethod> {
        None
    }

    /// Java `lockProcessingMethod(boolean)`: empty.
    fn lock_processing_method(&self, _lock: bool) {}

    /// Java `setMethod(ProcessingMethod)`.
    fn set_method(&self, processing_method: ProcessingMethod) {
        if let (Some(mediator), Some(this)) = (&self.mediator, self.this.upgrade()) {
            let origin: Rc<dyn ProcessInterface> = this;
            mediator.set_method_process_interface_processing_method(&origin, processing_method);
        }
    }

    /// Java `isUseGpu()`.
    fn is_use_gpu(&self) -> bool {
        false
    }

    /// Java `setUseQueueCheckBox(ButtonComponent)`: empty.
    fn set_use_queue_check_box(&self, _use_queue_checkbox: Option<Rc<dyn ButtonComponent>>) {}

    /// Java `addQueueTableListener(QueueTableListener)`: empty.
    fn add_queue_table_listener(&self, _listener: Rc<dyn QueueTableListener>) {}

    /// Java `removeQueueTableListener(QueueTableListener)`: empty.
    fn remove_queue_table_listener(&self, _listener: &Rc<dyn QueueTableListener>) {}
}
