//! `IMOD/Etomo/src/etomo/ui/swing/FilterFullVolumePanel.java`.
//!
//! The "Filter Full Volume" box of the anisotropic diffusion dialog: K value,
//! iterations and memory per chunk for running nad_eed_3d on the whole volume in
//! chunks (chunksetup, then processchunks), viewing the result, and cleaning up the
//! subdirectory.  An event dispatch thread object, created by
//! [`FilterFullVolumePanel::get_instance`].

use std::rc::{Rc, Weak};

use super::check_box::CheckBox;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::filter_full_volume_parent::FilterFullVolumeParent;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::{self, SpacedPanel};
use super::spinner::Spinner;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::anisotropic_diffusion_param::AnisotropicDiffusionParam;
use crate::imod::etomo::comscript::chunksetup_param::{self, ChunksetupParam};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::parallel_manager::ParallelManager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Number;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::extension;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::parallel_meta_data::ParallelMetaData;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java package-private static final `FILTER_FULL_VOLUME_LABEL`.
pub const FILTER_FULL_VOLUME_LABEL: &str = "Filter Full Volume";
/// Java package-private static final `MEMORY_PER_CHUNK_LABEL`.
pub const MEMORY_PER_CHUNK_LABEL: &str = "Memory per chunk";
/// Java package-private static final `MEMORY_PER_CHUNK_DEFAULT`.
pub const MEMORY_PER_CHUNK_DEFAULT: i32 = 14 * chunksetup_param::MEMORY_TO_VOXEL;
/// Java package-private static final `CLEANUP_LABEL`.
pub const CLEANUP_LABEL: &str = "Clean Up Subdirectory";

/// Java package-private `final class FilterFullVolumePanel implements
/// Run3dmodButtonContainer`.
pub struct FilterFullVolumePanel {
    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `btnRunFilterFullVolume`.
    btn_run_filter_full_volume: Rc<Run3dmodButton>,
    /// Java private final `ltfKValue`.
    ltf_k_value: Rc<LabeledTextField>,
    /// Java private final `spIteration`.
    sp_iteration: Rc<Spinner>,
    /// Java private final `spMemoryPerChunk`.
    sp_memory_per_chunk: Rc<Spinner>,
    /// Java private final `btnViewFilteredVolume`.
    btn_view_filtered_volume: Rc<Run3dmodButton>,
    /// Java private final `btnCleanup`.
    btn_cleanup: Rc<MultiLineButton>,
    /// Java private final `cbOverlapTimesFour`.
    cb_overlap_times_four: Rc<CheckBox>,

    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `manager`.
    manager: &'static ParallelManager,
    /// Java private final `parent`.
    parent: Weak<dyn FilterFullVolumeParent>,
}

impl FilterFullVolumePanel {
    /// Java private `FilterFullVolumePanel(ParallelManager, DialogType,
    /// FilterFullVolumeParent)`, with the field initializers.
    fn new(
        manager: &'static ParallelManager,
        dialog_type: DialogType,
        parent: Weak<dyn FilterFullVolumeParent>,
    ) -> Rc<FilterFullVolumePanel> {
        Rc::new_cyclic(|this: &Weak<FilterFullVolumePanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = this.clone();
            FilterFullVolumePanel {
                pnl_root: SpacedPanel::get_instance_void(),
                btn_run_filter_full_volume:
                    Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                        Some(FILTER_FULL_VOLUME_LABEL),
                        Some(container.clone()),
                    ),
                ltf_k_value: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some("K value: "),
                ),
                sp_iteration: Spinner::get_labeled_instance_string_int_int_int(
                    Some("Iterations: "),
                    10,
                    1,
                    200,
                ),
                sp_memory_per_chunk: Spinner::get_labeled_instance_string_int_int_int_int(
                    Some(&format!("{MEMORY_PER_CHUNK_LABEL} (MB): ")),
                    MEMORY_PER_CHUNK_DEFAULT,
                    chunksetup_param::MEMORY_TO_VOXEL,
                    30 * chunksetup_param::MEMORY_TO_VOXEL,
                    chunksetup_param::MEMORY_TO_VOXEL,
                ),
                btn_view_filtered_volume:
                    Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                        Some("View Filtered Volume"),
                        Some(container),
                    ),
                btn_cleanup: MultiLineButton::new_string(Some(CLEANUP_LABEL)),
                cb_overlap_times_four: CheckBox::new_string(Some(
                    "Overlap chunks by 4 times # of iterations",
                )),
                dialog_type,
                manager,
                parent,
            }
        })
    }

    /// Java package-private static `getInstance(ParallelManager, DialogType,
    /// FilterFullVolumeParent)`.
    pub fn get_instance(
        manager: &'static ParallelManager,
        dialog_type: DialogType,
        parent: Weak<dyn FilterFullVolumeParent>,
    ) -> Rc<FilterFullVolumePanel> {
        let instance = FilterFullVolumePanel::new(manager, dialog_type, parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners(&instance);
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self, this: &Rc<FilterFullVolumePanel>) {
        // new FilterFullVolumeActionListener(this)
        let adaptee = Rc::downgrade(this);
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event.get_action_command().unwrap_or(""), None, None);
            }
        });
        self.btn_run_filter_full_volume
            .add_action_listener(action_listener.clone());
        self.btn_view_filtered_volume
            .add_action_listener(action_listener.clone());
        self.btn_cleanup.add_action_listener(action_listener);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // initialization
        // Swing layout: ltfKValue.setTextPreferredWidth(
        // UIParameters.getInstance().getFourDigitWidth()).
        self.ltf_k_value.set_required(true);
        // local panels
        let pnl_fields = SpacedPanel::get_instance_void();
        let pnl_buttons = SpacedPanel::get_instance_void();
        let pnl_check_box = JComponent::new_panel();
        // root panel
        self.pnl_root.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_root
            .set_border(&EtchedBorder::new(Some(FILTER_FULL_VOLUME_LABEL)).get_border());
        self.pnl_root.add_spaced_panel(&pnl_fields);
        self.pnl_root.add_j_panel(&pnl_check_box);
        self.pnl_root.add_spaced_panel(&pnl_buttons);
        // fields panel
        pnl_fields.set_box_layout(spaced_panel::X_AXIS);
        pnl_fields.add_labeled_text_field(&self.ltf_k_value);
        pnl_fields.add_spinner(&self.sp_iteration);
        pnl_fields.add_spinner(&self.sp_memory_per_chunk);
        pnl_fields.add_horizontal_glue();
        // checkbox panel
        // Swing layout: pnlCheckBox BoxLayout X_AXIS, CENTER_ALIGNMENT; glue after.
        pnl_check_box.add(&self.cb_overlap_times_four.get_component());
        // buttons panel
        pnl_buttons.set_box_layout(spaced_panel::X_AXIS);
        self.btn_run_filter_full_volume
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_view_filtered_volume.clone() as Rc<dyn Deferred3dmodButton>,
            ));
        pnl_buttons.add_multi_line_button(&self.btn_run_filter_full_volume);
        pnl_buttons.add_multi_line_button(&self.btn_view_filtered_volume);
        pnl_buttons.add_multi_line_button(&self.btn_cleanup);
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `getParameters(ParallelMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &ParallelMetaData) {
        meta_data.set_k_value(self.ltf_k_value.get_text_void().as_deref());
        meta_data.set_iteration(Some(self.sp_iteration.get_value()));
        meta_data.set_memory_per_chunk(Some(self.sp_memory_per_chunk.get_value()));
        meta_data.set_overlap_times_four(self.cb_overlap_times_four.is_selected());
    }

    /// Java package-private `getMemoryPerChunk()`.
    pub fn get_memory_per_chunk(&self) -> Number {
        self.sp_memory_per_chunk.get_value()
    }

    /// Java package-private `setParameters(ParallelMetaData)`.
    pub fn set_parameters(&self, meta_data: &ParallelMetaData) {
        self.ltf_k_value
            .set_text_string(meta_data.get_k_value().as_deref());
        self.sp_iteration
            .set_value_const_etomo_number(&meta_data.get_iteration());
        self.sp_memory_per_chunk
            .set_value_const_etomo_number(&meta_data.get_memory_per_chunk());
        self.cb_overlap_times_four
            .set_selected_boolean(meta_data.is_overlap_times_four());
    }

    /// Java package-private `getParameters(AnisotropicDiffusionParam, boolean)`.
    pub fn get_parameters_anisotropic_diffusion_param(
        &self,
        param: &mut AnisotropicDiffusionParam,
        do_validation: bool,
    ) -> bool {
        match self.ltf_k_value.get_text_boolean(do_validation) {
            Ok(k_value) => {
                param.set_k_value(k_value.as_deref());
                param.set_iteration(Some(self.sp_iteration.get_value()));
                true
            }
            // catch (FieldValidationFailedException e)
            Err(_) => false,
        }
    }

    /// Java package-private `getParameters(ChunksetupParam)`.
    pub fn get_parameters_chunksetup_param(&self, param: &mut ChunksetupParam) {
        param.set_memory_per_chunk(self.sp_memory_per_chunk.get_value());
        param.set_overlap(self.sp_iteration.get_value());
        param.set_overlap_times_four(self.cb_overlap_times_four.is_selected());
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let image_filename_style = self
            .manager
            .get_base_meta_data()
            .map(|meta_data| meta_data.base().get_image_filename_style());
        self.btn_run_filter_full_volume
            .set_tool_tip_text(Some(&format!(
                "Run diffusion on the full volume in chunks, creates a {}file.",
                extension::CLASS.nad.get_suffix(image_filename_style)
            )));
        self.ltf_k_value
            .set_tool_tip_text(Some("K threshold value for running on full volume"));
        self.sp_iteration
            .set_tool_tip_text(Some("Number of iterations to run on full volume"));
        self.sp_memory_per_chunk.set_tool_tip_text(Some(
            "Maximum memory in megabytes to use while running diffusion on one chunk. Reduce if there is less memory per processor or if you want to break the job into more chunks.  The number of voxels in each chunk will be 1/36 of this memory limit or less.",
        ));
        self.btn_view_filtered_volume
            .set_tool_tip_text(Some("View filtered volume (filename.nad) in 3dmod"));
        self.btn_cleanup.set_tool_tip_text(Some(
            "Remove subdirectory with all temporary and test files (naddir.filename).",
        ));
        self.cb_overlap_times_four.set_tool_tip_text(Some(
            "Increase overlap to 4 times # of iterations (default is equal to # of iterations) to eliminate minor effects of cutting volume into chunks.",
        ));
    }
}

impl Run3dmodButtonContainer for FilterFullVolumePanel {
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let Some(parent) = self.parent.upgrade() else {
            return;
        };
        if Some(command)
            == self
                .btn_run_filter_full_volume
                .get_action_command()
                .as_deref()
        {
            if !parent.init_subdir() {
                return;
            }
            let processing_method = self
                .manager
                .get_processing_method_mediator(Some(AxisID::Only))
                .map(|mediator| {
                    mediator.get_run_method_for_process_interface(parent.get_processing_method())
                });
            self.manager.chunksetup(
                None,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                Some(self.dialog_type),
                processing_method,
            );
        } else if Some(command) == self.btn_cleanup.get_action_command().as_deref() {
            parent.clean_up();
        } else if Some(command)
            == self
                .btn_view_filtered_volume
                .get_action_command()
                .as_deref()
        {
            self.manager.imod_file_type(
                &file_type::CLASS.anisotropic_diffusion_output,
                run_3dmod_menu_options,
                parent.is_load_with_flipping(),
            );
        }
    }
}
