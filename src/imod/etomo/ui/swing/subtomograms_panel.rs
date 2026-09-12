//! `IMOD/Etomo/src/etomo/ui/swing/SubtomogramsPanel.java`.
//!
//! Native widget construction, file choosing, autodoc reading and manager calls
//! are boundaries.  The Java controller's state, action routing, visibility,
//! validation ordering and parameter protocol remain here.
#![allow(dead_code)]

use super::{
    check_box::CheckBox,
    labeled_text_field::LabeledTextField,
    radio_button::{RadioButton, RadioButtonGroup},
    spinner::Spinner,
    text_efield::TextEfield,
};
use crate::imod::etomo::{
    r#type::{axis_id::AxisID, dialog_type::DialogType},
    ui::field_type::FieldType,
};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
};

pub const VOLUME_MODELED_LABEL: &str = "Tomogram that was modeled: ";
pub const REORIENTATION_TYPE_LABEL: &str = "Specify reorientation of tomogram: ";
pub const CENTER_POSITION_FILE_LABEL: &str = "Model or point file: ";
pub const OBJECTS_TO_USE_LABEL: &str = "Objects with points to use: ";
pub const SIZE_IN_X_LABEL: &str = "Output size in X: ";
pub const SIZE_IN_Y_LABEL: &str = "Y: ";
pub const SIZE_IN_Z_LABEL: &str = "Z: ";
pub const DIRECTORY_FOR_OUTPUT_LABEL: &str = "Output directory for subvolumes: ";
pub const MAKE_VOLUME_STACKS_LABEL: &str = "Make MRC volume stacks with up to ";
pub const SKIP_SUBVOL_NUMBERS_LABEL: &str = "Skip numbers for skipped subvolumes near edge";
pub const NEW_ALIGNED_BINNING_LABEL: &str = "Make new aligned stack with binning";
pub const USE_UNALIGNED_IMAGES_LABEL: &str = "Reconstruct from raw images with reduction by";
pub const EXTENT_OF_ZLEVELS_IN_NM_LABEL: &str = "Do 3D CTF correction with division into ";
pub const ERASE_FIDUCIALS_LABEL: &str = "Erase gold";
pub const FILTER_IN_2D_LABEL: &str = "Apply 2D filter";

/// Source-visible subset of Java `ButtonControlTextEfield`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ButtonControlTextEfield {
    pub label: String,
    pub text: String,
    pub required: bool,
    pub directories_only: bool,
    pub displayed_path_limit: i32,
    pub tooltip: Option<String>,
    pub control_listener_count: usize,
}
impl ButtonControlTextEfield {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.into(),
            ..Self::default()
        }
    }
    pub fn get_text(&self) -> String {
        self.text.clone()
    }
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into()
    }
}
/// Direct `SubtomoSetupCpuGpuPanel` boundary, including its three radio choices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuGpuState {
    pub cpus_only: bool,
    pub gpu_recon_ctf: bool,
    pub mixed_cpu_gpu: bool,
    pub gpu_enabled: bool,
    pub mixed_enabled: bool,
    pub mix_action_event: bool,
    pub queue_checkbox: bool,
}
impl Default for CpuGpuState {
    fn default() -> Self {
        Self {
            cpus_only: true,
            gpu_recon_ctf: false,
            mixed_cpu_gpu: false,
            gpu_enabled: true,
            mixed_enabled: false,
            mix_action_event: false,
            queue_checkbox: false,
        }
    }
}
/// Java `ApplicationManager`, chooser and `UIHarness` calls.
pub trait SubtomogramsPanelManager {
    fn property_user_dir(&self) -> PathBuf;
    fn total_gpus(&self, axis: AxisID) -> i32;
    fn output_processchunks_exists(&self, axis: AxisID, directory: &Path) -> bool;
    fn confirm_existing_output(&mut self, axis: AxisID) -> bool;
    fn subtomo_setup(&mut self, axis: AxisID, dialog: DialogType, method: i32);
    fn open_files_in_imod(&mut self, axis: AxisID, names: Vec<String>, subdir: PathBuf);
    fn pack(&mut self, axis: AxisID);
}
/// All `SubtomoSetupParam` values touched by this panel are kept string-shaped.
pub trait SubtomoSetupParam {
    fn set(&mut self, key: &str, value: String);
    fn reset(&mut self, key: &str);
    fn is_set(&self, key: &str) -> bool;
    fn get(&self, key: &str) -> Option<String>;
}
/// Java `MetaData` writes in overloaded `getParameters(MetaData)`.
pub trait SubtomogramsMetaData {
    fn set_subtomo_reorientation_type_none(&mut self, value: bool);
    fn set_subtomo_reorientation_type_flipped(&mut self, value: bool);
    fn set_subtomo_reorientation_type_rotated(&mut self, value: bool);
    fn set_subtomo_make_volume_stacks(&mut self, value: i32);
    fn set_subtomo_new_aligned_binning(&mut self, value: i32);
    fn set_subtomo_fourier_reduce_by_factor(&mut self, value: i32);
    fn set_subtomo_extent_of_z_levels_in_nm(&mut self, value: String);
}
/// Java `ConstMetaData` reads in overloaded `setParameters(ConstMetaData)`.
pub trait ConstSubtomogramsMetaData {
    fn subtomo_reorientation_type_none(&self) -> bool;
    fn subtomo_reorientation_type_flipped(&self) -> bool;
    fn subtomo_reorientation_type_rotated(&self) -> bool;
    fn subtomo_make_volume_stacks(&self) -> bool;
    fn subtomo_make_volume_stacks_value(&self) -> i32;
    fn subtomo_new_aligned_binning(&self) -> i32;
    fn subtomo_fourier_reduce_by_factor(&self) -> i32;
    fn subtomo_extent_of_z_levels_in_nm(&self) -> String;
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SubtomogramsPanelLayout {
    pub root_visible: bool,
    pub reorientation_type_visible: bool,
    pub objects_to_use_visible: bool,
    pub adjust_for_align_z_shift_visible: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub component_order: Vec<&'static str>,
}

/// Java `SubtomogramsPanel` fields, with Java names mapped snake_case.
pub struct SubtomogramsPanel {
    pub pnl_root: SubtomogramsPanelLayout,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub bctf_volume_modeled: ButtonControlTextEfield,
    pub cb_reorientation_type: CheckBox,
    pub rb_reorientation_type_none: RadioButton,
    pub rb_reorientation_type_rotated: RadioButton,
    pub rb_reorientation_type_flipped: RadioButton,
    pub bctf_center_position_file: ButtonControlTextEfield,
    pub ltf_objects_to_use: LabeledTextField,
    pub ltf_size_in_x: LabeledTextField,
    pub ltf_size_in_y: LabeledTextField,
    pub ltf_size_in_z: LabeledTextField,
    pub bctf_directory_for_output: ButtonControlTextEfield,
    pub cb_make_volume_stacks: CheckBox,
    pub sp_make_volume_stacks: Spinner,
    pub cb_skip_sub_vol_numbers: CheckBox,
    pub rb_existing_aligned_stack: RadioButton,
    pub rb_new_aligned_binning: RadioButton,
    pub sp_new_aligned_binning: Spinner,
    pub rb_use_unaligned_images: RadioButton,
    pub sp_fourier_reduce_by_factor: Spinner,
    pub cb_extent_of_z_levels_in_nm: CheckBox,
    pub tf_extent_of_z_levels_in_nm: TextEfield,
    pub cb_adjust_for_align_z_shift: CheckBox,
    pub cb_erase_fiducials: CheckBox,
    pub cb_filter_in_2d: CheckBox,
    pub subtomo_setup_cpu_gpu_panel: CpuGpuState,
    pub subtomo_browsing_dir: PathBuf,
}
impl SubtomogramsPanel {
    pub fn new<M: SubtomogramsPanelManager>(
        manager: &M,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        let rg = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let ig = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut value = Self {
            pnl_root: SubtomogramsPanelLayout::default(),
            axis_id,
            dialog_type,
            bctf_volume_modeled: ButtonControlTextEfield::new(VOLUME_MODELED_LABEL),
            cb_reorientation_type: CheckBox::new_with_text(REORIENTATION_TYPE_LABEL),
            rb_reorientation_type_none: RadioButton::new_in_group("None", rg.clone()),
            rb_reorientation_type_rotated: RadioButton::new_in_group("Rotated", rg.clone()),
            rb_reorientation_type_flipped: RadioButton::new_in_group("Flipped", rg),
            bctf_center_position_file: ButtonControlTextEfield::new(CENTER_POSITION_FILE_LABEL),
            ltf_objects_to_use: LabeledTextField::new(FieldType::IntegerList, OBJECTS_TO_USE_LABEL),
            ltf_size_in_x: LabeledTextField::new(FieldType::Integer, SIZE_IN_X_LABEL),
            ltf_size_in_y: LabeledTextField::new(FieldType::Integer, SIZE_IN_Y_LABEL),
            ltf_size_in_z: LabeledTextField::new(FieldType::Integer, SIZE_IN_Z_LABEL),
            bctf_directory_for_output: ButtonControlTextEfield::new(DIRECTORY_FOR_OUTPUT_LABEL),
            cb_make_volume_stacks: CheckBox::new_with_text(MAKE_VOLUME_STACKS_LABEL),
            sp_make_volume_stacks: Spinner::get_instance(
                MAKE_VOLUME_STACKS_LABEL,
                100,
                1,
                10000,
                1,
            ),
            cb_skip_sub_vol_numbers: CheckBox::new_with_text(SKIP_SUBVOL_NUMBERS_LABEL),
            rb_existing_aligned_stack: RadioButton::new_in_group(
                "Use existing aligned stack",
                ig.clone(),
            ),
            rb_new_aligned_binning: RadioButton::new_in_group(
                NEW_ALIGNED_BINNING_LABEL,
                ig.clone(),
            ),
            sp_new_aligned_binning: Spinner::get_instance(NEW_ALIGNED_BINNING_LABEL, 1, 1, 100, 1),
            rb_use_unaligned_images: RadioButton::new_in_group(USE_UNALIGNED_IMAGES_LABEL, ig),
            sp_fourier_reduce_by_factor: Spinner::get_instance(
                USE_UNALIGNED_IMAGES_LABEL,
                1,
                1,
                100,
                1,
            ),
            cb_extent_of_z_levels_in_nm: CheckBox::new_with_text(EXTENT_OF_ZLEVELS_IN_NM_LABEL),
            tf_extent_of_z_levels_in_nm: TextEfield::new(
                EXTENT_OF_ZLEVELS_IN_NM_LABEL,
                Some(FieldType::Integer),
                true,
                false,
                true,
                true,
                false,
                false,
            ),
            cb_adjust_for_align_z_shift: CheckBox::new_with_text(
                "Adjust for Z shift in fine alignment and positioning",
            ),
            cb_erase_fiducials: CheckBox::new_with_text(ERASE_FIDUCIALS_LABEL),
            cb_filter_in_2d: CheckBox::new_with_text(FILTER_IN_2D_LABEL),
            subtomo_setup_cpu_gpu_panel: CpuGpuState::default(),
            subtomo_browsing_dir: manager.property_user_dir(),
        };
        value.create_panel(manager);
        value.set_tooltips();
        value.add_listeners();
        value
    }
    pub fn get_instance<M: SubtomogramsPanelManager>(
        manager: &M,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Self {
        Self::new(manager, axis_id, dialog_type)
    }
    fn create_panel<M: SubtomogramsPanelManager>(&mut self, manager: &M) {
        self.bctf_volume_modeled.required = true;
        self.bctf_center_position_file.required = true;
        self.bctf_volume_modeled.displayed_path_limit = 40;
        self.bctf_center_position_file.displayed_path_limit = 50;
        self.bctf_directory_for_output.displayed_path_limit = 28;
        self.bctf_directory_for_output.directories_only = true;
        self.ltf_size_in_x.set_required(true);
        self.ltf_size_in_x.set_number_must_be_positive(true);
        self.ltf_size_in_y.set_required(true);
        self.ltf_size_in_y.set_number_must_be_positive(true);
        self.ltf_size_in_z.set_required(true);
        self.ltf_size_in_z.set_number_must_be_positive(true);
        self.tf_extent_of_z_levels_in_nm.set_required(true);
        self.rb_reorientation_type_none.set_selected(true);
        self.rb_existing_aligned_stack.set_selected(true);
        if manager.total_gpus(self.axis_id) < 1 {
            self.subtomo_setup_cpu_gpu_panel.gpu_enabled = false;
            self.subtomo_setup_cpu_gpu_panel.mixed_enabled = false
        }
        self.pnl_root.root_visible = true;
        self.pnl_root.component_order = vec![
            "volumeModeled",
            "reorientationType",
            "centerPositionFile",
            "objectsToUse",
            "sizeInXYZ",
            "directoryForOutput",
            "makeVolumeStacks",
            "skipSubVolNumbers",
            "sourceOfImages",
            "extentOfZLevelsInNm",
            "adjustForAlignZShift",
            "eraseFiducials",
            "filterIn2D",
            "subtomoSetupCpuGpu",
            "buttons",
        ];
        self.update_display()
    }
    fn add_listeners(&mut self) {
        self.pnl_root.listener_count = 14
    }
    pub fn action_performed<M: SubtomogramsPanelManager>(
        &mut self,
        manager: &mut M,
        command: &str,
        selected_files: Option<Vec<PathBuf>>,
    ) {
        if self.cb_extent_of_z_levels_in_nm.get_action_command() == Some(command) {
            if !self.cb_extent_of_z_levels_in_nm.is_selected() {
                self.subtomo_setup_cpu_gpu_panel.mixed_enabled = false;
                if self.subtomo_setup_cpu_gpu_panel.mixed_cpu_gpu {
                    self.subtomo_setup_cpu_gpu_panel.cpus_only = true
                }
            } else {
                self.subtomo_setup_cpu_gpu_panel.mixed_enabled = true
            }
        } else if command == "Generate Subtomograms" {
            if manager.output_processchunks_exists(
                self.axis_id,
                Path::new(&self.bctf_directory_for_output.text),
            ) && !manager.confirm_existing_output(self.axis_id)
            {
                return;
            }
            manager.subtomo_setup(self.axis_id, self.dialog_type, self.get_processing_method())
        } else if command == "View Subtomograms In 3dmod" {
            self.open_files_in_imod(manager, selected_files.unwrap_or_default())
        }
        self.update_display()
    }
    fn open_files_in_imod<M: SubtomogramsPanelManager>(
        &self,
        manager: &mut M,
        files: Vec<PathBuf>,
    ) {
        if files.is_empty() {
            return;
        }
        let subdir = files[0]
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .to_path_buf();
        manager.open_files_in_imod(
            self.axis_id,
            files
                .iter()
                .filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned()))
                .collect(),
            subdir,
        )
    }
    pub fn get_component(&self) -> &SubtomogramsPanelLayout {
        &self.pnl_root
    }
    pub fn update_display(&mut self) {
        let reorient = self.cb_reorientation_type.is_selected();
        self.rb_reorientation_type_none.set_enabled(reorient);
        self.rb_reorientation_type_rotated.set_enabled(reorient);
        self.rb_reorientation_type_flipped.set_enabled(reorient);
        self.sp_make_volume_stacks
            .set_enabled(self.cb_make_volume_stacks.is_selected());
        self.sp_new_aligned_binning
            .set_enabled(self.rb_new_aligned_binning.is_selected());
        self.sp_fourier_reduce_by_factor
            .set_enabled(self.rb_use_unaligned_images.is_selected());
        self.tf_extent_of_z_levels_in_nm
            .set_enabled(self.cb_extent_of_z_levels_in_nm.is_selected());
        self.subtomo_setup_cpu_gpu_panel.mixed_enabled =
            self.cb_extent_of_z_levels_in_nm.is_selected();
        if !self.cb_extent_of_z_levels_in_nm.is_selected()
            && self.subtomo_setup_cpu_gpu_panel.mixed_cpu_gpu
            && !self.subtomo_setup_cpu_gpu_panel.mix_action_event
        {
            self.subtomo_setup_cpu_gpu_panel.cpus_only = true
        }
    }
    pub fn update_advanced(&mut self, advanced: bool) {
        self.pnl_root.reorientation_type_visible = advanced;
        self.pnl_root.objects_to_use_visible = advanced;
        self.pnl_root.adjust_for_align_z_shift_visible = advanced
    }
    /// Java overload `getParameters(MetaData)`.
    pub fn get_parameters_metadata<M: SubtomogramsMetaData>(&self, metadata: &mut M) {
        metadata.set_subtomo_reorientation_type_none(self.rb_reorientation_type_none.is_selected());
        metadata.set_subtomo_reorientation_type_flipped(
            self.rb_reorientation_type_flipped.is_selected(),
        );
        metadata.set_subtomo_reorientation_type_rotated(
            self.rb_reorientation_type_rotated.is_selected(),
        );
        metadata.set_subtomo_make_volume_stacks(self.sp_make_volume_stacks.get_value());
        if !self.rb_new_aligned_binning.is_selected() {
            metadata.set_subtomo_new_aligned_binning(self.sp_new_aligned_binning.get_value());
        }
        if !self.rb_use_unaligned_images.is_selected() {
            metadata
                .set_subtomo_fourier_reduce_by_factor(self.sp_fourier_reduce_by_factor.get_value());
        }
        metadata.set_subtomo_extent_of_z_levels_in_nm(self.tf_extent_of_z_levels_in_nm.get_text());
    }
    pub fn get_parameters<P: SubtomoSetupParam>(
        &self,
        param: &mut P,
        do_validation: bool,
        rootname: String,
        processor_number: Option<String>,
    ) -> bool {
        let fields = [
            (self.ltf_size_in_x.get_text(), SIZE_IN_X_LABEL),
            (self.ltf_size_in_y.get_text(), SIZE_IN_Y_LABEL),
            (self.ltf_size_in_z.get_text(), SIZE_IN_Z_LABEL),
        ];
        if do_validation && fields.iter().any(|(x, _)| x.trim().is_empty()) {
            return false;
        }
        param.set("volumeModeled", self.bctf_volume_modeled.get_text());
        if self.cb_reorientation_type.is_selected() {
            param.set(
                "reorientationType",
                if self.rb_reorientation_type_none.is_selected() {
                    "0"
                } else if self.rb_reorientation_type_flipped.is_selected() {
                    "1"
                } else {
                    "2"
                }
                .into(),
            )
        } else {
            param.reset("reorientationType")
        }
        param.set(
            "centerPositionFile",
            self.bctf_center_position_file.get_text(),
        );
        param.set("objectsToUse", self.ltf_objects_to_use.get_text());
        for (key, value) in [
            ("sizeInX", fields[0].0.clone()),
            ("sizeInY", fields[1].0.clone()),
            ("sizeInZ", fields[2].0.clone()),
            (
                "directoryForOutput",
                self.bctf_directory_for_output.get_text(),
            ),
            (
                "skipSubVolNumbers",
                self.cb_skip_sub_vol_numbers.is_selected().to_string(),
            ),
            (
                "useUnalignedImages",
                self.rb_use_unaligned_images.is_selected().to_string(),
            ),
            (
                "adjustForAlignZShift",
                self.cb_adjust_for_align_z_shift.is_selected().to_string(),
            ),
            (
                "eraseFiducials",
                self.cb_erase_fiducials.is_selected().to_string(),
            ),
            ("filterIn2D", self.cb_filter_in_2d.is_selected().to_string()),
            ("whenToUseGpu", self.get_processing_method().to_string()),
            ("rootname", rootname),
        ] {
            param.set(key, value)
        }
        if self.cb_make_volume_stacks.is_selected() {
            param.set(
                "makeVolumeStacks",
                self.sp_make_volume_stacks.get_value().to_string(),
            )
        } else {
            param.reset("makeVolumeStacks")
        }
        if self.rb_new_aligned_binning.is_selected() {
            param.set(
                "newAlignedBinning",
                self.sp_new_aligned_binning.get_value().to_string(),
            )
        } else {
            param.reset("newAlignedBinning")
        }
        if self.rb_use_unaligned_images.is_selected() {
            param.set(
                "fourierReduceByFactor",
                self.sp_fourier_reduce_by_factor.get_value().to_string(),
            )
        } else {
            param.reset("fourierReduceByFactor")
        }
        if self.cb_extent_of_z_levels_in_nm.is_selected() {
            let value = self.tf_extent_of_z_levels_in_nm.get_text();
            if do_validation && value.trim().is_empty() {
                return false;
            }
            param.set("extentOfZLevelsInNm", value)
        } else {
            param.reset("extentOfZLevelsInNm")
        }
        if let Some(value) = processor_number {
            param.set("processorNumber", value)
        }
        true
    }
    pub fn set_parameters<P: SubtomoSetupParam>(&mut self, param: &P) {
        if param.is_set("volumeModeled") {
            self.bctf_volume_modeled
                .set_text(param.get("volumeModeled").unwrap_or_default())
        }
        if param.is_set("reorientationType") {
            self.cb_reorientation_type.set_selected(true);
            match param.get("reorientationType").as_deref() {
                Some("0") => self.rb_reorientation_type_none.set_selected(true),
                Some("1") => self.rb_reorientation_type_flipped.set_selected(true),
                Some("2") => self.rb_reorientation_type_rotated.set_selected(true),
                _ => {}
            }
        }
        if param.is_set("centerPositionFile") {
            self.bctf_center_position_file
                .set_text(param.get("centerPositionFile").unwrap_or_default())
        }
        if param.is_set("objectsToUse") {
            self.ltf_objects_to_use
                .set_text(&param.get("objectsToUse").unwrap_or_default())
        }
        for (key, field) in [
            ("sizeInX", &mut self.ltf_size_in_x),
            ("sizeInY", &mut self.ltf_size_in_y),
            ("sizeInZ", &mut self.ltf_size_in_z),
        ] {
            if param.is_set(key) {
                field.set_text(&param.get(key).unwrap_or_default())
            }
        }
        if param.is_set("directoryForOutput") {
            self.bctf_directory_for_output
                .set_text(param.get("directoryForOutput").unwrap_or_default())
        }
        if param.is_set("makeVolumeStacks") {
            self.cb_make_volume_stacks.set_selected(true);
            self.sp_make_volume_stacks.set_value(
                param
                    .get("makeVolumeStacks")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(100),
            )
        }
        self.cb_skip_sub_vol_numbers
            .set_selected(param.get("skipSubVolNumbers").as_deref() == Some("true"));
        if param.is_set("newAlignedBinning") {
            self.rb_new_aligned_binning.set_selected(true);
            self.sp_new_aligned_binning.set_value(
                param
                    .get("newAlignedBinning")
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(1),
            )
        }
        self.rb_use_unaligned_images
            .set_selected(param.get("useUnalignedImages").as_deref() == Some("true"));
        if param.is_set("extentOfZLevelsInNm") {
            self.cb_extent_of_z_levels_in_nm.set_selected(true);
            self.tf_extent_of_z_levels_in_nm
                .set_text(param.get("extentOfZLevelsInNm").unwrap_or_default())
        }
        self.cb_adjust_for_align_z_shift
            .set_selected(param.get("adjustForAlignZShift").as_deref() == Some("true"));
        self.cb_erase_fiducials
            .set_selected(param.get("eraseFiducials").as_deref() == Some("true"));
        self.cb_filter_in_2d
            .set_selected(param.get("filterIn2D").as_deref() == Some("true"));
        self.update_display()
    }
    /// Java overload `setParameters(ConstMetaData)`.
    pub fn set_parameters_metadata<M: ConstSubtomogramsMetaData>(&mut self, metadata: &M) {
        if metadata.subtomo_reorientation_type_none() {
            self.rb_reorientation_type_none.set_selected(true);
        } else if metadata.subtomo_reorientation_type_flipped() {
            self.rb_reorientation_type_flipped.set_selected(true);
        } else if metadata.subtomo_reorientation_type_rotated() {
            self.rb_reorientation_type_rotated.set_selected(true);
        }
        if metadata.subtomo_make_volume_stacks() {
            self.sp_make_volume_stacks
                .set_value(metadata.subtomo_make_volume_stacks_value());
        }
        self.sp_new_aligned_binning
            .set_value(metadata.subtomo_new_aligned_binning());
        self.sp_fourier_reduce_by_factor
            .set_value(metadata.subtomo_fourier_reduce_by_factor());
        self.tf_extent_of_z_levels_in_nm
            .set_text(metadata.subtomo_extent_of_z_levels_in_nm());
    }
    fn set_tooltips(&mut self) {
        self.pnl_root.tooltip_initialized = true
    }
    pub fn expand(&mut self, advanced: bool) {
        self.update_advanced(advanced)
    }
    pub fn get_browsing_dir(&self) -> &Path {
        &self.subtomo_browsing_dir
    }
    pub fn set_browsing_dir(&mut self, file: PathBuf) {
        self.subtomo_browsing_dir = file
    }
    pub fn clear(&mut self) {}
    pub fn set_text(&mut self, _: &Path) {}
    pub fn set_text_files(&mut self, _: &[PathBuf]) {}
    pub fn get_label(&self) -> Option<&str> {
        None
    }
    pub fn set_component_control(&mut self, _: bool) {}
    pub fn set_enable_control(&mut self, _: bool) {}
    pub fn send_control_event<M: SubtomogramsPanelManager>(&self, m: &mut M) {
        m.pack(self.axis_id)
    }
    pub fn is_local_dir<M: SubtomogramsPanelManager>(&self, m: &M, p: &Path) -> bool {
        p == m.property_user_dir()
    }
    pub fn control_event<M: SubtomogramsPanelManager>(&self, m: &mut M) {
        m.pack(self.axis_id)
    }
    pub fn get_processing_method(&self) -> i32 {
        if self.subtomo_setup_cpu_gpu_panel.cpus_only {
            0
        } else if self.subtomo_setup_cpu_gpu_panel.gpu_recon_ctf {
            1
        } else {
            2
        }
    }
    pub fn get_subtomo_setup_display(&self) -> &Self {
        self
    }
    pub fn pop_up_context_menu(&self) {}
    pub fn update_gpu(&mut self, disable: bool) {
        self.update_display();
        if disable {
            self.subtomo_setup_cpu_gpu_panel.gpu_enabled = false;
            self.subtomo_setup_cpu_gpu_panel.mixed_enabled = false
        }
    }
    pub fn is_use_gpu(&self) -> bool {
        !self.subtomo_setup_cpu_gpu_panel.cpus_only
    }
    pub fn set_use_queue_check_box(&mut self) {
        self.subtomo_setup_cpu_gpu_panel.queue_checkbox = true
    }
    pub fn is_extent_of_z_levels_in_nm(&self) -> bool {
        self.cb_extent_of_z_levels_in_nm.is_selected()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct M {
        dir: PathBuf,
        opened: Vec<String>,
    }
    impl SubtomogramsPanelManager for M {
        fn property_user_dir(&self) -> PathBuf {
            self.dir.clone()
        }
        fn total_gpus(&self, _: AxisID) -> i32 {
            0
        }
        fn output_processchunks_exists(&self, _: AxisID, _: &Path) -> bool {
            false
        }
        fn confirm_existing_output(&mut self, _: AxisID) -> bool {
            true
        }
        fn subtomo_setup(&mut self, _: AxisID, _: DialogType, _: i32) {}
        fn open_files_in_imod(&mut self, _: AxisID, n: Vec<String>, _: PathBuf) {
            self.opened = n
        }
        fn pack(&mut self, _: AxisID) {}
    }
    #[test]
    fn display_tracks_controls() {
        let m = M::default();
        let mut p = SubtomogramsPanel::new(&m, AxisID::Only, DialogType::PostProcessing);
        p.cb_make_volume_stacks.set_selected(true);
        p.cb_extent_of_z_levels_in_nm.set_selected(true);
        p.update_display();
        assert!(p.sp_make_volume_stacks.is_enabled());
        assert!(p.tf_extent_of_z_levels_in_nm.text_field.enabled)
    }
    #[test]
    fn chooser_uses_basenames() {
        let mut m = M::default();
        let mut p = SubtomogramsPanel::new(&m, AxisID::Only, DialogType::PostProcessing);
        p.action_performed(
            &mut m,
            "View Subtomograms In 3dmod",
            Some(vec!["x/a.mrc".into(), "x/b.mrc".into()]),
        );
        assert_eq!(m.opened, ["a.mrc", "b.mrc"])
    }
    #[test]
    fn expand_controls_advanced() {
        let m = M::default();
        let mut p = SubtomogramsPanel::new(&m, AxisID::Only, DialogType::PostProcessing);
        p.expand(true);
        assert!(p.pnl_root.objects_to_use_visible)
    }
}
