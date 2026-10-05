//! `IMOD/Etomo/src/etomo/ui/swing/SubtomogramsPanel.java`.
//!
//! Java `public class SubtomogramsPanel implements ActionListener, Expandable,
//! SubtomoSetupDisplay, Run3dmodButtonContainer, BrowsingDirectory,
//! ControlTarget, ControlListener, ContextMenu`: the Subtomograms tab of the
//! Post Processing dialog.  An EDT object created as `Rc<Self>` by
//! [`SubtomogramsPanel::get_instance`]; every method takes `&self`.
//!
//! The panel is its own `ActionListener` in the Java; here that listener is one
//! closure ([`SubtomogramsPanel::action_performed`]) holding a weak reference.
//! The `ProcessInterface` (the Post Processing dialog that owns this panel) is
//! held weakly.  The panel hands itself to its three file fields as their
//! `BrowsingDirectory` and `ControlListener`, which the Rust widgets hold as
//! strong `Rc`s: that is a reference cycle, like the Java's object graph, and
//! is only released with the process (the dialog lives as long as the
//! manager).  Because those widgets need a strong handle, the Java
//! constructor's `setAltBrowsingDirectory(this)` calls are made right after the
//! panel's `Rc` exists (still inside construction, before `createPanel`).

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::button_component::ButtonComponent;
use super::button_control_text_efield::ButtonControlTextEfield;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::control_listener::ControlListener;
use super::control_state::ControlState;
use super::control_target::ControlTarget;
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::etched_border::EtchedBorder;
use super::expand_button::ExpandButton;
use super::expandable::Expandable;
use super::file_chooser::{self, FileChooser};
use super::generic_mouse_adapter::GenericMouseAdapter;
use super::global_expand_button::GlobalExpandButton;
use super::label::Label;
use super::labeled_text_field::LabeledTextField;
use super::process_control_panel;
use super::process_interface::ProcessInterface;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::select_file_extension::SelectFileExtension;
use super::spinner::Spinner;
use super::subtomo_setup_cpu_gpu_panel::SubtomoSetupCpuGpuPanel;
use super::subtomo_setup_display::SubtomoSetupDisplay;
use super::text_field::TextField;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::subtomo_setup_param::{self, SubtomoSetupParam};
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, JComponent, MouseEvent, MouseListener,
};
use crate::imod::etomo::process::imod_manager;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section::ReadOnlySection;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::processing_method::ProcessingMethod;
use crate::imod::etomo::ui::browsing_directory::BrowsingDirectory;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;

/// Java package-private static final `VOLUME_MODELED_LABEL`.
pub const VOLUME_MODELED_LABEL: &str = "Tomogram that was modeled: ";
/// Java package-private static final `REORIENTATION_TYPE_LABEL`.
pub const REORIENTATION_TYPE_LABEL: &str = "Specify reorientation of tomogram: ";
/// Java package-private static final `CENTER_POSITION_FILE_LABEL`.
pub const CENTER_POSITION_FILE_LABEL: &str = "Model or point file: ";
/// Java package-private static final `OBJECTS_TO_USE_LABEL`.
pub const OBJECTS_TO_USE_LABEL: &str = "Objects with points to use: ";
/// Java package-private static final `SIZE_IN_X_LABEL`.
pub const SIZE_IN_X_LABEL: &str = "Output size in X: ";
/// Java package-private static final `SIZE_IN_Y_LABEL`.
pub const SIZE_IN_Y_LABEL: &str = "Y: ";
/// Java package-private static final `SIZE_IN_Z_LABEL`.
pub const SIZE_IN_Z_LABEL: &str = "Z: ";
/// Java package-private static final `DIRECTORY_FOR_OUTPUT_LABEL`.
pub const DIRECTORY_FOR_OUTPUT_LABEL: &str = "Output directory for subvolumes: ";
/// Java package-private static final `MAKE_VOLUME_STACKS_LABEL`.
pub const MAKE_VOLUME_STACKS_LABEL: &str = "Make MRC volume stacks with up to ";
/// Java package-private static final `SKIP_SUBVOL_NUMBERS_LABEL`.
pub const SKIP_SUBVOL_NUMBERS_LABEL: &str = "Skip numbers for skipped subvolumes near edge";
/// Java package-private static final `NEW_ALIGNED_BINNING_LABEL`.
pub const NEW_ALIGNED_BINNING_LABEL: &str = "Make new aligned stack with binning";
/// Java package-private static final `USE_UNALIGNED_IMAGES_LABEL`.
pub const USE_UNALIGNED_IMAGES_LABEL: &str = "Reconstruct from raw images with reduction by";
/// Java package-private static final `EXTENT_OF_ZLEVELS_IN_NM_LABEL`.
pub const EXTENT_OF_ZLEVELS_IN_NM_LABEL: &str = "Do 3D CTF correction with division into ";
/// Java package-private static final `ERASE_FIDUCIALS_LABEL`.
pub const ERASE_FIDUCIALS_LABEL: &str = "Erase gold";
/// Java package-private static final `FILTER_IN_2D_LABEL`.
pub const FILTER_IN_2D_LABEL: &str = "Apply 2D filter";

/// Java `public class SubtomogramsPanel`.
pub struct SubtomogramsPanel {
    /// Rust-only: Java `this`.
    self_ref: Weak<SubtomogramsPanel>,
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java package-private `pnlReorientationType = new JPanel()`.
    pub pnl_reorientation_type: Rc<JComponent>,
    /// Java private final `bctfVolumeModeled`.
    bctf_volume_modeled: Rc<ButtonControlTextEfield>,
    /// Java private final `cbReorientationType`.
    cb_reorientation_type: Rc<CheckBox>,
    /// Java private `bgReorientationType` (never reassigned).
    bg_reorientation_type: Rc<ButtonGroup>,
    /// Java private final `rbReorientationTypeNone`.
    rb_reorientation_type_none: Rc<RadioButton>,
    /// Java private final `rbReorientationTypeRotated`.
    rb_reorientation_type_rotated: Rc<RadioButton>,
    /// Java private final `rbReorientationTypeFlipped`.
    rb_reorientation_type_flipped: Rc<RadioButton>,
    /// Java private final `bctfCenterPositionFile`.
    bctf_center_position_file: Rc<ButtonControlTextEfield>,
    /// Java private final `ltfObjectsToUse`.
    ltf_objects_to_use: Rc<LabeledTextField>,
    /// Java private final `ltfSizeInX`.
    ltf_size_in_x: Rc<LabeledTextField>,
    /// Java private final `ltfSizeInY`.
    ltf_size_in_y: Rc<LabeledTextField>,
    /// Java private final `ltfSizeInZ`.
    ltf_size_in_z: Rc<LabeledTextField>,
    /// Java private final `bctfDirectoryForOutput`.
    bctf_directory_for_output: Rc<ButtonControlTextEfield>,
    /// Java private final `cbMakeVolumeStacks`.
    cb_make_volume_stacks: Rc<CheckBox>,
    /// Java private final `spMakeVolumeStacks`.
    sp_make_volume_stacks: Rc<Spinner>,
    /// Java private final `strMakeVolumeStacks` (a `JLabel`).
    str_make_volume_stacks: Rc<JComponent>,
    /// Java private final `cbSkipSubVolNumbers`.
    cb_skip_sub_vol_numbers: Rc<CheckBox>,
    /// Java private `bgSourceOfImages` (never reassigned).
    bg_source_of_images: Rc<ButtonGroup>,
    /// Java private final `rbExistingAlignedStack`.
    rb_existing_aligned_stack: Rc<RadioButton>,
    /// Java private final `rbNewAlignedBinning`.
    rb_new_aligned_binning: Rc<RadioButton>,
    /// Java private final `spNewAlignedBinning`.
    sp_new_aligned_binning: Rc<Spinner>,
    /// Java private final `rbUseUnalignedImages`.
    rb_use_unaligned_images: Rc<RadioButton>,
    /// Java private final `spFourierReduceByFactor`.
    sp_fourier_reduce_by_factor: Rc<Spinner>,
    /// Java private final `cbExtentOfZLevelsInNm`.
    cb_extent_of_z_levels_in_nm: Rc<CheckBox>,
    /// Java private final `tfExtentOfZLevelsInNm`.
    tf_extent_of_z_levels_in_nm: Rc<TextField>,
    /// Java private final `strExtentOfZLevelsInNm` (a `JLabel`).
    str_extent_of_z_levels_in_nm: Rc<JComponent>,
    /// Java private final `cbAdjustForAlignZShift`.
    cb_adjust_for_align_z_shift: Rc<CheckBox>,
    /// Java private final `cbEraseFiducials`.
    cb_erase_fiducials: Rc<CheckBox>,
    /// Java private final `cbFilterIn2D`.
    cb_filter_in_2d: Rc<CheckBox>,
    /// Java private final `lEraseFiducials` (a `Label`).
    l_erase_fiducials: Rc<Label>,
    /// Java private final `lFilterIn2D` (a `Label`).
    l_filter_in_2d: Rc<Label>,
    /// Java private final `lctf3d` (a `JLabel`).
    lctf3d: Rc<JComponent>,

    // Use MultiLineButton w/o toggle
    /// Java private final `btnGenerateSubtomograms`.
    btn_generate_subtomograms: Rc<Run3dmodButton>,
    /// Java private final `btnViewSubtomogramsIn3dmod`.
    btn_view_subtomograms_in_3dmod: Rc<Run3dmodButton>,

    /// Java private final `manager`.
    manager: &'static ApplicationManager,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `subtomoSetupCpuGpuPanel`.
    subtomo_setup_cpu_gpu_panel: Rc<SubtomoSetupCpuGpuPanel>,
    /// Java private `subtomoBrowsingDir`.
    subtomo_browsing_dir: RefCell<Option<PathBuf>>,

    /// Rust-only: Java `this` as the `ActionListener` it registers.
    action_listener: ActionListener,
}

impl SubtomogramsPanel {
    /// Java private constructor `SubtomogramsPanel(ApplicationManager, AxisID,
    /// DialogType, ProcessInterface, GlobalExpandButton)`.
    fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        process_interface: Weak<dyn ProcessInterface>,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<SubtomogramsPanel> {
        let instance = Rc::new_cyclic(|self_ref: &Weak<SubtomogramsPanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Field initializers, in declaration order.
            let pnl_root = JComponent::new_panel();
            let pnl_reorientation_type = JComponent::new_panel();
            let cb_reorientation_type = CheckBox::new_string(Some(REORIENTATION_TYPE_LABEL));
            let bg_reorientation_type = ButtonGroup::new();
            let rb_reorientation_type_none =
                RadioButton::new_string_button_group(Some("None"), Some(&bg_reorientation_type));
            let rb_reorientation_type_rotated =
                RadioButton::new_string_button_group(Some("Rotated"), Some(&bg_reorientation_type));
            let rb_reorientation_type_flipped =
                RadioButton::new_string_button_group(Some("Flipped"), Some(&bg_reorientation_type));
            let ltf_objects_to_use = LabeledTextField::new_field_type_string(
                FieldType::IntegerList,
                Some(OBJECTS_TO_USE_LABEL),
            );
            let ltf_size_in_x =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some(SIZE_IN_X_LABEL));
            let ltf_size_in_y =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some(SIZE_IN_Y_LABEL));
            let ltf_size_in_z =
                LabeledTextField::new_field_type_string(FieldType::Integer, Some(SIZE_IN_Z_LABEL));
            let cb_make_volume_stacks = CheckBox::new_string(Some(MAKE_VOLUME_STACKS_LABEL));
            let sp_make_volume_stacks = Spinner::get_instance_string_int_int_int_int(
                Some(MAKE_VOLUME_STACKS_LABEL),
                subtomo_setup_param::MAKE_VOLUME_STACKS_SPINNER_DEFAULT,
                subtomo_setup_param::MAKE_VOLUME_STACKS_SPINNER_MIN,
                subtomo_setup_param::MAKE_VOLUME_STACKS_SPINNER_MAX,
                subtomo_setup_param::MAKE_VOLUME_STACKS_SPINNER_STEP,
            );
            let str_make_volume_stacks = JComponent::new_label(" subvolumes each");
            let cb_skip_sub_vol_numbers = CheckBox::new_string(Some(SKIP_SUBVOL_NUMBERS_LABEL));
            let bg_source_of_images = ButtonGroup::new();
            let rb_existing_aligned_stack = RadioButton::new_string_button_group(
                Some("Use existing aligned stack"),
                Some(&bg_source_of_images),
            );
            let rb_new_aligned_binning = RadioButton::new_string_button_group(
                Some(NEW_ALIGNED_BINNING_LABEL),
                Some(&bg_source_of_images),
            );
            let sp_new_aligned_binning = Spinner::get_instance_string_int_int_int(
                Some(NEW_ALIGNED_BINNING_LABEL),
                subtomo_setup_param::NEW_ALIGNED_BINNING_SPINNER_DEFAULT,
                subtomo_setup_param::NEW_ALIGNED_BINNING_SPINNER_MIN,
                subtomo_setup_param::NEW_ALIGNED_BINNING_SPINNER_MAX,
            );
            let rb_use_unaligned_images = RadioButton::new_string_button_group(
                Some(USE_UNALIGNED_IMAGES_LABEL),
                Some(&bg_source_of_images),
            );
            let sp_fourier_reduce_by_factor = Spinner::get_instance_string_int_int_int(
                Some(USE_UNALIGNED_IMAGES_LABEL),
                subtomo_setup_param::FOURIER_REDUCEBY_FACTOR_SPINNER_DEFAULT,
                subtomo_setup_param::FOURIER_REDUCEBY_FACTOR_SPINNER_MIN,
                subtomo_setup_param::FOURIER_REDUCEBY_FACTOR_SPINNER_MAX,
            );
            let cb_extent_of_z_levels_in_nm =
                CheckBox::new_string(Some(EXTENT_OF_ZLEVELS_IN_NM_LABEL));
            let tf_extent_of_z_levels_in_nm = TextField::new(
                FieldType::Integer,
                Some(EXTENT_OF_ZLEVELS_IN_NM_LABEL),
                None,
            );
            let str_extent_of_z_levels_in_nm = JComponent::new_label("nm extents");
            let cb_adjust_for_align_z_shift =
                CheckBox::new_string(Some("Adjust for Z shift in fine alignment and positioning"));
            let cb_erase_fiducials = CheckBox::new_string(Some(ERASE_FIDUCIALS_LABEL));
            let cb_filter_in_2d = CheckBox::new_string(Some(FILTER_IN_2D_LABEL));
            let l_erase_fiducials = Label::new_string_string(
                Some(ERASE_FIDUCIALS_LABEL),
                Some("   Parameters must already be set for gold erasing"),
            );
            let l_filter_in_2d = Label::new_string_string(
                Some(FILTER_IN_2D_LABEL),
                Some("   Parameters must already be set for 2-D filtering"),
            );
            let lctf3d = JComponent::new_label(&format!(
                "Parameters must already be set in Final Aligned Stack {}",
                shared_strings::CTF_CORRECTION_LABEL
            ));
            let btn_generate_subtomograms =
                Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                    Some("Generate Subtomograms"),
                    Some(container.clone()),
                );
            let btn_view_subtomograms_in_3dmod =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("View Subtomograms In 3dmod"),
                    Some(container),
                );
            // Java `this` as the ActionListener.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action_performed(Some(event));
                }
            });

            // Constructor body.
            let volume_modeled_file_extension = SelectFileExtension::new();
            volume_modeled_file_extension
                .set_file_chooser_title(Some("Select tomogram that was modeled"));
            let bctf_volume_modeled =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean_select_file_extension(
                    Some(VOLUME_MODELED_LABEL),
                    false,
                    false,
                    Some(volume_modeled_file_extension),
                );
            bctf_volume_modeled.set_file_filter(Some(
                crate::imod::etomo::storage::tomogram_file_filter::TomogramFileFilter::get_instance(
                    manager,
                ),
            ));
            bctf_volume_modeled.set_limit_displayed_file_path(40);
            // bctfVolumeModeled.setAltBrowsingDirectory(this): after the Rc exists
            // (see the module docs).
            let center_position_file_file_extension = SelectFileExtension::new();
            center_position_file_file_extension
                .set_file_chooser_title(Some("Select model or point file"));
            let bctf_center_position_file =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean_select_file_extension(
                    Some(CENTER_POSITION_FILE_LABEL),
                    false,
                    false,
                    Some(center_position_file_file_extension),
                );
            bctf_center_position_file.set_file_filter(Some(Rc::new(
                crate::imod::etomo::storage::model_or_point_file_filter::ModelOrPointFileFilter::get_instance(
                    Some(manager),
                    true,
                ),
            )));
            bctf_center_position_file.set_limit_displayed_file_path(50);
            // bctfCenterPositionFile.setAltBrowsingDirectory(this): see above.
            let directory_for_output_file_extension = SelectFileExtension::new();
            directory_for_output_file_extension
                .set_file_chooser_title(Some("Select output directory for subvolumes"));
            let bctf_directory_for_output =
                ButtonControlTextEfield::get_labeled_file_instance_string_boolean_boolean_select_file_extension(
                    Some(DIRECTORY_FOR_OUTPUT_LABEL),
                    true,
                    false,
                    Some(directory_for_output_file_extension),
                );
            bctf_directory_for_output.set_file_selection_mode(file_chooser::DIRECTORIES_ONLY);
            bctf_directory_for_output.set_limit_displayed_file_path(28);
            bctf_directory_for_output
                .set_select_file_dir(manager.get_property_user_dir().as_deref());
            let subtomo_setup_cpu_gpu_panel = SubtomoSetupCpuGpuPanel::get_subtomo_setup_instance(
                self_ref.clone(),
                manager,
                axis_id,
                process_interface,
            );
            // Java `new File(manager.getPropertyUserDir())`.
            let subtomo_browsing_dir = manager.get_property_user_dir().map(PathBuf::from);
            let expandable: Weak<dyn Expandable> = self_ref.clone();
            global_advanced_button.register_expandable(expandable);
            SubtomogramsPanel {
                self_ref: self_ref.clone(),
                pnl_root,
                pnl_reorientation_type,
                bctf_volume_modeled,
                cb_reorientation_type,
                bg_reorientation_type,
                rb_reorientation_type_none,
                rb_reorientation_type_rotated,
                rb_reorientation_type_flipped,
                bctf_center_position_file,
                ltf_objects_to_use,
                ltf_size_in_x,
                ltf_size_in_y,
                ltf_size_in_z,
                bctf_directory_for_output,
                cb_make_volume_stacks,
                sp_make_volume_stacks,
                str_make_volume_stacks,
                cb_skip_sub_vol_numbers,
                bg_source_of_images,
                rb_existing_aligned_stack,
                rb_new_aligned_binning,
                sp_new_aligned_binning,
                rb_use_unaligned_images,
                sp_fourier_reduce_by_factor,
                cb_extent_of_z_levels_in_nm,
                tf_extent_of_z_levels_in_nm,
                str_extent_of_z_levels_in_nm,
                cb_adjust_for_align_z_shift,
                cb_erase_fiducials,
                cb_filter_in_2d,
                l_erase_fiducials,
                l_filter_in_2d,
                lctf3d,
                btn_generate_subtomograms,
                btn_view_subtomograms_in_3dmod,
                manager,
                axis_id,
                dialog_type,
                subtomo_setup_cpu_gpu_panel,
                subtomo_browsing_dir: RefCell::new(subtomo_browsing_dir),
                action_listener,
            }
        });
        // The constructor's setAltBrowsingDirectory(this) calls (see the module
        // docs).
        let browsing_directory: Rc<dyn BrowsingDirectory> = instance.clone();
        instance
            .bctf_volume_modeled
            .set_alt_browsing_directory(Some(browsing_directory.clone()));
        instance
            .bctf_center_position_file
            .set_alt_browsing_directory(Some(browsing_directory));
        instance
    }

    /// Java static `getInstance(ApplicationManager, AxisID, DialogType,
    /// ProcessInterface, GlobalExpandButton)`.
    pub fn get_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        process_interface: Weak<dyn ProcessInterface>,
        global_advanced_button: &Rc<GlobalExpandButton>,
    ) -> Rc<SubtomogramsPanel> {
        let instance = SubtomogramsPanel::new(
            manager,
            axis_id,
            dialog_type,
            process_interface,
            global_advanced_button,
        );
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        let pnl_outer_panel = JComponent::new_panel();
        let pnl_size_in_xyz = JComponent::new_panel();
        let pnl_make_volume_stacks = JComponent::new_panel();
        let pnl_skip_sub_vol_numbers = JComponent::new_panel();
        let pnl_source_of_images = JComponent::new_panel();
        let pnl_existing_aligned_stack = JComponent::new_panel();
        let pnl_new_aligned_binning = JComponent::new_panel();
        let pnl_use_unaligned_images = JComponent::new_panel();
        let pnl_extent_of_z_levels_in_nm_warning = JComponent::new_panel();
        let pnl_extent_of_z_levels_in_nm = JComponent::new_panel();
        let pnl_adjust_for_align_z_shift = JComponent::new_panel();
        let pnl_erase_fiducials = JComponent::new_panel();
        let pnl_filter_in_2d = JComponent::new_panel();
        // Java also creates pnlCpusOnly, pnlGpuForReconAndCtfCorrect,
        // pnlCpuForReconAndGpuForCtfCorrect here and never uses them.
        let _pnl_cpus_only = JComponent::new_panel();
        let _pnl_gpu_for_recon_and_ctf_correct = JComponent::new_panel();
        let _pnl_cpu_for_recon_and_gpu_for_ctf_correct = JComponent::new_panel();
        let pnl_buttons = JComponent::new_panel();
        // init
        self.bctf_volume_modeled.set_required(true);
        self.bctf_center_position_file.set_required(true);
        self.rb_reorientation_type_none.set_selected_boolean(true);
        self.rb_existing_aligned_stack.set_selected_boolean(true);
        self.subtomo_setup_cpu_gpu_panel
            .set_selected_cpus_only(true);
        self.ltf_size_in_x.set_required(true);
        self.ltf_size_in_x.set_number_must_be_positive(true);
        self.ltf_size_in_y.set_required(true);
        self.ltf_size_in_y.set_number_must_be_positive(true);
        self.ltf_size_in_z.set_required(true);
        self.ltf_size_in_z.set_number_must_be_positive(true);
        self.cb_make_volume_stacks.set_selected_boolean(false);
        self.tf_extent_of_z_levels_in_nm.set_required(true);
        self.subtomo_setup_cpu_gpu_panel
            .set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(false);
        let deferred: Rc<dyn Deferred3dmodButton> = self.btn_view_subtomograms_in_3dmod.clone();
        self.btn_generate_subtomograms
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(deferred));
        self.lctf3d
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_filter_in_2d
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        self.l_erase_fiducials
            .get_component()
            .set_foreground(Some(process_control_panel::COLOR_COMPLETE));
        let base_manager: &'static dyn BaseManager = self.manager;
        if Network::get_total_gpus(
            base_manager,
            self.axis_id,
            self.manager.get_property_user_dir().as_deref(),
        ) < 1
        {
            self.subtomo_setup_cpu_gpu_panel
                .set_enabled_gpu_for_recon_and_ctf_correct(false);
            self.subtomo_setup_cpu_gpu_panel
                .set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(false);
        }
        // Root
        // Swing layout: pnlRoot BoxLayout X_AXIS.
        self.pnl_root.add(&pnl_outer_panel);
        // Swing layout: pnlOuterPanel BoxLayout Y_AXIS; rigid areas x0_y10 at the
        // top and x0_y5 between most rows (none between the extent, Z shift,
        // erase and filter rows).
        pnl_outer_panel.set_border_title(
            BeveledBorder::new(Some("Subtomogram Generation"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_outer_panel.add(&self.bctf_volume_modeled.get_component());
        pnl_outer_panel.add(&self.pnl_reorientation_type);
        pnl_outer_panel.add(&self.bctf_center_position_file.get_component());
        pnl_outer_panel.add(&self.ltf_objects_to_use.get_component());
        pnl_outer_panel.add(&pnl_size_in_xyz);
        pnl_outer_panel.add(&self.bctf_directory_for_output.get_component());
        pnl_outer_panel.add(&pnl_make_volume_stacks);
        pnl_outer_panel.add(&pnl_skip_sub_vol_numbers);
        pnl_outer_panel.add(&pnl_source_of_images);
        pnl_outer_panel.add(&pnl_extent_of_z_levels_in_nm_warning);
        pnl_outer_panel.add(&pnl_extent_of_z_levels_in_nm);
        pnl_outer_panel.add(&pnl_adjust_for_align_z_shift);
        pnl_outer_panel.add(&pnl_erase_fiducials);
        pnl_outer_panel.add(&pnl_filter_in_2d);
        pnl_outer_panel.add(&self.subtomo_setup_cpu_gpu_panel.get_component());
        pnl_outer_panel.add(&pnl_buttons);
        // Reorientation type panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        self.pnl_reorientation_type
            .add(&self.cb_reorientation_type.get_component());
        self.pnl_reorientation_type
            .add(&self.rb_reorientation_type_none.get_component());
        self.pnl_reorientation_type
            .add(&self.rb_reorientation_type_flipped.get_component());
        self.pnl_reorientation_type
            .add(&self.rb_reorientation_type_rotated.get_component());
        // Size in XYZ panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_size_in_xyz.add(&self.ltf_size_in_x.get_component());
        pnl_size_in_xyz.add(&self.ltf_size_in_y.get_component());
        pnl_size_in_xyz.add(&self.ltf_size_in_z.get_component());
        // Make volume stacks
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_make_volume_stacks.add(&self.cb_make_volume_stacks.get_component());
        pnl_make_volume_stacks.add(&self.sp_make_volume_stacks.get_component());
        pnl_make_volume_stacks.add(&self.str_make_volume_stacks);
        // pnlSkipSubVolNumbers
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_skip_sub_vol_numbers.add(&self.cb_skip_sub_vol_numbers.get_component());
        // Source of images panel
        // Swing layout: BoxLayout Y_AXIS; rigid areas x0_y2 between the rows,
        // horizontal glue at the end.
        pnl_source_of_images.set_border_title(
            EtchedBorder::new(Some("Source of images"))
                .get_border()
                .get_title()
                .as_deref(),
        );
        pnl_source_of_images.add(&pnl_existing_aligned_stack);
        pnl_source_of_images.add(&pnl_new_aligned_binning);
        pnl_source_of_images.add(&pnl_use_unaligned_images);
        // Swing layout: the three rows BoxLayout X_AXIS, horizontal glue at the end.
        pnl_existing_aligned_stack.add(&self.rb_existing_aligned_stack.get_component());
        pnl_new_aligned_binning.add(&self.rb_new_aligned_binning.get_component());
        pnl_new_aligned_binning.add(&self.sp_new_aligned_binning.get_component());
        pnl_use_unaligned_images.add(&self.rb_use_unaligned_images.get_component());
        pnl_use_unaligned_images.add(&self.sp_fourier_reduce_by_factor.get_component());
        // Warning message for Extent Of ZLevels In Nm panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_extent_of_z_levels_in_nm_warning.add(&self.lctf3d);
        // Extent Of ZLevels In Nm panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_extent_of_z_levels_in_nm.add(&self.cb_extent_of_z_levels_in_nm.get_component());
        pnl_extent_of_z_levels_in_nm.add(&self.tf_extent_of_z_levels_in_nm.get_component());
        pnl_extent_of_z_levels_in_nm.add(&self.str_extent_of_z_levels_in_nm);
        // AdjustForAlignZShift
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_adjust_for_align_z_shift.add(&self.cb_adjust_for_align_z_shift.get_component());
        // Erase fiducials panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_erase_fiducials.add(&self.cb_erase_fiducials.get_component());
        pnl_erase_fiducials.add(&self.l_erase_fiducials.get_component());
        // Filter in 2D panel
        // Swing layout: BoxLayout X_AXIS; horizontal glue at the end.
        pnl_filter_in_2d.add(&self.cb_filter_in_2d.get_component());
        pnl_filter_in_2d.add(&self.l_filter_in_2d.get_component());
        // Buttons panel
        // Swing layout: BoxLayout X_AXIS; rigid area x70_y0 between the buttons.
        pnl_buttons.add(&self.btn_generate_subtomograms.get_component());
        pnl_buttons.add(&self.btn_view_subtomograms_in_3dmod.get_component());

        self.update_display();
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        let context_menu: Weak<dyn ContextMenu> = self.self_ref.clone();
        let mouse_adapter: Rc<dyn MouseListener> = GenericMouseAdapter::new(context_menu);
        self.pnl_root.add_mouse_listener(mouse_adapter);
        if let Some(this) = self.self_ref.upgrade() {
            let control_listener: Rc<dyn ControlListener> = this;
            self.bctf_volume_modeled
                .add_control_listener(Some(control_listener.clone()));
            self.bctf_center_position_file
                .add_control_listener(Some(control_listener.clone()));
            self.bctf_directory_for_output
                .add_control_listener(Some(control_listener));
        }
        let listener = || Some(self.action_listener.clone());
        self.cb_reorientation_type.add_action_listener(listener());
        self.rb_reorientation_type_none
            .add_action_listener(self.action_listener.clone());
        self.rb_reorientation_type_flipped
            .add_action_listener(self.action_listener.clone());
        self.rb_reorientation_type_rotated
            .add_action_listener(self.action_listener.clone());
        self.cb_skip_sub_vol_numbers.add_action_listener(listener());
        self.cb_make_volume_stacks.add_action_listener(listener());
        self.rb_existing_aligned_stack
            .add_action_listener(self.action_listener.clone());
        self.rb_new_aligned_binning
            .add_action_listener(self.action_listener.clone());
        self.rb_use_unaligned_images
            .add_action_listener(self.action_listener.clone());
        self.cb_extent_of_z_levels_in_nm
            .add_action_listener(listener());
        self.cb_erase_fiducials.add_action_listener(listener());
        self.cb_filter_in_2d.add_action_listener(listener());
        self.btn_generate_subtomograms
            .add_action_listener(self.action_listener.clone());
        self.btn_view_subtomograms_in_3dmod
            .add_action_listener(self.action_listener.clone());
    }

    /// Java `actionPerformed(ActionEvent)` (the panel is its own
    /// `ActionListener`).
    pub fn action_performed(&self, event: Option<&ActionEvent>) {
        let action_command = event.and_then(|event| event.get_action_command().map(str::to_owned));
        self.action_option(action_command.as_deref(), None, None);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)` with a
    /// possibly-null command.  Java dereferences the command
    /// (`actionCommand.equals(...)`); a null command (only reachable from
    /// `actionPerformed(null)`) is treated as matching no branch.
    fn action_option(
        &self,
        action_command: Option<&str>,
        _deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        let matches = |command: Option<String>| {
            action_command.is_some() && action_command == command.as_deref()
        };
        if matches(self.cb_extent_of_z_levels_in_nm.get_action_command()) {
            if !self.cb_extent_of_z_levels_in_nm.is_selected() {
                self.subtomo_setup_cpu_gpu_panel
                    .set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(false);
                if self
                    .subtomo_setup_cpu_gpu_panel
                    .is_selected_cpu_for_recon_and_gpu_for_ctf_correct()
                {
                    self.subtomo_setup_cpu_gpu_panel
                        .set_selected_cpus_only(true);
                }
            } else {
                self.subtomo_setup_cpu_gpu_panel
                    .set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(true);
            }
        } else if matches(self.btn_generate_subtomograms.get_action_command()) {
            // Java `new File(bctfDirectoryForOutput.getText(),
            // FileType.PROCESSCHUNKS_MRC.getFileName(manager, manager.getName(),
            // axisID)).exists()`; a null parent is no parent.
            let base_manager: &'static dyn BaseManager = self.manager;
            let child = file_type::CLASS
                .processchunks_mrc
                .get_file_name_with_root_name(
                    Some(base_manager),
                    self.manager.get_name().as_deref(),
                    Some(self.axis_id),
                )
                .unwrap_or_else(|| "null".to_string());
            let file = match self.bctf_directory_for_output.get_text_void() {
                Some(parent) => Path::new(&parent).join(child),
                None => PathBuf::from(child),
            };
            let file_exists = file.exists();
            if file_exists {
                let file_exists_warning = "The output directory already has some files in it, \
                                           are you sure you want to write to the same directory \
                                           instead of choosing a new one?";
                if !ui_harness::with(|harness| {
                    harness.open_yes_no_dialog_with_default_no(
                        Some(base_manager),
                        file_exists_warning,
                        "Directory has output files",
                        Some(self.axis_id),
                    )
                }) {
                    return;
                }
            }
            self.manager.subtomo_setup(
                self.axis_id,
                self.dialog_type,
                Some(self.subtomo_setup_cpu_gpu_panel.get_processing_method()),
                self,
            );
        } else if matches(self.btn_view_subtomograms_in_3dmod.get_action_command()) {
            self.open_files_in_imod(run_3dmod_menu_options);
        }
        self.update_display();
    }

    /// Java private `openFilesInImod(Run3dmodMenuOptions)`.
    fn open_files_in_imod(&self, run_3dmod_menu_options: Option<Run3dmodMenuOptions>) {
        let base_manager: &'static dyn BaseManager = self.manager;
        let chooser = FileChooser::new_base_manager(Some(base_manager));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        chooser.set_multi_selection_enabled(true);
        chooser.set_file_filter(Some(Rc::new(
            crate::imod::etomo::storage::chunk_file_filter::ChunkFileFilter::get_instance(
                base_manager,
                true,
            ),
        )));
        let return_val = chooser.show_open_dialog(Some(&self.pnl_root));
        if return_val != file_chooser::APPROVE_OPTION {
            return;
        }
        let file_list = chooser.get_selected_files();
        if file_list.is_empty() {
            return;
        }
        let subdir_name = file_list[0]
            .parent()
            .map(|parent| parent.to_string_lossy().into_owned());
        let file_name_array: Vec<String> = file_list
            .iter()
            .map(|file| {
                file.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        // Java passes `run3dmodMenuOptions` through; a null one is the default
        // (no) options.
        self.manager
            .open_files_in_imod_axis_id_string_string_array_string_run3dmod_menu_options(
                self.axis_id,
                imod_manager::SUBTOMO_SETUP_KEY,
                &file_name_array,
                subdir_name.as_deref(),
                run_3dmod_menu_options.unwrap_or_default(),
            );
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java private `updateDisplay()`.
    fn update_display(&self) {
        self.rb_reorientation_type_none
            .set_enabled(self.cb_reorientation_type.is_selected());
        self.rb_reorientation_type_rotated
            .set_enabled(self.cb_reorientation_type.is_selected());
        self.rb_reorientation_type_flipped
            .set_enabled(self.cb_reorientation_type.is_selected());
        self.sp_make_volume_stacks
            .set_enabled(self.cb_make_volume_stacks.is_selected());
        self.sp_new_aligned_binning
            .set_enabled(self.rb_new_aligned_binning.is_selected());
        self.sp_fourier_reduce_by_factor
            .set_enabled(self.rb_use_unaligned_images.is_selected());
        self.tf_extent_of_z_levels_in_nm
            .set_enabled(self.cb_extent_of_z_levels_in_nm.is_selected());
        self.l_erase_fiducials
            .get_component()
            .set_enabled(self.cb_erase_fiducials.is_selected());
        self.l_filter_in_2d
            .get_component()
            .set_enabled(self.cb_filter_in_2d.is_selected());
        self.subtomo_setup_cpu_gpu_panel
            .set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(
                self.cb_extent_of_z_levels_in_nm.is_selected(),
            );
        if !self.cb_extent_of_z_levels_in_nm.is_selected()
            && self
                .subtomo_setup_cpu_gpu_panel
                .is_selected_cpu_for_recon_and_gpu_for_ctf_correct()
            && !self
                .subtomo_setup_cpu_gpu_panel
                .is_mix_cpu_gpu_action_event()
        {
            self.subtomo_setup_cpu_gpu_panel
                .set_selected_cpus_only(true);
        }
    }

    /// Java package-private `updateAdvanced(boolean)`.
    pub fn update_advanced(&self, advanced: bool) {
        self.pnl_reorientation_type.set_visible(advanced);
        self.ltf_objects_to_use.set_visible(advanced);
        self.cb_adjust_for_align_z_shift.set_visible(advanced);
    }

    /// Java package-private `getParameters(MetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data
            .set_subtomo_reorientation_type_none(self.rb_reorientation_type_none.is_selected());
        meta_data.set_subtomo_reorientation_type_flipped(
            self.rb_reorientation_type_flipped.is_selected(),
        );
        meta_data.set_subtomo_reorientation_type_rotated(
            self.rb_reorientation_type_rotated.is_selected(),
        );
        meta_data.set_subtomo_make_volume_stacks(Some(self.sp_make_volume_stacks.get_value()));
        if !self.rb_new_aligned_binning.is_selected() {
            meta_data
                .set_subtomo_new_aligned_binning(Some(self.sp_new_aligned_binning.get_value()));
        }
        if !self.rb_use_unaligned_images.is_selected() {
            meta_data.set_subtomo_fourier_reduce_by_factor(Some(
                self.sp_fourier_reduce_by_factor.get_value(),
            ));
        }
        meta_data.set_subtomo_extent_of_z_levels_in_nm(
            self.tf_extent_of_z_levels_in_nm.get_text_void().as_deref(),
        );
    }

    /// Java package-private `setParameters(ConstMetaData)`.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        if meta_data.is_subtomo_reorientation_type_none() {
            self.rb_reorientation_type_none.set_selected_boolean(true);
        } else if meta_data.is_subtomo_reorientation_type_flipped() {
            self.rb_reorientation_type_flipped
                .set_selected_boolean(true);
        } else if meta_data.is_subtomo_reorientation_type_rotated() {
            self.rb_reorientation_type_rotated
                .set_selected_boolean(true);
        }
        if meta_data.is_subtomo_make_volume_stacks() {
            self.sp_make_volume_stacks
                .set_value_string(Some(&meta_data.get_subtomo_make_volume_stacks()));
        }
        self.sp_new_aligned_binning
            .set_value_string(Some(&meta_data.get_subtomo_new_aligned_binning()));
        self.sp_fourier_reduce_by_factor
            .set_value_string(Some(&meta_data.get_subtomo_fourier_reduce_by_factor()));
        self.tf_extent_of_z_levels_in_nm
            .set_text_string(Some(&meta_data.get_subtomo_extent_of_z_levels_in_nm()));
    }

    /// Java package-private `setParameters(SubtomoSetupParam)`.
    pub fn set_parameters_subtomo_setup_param(&self, param: &SubtomoSetupParam) {
        if param.is_volume_modeled() {
            self.bctf_volume_modeled
                .set_text_string(Some(&param.get_volume_modeled()));
        }
        if param.is_reorientation_type() {
            self.cb_reorientation_type.set_selected_boolean(true);
            let int_reorientation_type = param.get_reorientation_type();
            if int_reorientation_type == subtomo_setup_param::REORIENTATION_TYPE_NONE {
                self.rb_reorientation_type_none.set_selected_boolean(true);
            } else if int_reorientation_type == subtomo_setup_param::REORIENTATION_TYPE_FLIPPED {
                self.rb_reorientation_type_flipped
                    .set_selected_boolean(true);
            } else if int_reorientation_type == subtomo_setup_param::REORIENTATION_TYPE_ROTATED {
                self.rb_reorientation_type_rotated
                    .set_selected_boolean(true);
            }
        }
        if param.is_center_position_file() {
            self.bctf_center_position_file
                .set_text_string(Some(&param.get_center_position_file()));
        }
        if param.is_objects_to_use() {
            self.ltf_objects_to_use
                .set_text_string(Some(&param.get_ojects_to_use()));
        }
        if param.is_size_in_xyz() {
            self.ltf_size_in_x.set_text_int(param.get_size_in_x());
            self.ltf_size_in_y.set_text_int(param.get_size_in_y());
            self.ltf_size_in_z.set_text_int(param.get_size_in_z());
        }
        if param.is_directory_for_output() {
            self.bctf_directory_for_output
                .set_text_string(Some(&param.get_directory_for_output()));
        }
        if param.is_make_volume_stacks() {
            self.cb_make_volume_stacks.set_selected_boolean(true);
            self.sp_make_volume_stacks
                .set_value_string(Some(&param.get_make_volume_stacks()));
        }
        self.cb_skip_sub_vol_numbers
            .set_selected_boolean(param.is_skip_sub_vol_numbers());
        if param.is_new_aligned_binning() {
            self.rb_new_aligned_binning.set_selected_boolean(true);
            self.sp_new_aligned_binning
                .set_value_int(param.get_new_aligned_binning());
        }
        self.rb_use_unaligned_images
            .set_selected_boolean(param.is_use_unaligned_images());
        if self.rb_use_unaligned_images.is_selected() && param.is_fourier_reduce_by_factor() {
            self.sp_fourier_reduce_by_factor
                .set_value_int(param.get_fourier_reduce_by_factor());
        }
        if param.is_extent_of_z_levels_in_nm() {
            self.cb_extent_of_z_levels_in_nm.set_selected_boolean(true);
            self.tf_extent_of_z_levels_in_nm
                .set_text_string(Some(&param.get_extent_of_z_levels_in_nm()));
            self.subtomo_setup_cpu_gpu_panel
                .set_enabled_cpu_for_recon_and_gpu_for_ctf_correct(true);
        }
        self.cb_adjust_for_align_z_shift
            .set_selected_boolean(param.is_adjust_for_align_z_shift());
        self.cb_erase_fiducials
            .set_selected_boolean(param.is_erase_fiducials());
        self.cb_filter_in_2d
            .set_selected_boolean(param.is_filter_in_2d());
        if param.is_when_to_use_gpu() {
            if param.get_when_to_use_gpu() == subtomo_setup_param::WHEN_TO_USE_GPU_VAL_0 {
                self.subtomo_setup_cpu_gpu_panel
                    .set_selected_cpus_only(true);
            } else if param.get_when_to_use_gpu() == subtomo_setup_param::WHEN_TO_USE_GPU_VAL_1 {
                self.subtomo_setup_cpu_gpu_panel
                    .set_selected_gpu_for_recon_and_ctf_correct(true);
            } else if param.get_when_to_use_gpu() == subtomo_setup_param::WHEN_TO_USE_GPU_VAL_2 {
                self.subtomo_setup_cpu_gpu_panel
                    .set_selected_cpu_for_recon_and_gpu_for_ctf_correct(true);
            }
        }

        self.update_display();
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        // Java `ReadOnlyAutodoc autodoc = null;` then the try/catch.  The Java
        // passes a null AxisID; SUBTOMO_SETUP is not a per-axis autodoc, so
        // `AxisID::Only` stands in for it.
        let mut autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        // SAFETY: the factory keeps every autodoc it returns (and its sections)
        // for the life of the process.
        match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::SUBTOMO_SETUP),
                AxisID::Only,
                false,
            )
        } {
            Ok(instance) => autodoc = instance as *const Autodoc,
            // catch (final LockException except) {}
            Err(LogFileError::Lock(_)) => {}
            // catch (final LogFileException | IOException except):
            // except.printStackTrace().
            Err(except) => eprintln!("{except}"),
        }
        if autodoc.is_null() {
            return;
        }
        // SAFETY: see above.
        let autodoc: &dyn ReadOnlyAutodoc = unsafe { &*autodoc };
        let autodoc_name = autodoc.get_autodoc_name();
        let get_tooltip =
            |field_name: &str| etomo_autodoc::get_tooltip(Some(autodoc), Some(field_name));
        self.bctf_volume_modeled
            .set_tooltip_string(get_tooltip(subtomo_setup_param::VOLUME_MODELED).as_deref());
        self.cb_reorientation_type.set_tool_tip_text_string(Some(
            "If Subtomosetup cannot determine how the modeled volume was reoriented, select \
             this option to specify that",
        ));
        let section_reorientation_type = unsafe {
            autodoc.get_section(
                Some(etomo_autodoc::FIELD_SECTION_NAME),
                Some(subtomo_setup_param::REORIENTATION_TYPE),
            )
        };
        // Java `EtomoAutodoc.getTooltip(String, ReadOnlySection, String)`: a null
        // section is caught (NullPointerException) and answers null.
        let enum_tooltip = |value: i32| -> Option<String> {
            if section_reorientation_type.is_null() {
                return None;
            }
            // SAFETY: see above.
            let section: &dyn ReadOnlySection = unsafe { &*section_reorientation_type };
            etomo_autodoc::get_tooltip_enum_value_name(
                Some(&autodoc_name),
                section,
                Some(&value.to_string()),
            )
        };
        self.rb_reorientation_type_none.set_tool_tip_text_string(
            enum_tooltip(subtomo_setup_param::REORIENTATION_TYPE_NONE).as_deref(),
        );
        self.rb_reorientation_type_flipped.set_tool_tip_text_string(
            enum_tooltip(subtomo_setup_param::REORIENTATION_TYPE_FLIPPED).as_deref(),
        );
        self.rb_reorientation_type_rotated.set_tool_tip_text_string(
            enum_tooltip(subtomo_setup_param::REORIENTATION_TYPE_ROTATED).as_deref(),
        );
        self.bctf_center_position_file
            .set_tooltip_string(get_tooltip(subtomo_setup_param::CENTER_POSITION_FILE).as_deref());
        self.ltf_objects_to_use
            .set_tool_tip_text(get_tooltip(subtomo_setup_param::OBJECTS_TO_USE).as_deref());
        self.ltf_size_in_x
            .set_tool_tip_text(get_tooltip(subtomo_setup_param::SIZE_IN_XYZ).as_deref());
        self.ltf_size_in_y
            .set_tool_tip_text(get_tooltip(subtomo_setup_param::SIZE_IN_XYZ).as_deref());
        self.ltf_size_in_z
            .set_tool_tip_text(get_tooltip(subtomo_setup_param::SIZE_IN_XYZ).as_deref());
        self.bctf_directory_for_output
            .set_tooltip_string(get_tooltip(subtomo_setup_param::DIRECTORY_FOR_OUTPUT).as_deref());
        self.cb_make_volume_stacks.set_tool_tip_text_string(
            get_tooltip(subtomo_setup_param::MAKE_VOLUME_STACKS).as_deref(),
        );
        self.sp_make_volume_stacks
            .set_tool_tip_text(Some("Maximum number of subvolumes to place in each stack"));
        self.cb_skip_sub_vol_numbers.set_tool_tip_text_string(
            get_tooltip(subtomo_setup_param::SKIP_SUBVOL_NUMBERS).as_deref(),
        );
        self.rb_existing_aligned_stack
            .set_tool_tip_text_string(Some(
                "Use the existing aligned stack as the basis for CTF correction and \
             reconstructions",
            ));
        self.rb_new_aligned_binning.set_tool_tip_text_string(
            get_tooltip(subtomo_setup_param::NEW_ALIGNED_BINNING).as_deref(),
        );
        self.sp_new_aligned_binning
            .set_tool_tip_text(Some("Specify the binning for making a new aligned stack"));
        self.rb_use_unaligned_images.set_tool_tip_text_string(
            get_tooltip(subtomo_setup_param::USE_UNALIGNED_IMAGES).as_deref(),
        );
        self.sp_fourier_reduce_by_factor.set_tool_tip_text(
            get_tooltip(subtomo_setup_param::FOURIER_REDUCEBY_FACTOR).as_deref(),
        );
        self.cb_extent_of_z_levels_in_nm
            .set_tool_tip_text_string(Some(
                "Do CTF correction in 3-D, with corrections computed using the existing command \
             file for a set of extents in Z",
            ));
        self.tf_extent_of_z_levels_in_nm.set_tool_tip_text(
            get_tooltip(subtomo_setup_param::EXTENT_OF_ZLEVELS_IN_NM).as_deref(),
        );
        self.cb_adjust_for_align_z_shift
            .set_tool_tip_text_string(Some(
                "Adjust for Z shift of tomogram away from cross-correlation alignment; use this if \
             the average material determining the CTF is not centered in the tomogram but \
             would probably have been centered with the cross-correlation alignment.",
            ));
        self.cb_erase_fiducials
            .set_tool_tip_text_string(get_tooltip(subtomo_setup_param::ERASE_FIDUCIALS).as_deref());
        self.cb_filter_in_2d
            .set_tool_tip_text_string(get_tooltip(subtomo_setup_param::FILTER_IN_2D).as_deref());
        self.subtomo_setup_cpu_gpu_panel.set_tooltips();
        self.btn_generate_subtomograms.set_tool_tip_text(Some(
            "Run Subtomosetup to create command files and make reconstructions with the \
             selected resources",
        ));
        self.btn_view_subtomograms_in_3dmod
            .set_tool_tip_text(Some("Select a set of reconstructions to view in 3dmod"));
    }

    /// Java public `getProcessingMethod()`.
    pub fn get_processing_method(&self) -> ProcessingMethod {
        self.subtomo_setup_cpu_gpu_panel.get_processing_method()
    }

    /// Java public `getSubtomoSetupDisplay()`: `this`.
    pub fn get_subtomo_setup_display(&self) -> &dyn SubtomoSetupDisplay {
        self
    }

    /// Java package-private `updateGpu(boolean)`.
    pub fn update_gpu(&self, disable: bool) {
        self.update_display();
        self.subtomo_setup_cpu_gpu_panel.update_gpu(disable);
    }

    /// Java package-private `isUseGpu()`.
    pub fn is_use_gpu(&self) -> bool {
        self.subtomo_setup_cpu_gpu_panel.is_use_gpu()
    }

    /// Java package-private `setUseQueueCheckBox(ButtonComponent)`.
    pub fn set_use_queue_check_box(&self, use_queue_check_box: Option<Rc<dyn ButtonComponent>>) {
        self.subtomo_setup_cpu_gpu_panel
            .set_use_queue_check_box(use_queue_check_box);
    }

    /// Java public `isExtentOfZLevelsInNm()`.
    pub fn is_extent_of_z_levels_in_nm(&self) -> bool {
        self.cb_extent_of_z_levels_in_nm.is_selected()
    }
}

impl SubtomoSetupDisplay for SubtomogramsPanel {
    /// Java override `getParameters(SubtomoSetupParam, boolean)`.
    fn get_parameters(&self, param: &mut SubtomoSetupParam, do_validation: bool) -> bool {
        // try { ... } catch (FieldValidationFailedException e) {
        // e.printStackTrace(); return false; }
        let result = (|| -> Result<(), crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException> {
            param.set_volume_modeled(self.bctf_volume_modeled.get_text_void().as_deref());
            if self.cb_reorientation_type.is_selected() {
                if self.rb_reorientation_type_none.is_selected() {
                    param.set_reorientation_type(Some(
                        &subtomo_setup_param::REORIENTATION_TYPE_NONE.to_string(),
                    ));
                } else if self.rb_reorientation_type_flipped.is_selected() {
                    param.set_reorientation_type(Some(
                        &subtomo_setup_param::REORIENTATION_TYPE_FLIPPED.to_string(),
                    ));
                } else if self.rb_reorientation_type_rotated.is_selected() {
                    param.set_reorientation_type(Some(
                        &subtomo_setup_param::REORIENTATION_TYPE_ROTATED.to_string(),
                    ));
                }
            } else {
                param.reset_reorientation_type();
            }
            param.set_center_position_file(self.bctf_center_position_file.get_text_void().as_deref());
            param.set_ojects_to_use(self.ltf_objects_to_use.get_text_boolean(do_validation)?.as_deref());
            param.set_size_in_x(self.ltf_size_in_x.get_text_boolean(do_validation)?.as_deref());
            param.set_size_in_y(self.ltf_size_in_y.get_text_boolean(do_validation)?.as_deref());
            param.set_size_in_z(self.ltf_size_in_z.get_text_boolean(do_validation)?.as_deref());
            param.set_directory_for_output(self.bctf_directory_for_output.get_text_void().as_deref());
            if self.cb_make_volume_stacks.is_selected() {
                param.set_make_volume_stacks(Some(&self.sp_make_volume_stacks.get_value().to_string()));
            } else {
                param.reset_make_volume_stacks();
            }
            param.set_skip_sub_vol_numbers(self.cb_skip_sub_vol_numbers.is_selected());
            if self.rb_new_aligned_binning.is_selected() {
                param.set_new_aligned_binning(Some(&self.sp_new_aligned_binning.get_value().to_string()));
            } else {
                param.reset_new_aligned_binning();
            }
            param.set_use_unaligned_images(self.rb_use_unaligned_images.is_selected());
            if self.rb_use_unaligned_images.is_selected() {
                param.set_fourier_reduce_by_factor(Some(
                    &self.sp_fourier_reduce_by_factor.get_value().to_string(),
                ));
            } else {
                param.reset_fourier_reduce_by_factor();
            }
            if self.cb_extent_of_z_levels_in_nm.is_selected() {
                param.set_extent_of_z_levels_in_nm(
                    self.tf_extent_of_z_levels_in_nm.get_text_boolean(do_validation)?.as_deref(),
                );
            } else {
                param.reset_extent_of_z_levels_in_nm();
            }
            param.set_adjust_for_align_z_shift(self.cb_adjust_for_align_z_shift.is_selected());
            param.set_erase_fiducials(self.cb_erase_fiducials.is_selected());
            param.set_filter_in_2d(self.cb_filter_in_2d.is_selected());
            if self.subtomo_setup_cpu_gpu_panel.is_selected_cpus_only() {
                param.set_when_to_use_gpu(Some(&subtomo_setup_param::WHEN_TO_USE_GPU_VAL_0.to_string()));
            } else if self
                .subtomo_setup_cpu_gpu_panel
                .is_selected_gpu_for_recon_and_ctf_correct()
            {
                param.set_when_to_use_gpu(Some(&subtomo_setup_param::WHEN_TO_USE_GPU_VAL_1.to_string()));
            } else if self
                .subtomo_setup_cpu_gpu_panel
                .is_selected_cpu_for_recon_and_gpu_for_ctf_correct()
            {
                param.set_when_to_use_gpu(Some(&subtomo_setup_param::WHEN_TO_USE_GPU_VAL_2.to_string()));
            }
            param.set_rootname(self.manager.get_name().as_deref());
            // Java dereferences manager.getMainPanel() without a null check; a
            // missing main panel is treated like a missing parallel panel.
            let base_manager: &'static dyn BaseManager = self.manager;
            let parallel_panel = base_manager
                .get_main_panel()
                .and_then(|main_panel| main_panel.main_panel().get_parallel_panel(self.axis_id));
            if let Some(parallel_panel) = parallel_panel {
                param.set_processor_number(parallel_panel.get_cpus_selected(do_validation)?.as_deref());
            }
            Ok(())
        })();
        if let Err(e) = result {
            // e.printStackTrace()
            eprintln!("{e}");
            return false;
        }
        true
    }
}

impl Expandable for SubtomogramsPanel {
    /// Java override `expand(ExpandButton)`: empty.
    fn expand_expand_button(&self, _button: &Rc<ExpandButton>) {}

    /// Java override `expand(GlobalExpandButton)`.
    fn expand_global_expand_button(&self, button: &Rc<GlobalExpandButton>) {
        self.update_advanced(button.is_expanded());
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }
}

impl Run3dmodButtonContainer for SubtomogramsPanel {
    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        action_command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        self.action_option(
            Some(action_command),
            deferred_3dmod_button,
            run_3dmod_menu_options,
        );
    }
}

impl BrowsingDirectory for SubtomogramsPanel {
    /// Java override `getBrowsingDir()`.
    fn get_browsing_dir(&self) -> Option<PathBuf> {
        self.subtomo_browsing_dir.borrow().clone()
    }

    /// Java override `setBrowsingDir(File)`.
    fn set_browsing_dir(&self, file: Option<&Path>) {
        *self.subtomo_browsing_dir.borrow_mut() = file.map(Path::to_path_buf);
    }
}

impl ControlTarget for SubtomogramsPanel {
    /// Java override `clear()`: empty.
    fn clear(&self) {}

    /// Java override `setText(File)`: empty.
    fn set_text_file(&self, _file: Option<&Path>) {}

    /// Java override `setText(File[])`: empty.
    fn set_text_file_array(&self, _files: Option<&[PathBuf]>) {}

    /// Java override `getLabel()`: null.
    fn get_label(&self) -> Option<String> {
        None
    }

    /// Java override `setComponentControl(boolean, ControlState)`: empty.
    fn set_component_control(&self, _control: bool, _state: Option<&'static ControlState>) {}

    /// Java override `setEnableControl(boolean, ControlState)`: empty.
    fn set_enable_control(&self, _control: bool, _state: Option<&'static ControlState>) {}

    /// Java override `sendControlEvent()`: empty.
    fn send_control_event(&self) {}

    /// Java override `isLocalDir(String)`.
    fn is_local_dir(&self, current_directory: Option<&str>) -> bool {
        // Upstream bug fixed in translation (SubtomogramsPanel.java:778): the
        // Java compares the two directory Strings with `==` (reference
        // identity), so equal paths held in different String objects compare
        // unequal.  We compare the contents, which is the evident intent.
        current_directory == self.manager.get_property_user_dir().as_deref()
    }
}

impl ControlListener for SubtomogramsPanel {
    /// Java override `controlEvent()`.
    fn control_event(&self) {
        let manager: &'static dyn BaseManager = self.manager;
        ui_harness::with(|harness| {
            harness.pack_axis_id_base_manager(Some(self.axis_id), Some(manager))
        });
    }
}

impl ContextMenu for SubtomogramsPanel {
    /// Java override `popUpContextMenu(MouseEvent)`.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel: Vec<String> = vec!["SubtomoSetup".to_string(), "Tilt".to_string()];
        let man_page: Vec<String> = vec!["subtomosetup.html".to_string(), "tilt.html".to_string()];

        let log_file_label: Vec<String> = vec!["SubtomoSetup".to_string()];
        let manager: &'static dyn BaseManager = self.manager;
        let log_file: Vec<String> = vec![
            file_type::CLASS
                .subtomo_setup_log
                .get_file_name(Some(manager), Some(self.axis_id))
                .unwrap_or_else(|| "null".to_string()),
        ];

        if let Err(except) = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root,
            mouse_event,
            Some("Subtomograms"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(log_file_label.as_slice()),
            Some(log_file.as_slice()),
            manager,
            self.axis_id,
        ) {
            // An exception thrown by the ContextPopup constructor propagates to
            // the Swing event dispatch thread, which prints it.
            eprintln!("{except}");
        }
    }
}
