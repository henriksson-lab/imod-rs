//! `IMOD/Etomo/src/etomo/ui/swing/FlattenVolumePanel.java`.
//!
//! Java `final class FlattenVolumePanel implements Run3dmodButtonContainer,
//! WarpVolDisplay, FlattenWarpDisplay, SmoothingAssessmentParent,
//! ContextMenu, ToolPanel`: the Flatten tab of the Post Processing dialog
//! (make a surface model, run flattenwarp, run warpvol), also used as the
//! Flatten Volume tool of the Tools dialog.
//!
//! An EDT object (`ui.md`): created as `Rc<Self>` by
//! [`FlattenVolumePanel::get_post_instance`] /
//! [`FlattenVolumePanel::get_tools_instance`]; every method takes `&self`.
//! The listener class `FlattenVolumeActionListener` is a closure holding a
//! weak reference to the panel.
//!
//! The panel implements two Java interfaces that each declare
//! `getParameters(param, boolean)` (`WarpVolDisplay`, `FlattenWarpDisplay`):
//! those are the two trait impls; the class's own `getParameters(MetaData)`
//! overload is [`FlattenVolumePanel::get_parameters_meta_data`].

use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::beveled_border::BeveledBorder;
use super::binned_xy_3dmod_button::BinnedXY3dmodButton;
use super::check_box::CheckBox;
use super::context_menu::ContextMenu;
use super::context_popup::{self, ContextPopup};
use super::deferred_3dmod_button::Deferred3dmodButton;
use super::file_chooser::{self, FileChooser};
use super::file_text_field::FileTextField;
use super::file_text_field_interface::FileTextFieldInterface;
use super::flatten_warp_display::FlattenWarpDisplay;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::radio_button::RadioButton;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::smoothing_assessment_panel::SmoothingAssessmentPanel;
use super::smoothing_assessment_parent::SmoothingAssessmentParent;
use super::spaced_panel::{self, SpacedPanel};
use super::tool_panel::ToolPanel;
use super::ui_harness;
use super::warp_vol_display::WarpVolDisplay;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::comscript::const_warp_vol_param::ConstWarpVolParam;
use crate::imod::etomo::comscript::flatten_warp_param::{self, FlattenWarpParam};
use crate::imod::etomo::comscript::warp_vol_param::{self, WarpVolParam};
use crate::imod::etomo::jdk::{
    ActionEvent, ActionListener, ButtonGroup, FileFilter, JComponent, MouseEvent,
};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc::Autodoc;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
// TODO(unit): needs etomo/storage/ReduceFiltVolFileFilter.java - the reducefiltvol
// output file name filter `getInputFile` lists the dataset directory with.
use crate::imod::etomo::storage::reduce_filt_vol_file_filter::ReduceFiltVolFileFilter;
// TODO(unit): needs etomo/storage/TomogramFileFilter.java - the file chooser filter of
// the tools input file.
use crate::imod::etomo::storage::tomogram_file_filter::TomogramFileFilter;
use crate::imod::etomo::tools_manager::ToolsManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::data_file_type::DataFileType;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::file_type::{self, FileType};
use crate::imod::etomo::r#type::image_file_type::ImageFileType;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::panel_id::PanelId;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException;
// TODO(unit): needs etomo/util/FrontEndLogic.java - `isRotated(BaseManager, AxisID,
// File)`, which reads the file's MRC header.
use crate::imod::etomo::util::front_end_logic;
use crate::imod::etomo::util::utilities;

/// Java private static final `OUTPUT_SIZE_Z_LABEL`.
const OUTPUT_SIZE_Z_LABEL: &str = "Output thickness in Z";
/// Java static final `FLATTEN_LABEL`.
pub const FLATTEN_LABEL: &str = "Flatten";
/// Java static final `WARP_SPACING_X_LABEL`.
pub const WARP_SPACING_X_LABEL: &str = "Spacing in X";
/// Java static final `WARP_SPACING_Y_LABEL`.
pub const WARP_SPACING_Y_LABEL: &str = "and Y";
/// Java private static final `LAMBDA_FOR_SMOOTHING_LABEL`.
const LAMBDA_FOR_SMOOTHING_LABEL: &str = "Smoothing factor";
/// Java static final `FLATTEN_WARP_LABEL`.
pub const FLATTEN_WARP_LABEL: &str = "Run Flattenwarp";

/// Java `final class FlattenVolumePanel`.
pub struct FlattenVolumePanel {
    /// Rust-only: Java `this` (for `getFlattenWarpDisplay`).
    self_ref: Weak<FlattenVolumePanel>,

    /// Java private final `pnlRoot = SpacedPanel.getInstance()`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `btnMakeSurfaceModel = new
    /// BinnedXY3dmodButton("Make Surface Model", this)`.
    btn_make_surface_model: Rc<BinnedXY3dmodButton>,
    /// Java private final `cbOneSurface`.
    cb_one_surface: Rc<CheckBox>,
    /// Java private final `ltfWarpSpacingX`.
    ltf_warp_spacing_x: Rc<LabeledTextField>,
    /// Java private final `ltfWarpSpacingY`.
    ltf_warp_spacing_y: Rc<LabeledTextField>,
    /// Java private final `ltfLambdaForSmoothing`.
    ltf_lambda_for_smoothing: Rc<LabeledTextField>,
    /// Java package-private `actionListener = new
    /// FlattenVolumeActionListener(this)`.
    action_listener: ActionListener,
    /// Java private final `bgInputFile = new ButtonGroup()`.
    bg_input_file: Rc<ButtonGroup>,
    /// Java private final `rbInputFileTrimVol`.
    rb_input_file_trim_vol: Rc<RadioButton>,
    /// Java private final `rbInputFileSqueezeVol`.
    rb_input_file_squeeze_vol: Rc<RadioButton>,
    /// Java private final `cbInterpolationOrderLinear`.
    cb_interpolation_order_linear: Rc<CheckBox>,
    /// Java private final `ltfOutputSizeZ`.
    ltf_output_size_z: Rc<LabeledTextField>,
    /// Java private final `btnImodFlatten`.
    btn_imod_flatten: Rc<Run3dmodButton>,
    /// Java private final `ftfTemporaryDirectory`.
    ftf_temporary_directory: Rc<FileTextField>,
    /// Java private final `ftfInputFile`.
    ftf_input_file: Rc<FileTextField>,

    /// Java private final `panelId`.
    panel_id: PanelId,
    /// Java private final `btnFlatten`.
    btn_flatten: Rc<Run3dmodButton>,
    /// Java private final `axisID`.
    axis_id: AxisID,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `applicationManager`; null for the tools instance.
    application_manager: Option<&'static ApplicationManager>,
    /// Java private final `toolsManager`; null for the post-processing
    /// instance.
    tools_manager: Option<&'static ToolsManager>,
    /// Java private final `dialogType`.
    dialog_type: DialogType,
    /// Java private final `smoothingAssessmentPanel`.
    smoothing_assessment_panel: Rc<SmoothingAssessmentPanel>,
    /// Java private final `btnFlattenWarp`.
    btn_flatten_warp: Rc<MultiLineButton>,
}

/// What differs between the two Java constructors (Rust-only).
enum ConstructorKind {
    /// `FlattenVolumePanel(ApplicationManager, AxisID, DialogType)`.
    Post(&'static ApplicationManager),
    /// `FlattenVolumePanel(ToolsManager, AxisID, DialogType)`.
    Tools(&'static ToolsManager),
}

impl FlattenVolumePanel {
    /// The two Java constructors: field initializers, then the constructor
    /// body for `kind`.
    fn new(
        kind: ConstructorKind,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<FlattenVolumePanel> {
        Rc::new_cyclic(|self_ref: &Weak<FlattenVolumePanel>| {
            let container: Weak<dyn Run3dmodButtonContainer> = self_ref.clone();
            // Field initializers.
            let pnl_root = SpacedPanel::get_instance_void();
            let btn_make_surface_model =
                BinnedXY3dmodButton::new(Some("Make Surface Model"), Some(container.clone()));
            let cb_one_surface = CheckBox::new_string(Some("Contours are all on one surface"));
            let ltf_warp_spacing_x = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(&format!("{WARP_SPACING_X_LABEL}: ")),
            );
            let ltf_warp_spacing_y = LabeledTextField::new_field_type_string(
                FieldType::FloatingPoint,
                Some(&format!(" {WARP_SPACING_Y_LABEL}: ")),
            );
            let ltf_lambda_for_smoothing = LabeledTextField::new_field_type_string(
                FieldType::FloatingPointArray,
                Some(&format!("{LAMBDA_FOR_SMOOTHING_LABEL}: ")),
            );
            // Java `new FlattenVolumeActionListener(this)`.
            let adaptee = self_ref.clone();
            let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
                if let Some(adaptee) = adaptee.upgrade() {
                    adaptee.action(event.get_action_command().unwrap_or(""), None, None);
                }
            });
            let bg_input_file = ButtonGroup::new();
            let rb_input_file_trim_vol = RadioButton::new_string_button_group(
                Some("Flatten the trimvol output"),
                Some(&bg_input_file),
            );
            let rb_input_file_squeeze_vol = RadioButton::new_string_button_group(
                Some("Flatten the reducefiltvol output"),
                Some(&bg_input_file),
            );
            let cb_interpolation_order_linear = CheckBox::new_string(Some("Linear interpolation"));
            let ltf_output_size_z = LabeledTextField::new_field_type_string(
                FieldType::Integer,
                Some(&format!("{OUTPUT_SIZE_Z_LABEL}: ")),
            );
            let btn_imod_flatten =
                Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
                    Some("Open Flattened Tomogram"),
                    Some(container.clone()),
                );
            let ftf_temporary_directory = FileTextField::new("Temporary directory:");
            let ftf_input_file = FileTextField::new("Input file:");
            // Constructor body.
            let parent: Weak<dyn SmoothingAssessmentParent> = self_ref.clone();
            let (
                manager,
                application_manager,
                tools_manager,
                panel_id,
                btn_flatten,
                btn_flatten_warp,
                smoothing_assessment_panel,
            ): (
                &'static dyn BaseManager,
                Option<&'static ApplicationManager>,
                Option<&'static ToolsManager>,
                PanelId,
                Rc<Run3dmodButton>,
                Rc<MultiLineButton>,
                Rc<SmoothingAssessmentPanel>,
            ) = match kind {
                ConstructorKind::Post(manager) => {
                    // this.manager = manager; applicationManager = manager;
                    // toolsManager = null; ...
                    let panel_id = PanelId::PostFlattenVolume;
                    let factory = manager.get_process_result_display_factory(axis_id);
                    let btn_flatten = factory.get_flatten();
                    let btn_flatten_warp = factory.get_flatten_warp();
                    let smoothing_assessment_panel = SmoothingAssessmentPanel::get_post_instance(
                        manager,
                        axis_id,
                        dialog_type,
                        panel_id,
                        parent,
                    );
                    (
                        manager as &'static dyn BaseManager,
                        Some(manager),
                        None,
                        panel_id,
                        btn_flatten,
                        btn_flatten_warp,
                        smoothing_assessment_panel,
                    )
                }
                ConstructorKind::Tools(manager) => {
                    // this.manager = manager; applicationManager = null;
                    // toolsManager = manager; ...
                    let panel_id = PanelId::ToolsFlattenVolume;
                    let btn_flatten =
                        Run3dmodButton::get_deferred_3dmod_instance_string_run_3dmod_button_container(
                            Some(FLATTEN_LABEL),
                            Some(container.clone()),
                        );
                    let btn_flatten_warp = MultiLineButton::new_string(Some(FLATTEN_WARP_LABEL));
                    let smoothing_assessment_panel = SmoothingAssessmentPanel::get_tools_instance(
                        manager,
                        axis_id,
                        dialog_type,
                        panel_id,
                        parent,
                    );
                    (
                        manager as &'static dyn BaseManager,
                        None,
                        Some(manager),
                        panel_id,
                        btn_flatten,
                        btn_flatten_warp,
                        smoothing_assessment_panel,
                    )
                }
            };
            FlattenVolumePanel {
                self_ref: self_ref.clone(),
                pnl_root,
                btn_make_surface_model,
                cb_one_surface,
                ltf_warp_spacing_x,
                ltf_warp_spacing_y,
                ltf_lambda_for_smoothing,
                action_listener,
                bg_input_file,
                rb_input_file_trim_vol,
                rb_input_file_squeeze_vol,
                cb_interpolation_order_linear,
                ltf_output_size_z,
                btn_imod_flatten,
                ftf_temporary_directory,
                ftf_input_file,
                panel_id,
                btn_flatten,
                axis_id,
                manager,
                application_manager,
                tools_manager,
                dialog_type,
                smoothing_assessment_panel,
                btn_flatten_warp,
            }
        })
    }

    /// Java package-private static `getPostInstance(ApplicationManager, AxisID,
    /// DialogType)`.
    pub fn get_post_instance(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<FlattenVolumePanel> {
        let instance =
            FlattenVolumePanel::new(ConstructorKind::Post(manager), axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java package-private static `getToolsInstance(ToolsManager, AxisID,
    /// DialogType)`.
    pub fn get_tools_instance(
        manager: &'static ToolsManager,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) -> Rc<FlattenVolumePanel> {
        let instance =
            FlattenVolumePanel::new(ConstructorKind::Tools(manager), axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self) {
        // Swing mouse: pnlRoot.addMouseListener(new GenericMouseAdapter(this)).
        // Mouse events are not modelled; the adapter's only effect is to call
        // popUpContextMenu on a right-button press, which a driver calls
        // directly.
        self.btn_make_surface_model
            .add_action_listener(self.action_listener.clone());
        self.btn_flatten_warp
            .add_action_listener(self.action_listener.clone());
        self.btn_flatten
            .add_action_listener(self.action_listener.clone());
        self.btn_imod_flatten
            .add_action_listener(self.action_listener.clone());
        self.ftf_input_file
            .add_action_listener(self.action_listener.clone());
    }

    /// Java package-private `done()`.
    pub fn done(&self) {
        self.btn_flatten_warp
            .remove_action_listener(&self.action_listener);
        self.btn_flatten
            .remove_action_listener(&self.action_listener);
        self.smoothing_assessment_panel.done();
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        let pnl_input_file = JComponent::new_panel();
        let pnl_interpolation_order = JComponent::new_panel();
        let pnl_flatten = SpacedPanel::get_instance_void();
        let pnl_flatten_warp = SpacedPanel::get_instance_void();
        let pnl_one_surface = JComponent::new_panel();
        let pnl_warp_spacing = JComponent::new_panel();
        let pnl_flatten_warp_buttons = JComponent::new_panel();
        // initialize
        self.rb_input_file_trim_vol.set_selected_boolean(true);
        let container: Weak<dyn Run3dmodButtonContainer> = self.self_ref.clone();
        self.btn_flatten.set_container(Some(container));
        self.btn_flatten
            .set_deferred_3dmod_button_deferred_3dmod_button(Some(
                self.btn_imod_flatten.clone() as Rc<dyn Deferred3dmodButton>
            ));
        self.ftf_input_file.set_field_editable(false);
        self.ftf_temporary_directory.add_action(
            self.manager.get_property_user_dir().as_deref(),
            Some(ToolPanel::get_component(self)),
            file_chooser::DIRECTORIES_ONLY,
        );
        // Root panel
        self.pnl_root.set_box_layout(spaced_panel::Y_AXIS);
        self.pnl_root
            .set_border(&BeveledBorder::new(Some("Flatten Volume")).get_border());
        // Swing layout: pnlRoot.setAlignmentX(Box.CENTER_ALIGNMENT).
        if self.panel_id == PanelId::PostFlattenVolume {
            self.pnl_root.add_j_panel(&pnl_input_file);
        } else if self.panel_id == PanelId::ToolsFlattenVolume {
            self.pnl_root.add_file_text_field(&self.ftf_input_file);
        }
        self.pnl_root.add_spaced_panel(&pnl_flatten_warp);
        self.pnl_root.add_j_panel(&pnl_interpolation_order);
        self.pnl_root
            .add_container(&self.ltf_output_size_z.get_container());
        self.pnl_root
            .add_file_text_field(&self.ftf_temporary_directory);
        self.pnl_root.add_spaced_panel(&pnl_flatten);
        // Input file panel
        if self.panel_id == PanelId::PostFlattenVolume {
            // Swing layout: pnlInputFile Y_AXIS BoxLayout, CENTER_ALIGNMENT.
            pnl_input_file.set_border_title(
                BeveledBorder::new(Some("Set Input File"))
                    .get_border()
                    .get_title()
                    .as_deref(),
            );
            pnl_input_file.add(&self.rb_input_file_trim_vol.get_component());
            pnl_input_file.add(&self.rb_input_file_squeeze_vol.get_component());
        }
        // Flatten warp panel
        pnl_flatten_warp.set_box_layout(spaced_panel::Y_AXIS);
        // Swing layout: pnlFlattenWarp.setAlignmentX(Box.CENTER_ALIGNMENT).
        pnl_flatten_warp.add_container(&self.btn_make_surface_model.get_container());
        pnl_flatten_warp.add_j_panel(&pnl_one_surface);
        pnl_flatten_warp.add_j_panel(&pnl_warp_spacing);
        pnl_flatten_warp.add_component(&self.smoothing_assessment_panel.get_component());
        pnl_flatten_warp.add_container(&self.ltf_lambda_for_smoothing.get_container());
        pnl_flatten_warp.add_j_panel(&pnl_flatten_warp_buttons);
        // One surface panel
        // Swing layout: pnlOneSurface Y_AXIS BoxLayout, CENTER_ALIGNMENT.
        pnl_one_surface.add(&self.cb_one_surface.get_component());
        // Swing layout: pnlOneSurface.add(Box.createHorizontalGlue()).
        // Warp Spacing panel
        // Swing layout: pnlWarpSpacing X_AXIS BoxLayout, CENTER_ALIGNMENT.
        pnl_warp_spacing.add(&self.ltf_warp_spacing_x.get_container());
        pnl_warp_spacing.add(&self.ltf_warp_spacing_y.get_container());
        // Flatten warp buttons panel
        // Swing layout: pnlFlattenWarpButtons Y_AXIS BoxLayout, CENTER_ALIGNMENT.
        pnl_flatten_warp_buttons.add(&self.btn_flatten_warp.get_component());
        // Interpolation order panel
        // Swing layout: pnlInterpolationOrder X_AXIS BoxLayout, CENTER_ALIGNMENT.
        pnl_interpolation_order.add(&self.cb_interpolation_order_linear.get_component());
        // Swing layout: pnlInterpolationOrder.add(Box.createHorizontalGlue()).
        // Flatten panel
        pnl_flatten.set_box_layout(spaced_panel::X_AXIS);
        // Swing layout: pnlFlatten.setAlignmentX(Box.CENTER_ALIGNMENT).
        pnl_flatten.add_component(&self.btn_flatten.get_component());
        pnl_flatten.add_component(&self.btn_imod_flatten.get_component());
    }

    /// Java package-private `getFlattenWarpDisplay()`: `this`.
    pub fn get_flatten_warp_display(&self) -> Rc<dyn FlattenWarpDisplay> {
        self.self_ref
            .upgrade()
            .expect("FlattenVolumePanel used after it was dropped")
    }

    /// Java package-private `setParameters(ConstMetaData)`.  Sets values from
    /// the reconstruction metadata.  Not used by the tools manager.
    pub fn set_parameters_const_meta_data(&self, meta_data: &dyn ConstMetaData) {
        self.rb_input_file_trim_vol
            .set_selected_boolean(meta_data.is_post_flatten_warp_input_trim_vol());
        if !self.rb_input_file_trim_vol.is_selected() {
            self.rb_input_file_squeeze_vol.set_selected_boolean(true);
        }
        self.cb_one_surface
            .set_selected_boolean(meta_data.is_post_flatten_warp_contours_on_one_surface());
        self.ltf_warp_spacing_x
            .set_text_string(Some(&meta_data.get_post_flatten_warp_spacing_in_x()));
        self.ltf_warp_spacing_y
            .set_text_string(Some(&meta_data.get_post_flatten_warp_spacing_in_y()));
        self.ltf_lambda_for_smoothing
            .set_text_string(Some(&meta_data.get_lambda_for_smoothing()));
        self.smoothing_assessment_panel.set_parameters(meta_data);
    }

    /// Java package-private `getParameters(MetaData)`.  Puts values into the
    /// reconstruction metadata.  Not used by the tools manager.
    pub fn get_parameters_meta_data(&self, meta_data: &MetaData) {
        meta_data.set_post_flatten_warp_input_trim_vol(self.rb_input_file_trim_vol.is_selected());
        meta_data.set_post_flatten_warp_contours_on_one_surface(self.cb_one_surface.is_selected());
        meta_data
            .set_post_flatten_warp_spacing_in_x(self.ltf_warp_spacing_x.get_text_void().as_deref());
        meta_data
            .set_post_flatten_warp_spacing_in_y(self.ltf_warp_spacing_y.get_text_void().as_deref());
        meta_data
            .set_lambda_for_smoothing(self.ltf_lambda_for_smoothing.get_text_void().as_deref());
        self.smoothing_assessment_panel
            .get_parameters_meta_data(meta_data);
    }

    /// Java private `validateFlattenWarp()`.
    fn validate_flatten_warp(&self) -> bool {
        let lambda_for_smoothing = self.ltf_lambda_for_smoothing.get_text_void();
        if lambda_for_smoothing.is_none()
            || lambda_for_smoothing.as_deref().is_some_and(
                crate::imod::etomo::r#type::const_etomo_number::java_lang_string_matches_whitespace,
            )
        {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string_axis_id(
                    Some(self.manager),
                    &format!("{LAMBDA_FOR_SMOOTHING_LABEL} is a required field."),
                    "Entry Error",
                    Some(self.axis_id),
                )
            });
            return false;
        }
        true
    }

    /// Java package-private `setParameters(ConstWarpVolParam)`.  Sets values
    /// from the param if this panel in post processing.
    pub fn set_parameters_const_warp_vol_param(&self, param: &dyn ConstWarpVolParam) {
        self.cb_interpolation_order_linear
            .set_selected_boolean(param.is_interpolation_order_linear());
        self.ltf_output_size_z
            .set_text_string(Some(&param.get_output_size_z()));
        self.ftf_temporary_directory
            .set_text_string(Some(&param.get_temporary_directory()));
    }

    /// Java package-private `getInputFileType()`; `None` is Java null (the
    /// tools panel).
    pub fn get_input_file_type(&self) -> Option<Arc<FileType>> {
        if self.panel_id == PanelId::PostFlattenVolume {
            if self.rb_input_file_trim_vol.is_selected() {
                return Some(file_type::CLASS.trim_vol_output.clone());
            }
            return Some(file_type::CLASS.flatten_reduce_filt_vol_file.clone());
        }
        None
    }

    /// Java package-private `getInputFile()`: the most recently modified
    /// reducefiltvol output file in the dataset directory; `None` is Java
    /// null.
    pub fn get_input_file(&self) -> Option<PathBuf> {
        let reduce_filt_vol_dir = PathBuf::from(
            self.manager
                .get_property_user_dir()
                .unwrap_or_else(|| "null".to_string()),
        );
        // `reduceFiltVolDir.listFiles((FilenameFilter)
        // ReduceFiltVolFileFilter.getInstance(manager))`.
        let filter = ReduceFiltVolFileFilter::get_instance(self.manager);
        // Upstream bug fixed in translation (FlattenVolumePanel.java:403): Java's
        // `listFiles` returns null for an unreadable directory and the loop then
        // throws NullPointerException; the translation reads that as no files.
        let reduce_filt_vol_files: Vec<PathBuf> = match std::fs::read_dir(&reduce_filt_vol_dir) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    filter.accept_in_dir(
                        Some(reduce_filt_vol_dir.as_path()),
                        Some(&entry.file_name().to_string_lossy()),
                    )
                })
                .map(|entry| reduce_filt_vol_dir.join(entry.file_name()))
                .collect(),
            Err(_) => Vec::new(),
        };
        let mut last_modified: i64 = 0;
        let mut last_modified_index: i32 = -1;
        for (i, file) in reduce_filt_vol_files.iter().enumerate() {
            let curr_last_modified = utilities::java_io_file_last_modified(&file.to_string_lossy());
            if curr_last_modified >= last_modified {
                last_modified = curr_last_modified;
                last_modified_index = i as i32;
            }
        }
        if last_modified_index == -1 {
            return None;
        }
        let last_modified_file = reduce_filt_vol_files[last_modified_index as usize].clone();
        Some(last_modified_file)
    }

    /// Java private `inputFileAction()`.  Set the input file.  In tools version
    /// this checks for conflicting dataset names.  Also pops up a warning if
    /// the file was not rotated.
    fn input_file_action(&self) {
        // Open up the file chooser in the current working directory
        let chooser = FileChooser::new_base_manager(Some(self.manager));
        // Swing layout: chooser.setPreferredSize(UIParameters.getInstance()
        // .getFileChooserDimension()).
        chooser.set_file_selection_mode(file_chooser::FILES_ONLY);
        chooser.set_file_filter(Some(
            TomogramFileFilter::get_instance(self.manager) as Rc<dyn FileFilter>
        ));
        let return_val = chooser.show_open_dialog(Some(&self.pnl_root.get_container()));
        if return_val == file_chooser::APPROVE_OPTION {
            let file = chooser.get_selected_file();
            let Some(file) = file else {
                return;
            };
            if file.is_dir() || !file.exists() {
                return;
            }
            if let Some(tools_manager) = self.tools_manager {
                if !dataset_tool::validate_dataset_name_input_file(
                    tools_manager,
                    self.axis_id,
                    Some(&file),
                    DataFileType::Tools,
                    None,
                ) {
                    return;
                }
                if tools_manager.is_conflicting_dataset_name(self.axis_id, &file) {
                    return;
                }
            }
            // try { ... } catch (Exception excep) { excep.printStackTrace(); }:
            // nothing in the block throws in the translation.
            self.ftf_input_file
                .set_text_string(Some(&utilities::java_io_file_get_absolute_path(
                    &file.to_string_lossy(),
                )));
            self.ftf_input_file.set_button_enabled(false);
            if let Some(tools_manager) = self.tools_manager {
                tools_manager.set_name(&file);
            }
            let manager = self.manager;
            ui_harness::with(|harness| harness.pack_base_manager(Some(manager)));
            self.check_rotated(&file);
        }
    }

    /// Java private `checkRotated(File)`.
    fn check_rotated(&self, file: &Path) {
        let rotated = front_end_logic::is_rotated(self.manager, self.axis_id, file);
        match rotated {
            None => {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "The MRC header of this file, {}, is unreadable.",
                            utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                        ),
                        "Warning",
                        Some(self.axis_id),
                    )
                });
            }
            Some(rotated) if !rotated.is() => {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!(
                            "This tomogram, {}, looks like the volume hasn't been reoriented.   \
                             Flattening won't work on a volume that hasn't been reoriented.",
                            utilities::java_io_file_get_absolute_path(&file.to_string_lossy())
                        ),
                        "Warning",
                        Some(self.axis_id),
                    )
                });
            }
            Some(_) => {}
        }
    }

    /// Java private `setToolTipText()`.
    fn set_tool_tip_text(&self) {
        // Java `ReadOnlyAutodoc flattenWarpAutodoc = null; warpVolAutodoc = null;`
        // and one try block around both getInstance calls: a failure of the
        // first skips the second.
        let mut flatten_warp_autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let mut warp_vol_autodoc: *const dyn ReadOnlyAutodoc = std::ptr::null::<Autodoc>();
        let result = (|| -> Result<(), LogFileError> {
            flatten_warp_autodoc = unsafe {
                autodoc_factory::get_instance(
                    Some(self.manager),
                    Some(autodoc_factory::FLATTEN_WARP),
                    self.axis_id,
                    false,
                )
            }? as *const Autodoc;
            warp_vol_autodoc = unsafe {
                autodoc_factory::get_instance(
                    Some(self.manager),
                    Some(autodoc_factory::WARP_VOL),
                    self.axis_id,
                    false,
                )
            }? as *const Autodoc;
            Ok(())
        })();
        match result {
            Ok(()) => {}
            // `catch (final LockException except) {}`.
            Err(LogFileError::Lock(_)) => {}
            // `catch (final LogFileException | IOException except)`:
            // `except.printStackTrace()`.
            Err(except) => eprintln!("{}", except),
        }
        // SAFETY: each pointer is null or an autodoc the factory keeps for the
        // life of the process.
        let flatten_warp_autodoc: Option<&dyn ReadOnlyAutodoc> = if flatten_warp_autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*flatten_warp_autodoc })
        };
        let warp_vol_autodoc: Option<&dyn ReadOnlyAutodoc> = if warp_vol_autodoc.is_null() {
            None
        } else {
            Some(unsafe { &*warp_vol_autodoc })
        };
        if flatten_warp_autodoc.is_some() {
            self.ltf_lambda_for_smoothing.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    flatten_warp_autodoc,
                    Some(flatten_warp_param::LAMBDA_FOR_SMOOTHING_OPTION),
                )
                .as_deref(),
            );
            self.cb_one_surface.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(
                    flatten_warp_autodoc,
                    Some(flatten_warp_param::ONE_SURFACE_OPTION),
                )
                .as_deref(),
            );
            self.ltf_warp_spacing_x.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    flatten_warp_autodoc,
                    Some(flatten_warp_param::WARP_SPACING_X_AND_Y_OPTION),
                )
                .as_deref(),
            );
            self.ltf_warp_spacing_y.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    flatten_warp_autodoc,
                    Some(flatten_warp_param::WARP_SPACING_X_AND_Y_OPTION),
                )
                .as_deref(),
            );
        }
        if warp_vol_autodoc.is_some() {
            self.rb_input_file_trim_vol.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(
                    warp_vol_autodoc,
                    Some(warp_vol_param::INPUT_FILE_OPTION),
                )
                .as_deref(),
            );
            self.rb_input_file_squeeze_vol.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(
                    warp_vol_autodoc,
                    Some(warp_vol_param::INPUT_FILE_OPTION),
                )
                .as_deref(),
            );
            self.cb_interpolation_order_linear.set_tool_tip_text_string(
                etomo_autodoc::get_tooltip(
                    warp_vol_autodoc,
                    Some(warp_vol_param::INTERPOLATION_ORDER_OPTION),
                )
                .as_deref(),
            );
            self.ltf_output_size_z.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    warp_vol_autodoc,
                    Some(warp_vol_param::OUTPUT_SIZE_X_Y_Z_OPTION),
                )
                .as_deref(),
            );
            self.ftf_temporary_directory.set_tool_tip_text(
                etomo_autodoc::get_tooltip(
                    warp_vol_autodoc,
                    Some(warp_vol_param::TEMPORARY_DIRECTORY_OPTION),
                )
                .as_deref(),
            );
        }
        self.btn_make_surface_model.set_button_tool_tip_text(Some(
            "Add contours to describe the location of the sectioned material.",
        ));
        self.btn_flatten_warp
            .set_tool_tip_text(Some("Run flattenwarp."));
        self.btn_flatten.set_tool_tip_text(Some("Run warpvol."));
        self.btn_imod_flatten
            .set_tool_tip_text(Some("Open warpvol output in 3dmod."));
    }
}

impl Run3dmodButtonContainer for FlattenVolumePanel {
    /// Java public `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    fn action(
        &self,
        command: &str,
        deferred_3dmod_button: Option<Rc<dyn Deferred3dmodButton>>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        // Java passes a null Run3dmodMenuOptions from the action listener; the
        // ApplicationManager methods take the options by value, and
        // ImodState.open replaces null with a new Run3dmodMenuOptions() (the
        // default value).
        // Reconstruction
        if self.panel_id == PanelId::PostFlattenVolume {
            let Some(application_manager) = self.application_manager else {
                return;
            };
            if Some(command) == self.btn_flatten.get_action_command().as_deref() {
                application_manager.flatten(
                    Some(self.btn_flatten.clone() as ProcessResultDisplayHandle),
                    None,
                    deferred_3dmod_button,
                    run_3dmod_menu_options.unwrap_or_default(),
                    self.dialog_type,
                    self.axis_id,
                    self,
                );
            } else if Some(command) == self.btn_imod_flatten.get_action_command().as_deref() {
                application_manager
                    .imod_flatten(run_3dmod_menu_options.unwrap_or_default(), self.axis_id);
            } else if Some(command) == self.btn_make_surface_model.get_action_command().as_deref() {
                let input_file: Option<PathBuf> =
                    if self.get_input_file_type().is_some_and(|file_type| {
                        Arc::ptr_eq(&file_type, &file_type::CLASS.trim_vol_output)
                    }) {
                        file_type::CLASS
                            .trim_vol_output
                            .get_file(Some(self.manager), Some(self.axis_id))
                    } else {
                        self.get_input_file()
                    };
                let Some(input_file) = input_file else {
                    let message = "File you intend to open does not exist";
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.manager),
                            message,
                            "File Not Found Error",
                            Some(self.axis_id),
                        )
                    });
                    return;
                };
                self.check_rotated(&input_file);
                // getInputFileType() is never null for the post-processing panel.
                if let Some(input_file_type) = self.get_input_file_type() {
                    application_manager.imod_make_surface_model(
                        run_3dmod_menu_options.unwrap_or_default(),
                        self.axis_id,
                        self.btn_make_surface_model.get_binning_in_xand_y(),
                        &input_file_type,
                        Some(input_file.as_path()),
                    );
                }
            } else if Some(command) == self.btn_flatten_warp.get_action_command().as_deref() {
                if self.validate_flatten_warp() {
                    application_manager.flatten_warp(
                        Some(self.btn_flatten_warp.clone() as ProcessResultDisplayHandle),
                        None,
                        deferred_3dmod_button,
                        run_3dmod_menu_options.unwrap_or_default(),
                        self.dialog_type,
                        self.axis_id,
                        self,
                    );
                }
            } else {
                // Java throws IllegalStateException, which Swing reports on the EDT and
                // survives; the translation reports it and returns.
                eprintln!("java.lang.IllegalStateException: Unknown command {command}");
            }
        }
        // Tools
        else if self.panel_id == PanelId::ToolsFlattenVolume {
            let Some(tools_manager) = self.tools_manager else {
                return;
            };
            if Some(command) == self.ftf_input_file.get_action_command().as_deref() {
                self.input_file_action();
            } else {
                // Must keep checking the dataset directory because a tools
                // interface cannot take pocession of a directory.
                let file = FileTextFieldInterface::get_file(&*self.ftf_input_file);
                if !dataset_tool::validate_dataset_name_input_file(
                    tools_manager,
                    self.axis_id,
                    file.as_deref(),
                    DataFileType::Tools,
                    None,
                ) {
                    return;
                }
                if Some(command) == self.btn_flatten.get_action_command().as_deref() {
                    tools_manager.flatten(
                        Some(self.btn_flatten.clone() as ProcessResultDisplayHandle),
                        None,
                        deferred_3dmod_button,
                        run_3dmod_menu_options,
                        Some(self.dialog_type),
                        self.axis_id,
                        self,
                    );
                } else if Some(command) == self.btn_imod_flatten.get_action_command().as_deref() {
                    tools_manager.imod_flatten(run_3dmod_menu_options, self.axis_id);
                } else if Some(command)
                    == self.btn_make_surface_model.get_action_command().as_deref()
                {
                    tools_manager.imod_make_surface_model(
                        run_3dmod_menu_options,
                        self.axis_id,
                        self.btn_make_surface_model.get_binning_in_xand_y(),
                        FileTextFieldInterface::get_file(&*self.ftf_input_file).as_deref(),
                    );
                } else if Some(command) == self.btn_flatten_warp.get_action_command().as_deref() {
                    if self.validate_flatten_warp() {
                        tools_manager.flatten_warp(
                            Some(self.btn_flatten_warp.clone() as ProcessResultDisplayHandle),
                            None,
                            deferred_3dmod_button,
                            run_3dmod_menu_options,
                            Some(self.dialog_type),
                            self.axis_id,
                            self,
                        );
                    }
                } else {
                    // Java throws IllegalStateException, which Swing reports on the EDT and
                    // survives; the translation reports it and returns.
                    eprintln!("java.lang.IllegalStateException: Unknown command {command}");
                }
            }
        } else {
            // Java throws IllegalStateException, which Swing reports on the EDT and
            // survives; the translation reports it and returns.
            eprintln!(
                "java.lang.IllegalStateException: Unknown panel ID {}",
                self.panel_id
            );
        }
    }
}

impl WarpVolDisplay for FlattenVolumePanel {
    /// Java public `getParameters(WarpVolParam, boolean)`.
    fn get_parameters(&self, param: &mut WarpVolParam, do_validation: bool) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            param.set_interpolation_order_linear(self.cb_interpolation_order_linear.is_selected());
            let error_message = param.set_output_size_z(
                self.ltf_output_size_z
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Error in {OUTPUT_SIZE_Z_LABEL}:  {message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            // The model contains coordinates so it can match either input file.
            if self.panel_id == PanelId::PostFlattenVolume {
                let input_file: Option<PathBuf> =
                    if self.get_input_file_type().is_some_and(|file_type| {
                        Arc::ptr_eq(&file_type, &file_type::CLASS.trim_vol_output)
                    }) {
                        file_type::CLASS
                            .trim_vol_output
                            .get_file(Some(self.manager), Some(self.axis_id))
                    } else {
                        self.get_input_file()
                    };
                let Some(input_file) = input_file else {
                    let message = "File you intend to open does not exist";
                    ui_harness::with(|harness| {
                        harness.open_message_dialog_base_manager_string_string_axis_id(
                            Some(self.manager),
                            message,
                            "File Not Found Error",
                            Some(self.axis_id),
                        )
                    });
                    return Ok(false);
                };
                param.set_input_file_file(&input_file);
                param.set_output_file(
                    ImageFileType::FlattenOutput
                        .get_file_name(self.manager)
                        .as_deref(),
                );
            } else if self.panel_id == PanelId::ToolsFlattenVolume {
                // Upstream bug fixed in translation (FlattenVolumePanel.java:365):
                // with no input file chosen, Java's `setInputFile(null)` throws
                // NullPointerException (`file.getAbsolutePath()`).  The translation
                // leaves the input file unset.
                if let Some(file) = FileTextFieldInterface::get_file(&*self.ftf_input_file) {
                    param.set_input_file_file(&file);
                }
                param.set_output_file(
                    file_type::CLASS
                        .flatten_tool_output
                        .get_file_name(Some(self.manager), Some(AxisID::Only))
                        .as_deref(),
                );
            }
            param.set_temporary_directory(self.ftf_temporary_directory.get_text().as_deref());
            Ok(true)
        })();
        result.unwrap_or(false)
    }
}

impl FlattenWarpDisplay for FlattenVolumePanel {
    /// Java public `getParameters(FlattenWarpParam, boolean)`.
    fn get_parameters(&self, param: &mut FlattenWarpParam, do_validation: bool) -> bool {
        // try { ... } catch (FieldValidationFailedException e) { return false; }
        let result = (|| -> Result<bool, FieldValidationFailedException> {
            let mut error_message = param.set_lambda_for_smoothing(
                self.ltf_lambda_for_smoothing
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Error in {LAMBDA_FOR_SMOOTHING_LABEL}:  {message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            param.set_one_surface(self.cb_one_surface.is_selected());
            error_message = param.set_warp_spacing_x(
                self.ltf_warp_spacing_x
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Error in {WARP_SPACING_X_LABEL}:  {message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            error_message = param.set_warp_spacing_y(
                self.ltf_warp_spacing_y
                    .get_text_boolean(do_validation)?
                    .as_deref(),
            );
            if let Some(message) = &error_message {
                ui_harness::with(|harness| {
                    harness.open_message_dialog_base_manager_string_string_axis_id(
                        Some(self.manager),
                        &format!("Error in {WARP_SPACING_Y_LABEL}:  {message}"),
                        "Entry Error",
                        Some(self.axis_id),
                    )
                });
                return Ok(false);
            }
            Ok(true)
        })();
        result.unwrap_or(false)
    }
}

impl SmoothingAssessmentParent for FlattenVolumePanel {
    /// Java public `isOneSurface()`.
    fn is_one_surface(&self) -> bool {
        self.cb_one_surface.is_selected()
    }

    /// Java public `getWarpSpacingX(boolean)`.
    fn get_warp_spacing_x(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_warp_spacing_x.get_text_boolean(do_validation)
    }

    /// Java public `getWarpSpacingY(boolean)`.
    fn get_warp_spacing_y(
        &self,
        do_validation: bool,
    ) -> Result<Option<String>, FieldValidationFailedException> {
        self.ltf_warp_spacing_y.get_text_boolean(do_validation)
    }
}

impl ContextMenu for FlattenVolumePanel {
    /// Java public `popUpContextMenu(MouseEvent)`.  Right mouse button context
    /// menu.
    fn pop_up_context_menu(&self, mouse_event: &MouseEvent) {
        let man_pagelabel = ["Flattenwarp".to_string(), "Warpvol".to_string()];
        let man_page = ["flattenwarp.html".to_string(), "warpvol.html".to_string()];
        let log_file_label = ["Flatten".to_string()];
        let log_file = [format!("{}{}", "flatten", ".log")];
        // `applicationManager == null ? toolsManager : applicationManager` is
        // the `manager` field in either case.
        let _context_popup = ContextPopup::new_component_mouse_event_string_string_string_array_string_array_string_array_string_array_base_manager_axis_id(
            &self.pnl_root.get_container(),
            mouse_event,
            Some("Flattening"),
            Some(context_popup::TOMO_GUIDE),
            &man_pagelabel,
            &man_page,
            Some(&log_file_label),
            Some(&log_file),
            self.manager,
            self.axis_id,
        );
    }
}

impl ToolPanel for FlattenVolumePanel {
    /// Java public `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }
}
