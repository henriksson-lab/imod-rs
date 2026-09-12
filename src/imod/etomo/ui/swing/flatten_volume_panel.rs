//! `IMOD/Etomo/src/etomo/ui/swing/FlattenVolumePanel.java`.
//!
//! Swing construction, file choosing/autodoc lookup, and the concrete manager
//! calls remain explicit boundaries.  This unit owns the source panel state,
//! its validation order, input-file selection, and post-processing/tools action
//! dispatch.
#![allow(dead_code)]

use std::{
    cell::Cell,
    path::{Path, PathBuf},
};

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::abstract_frame::ComponentState;
use super::check_box::CheckBox;
use super::flatten_warp_display::FlattenWarpDisplay;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonGroup};
use super::tilt_panel::Deferred3dmodButton;
use super::tool_panel::ToolPanel;

pub const OUTPUT_SIZE_Z_LABEL: &str = "Output thickness in Z";
pub const FLATTEN_LABEL: &str = "Flatten";
pub const WARP_SPACING_X_LABEL: &str = "Spacing in X";
pub const WARP_SPACING_Y_LABEL: &str = "and Y";
pub const LAMBDA_FOR_SMOOTHING_LABEL: &str = "Smoothing factor";
pub const FLATTEN_WARP_LABEL: &str = "Run Flattenwarp";

/// Java `PanelId` values selected by the two constructors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlattenVolumePanelId {
    PostFlattenVolume,
    ToolsFlattenVolume,
}

/// Java `FileType` result from `getInputFileType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlattenVolumeInputFileType {
    TrimVolOutput,
    FlattenReduceFiltVolFile,
}

/// Direct `FileTextField` widget/file-chooser boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FileTextFieldBoundary {
    pub label: String,
    pub text: String,
    pub editable: bool,
    pub button_enabled: bool,
    pub action_listener_count: usize,
    pub tooltip: Option<String>,
}
impl FileTextFieldBoundary {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.into(),
            editable: true,
            button_enabled: true,
            ..Self::default()
        }
    }
    pub fn get_file(&self) -> Option<PathBuf> {
        (!self.text.is_empty()).then(|| PathBuf::from(&self.text))
    }
    pub fn set_text(&mut self, text: &str) {
        self.text = text.into();
    }
    pub fn get_action_command(&self) -> &str {
        &self.label
    }
}

/// The fields of Java `ConstMetaData`/`MetaData` reached by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FlattenVolumeMetaData {
    pub post_flatten_warp_input_trim_vol: bool,
    pub post_flatten_warp_contours_on_one_surface: bool,
    pub post_flatten_warp_spacing_in_x: String,
    pub post_flatten_warp_spacing_in_y: String,
    pub lambda_for_smoothing: String,
}

/// Java `FlattenWarpParam` fields reached by `getParameters`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FlattenWarpParamBoundary {
    pub lambda_for_smoothing: String,
    pub one_surface: bool,
    pub warp_spacing_x: String,
    pub warp_spacing_y: String,
}
impl FlattenWarpParamBoundary {
    pub fn set_lambda_for_smoothing(&mut self, value: String) -> Option<String> {
        self.lambda_for_smoothing = value;
        None
    }
    pub fn set_one_surface(&mut self, value: bool) {
        self.one_surface = value;
    }
    pub fn set_warp_spacing_x(&mut self, value: String) -> Option<String> {
        self.warp_spacing_x = value;
        None
    }
    pub fn set_warp_spacing_y(&mut self, value: String) -> Option<String> {
        self.warp_spacing_y = value;
        None
    }
}

/// Java `ConstWarpVolParam`/`WarpVolParam` state selected by this panel.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WarpVolParamBoundary {
    pub interpolation_order_linear: bool,
    pub output_size_z: String,
    pub input_file: Option<PathBuf>,
    pub output_file: Option<String>,
    pub temporary_directory: String,
}
impl WarpVolParamBoundary {
    pub fn set_output_size_z(&mut self, value: String) -> Option<String> {
        self.output_size_z = value;
        None
    }
}

/// Source-visible Swing hierarchy/listener state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FlattenVolumePanelLayout {
    pub root_border_title: Option<String>,
    pub root_component_order: Vec<String>,
    pub input_file_component_order: Vec<String>,
    pub flatten_warp_component_order: Vec<String>,
    pub root_mouse_listener_count: usize,
    pub action_listener_count: usize,
    pub context_popup_title: Option<String>,
}

/// Direct `SmoothingAssessmentPanel` dependency state.  Its own source unit
/// owns the assessment command construction; these are exactly the calls made
/// by `FlattenVolumePanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SmoothingAssessmentPanelBoundary {
    pub post_instance: bool,
    pub tools_instance: bool,
    pub component_present: bool,
    pub done_count: usize,
    pub metadata_set_count: usize,
    pub metadata_get_count: Cell<usize>,
}
impl SmoothingAssessmentPanelBoundary {
    pub fn done(&mut self) {
        self.done_count += 1;
    }
    pub fn set_parameters(&mut self) {
        self.metadata_set_count += 1;
    }
    pub fn get_parameters(&self) {
        self.metadata_get_count
            .set(self.metadata_get_count.get() + 1);
    }
}

/// Direct `ApplicationManager`/`ToolsManager` action and message boundary.
pub trait FlattenVolumePanelManager {
    fn property_user_dir(&self) -> &Path;
    fn trim_vol_output_file(&self, axis_id: AxisID) -> Option<PathBuf>;
    fn flatten_output_file_name(&self) -> String;
    fn flatten_tool_output_file_name(&self) -> String;
    fn reduce_filt_vol_files(&self) -> Vec<(PathBuf, u64)>;
    fn validate_tools_dataset_name(&mut self, _axis_id: AxisID, _file: &Path) -> bool {
        true
    }
    fn conflicting_dataset_name(&self, _axis_id: AxisID, _file: &Path) -> bool {
        false
    }
    fn set_name(&mut self, _file: &Path) {}
    fn flatten(
        &mut self,
        _tools: bool,
        _options: Option<Run3dmodMenuOptions>,
        _dialog_type: DialogType,
        _axis_id: AxisID,
    ) {
    }
    fn imod_flatten(
        &mut self,
        _tools: bool,
        _options: Option<Run3dmodMenuOptions>,
        _axis_id: AxisID,
    ) {
    }
    fn imod_make_surface_model(
        &mut self,
        _tools: bool,
        _options: Option<Run3dmodMenuOptions>,
        _axis_id: AxisID,
        _binning: i32,
        _input_type: Option<FlattenVolumeInputFileType>,
        _input: &Path,
    ) {
    }
    fn flatten_warp(
        &mut self,
        _tools: bool,
        _options: Option<Run3dmodMenuOptions>,
        _dialog_type: DialogType,
        _axis_id: AxisID,
    ) {
    }
    fn is_rotated(&self, _axis_id: AxisID, _file: &Path) -> Option<bool> {
        Some(true)
    }
    fn open_message_dialog(&mut self, _message: String, _title: &str, _axis_id: AxisID) {}
    fn pack(&mut self) {}
}

/// Java final `FlattenVolumePanel`.
#[derive(Clone, Debug)]
pub struct FlattenVolumePanel {
    pub component: ComponentState,
    pub pnl_root: FlattenVolumePanelLayout,
    pub btn_make_surface_model: MultiLineButton,
    pub cb_one_surface: CheckBox,
    pub ltf_warp_spacing_x: LabeledTextField,
    pub ltf_warp_spacing_y: LabeledTextField,
    pub ltf_lambda_for_smoothing: LabeledTextField,
    pub bg_input_file: std::rc::Rc<std::cell::RefCell<RadioButtonGroup>>,
    pub rb_input_file_trim_vol: RadioButton,
    pub rb_input_file_squeeze_vol: RadioButton,
    pub cb_interpolation_order_linear: CheckBox,
    pub ltf_output_size_z: LabeledTextField,
    pub btn_imod_flatten: MultiLineButton,
    pub ftf_temporary_directory: FileTextFieldBoundary,
    pub ftf_input_file: FileTextFieldBoundary,
    pub panel_id: FlattenVolumePanelId,
    pub btn_flatten: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
    pub btn_flatten_warp: MultiLineButton,
    pub surface_model_binning_in_x_and_y: i32,
    pub smoothing_assessment_panel: SmoothingAssessmentPanelBoundary,
}

impl FlattenVolumePanel {
    /// Java private `FlattenVolumePanel(ApplicationManager, AxisID, DialogType)`.
    pub fn new_post(
        axis_id: AxisID,
        dialog_type: DialogType,
        btn_flatten: MultiLineButton,
        btn_flatten_warp: MultiLineButton,
    ) -> Self {
        Self::new(
            axis_id,
            dialog_type,
            FlattenVolumePanelId::PostFlattenVolume,
            btn_flatten,
            btn_flatten_warp,
        )
    }
    /// Java private `FlattenVolumePanel(ToolsManager, AxisID, DialogType)`.
    pub fn new_tools(axis_id: AxisID, dialog_type: DialogType) -> Self {
        Self::new(
            axis_id,
            dialog_type,
            FlattenVolumePanelId::ToolsFlattenVolume,
            MultiLineButton::new_with_label(Some(FLATTEN_LABEL)),
            MultiLineButton::new_with_label(Some(FLATTEN_WARP_LABEL)),
        )
    }
    /// Java shared constructor body after manager identity is kept at action boundary.
    pub fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        panel_id: FlattenVolumePanelId,
        mut btn_flatten: MultiLineButton,
        mut btn_flatten_warp: MultiLineButton,
    ) -> Self {
        use std::{cell::RefCell, rc::Rc};
        let bg_input_file = Rc::new(RefCell::new(RadioButtonGroup::new()));
        btn_flatten.set_action_command(Some(FLATTEN_LABEL));
        btn_flatten_warp.set_action_command(Some(FLATTEN_WARP_LABEL));
        let mut btn_make_surface_model =
            MultiLineButton::new_with_label(Some("Make Surface Model"));
        btn_make_surface_model.set_action_command(Some("Make Surface Model"));
        let mut btn_imod_flatten = MultiLineButton::new_with_label(Some("Open Flattened Tomogram"));
        btn_imod_flatten.set_action_command(Some("Open Flattened Tomogram"));
        Self {
            component: ComponentState::default(),
            pnl_root: FlattenVolumePanelLayout::default(),
            btn_make_surface_model,
            cb_one_surface: CheckBox::new_with_text("Contours are all on one surface"),
            ltf_warp_spacing_x: LabeledTextField::new(FieldType::FloatingPoint, "Spacing in X: "),
            ltf_warp_spacing_y: LabeledTextField::new(FieldType::FloatingPoint, " and Y: "),
            ltf_lambda_for_smoothing: LabeledTextField::new(
                FieldType::FloatingPointArray,
                "Smoothing factor: ",
            ),
            bg_input_file: bg_input_file.clone(),
            rb_input_file_trim_vol: RadioButton::new_in_group(
                "Flatten the trimvol output",
                bg_input_file.clone(),
            ),
            rb_input_file_squeeze_vol: RadioButton::new_in_group(
                "Flatten the reducefiltvol output",
                bg_input_file,
            ),
            cb_interpolation_order_linear: CheckBox::new_with_text("Linear interpolation"),
            ltf_output_size_z: LabeledTextField::new(FieldType::Integer, "Output thickness in Z: "),
            btn_imod_flatten,
            ftf_temporary_directory: FileTextFieldBoundary::new("Temporary directory:"),
            ftf_input_file: FileTextFieldBoundary::new("Input file:"),
            panel_id,
            btn_flatten,
            axis_id,
            dialog_type,
            btn_flatten_warp,
            surface_model_binning_in_x_and_y: 1,
            smoothing_assessment_panel: SmoothingAssessmentPanelBoundary {
                post_instance: panel_id == FlattenVolumePanelId::PostFlattenVolume,
                tools_instance: panel_id == FlattenVolumePanelId::ToolsFlattenVolume,
                component_present: true,
                ..Default::default()
            },
        }
    }
    /// Java static `getPostInstance`.
    pub fn get_post_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        btn_flatten: MultiLineButton,
        btn_flatten_warp: MultiLineButton,
    ) -> Self {
        let mut instance = Self::new_post(axis_id, dialog_type, btn_flatten, btn_flatten_warp);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }
    /// Java static `getToolsInstance`.
    pub fn get_tools_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new_tools(axis_id, dialog_type);
        instance.create_panel();
        instance.set_tool_tip_text();
        instance.add_listeners();
        instance
    }
    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.pnl_root.root_mouse_listener_count += 1;
        self.btn_make_surface_model.add_action_listener();
        self.btn_flatten_warp.add_action_listener();
        self.btn_flatten.add_action_listener();
        self.btn_imod_flatten.add_action_listener();
        self.ftf_input_file.action_listener_count += 1;
        self.pnl_root.action_listener_count += 5;
    }
    /// Java `popUpContextMenu`.
    pub fn pop_up_context_menu(&mut self) {
        self.pnl_root.context_popup_title = Some("Flattening".into());
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.btn_flatten_warp.remove_action_listener();
        self.btn_flatten.remove_action_listener();
        self.smoothing_assessment_panel.done();
    }
    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.rb_input_file_trim_vol.set_selected(true);
        self.ftf_input_file.editable = false;
        self.pnl_root.root_border_title = Some("Flatten Volume".into());
        self.pnl_root.root_component_order =
            if self.panel_id == FlattenVolumePanelId::PostFlattenVolume {
                vec![
                    "pnlInputFile",
                    "pnlFlattenWarp",
                    "pnlInterpolationOrder",
                    "ltfOutputSizeZ",
                    "ftfTemporaryDirectory",
                    "pnlFlatten",
                ]
            } else {
                vec![
                    "ftfInputFile",
                    "pnlFlattenWarp",
                    "pnlInterpolationOrder",
                    "ltfOutputSizeZ",
                    "ftfTemporaryDirectory",
                    "pnlFlatten",
                ]
            }
            .into_iter()
            .map(String::from)
            .collect();
        self.pnl_root.input_file_component_order =
            vec!["rbInputFileTrimVol".into(), "rbInputFileSqueezeVol".into()];
        self.pnl_root.flatten_warp_component_order = vec![
            "btnMakeSurfaceModel",
            "cbOneSurface",
            "warpSpacing",
            "smoothingAssessmentPanel",
            "ltfLambdaForSmoothing",
            "btnFlattenWarp",
        ]
        .into_iter()
        .map(String::from)
        .collect();
    }
    /// Java `getFlattenWarpDisplay`.
    pub fn get_flatten_warp_display<M: FlattenVolumePanelManager>(
        &self,
    ) -> &dyn FlattenWarpDisplay<M> {
        self
    }
    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_metadata(&mut self, meta_data: &FlattenVolumeMetaData) {
        self.rb_input_file_trim_vol
            .set_selected(meta_data.post_flatten_warp_input_trim_vol);
        if !self.rb_input_file_trim_vol.is_selected() {
            self.rb_input_file_squeeze_vol.set_selected(true);
        }
        self.cb_one_surface
            .set_selected(meta_data.post_flatten_warp_contours_on_one_surface);
        self.ltf_warp_spacing_x
            .set_text(&meta_data.post_flatten_warp_spacing_in_x);
        self.ltf_warp_spacing_y
            .set_text(&meta_data.post_flatten_warp_spacing_in_y);
        self.ltf_lambda_for_smoothing
            .set_text(&meta_data.lambda_for_smoothing);
        self.smoothing_assessment_panel.set_parameters();
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_metadata(&self, meta_data: &mut FlattenVolumeMetaData) {
        meta_data.post_flatten_warp_input_trim_vol = self.rb_input_file_trim_vol.is_selected();
        meta_data.post_flatten_warp_contours_on_one_surface = self.cb_one_surface.is_selected();
        meta_data.post_flatten_warp_spacing_in_x = self.ltf_warp_spacing_x.get_text();
        meta_data.post_flatten_warp_spacing_in_y = self.ltf_warp_spacing_y.get_text();
        meta_data.lambda_for_smoothing = self.ltf_lambda_for_smoothing.get_text();
        self.smoothing_assessment_panel.get_parameters();
    }
    /// Java private `validateFlattenWarp`.
    pub fn validate_flatten_warp<M: FlattenVolumePanelManager>(&self, manager: &mut M) -> bool {
        if self.ltf_lambda_for_smoothing.is_empty() {
            manager.open_message_dialog(
                format!("{LAMBDA_FOR_SMOOTHING_LABEL} is a required field."),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        true
    }
    /// Java `getParameters(FlattenWarpParam, boolean)`.
    pub fn get_parameters_flatten_warp<M: FlattenVolumePanelManager>(
        &self,
        param: &mut FlattenWarpParamBoundary,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        let lambda = match self
            .ltf_lambda_for_smoothing
            .get_text_validated(do_validation)
        {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error) = param.set_lambda_for_smoothing(lambda) {
            manager.open_message_dialog(
                format!("Error in {LAMBDA_FOR_SMOOTHING_LABEL}:  {error}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        param.set_one_surface(self.cb_one_surface.is_selected());
        let x = match self.ltf_warp_spacing_x.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error) = param.set_warp_spacing_x(x) {
            manager.open_message_dialog(
                format!("Error in {WARP_SPACING_X_LABEL}:  {error}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        let y = match self.ltf_warp_spacing_y.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error) = param.set_warp_spacing_y(y) {
            manager.open_message_dialog(
                format!("Error in {WARP_SPACING_Y_LABEL}:  {error}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        true
    }
    /// Java `getParameters(WarpVolParam, boolean)`.
    pub fn get_parameters_warp_vol<M: FlattenVolumePanelManager>(
        &self,
        param: &mut WarpVolParamBoundary,
        do_validation: bool,
        manager: &mut M,
    ) -> bool {
        param.interpolation_order_linear = self.cb_interpolation_order_linear.is_selected();
        let size = match self.ltf_output_size_z.get_text_validated(do_validation) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if let Some(error) = param.set_output_size_z(size) {
            manager.open_message_dialog(
                format!("Error in {OUTPUT_SIZE_Z_LABEL}:  {error}"),
                "Entry Error",
                self.axis_id,
            );
            return false;
        }
        match self.panel_id {
            FlattenVolumePanelId::PostFlattenVolume => {
                let input = if self.get_input_file_type()
                    == Some(FlattenVolumeInputFileType::TrimVolOutput)
                {
                    manager.trim_vol_output_file(self.axis_id)
                } else {
                    self.get_input_file(manager)
                };
                let Some(input) = input else {
                    manager.open_message_dialog(
                        "File you intend to open does not exist".into(),
                        "File Not Found Error",
                        self.axis_id,
                    );
                    return false;
                };
                param.input_file = Some(input);
                param.output_file = Some(manager.flatten_output_file_name());
            }
            FlattenVolumePanelId::ToolsFlattenVolume => {
                param.input_file = self.ftf_input_file.get_file();
                param.output_file = Some(manager.flatten_tool_output_file_name());
            }
        }
        param.temporary_directory = self.ftf_temporary_directory.text.clone();
        true
    }
    /// Java `setParameters(ConstWarpVolParam)`.
    pub fn set_parameters_warp_vol(&mut self, param: &WarpVolParamBoundary) {
        self.cb_interpolation_order_linear
            .set_selected(param.interpolation_order_linear);
        self.ltf_output_size_z.set_text(&param.output_size_z);
        self.ftf_temporary_directory
            .set_text(&param.temporary_directory);
    }
    /// Java `getInputFileType`.
    pub fn get_input_file_type(&self) -> Option<FlattenVolumeInputFileType> {
        (self.panel_id == FlattenVolumePanelId::PostFlattenVolume).then(|| {
            if self.rb_input_file_trim_vol.is_selected() {
                FlattenVolumeInputFileType::TrimVolOutput
            } else {
                FlattenVolumeInputFileType::FlattenReduceFiltVolFile
            }
        })
    }
    /// Java `getInputFile`.
    pub fn get_input_file<M: FlattenVolumePanelManager>(&self, manager: &M) -> Option<PathBuf> {
        manager
            .reduce_filt_vol_files()
            .into_iter()
            .max_by_key(|(_, modified)| *modified)
            .map(|(file, _)| file)
    }
    /// Java `isOneSurface`.
    pub fn is_one_surface(&self) -> bool {
        self.cb_one_surface.is_selected()
    }
    /// Java `getWarpSpacingX`.
    pub fn get_warp_spacing_x(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_warp_spacing_x.get_text_validated(do_validation)
    }
    /// Java `getWarpSpacingY`.
    pub fn get_warp_spacing_y(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_warp_spacing_y.get_text_validated(do_validation)
    }
    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: FlattenVolumePanelManager>(
        &mut self,
        command: &str,
        _deferred_3dmod_button: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
        manager: &mut M,
    ) {
        if self.panel_id == FlattenVolumePanelId::ToolsFlattenVolume
            && command == self.ftf_input_file.get_action_command()
        {
            self.input_file_action(None, manager);
            return;
        }
        if self.panel_id == FlattenVolumePanelId::ToolsFlattenVolume
            && !manager.validate_tools_dataset_name(
                self.axis_id,
                self.ftf_input_file
                    .get_file()
                    .as_deref()
                    .unwrap_or(Path::new("")),
            )
        {
            return;
        }
        let tools = self.panel_id == FlattenVolumePanelId::ToolsFlattenVolume;
        if command == self.btn_flatten.get_action_command().unwrap_or_default() {
            manager.flatten(tools, options, self.dialog_type, self.axis_id);
        } else if command
            == self
                .btn_imod_flatten
                .get_action_command()
                .unwrap_or_default()
        {
            manager.imod_flatten(tools, options, self.axis_id);
        } else if command
            == self
                .btn_make_surface_model
                .get_action_command()
                .unwrap_or_default()
        {
            let input = if tools {
                self.ftf_input_file.get_file()
            } else if self.get_input_file_type() == Some(FlattenVolumeInputFileType::TrimVolOutput)
            {
                manager.trim_vol_output_file(self.axis_id)
            } else {
                self.get_input_file(manager)
            };
            let Some(input) = input else {
                manager.open_message_dialog(
                    "File you intend to open does not exist".into(),
                    "File Not Found Error",
                    self.axis_id,
                );
                return;
            };
            self.check_rotated(&input, manager);
            manager.imod_make_surface_model(
                tools,
                options,
                self.axis_id,
                self.surface_model_binning_in_x_and_y,
                self.get_input_file_type(),
                &input,
            );
        } else if command
            == self
                .btn_flatten_warp
                .get_action_command()
                .unwrap_or_default()
        {
            if self.validate_flatten_warp(manager) {
                manager.flatten_warp(tools, options, self.dialog_type, self.axis_id);
            }
        } else {
            panic!("Unknown command {command}");
        }
    }
    /// Java private `inputFileAction`; chooser selection arrives at the native presentation boundary.
    pub fn input_file_action<M: FlattenVolumePanelManager>(
        &mut self,
        selected_file: Option<PathBuf>,
        manager: &mut M,
    ) {
        let Some(file) = selected_file.filter(|file| file.is_file()) else {
            return;
        };
        if !manager.validate_tools_dataset_name(self.axis_id, &file)
            || manager.conflicting_dataset_name(self.axis_id, &file)
        {
            return;
        }
        self.ftf_input_file.set_text(&file.to_string_lossy());
        self.ftf_input_file.button_enabled = false;
        manager.set_name(&file);
        manager.pack();
        self.check_rotated(&file, manager);
    }
    /// Java private `checkRotated`.
    pub fn check_rotated<M: FlattenVolumePanelManager>(&self, file: &Path, manager: &mut M) {
        match manager.is_rotated(self.axis_id, file) { None => manager.open_message_dialog(format!("The MRC header of this file, {}, is unreadable.", file.display()), "Warning", self.axis_id), Some(false) => manager.open_message_dialog(format!("This tomogram, {}, looks like the volume hasn't been reoriented.   Flattening won't work on a volume that hasn't been reoriented.", file.display()), "Warning", self.axis_id), Some(true) => {} }
    }
    /// Java private `setToolTipText`; autodoc lookup is a storage boundary.
    pub fn set_tool_tip_text(&mut self) {
        self.btn_make_surface_model.set_tool_tip_text(Some(
            "Add contours to describe the location of the sectioned material.",
        ));
        self.btn_flatten_warp
            .set_tool_tip_text(Some("Run flattenwarp."));
        self.btn_flatten.set_tool_tip_text(Some("Run warpvol."));
        self.btn_imod_flatten
            .set_tool_tip_text(Some("Open warpvol output in 3dmod."));
    }
}

impl ToolPanel for FlattenVolumePanel {
    fn get_component(&self) -> &ComponentState {
        &self.component
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        files: Vec<(PathBuf, u64)>,
        messages: Vec<String>,
        actions: Vec<String>,
    }
    impl FlattenVolumePanelManager for Manager {
        fn property_user_dir(&self) -> &Path {
            Path::new(".")
        }
        fn trim_vol_output_file(&self, _: AxisID) -> Option<PathBuf> {
            Some("trim.rec".into())
        }
        fn flatten_output_file_name(&self) -> String {
            "flat.rec".into()
        }
        fn flatten_tool_output_file_name(&self) -> String {
            "flatten.rec".into()
        }
        fn reduce_filt_vol_files(&self) -> Vec<(PathBuf, u64)> {
            self.files.clone()
        }
        fn open_message_dialog(&mut self, message: String, _: &str, _: AxisID) {
            self.messages.push(message)
        }
        fn flatten(
            &mut self,
            tools: bool,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
            _: AxisID,
        ) {
            self.actions.push(format!("flatten:{tools}"));
        }
    }
    #[test]
    fn post_metadata_and_warpvol_follow_source_input_selection() {
        let flatten = MultiLineButton::new_with_label(Some(FLATTEN_LABEL));
        let warp = MultiLineButton::new_with_label(Some(FLATTEN_WARP_LABEL));
        let mut panel = FlattenVolumePanel::get_post_instance(
            AxisID::Only,
            DialogType::PostProcessing,
            flatten,
            warp,
        );
        panel.ltf_output_size_z.set_text("100");
        let mut manager = Manager::default();
        let mut param = WarpVolParamBoundary::default();
        assert!(panel.get_parameters_warp_vol(&mut param, true, &mut manager));
        assert_eq!(param.input_file, Some("trim.rec".into()));
        assert_eq!(param.output_file.as_deref(), Some("flat.rec"));
        assert_eq!(param.output_size_z, "100");
    }
    #[test]
    fn latest_reducefiltvol_file_is_selected() {
        let panel = FlattenVolumePanel::new_tools(AxisID::Only, DialogType::Tools);
        let manager = Manager {
            files: vec![("old.rec".into(), 1), ("new.rec".into(), 2)],
            ..Default::default()
        };
        assert_eq!(panel.get_input_file(&manager), Some("new.rec".into()));
    }
    #[test]
    fn blank_smoothing_is_rejected_before_flattenwarp_dispatch() {
        let flatten = MultiLineButton::new_with_label(Some(FLATTEN_LABEL));
        let warp = MultiLineButton::new_with_label(Some(FLATTEN_WARP_LABEL));
        let mut panel = FlattenVolumePanel::get_post_instance(
            AxisID::Only,
            DialogType::PostProcessing,
            flatten,
            warp,
        );
        let mut manager = Manager::default();
        assert!(!panel.validate_flatten_warp(&mut manager));
        assert_eq!(
            manager.messages,
            vec!["Smoothing factor is a required field."]
        );
    }
}
