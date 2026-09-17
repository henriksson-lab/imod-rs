//! `IMOD/Etomo/src/etomo/ui/swing/MultifiltPanel.java`.
//!
//! File chooser construction, autodoc I/O, process execution, and widget
//! painting are frontend boundaries.  The selection, display, validation, and
//! parameter-routing policy remains the source-owned Rust state in this unit.
#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};

use super::{
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    multi_line_button::MultiLineButton,
    radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup},
    text_efield::TextEfield,
};
use crate::imod::etomo::{r#type::axis_id::AxisID, ui::field_type::FieldType};

pub const FAKE_SIRT_ITERATIONS_LABEL: &str = "SIRT-like filter with iterations:";
pub const EXACT_OBJECT_SIZES_LABEL: &str = "'Exact filter' function with 'object sizes':";
pub const GAUSSIAN_LABEL: &str = "Standard Gaussian with cutoffs:";
pub const HAMMING_LIKE_STARTS_LABEL: &str = "Hamming-like filter with start frequencies:";

/// Java `FilterType`, with the Java array index retained for output-file filters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultifiltFilterType {
    FakeSirtIterations,
    ExactObjectSizes,
    Gaussian,
    HammingLikeStarts,
}
impl MultifiltFilterType {
    pub const ARRAY_SIZE: usize = 5;
    pub const fn index(self) -> usize {
        match self {
            Self::FakeSirtIterations => 0,
            Self::ExactObjectSizes => 1,
            Self::Gaussian => 2,
            Self::HammingLikeStarts => 3,
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::FakeSirtIterations => FAKE_SIRT_ITERATIONS_LABEL,
            Self::ExactObjectSizes => EXACT_OBJECT_SIZES_LABEL,
            Self::Gaussian => GAUSSIAN_LABEL,
            Self::HammingLikeStarts => HAMMING_LIKE_STARTS_LABEL,
        }
    }
    pub const fn is_default(self) -> bool {
        matches!(self, Self::FakeSirtIterations)
    }
    #[allow(non_snake_case)]
    pub fn getLabel(self) -> &'static str {
        self.label()
    }
    #[allow(non_snake_case)]
    pub fn isDefault(self) -> bool {
        self.is_default()
    }
    #[allow(non_snake_case)]
    pub fn getValue(self) -> &'static str {
        self.label()
    }
    pub const fn is_radial_filter(self) -> bool {
        matches!(self, Self::FakeSirtIterations | Self::ExactObjectSizes)
    }
    pub const fn is_high_frequency_filter(self) -> bool {
        matches!(self, Self::Gaussian | Self::HammingLikeStarts)
    }
}

/// Java `MultifiltOutputFileFilter` retained at its storage/file chooser boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultifiltOutputFileFilterBoundary {
    pub filter_type: Option<MultifiltFilterType>,
}

/// Java `TomogramGenerationDialog` calls used directly by this panel.
pub trait MultifiltPanelParent {
    fn is_multifilt(&self) -> bool;
    fn processing_method(&self) -> String;
}
/// Java `ApplicationManager` process and 3dmod calls used directly by this panel.
pub trait MultifiltPanelApplicationManager {
    fn multifilt_setup(&mut self, axis_id: AxisID, dialog_type: &str, processing_method: String);
    fn open_files_in_imod(&mut self, axis_id: AxisID, files: Vec<String>);
    fn pack(&mut self, axis_id: AxisID);
}
/// Java `MultifiltSetupParam` calls made by `setParameters` and `getParameters`.
pub trait MultifiltSetupParam {
    fn is_fake_sirt_iterations(&self) -> bool;
    fn fake_sirt_iterations(&self) -> String;
    fn is_exact_object_sizes(&self) -> bool;
    fn exact_object_sizes(&self) -> String;
    fn is_gaussian_cutoffs(&self) -> bool;
    fn gaussian_cutoffs(&self) -> String;
    fn is_gaussian_falloffs(&self) -> bool;
    fn gaussian_falloffs(&self) -> String;
    fn is_hamming_like_starts(&self) -> bool;
    fn hamming_like_starts(&self) -> String;
    fn width_in_x(&self) -> String;
    fn shift_in_x(&self) -> String;
    fn size_in_y(&self) -> String;
    fn shift_in_y(&self) -> String;
    fn thickness_in_z(&self) -> String;
    fn shift_in_depth(&self) -> String;
    fn set_fake_sirt_iterations(&mut self, value: String);
    fn reset_fake_sirt_iterations(&mut self);
    fn set_exact_object_sizes(&mut self, value: String);
    fn reset_exact_object_sizes(&mut self);
    fn set_gaussian_cutoffs(&mut self, value: String);
    fn reset_gaussian_cutoffs(&mut self);
    fn set_gaussian_falloffs(&mut self, value: String);
    fn reset_gaussian_falloffs(&mut self);
    fn set_hamming_like_starts(&mut self, value: String);
    fn reset_hamming_like_starts(&mut self);
    fn set_width_in_x(&mut self, value: String);
    fn set_shift_in_x(&mut self, value: String);
    fn set_size_in_y(&mut self, value: String);
    fn set_shift_in_y(&mut self, value: String);
    fn set_thickness_in_z(&mut self, value: String);
    fn set_shift_in_depth(&mut self, value: String);
}
/// Java `ConstMetaData`/`MetaData` calls owned by this source unit.
pub trait MultifiltMetaData {
    fn fake_sirt_iterations(&self, axis_id: AxisID) -> String;
    fn exact_object_sizes(&self, axis_id: AxisID) -> String;
    fn gaussian_cutoffs(&self, axis_id: AxisID) -> String;
    fn gaussian_falloffs(&self, axis_id: AxisID) -> String;
    fn hamming_like_starts(&self, axis_id: AxisID) -> String;
    fn set_fake_sirt_iterations(&mut self, axis_id: AxisID, value: String);
    fn set_exact_object_sizes(&mut self, axis_id: AxisID, value: String);
    fn set_gaussian_cutoffs(&mut self, axis_id: AxisID, value: String);
    fn set_gaussian_falloffs(&mut self, axis_id: AxisID, value: String);
    fn set_hamming_like_starts(&mut self, axis_id: AxisID, value: String);
}

/// Source-visible component/layout state at the Swing boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultifiltPanelLayout {
    pub root_visible: bool,
    pub parameters_body_visible: bool,
    pub component_order: Vec<&'static str>,
    pub listener_count: usize,
    pub listener_signal_count: usize,
    pub tooltip_initialized: bool,
    pub selected_output_filter: Option<MultifiltFilterType>,
}

/// Java `MultifiltPanel`.
pub struct MultifiltPanel {
    pub axis_id: AxisID,
    pub dialog_type: String,
    pub layout: MultifiltPanelLayout,
    pub rb_fake_sirt_iterations: RadioButton,
    pub rb_exact_object_sizes: RadioButton,
    pub rb_gaussian: RadioButton,
    pub rb_hamming_like_starts: RadioButton,
    pub tf_fake_sirt_iterations: TextEfield,
    pub tf_exact_object_sizes: TextEfield,
    pub tf_gaussian_cutoffs: TextEfield,
    pub ltf_gaussian_falloffs: LabeledTextField,
    pub tf_hamming_like_starts: TextEfield,
    pub ltf_width_in_x: LabeledTextField,
    pub ltf_shift_in_x: LabeledTextField,
    pub ltf_size_in_y: LabeledTextField,
    pub ltf_shift_in_y: LabeledTextField,
    pub ltf_thickness_in_z: LabeledTextField,
    pub ltf_shift_in_depth: LabeledTextField,
    pub btn_multifilt_setup: MultiLineButton,
    pub btn_3dmod_multifilt_setup: MultiLineButton,
    pub file_filter_array:
        [Option<MultifiltOutputFileFilterBoundary>; MultifiltFilterType::ARRAY_SIZE],
}

impl MultifiltPanel {
    /// Java private constructor plus `getInstance` initialization sequence.
    pub fn get_instance(axis_id: AxisID, dialog_type: impl Into<String>) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let radio = |filter_type: MultifiltFilterType| {
            RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: filter_type.label().into(),
                    default: filter_type.is_default(),
                    value: None,
                },
                Some(group.clone()),
            )
        };
        let mut result = Self {
            axis_id,
            dialog_type: dialog_type.into(),
            layout: MultifiltPanelLayout {
                root_visible: true,
                parameters_body_visible: true,
                component_order: vec![
                    "parameters",
                    "filterToTry",
                    "fakeSirtIterations",
                    "exactObjectSizes",
                    "gaussian",
                    "hammingLikeStarts",
                    "subarea",
                    "buttons",
                ],
                listener_count: 0,
                listener_signal_count: 0,
                tooltip_initialized: false,
                selected_output_filter: None,
            },
            rb_fake_sirt_iterations: radio(MultifiltFilterType::FakeSirtIterations),
            rb_exact_object_sizes: radio(MultifiltFilterType::ExactObjectSizes),
            rb_gaussian: radio(MultifiltFilterType::Gaussian),
            rb_hamming_like_starts: radio(MultifiltFilterType::HammingLikeStarts),
            tf_fake_sirt_iterations: TextEfield::get_labeled_instance(
                FAKE_SIRT_ITERATIONS_LABEL,
                FieldType::IntegerList,
            ),
            tf_exact_object_sizes: TextEfield::get_labeled_instance(
                EXACT_OBJECT_SIZES_LABEL,
                FieldType::IntegerList,
            ),
            tf_gaussian_cutoffs: TextEfield::get_labeled_instance(
                GAUSSIAN_LABEL,
                FieldType::FloatingPointArray,
            ),
            ltf_gaussian_falloffs: LabeledTextField::new(
                FieldType::FloatingPointArray,
                " falloffs:",
            ),
            tf_hamming_like_starts: TextEfield::get_labeled_instance(
                HAMMING_LIKE_STARTS_LABEL,
                FieldType::FloatingPointArray,
            ),
            ltf_width_in_x: LabeledTextField::new(FieldType::Integer, "Tomogram width in X: "),
            ltf_shift_in_x: LabeledTextField::new(FieldType::FloatingPoint, "X shift: "),
            ltf_size_in_y: LabeledTextField::new(FieldType::Integer, "Tomogram height in Y: "),
            ltf_shift_in_y: LabeledTextField::new(FieldType::Integer, " Y shift: "),
            ltf_thickness_in_z: LabeledTextField::new(
                FieldType::Integer,
                "Tomogram thickness in Z: ",
            ),
            ltf_shift_in_depth: LabeledTextField::new(FieldType::FloatingPoint, " Z shift: "),
            btn_multifilt_setup: MultiLineButton::new_with_label(Some("Run Filter Trials")),
            btn_3dmod_multifilt_setup: MultiLineButton::new_with_label(Some(
                "View Tomogram(s) In 3dmod",
            )),
            file_filter_array: std::array::from_fn(|_| None),
        };
        result.create_panel();
        result.set_tooltips();
        result.add_listeners();
        result
    }
    #[allow(non_snake_case)]
    pub fn getUIComponent(&self) -> &Self {
        self
    }
    #[allow(non_snake_case)]
    pub fn isEnabled(&self) -> bool {
        self.layout.root_visible
    }
    #[allow(non_snake_case)]
    pub fn isEditable(&self) -> bool {
        self.layout.root_visible
    }
    /// Java `createPanel`; physical Swing composition is retained in `component_order`.
    fn create_panel(&mut self) {
        self.rb_fake_sirt_iterations.set_selected(true);
        self.tf_fake_sirt_iterations.set_required(true);
        self.tf_exact_object_sizes.set_required(true);
        self.tf_hamming_like_starts.set_required(true);
        self.ltf_width_in_x.set_preferred_width(163, None);
        self.ltf_size_in_y.set_preferred_width(159, None);
        self.tf_gaussian_cutoffs.set_preferred_width(108);
        self.ltf_gaussian_falloffs.set_preferred_width(109, None);
        self.create_file_filter(Some(MultifiltFilterType::FakeSirtIterations));
        self.create_file_filter(Some(MultifiltFilterType::ExactObjectSizes));
        self.create_file_filter(Some(MultifiltFilterType::Gaussian));
        self.create_file_filter(Some(MultifiltFilterType::HammingLikeStarts));
        self.create_file_filter(None);
        self.update_display();
    }
    /// Java `createFileFilter`.
    fn create_file_filter(&mut self, filter_type: Option<MultifiltFilterType>) {
        let index = filter_type.map_or(
            MultifiltFilterType::ARRAY_SIZE - 1,
            MultifiltFilterType::index,
        );
        self.file_filter_array[index] = Some(MultifiltOutputFileFilterBoundary { filter_type });
    }
    /// Java `addListeners`; actual native listener attachment is a GUI boundary.
    fn add_listeners(&mut self) {
        self.rb_fake_sirt_iterations.add_action_listener();
        self.rb_exact_object_sizes.add_action_listener();
        self.rb_gaussian.add_action_listener();
        self.rb_hamming_like_starts.add_action_listener();
        self.layout.listener_count = 6;
    }
    /// Java `addActionListener`; listener callbacks cross the frontend boundary.
    pub fn add_action_listener(&mut self) {
        self.layout.listener_signal_count += 1;
    }
    /// Java `actionPerformed`.
    pub fn action_performed<M: MultifiltPanelApplicationManager, P: MultifiltPanelParent>(
        &mut self,
        manager: &mut M,
        parent: &P,
        action_command: Option<&str>,
        selected_files: Option<Vec<String>>,
    ) {
        self.action(manager, parent, action_command, selected_files)
    }
    /// Java `action`.
    pub fn action<M: MultifiltPanelApplicationManager, P: MultifiltPanelParent>(
        &mut self,
        manager: &mut M,
        parent: &P,
        action_command: Option<&str>,
        selected_files: Option<Vec<String>>,
    ) {
        if action_command == self.btn_multifilt_setup.get_text() {
            manager.multifilt_setup(self.axis_id, &self.dialog_type, parent.processing_method());
        } else if action_command == self.btn_3dmod_multifilt_setup.get_text() {
            self.open_files_in_imod(manager, selected_files.unwrap_or_default());
        } else {
            self.update_display();
        }
    }
    /// Java `isRadialFilter`.
    pub fn is_radial_filter(&self) -> bool {
        self.rb_fake_sirt_iterations.is_selected() || self.rb_exact_object_sizes.is_selected()
    }
    /// Java `isHighFrequencyFilter`.
    pub fn is_high_frequency_filter(&self) -> bool {
        self.rb_gaussian.is_selected() || self.rb_hamming_like_starts.is_selected()
    }
    /// Java `expand(ExpandButton)`; header equality belongs to the GUI adapter.
    pub fn expand<M: MultifiltPanelApplicationManager>(&mut self, manager: &mut M, expanded: bool) {
        self.layout.parameters_body_visible = expanded;
        manager.pack(self.axis_id);
    }
    /// Java empty `expand(GlobalExpandButton)`.
    pub fn expand_global(&mut self) {}
    /// Java `openFilesInImod`; chooser selection is supplied by the native GUI boundary.
    pub fn open_files_in_imod<M: MultifiltPanelApplicationManager>(
        &mut self,
        manager: &mut M,
        file_list: Vec<String>,
    ) {
        if file_list.is_empty() {
            return;
        }
        self.layout.selected_output_filter = self.selected_filter_type();
        manager.open_files_in_imod(self.axis_id, file_list);
    }
    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        self.tf_fake_sirt_iterations
            .set_enabled(self.rb_fake_sirt_iterations.is_selected());
        self.tf_exact_object_sizes
            .set_enabled(self.rb_exact_object_sizes.is_selected());
        let enabled = self.rb_gaussian.is_selected();
        self.tf_gaussian_cutoffs.set_enabled(enabled);
        self.ltf_gaussian_falloffs.set_enabled(enabled);
        self.tf_hamming_like_starts
            .set_enabled(self.rb_hamming_like_starts.is_selected());
    }
    /// Java `msgMethodChanged`.
    pub fn msg_method_changed<P: MultifiltPanelParent>(&mut self, parent: &P) {
        self.layout.root_visible = parent.is_multifilt();
    }
    /// Java `done`.
    pub fn done(&mut self) {
        self.layout.listener_count = 0;
    }
    /// Java `getParameters(ReconScreenState)` / `setParameters(ReconScreenState)` state boundary.
    pub fn get_button_state(&self) -> bool {
        self.btn_multifilt_setup.get_button_state()
    }
    pub fn set_button_state(&mut self, state: bool) {
        self.btn_multifilt_setup.set_button_state(state);
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_metadata<M: MultifiltMetaData>(&self, metadata: &mut M) {
        metadata.set_fake_sirt_iterations(self.axis_id, self.tf_fake_sirt_iterations.get_text());
        metadata.set_exact_object_sizes(self.axis_id, self.tf_exact_object_sizes.get_text());
        metadata.set_gaussian_cutoffs(self.axis_id, self.tf_gaussian_cutoffs.get_text());
        metadata.set_gaussian_falloffs(self.axis_id, self.ltf_gaussian_falloffs.get_text());
        metadata.set_hamming_like_starts(self.axis_id, self.tf_hamming_like_starts.get_text());
    }
    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_metadata<M: MultifiltMetaData>(&mut self, metadata: &M) {
        self.tf_fake_sirt_iterations
            .set_text(metadata.fake_sirt_iterations(self.axis_id));
        self.tf_exact_object_sizes
            .set_text(metadata.exact_object_sizes(self.axis_id));
        self.tf_gaussian_cutoffs
            .set_text(metadata.gaussian_cutoffs(self.axis_id));
        self.ltf_gaussian_falloffs
            .set_text(&metadata.gaussian_falloffs(self.axis_id));
        self.tf_hamming_like_starts
            .set_text(metadata.hamming_like_starts(self.axis_id));
        self.signal_listeners();
    }
    /// Java `signalListeners`; native callback dispatch is a frontend boundary.
    fn signal_listeners(&mut self) {
        self.layout.listener_signal_count = self.layout.listener_signal_count.saturating_add(1);
    }
    /// Java `setParameters(MultifiltSetupParam)`.
    pub fn set_parameters<P: MultifiltSetupParam>(&mut self, param: &P) {
        if param.is_fake_sirt_iterations() {
            self.rb_fake_sirt_iterations.set_selected(true);
            self.tf_fake_sirt_iterations
                .set_text(param.fake_sirt_iterations());
        } else if param.is_exact_object_sizes() {
            self.rb_exact_object_sizes.set_selected(true);
            self.tf_exact_object_sizes
                .set_text(param.exact_object_sizes());
        } else if param.is_gaussian_cutoffs() || param.is_gaussian_falloffs() {
            self.rb_gaussian.set_selected(true);
            self.tf_gaussian_cutoffs.set_text(param.gaussian_cutoffs());
            self.ltf_gaussian_falloffs
                .set_text(&param.gaussian_falloffs());
        } else if param.is_hamming_like_starts() {
            self.rb_hamming_like_starts.set_selected(true);
            self.tf_hamming_like_starts
                .set_text(param.hamming_like_starts());
        }
        self.ltf_width_in_x.set_text(&param.width_in_x());
        self.ltf_shift_in_x.set_text(&param.shift_in_x());
        self.ltf_size_in_y.set_text(&param.size_in_y());
        self.ltf_shift_in_y.set_text(&param.shift_in_y());
        self.ltf_thickness_in_z.set_text(&param.thickness_in_z());
        self.ltf_shift_in_depth.set_text(&param.shift_in_depth());
        self.update_display();
        self.signal_listeners();
    }
    /// Java `getParameters(MultifiltSetupParam,boolean)`.
    pub fn get_parameters<P: MultifiltSetupParam>(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            if self.rb_fake_sirt_iterations.is_selected() {
                param.set_fake_sirt_iterations(
                    self.tf_fake_sirt_iterations
                        .get_text_validated(do_validation)
                        .map_err(FieldValidationFailedException)?,
                );
            } else {
                param.reset_fake_sirt_iterations();
            }
            if self.rb_exact_object_sizes.is_selected() {
                param.set_exact_object_sizes(
                    self.tf_exact_object_sizes
                        .get_text_validated(do_validation)
                        .map_err(FieldValidationFailedException)?,
                );
            } else {
                param.reset_exact_object_sizes();
            }
            if self.rb_gaussian.is_selected() {
                let cutoffs = self
                    .tf_gaussian_cutoffs
                    .get_text_validated(do_validation)
                    .map_err(FieldValidationFailedException)?;
                let falloffs = self
                    .ltf_gaussian_falloffs
                    .get_text_validated(do_validation)?;
                if do_validation && cutoffs.split(',').count() != falloffs.split(',').count() {
                    return Err(FieldValidationFailedException(
                        "Gaussian cutoffs and falloffs must have equal array lengths".into(),
                    ));
                }
                param.set_gaussian_cutoffs(cutoffs);
                param.set_gaussian_falloffs(falloffs);
            } else {
                param.reset_gaussian_cutoffs();
                param.reset_gaussian_falloffs();
            }
            if self.rb_hamming_like_starts.is_selected() {
                param.set_hamming_like_starts(
                    self.tf_hamming_like_starts
                        .get_text_validated(do_validation)
                        .map_err(FieldValidationFailedException)?,
                );
            } else {
                param.reset_hamming_like_starts();
            }
            param.set_width_in_x(self.ltf_width_in_x.get_text_validated(do_validation)?);
            param.set_shift_in_x(self.ltf_shift_in_x.get_text_validated(do_validation)?);
            param.set_size_in_y(self.ltf_size_in_y.get_text_validated(do_validation)?);
            param.set_shift_in_y(self.ltf_shift_in_y.get_text_validated(do_validation)?);
            param.set_thickness_in_z(self.ltf_thickness_in_z.get_text_validated(do_validation)?);
            param.set_shift_in_depth(self.ltf_shift_in_depth.get_text_validated(do_validation)?);
            Ok(())
        })();
        result.is_ok()
    }
    /// Java `setTooltips`; autodoc lookup stays at the autodoc I/O boundary.
    fn set_tooltips(&mut self) {
        self.rb_fake_sirt_iterations
            .set_tool_tip_text(Some("Use SIRT-like filter"));
        self.rb_exact_object_sizes
            .set_tool_tip_text(Some("Use exact filter"));
        self.rb_gaussian
            .set_tool_tip_text(Some("Use Gaussian filter"));
        self.rb_hamming_like_starts
            .set_tool_tip_text(Some("Use Hamming-like filter"));
        self.btn_multifilt_setup.set_tool_tip_text(Some(
            "Run multifiltsetup, and then run the resulting .com files with processchunks.",
        ));
        self.btn_3dmod_multifilt_setup.set_tool_tip_text(Some(
            "Opens a file chooser for picking filter trial output to open together in 3dmod",
        ));
        self.layout.tooltip_initialized = true;
    }
    fn selected_filter_type(&self) -> Option<MultifiltFilterType> {
        [
            MultifiltFilterType::FakeSirtIterations,
            MultifiltFilterType::ExactObjectSizes,
            MultifiltFilterType::Gaussian,
            MultifiltFilterType::HammingLikeStarts,
        ]
        .into_iter()
        .find(|filter| match filter {
            MultifiltFilterType::FakeSirtIterations => self.rb_fake_sirt_iterations.is_selected(),
            MultifiltFilterType::ExactObjectSizes => self.rb_exact_object_sizes.is_selected(),
            MultifiltFilterType::Gaussian => self.rb_gaussian.is_selected(),
            MultifiltFilterType::HammingLikeStarts => self.rb_hamming_like_starts.is_selected(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_default_and_display_follow_selected_filter() {
        let mut panel = MultifiltPanel::get_instance(AxisID::Only, "TomogramGeneration");
        assert!(panel.rb_fake_sirt_iterations.is_selected());
        assert!(panel.tf_fake_sirt_iterations.text_field.enabled);
        panel.rb_gaussian.set_selected(true);
        panel.update_display();
        assert!(panel.tf_gaussian_cutoffs.text_field.enabled);
        assert!(panel.ltf_gaussian_falloffs.enabled);
        assert!(!panel.tf_fake_sirt_iterations.text_field.enabled);
    }
    #[test]
    fn source_file_filter_array_includes_all_filter_slot() {
        let panel = MultifiltPanel::get_instance(AxisID::First, "TomogramGeneration");
        assert_eq!(
            panel.file_filter_array.len(),
            MultifiltFilterType::ARRAY_SIZE
        );
        assert_eq!(
            panel.file_filter_array[4].as_ref().unwrap().filter_type,
            None
        );
    }
}
