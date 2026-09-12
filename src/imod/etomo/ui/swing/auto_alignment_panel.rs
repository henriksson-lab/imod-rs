//! `IMOD/Etomo/src/etomo/ui/swing/AutoAlignmentPanel.java`.
//!
//! Swing layout and the `AutoAlignmentController`/autodoc services are native
//! UI and manager boundaries.  This module keeps the Java panel's field state,
//! parameter transfer, action dispatch, and enablement policy intact.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;

/// Java `TransformChooserPanel` transform selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AlignTransform {
    #[default]
    Search,
    Existing,
}

/// The `AutoAlignmentMetaData` calls made by this source unit.
pub trait AutoAlignmentMetaData {
    fn sigma_low_frequency(&self) -> Option<String>;
    fn set_sigma_low_frequency(&mut self, value: String);
    fn set_sigma_low_frequency_enabled(&mut self, value: bool);
    fn cutoff_high_frequency(&self) -> Option<String>;
    fn set_cutoff_high_frequency(&mut self, value: String);
    fn set_cutoff_high_frequency_enabled(&mut self, value: bool);
    fn sigma_high_frequency(&self) -> Option<String>;
    fn set_sigma_high_frequency(&mut self, value: String);
    fn set_sigma_high_frequency_enabled(&mut self, value: bool);
    fn align_transform(&self) -> AlignTransform;
    fn set_align_transform(&mut self, value: AlignTransform);
    fn find_warping(&self) -> bool;
    fn set_find_warping(&mut self, value: bool);
    fn warp_patch_size_x(&self) -> String;
    fn set_warp_patch_size_x(&mut self, value: String);
    fn warp_patch_size_y(&self) -> String;
    fn set_warp_patch_size_y(&mut self, value: String);
    fn boundary_model(&self) -> bool;
    fn set_boundary_model(&mut self, value: bool);
    fn shift_limits_for_warp_x(&self) -> String;
    fn set_shift_limits_for_warp_x(&mut self, value: String);
    fn shift_limits_for_warp_y(&self) -> String;
    fn set_shift_limits_for_warp_y(&mut self, value: String);
    fn pre_cross_correlation(&self) -> bool;
    fn set_pre_cross_correlation(&mut self, value: bool);
    fn skip_sections_from_1(&self) -> String;
    fn set_skip_sections_from_1(&mut self, value: String);
    fn edge_to_ignore(&self) -> Option<String>;
    fn set_edge_to_ignore(&mut self, value: String);
    fn reduce_by_binning(&self) -> Option<i32>;
    fn set_reduce_by_binning(&mut self, value: i32);
    fn midas_binning(&self) -> Option<i32>;
    fn set_midas_binning(&mut self, value: i32);
    fn sobel_filter(&self) -> bool;
    fn set_sobel_filter(&mut self, value: bool);
}

/// `XfalignParam` calls owned by this panel.
pub trait XfalignParam {
    fn set_pre_cross_correlation(&mut self, value: bool);
    fn set_skip_sections_from_1(&mut self, value: String);
    fn reset_skip_sections_from_1(&mut self);
    fn set_edge_to_ignore(&mut self, value: String);
    fn reset_edge_to_ignore(&mut self);
    fn set_reduce_by_binning(&mut self, value: i32);
    fn reset_reduce_by_binning(&mut self);
    fn set_warp_patch_size(&mut self, x: String, y: String);
    fn reset_warp_patch_size(&mut self);
    fn set_boundary_model(&mut self, value: bool);
    fn reset_boundary_model(&mut self);
    fn set_shift_limits_for_warp(&mut self, x: String, y: String);
    fn reset_shift_limits_for_warp(&mut self);
    fn set_sobel_filter(&mut self, value: bool);
    fn reset_sobel_filter(&mut self);
}

/// `MidasParam` call owned by this panel.
pub trait MidasParam {
    fn set_binning(&mut self, value: i32);
}

/// Direct `AutoAlignmentController` calls triggered by Java `action`.
pub trait AutoAlignmentController {
    fn xfalign_initial(&mut self, join_interface: bool);
    fn midas_sample(&mut self, label: Option<String>);
    fn xfalign_refine(&mut self, join_interface: bool, label: Option<String>);
    fn revert_xf_file_to_midas(&mut self);
    fn revert_xf_file_to_empty(&mut self);
    fn imod_boundary_model(&mut self, options: Option<&Run3dmodMenuOptions>);
}

/// Concrete data holder for tests and a Rust UI adapter.  The production
/// metadata class remains the explicit translated-type boundary above.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AutoAlignmentPanelMetaData {
    pub sigma_low_frequency: Option<String>,
    pub sigma_low_frequency_enabled: bool,
    pub cutoff_high_frequency: Option<String>,
    pub cutoff_high_frequency_enabled: bool,
    pub sigma_high_frequency: Option<String>,
    pub sigma_high_frequency_enabled: bool,
    pub align_transform: AlignTransform,
    pub find_warping: bool,
    pub warp_patch_size_x: String,
    pub warp_patch_size_y: String,
    pub boundary_model: bool,
    pub shift_limits_for_warp_x: String,
    pub shift_limits_for_warp_y: String,
    pub pre_cross_correlation: bool,
    pub skip_sections_from_1: String,
    pub edge_to_ignore: Option<String>,
    pub reduce_by_binning: Option<i32>,
    pub midas_binning: Option<i32>,
    pub sobel_filter: bool,
}
impl AutoAlignmentMetaData for AutoAlignmentPanelMetaData {
    fn sigma_low_frequency(&self) -> Option<String> {
        self.sigma_low_frequency.clone()
    }
    fn set_sigma_low_frequency(&mut self, v: String) {
        self.sigma_low_frequency = Some(v)
    }
    fn set_sigma_low_frequency_enabled(&mut self, v: bool) {
        self.sigma_low_frequency_enabled = v
    }
    fn cutoff_high_frequency(&self) -> Option<String> {
        self.cutoff_high_frequency.clone()
    }
    fn set_cutoff_high_frequency(&mut self, v: String) {
        self.cutoff_high_frequency = Some(v)
    }
    fn set_cutoff_high_frequency_enabled(&mut self, v: bool) {
        self.cutoff_high_frequency_enabled = v
    }
    fn sigma_high_frequency(&self) -> Option<String> {
        self.sigma_high_frequency.clone()
    }
    fn set_sigma_high_frequency(&mut self, v: String) {
        self.sigma_high_frequency = Some(v)
    }
    fn set_sigma_high_frequency_enabled(&mut self, v: bool) {
        self.sigma_high_frequency_enabled = v
    }
    fn align_transform(&self) -> AlignTransform {
        self.align_transform
    }
    fn set_align_transform(&mut self, v: AlignTransform) {
        self.align_transform = v
    }
    fn find_warping(&self) -> bool {
        self.find_warping
    }
    fn set_find_warping(&mut self, v: bool) {
        self.find_warping = v
    }
    fn warp_patch_size_x(&self) -> String {
        self.warp_patch_size_x.clone()
    }
    fn set_warp_patch_size_x(&mut self, v: String) {
        self.warp_patch_size_x = v
    }
    fn warp_patch_size_y(&self) -> String {
        self.warp_patch_size_y.clone()
    }
    fn set_warp_patch_size_y(&mut self, v: String) {
        self.warp_patch_size_y = v
    }
    fn boundary_model(&self) -> bool {
        self.boundary_model
    }
    fn set_boundary_model(&mut self, v: bool) {
        self.boundary_model = v
    }
    fn shift_limits_for_warp_x(&self) -> String {
        self.shift_limits_for_warp_x.clone()
    }
    fn set_shift_limits_for_warp_x(&mut self, v: String) {
        self.shift_limits_for_warp_x = v
    }
    fn shift_limits_for_warp_y(&self) -> String {
        self.shift_limits_for_warp_y.clone()
    }
    fn set_shift_limits_for_warp_y(&mut self, v: String) {
        self.shift_limits_for_warp_y = v
    }
    fn pre_cross_correlation(&self) -> bool {
        self.pre_cross_correlation
    }
    fn set_pre_cross_correlation(&mut self, v: bool) {
        self.pre_cross_correlation = v
    }
    fn skip_sections_from_1(&self) -> String {
        self.skip_sections_from_1.clone()
    }
    fn set_skip_sections_from_1(&mut self, v: String) {
        self.skip_sections_from_1 = v
    }
    fn edge_to_ignore(&self) -> Option<String> {
        self.edge_to_ignore.clone()
    }
    fn set_edge_to_ignore(&mut self, v: String) {
        self.edge_to_ignore = Some(v)
    }
    fn reduce_by_binning(&self) -> Option<i32> {
        self.reduce_by_binning
    }
    fn set_reduce_by_binning(&mut self, v: i32) {
        self.reduce_by_binning = Some(v)
    }
    fn midas_binning(&self) -> Option<i32> {
        self.midas_binning
    }
    fn set_midas_binning(&mut self, v: i32) {
        self.midas_binning = Some(v)
    }
    fn sobel_filter(&self) -> bool {
        self.sobel_filter
    }
    fn set_sobel_filter(&mut self, v: bool) {
        self.sobel_filter = v
    }
}

/// Java `AutoAlignmentPanel`, including source-visible layout and listener state.
pub struct AutoAlignmentPanel {
    pub join_interface: bool,
    pub root_focusable: bool,
    pub transform: AlignTransform,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub sigma_low_frequency: LabeledTextField,
    pub cutoff_high_frequency: LabeledTextField,
    pub sigma_high_frequency: LabeledTextField,
    pub initial_auto_alignment: MultiLineButton,
    pub midas: MultiLineButton,
    pub refine_auto_alignment: MultiLineButton,
    pub revert_to_midas: MultiLineButton,
    pub revert_to_empty: MultiLineButton,
    pub reduce_by_binning: LabeledSpinner,
    pub skip_sections_from_1: LabeledTextField,
    pub pre_cross_correlation: CheckBox,
    pub edge_to_ignore: LabeledTextField,
    pub midas_binning: LabeledSpinner,
    pub find_warping: Option<CheckBox>,
    pub warp_patch_size_x: Option<LabeledTextField>,
    pub warp_patch_size_y: Option<LabeledTextField>,
    pub boundary_model: Option<CheckBox>,
    pub boundary_model_button: Option<MultiLineButton>,
    pub shift_limits_for_warp_x: Option<LabeledTextField>,
    pub shift_limits_for_warp_y: Option<LabeledTextField>,
    pub sobel_filter: Option<CheckBox>,
}

impl AutoAlignmentPanel {
    pub fn get_join_instance() -> Self {
        let mut value = Self::new(true);
        value.set_tooltips();
        value
    }
    pub fn get_serial_sections_instance() -> Self {
        let mut value = Self::new(false);
        value.set_tooltips();
        value
    }
    fn new(join_interface: bool) -> Self {
        let mut value = Self {
            join_interface,
            root_focusable: true,
            transform: AlignTransform::Search,
            listener_count: 0,
            tooltip_initialized: false,
            sigma_low_frequency: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Sigma for low-frequency filter: ",
            ),
            cutoff_high_frequency: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Cutoff for high-frequency filter: ",
            ),
            sigma_high_frequency: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Sigma for high-frequency filter: ",
            ),
            initial_auto_alignment: MultiLineButton::new_with_label(Some("Initial Auto Alignment")),
            midas: MultiLineButton::new_with_label(Some("Midas")),
            refine_auto_alignment: MultiLineButton::new_with_label(Some(
                "Refine with Auto Alignment",
            )),
            revert_to_midas: MultiLineButton::new_with_label(Some(
                "Revert Auto Alignment to Midas",
            )),
            revert_to_empty: MultiLineButton::new_with_label(Some("Revert to No Transforms")),
            reduce_by_binning: LabeledSpinner::get_defaulted_instance(
                "Binning in search: ",
                2,
                1,
                50,
                1,
                1,
            ),
            skip_sections_from_1: LabeledTextField::new(
                FieldType::IntegerList,
                "Sections to skip: ",
            ),
            pre_cross_correlation: CheckBox::new_with_text(
                "Find initial shifts with cross-correlation",
            ),
            edge_to_ignore: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Fraction to ignore on edges: ",
            ),
            midas_binning: LabeledSpinner::get_instance("Binning in Midas: ", 1, 1, 8, 1),
            find_warping: (!join_interface)
                .then(|| CheckBox::new_with_text("Find warping transformations")),
            warp_patch_size_x: (!join_interface).then(|| {
                LabeledTextField::new(FieldType::Integer, "Correlation patch size in X: ")
            }),
            warp_patch_size_y: (!join_interface)
                .then(|| LabeledTextField::new(FieldType::Integer, " Y: ")),
            boundary_model: (!join_interface)
                .then(|| CheckBox::new_with_text("Use boundary model:")),
            boundary_model_button: (!join_interface)
                .then(|| MultiLineButton::new_with_label(Some("Create/View Boundary Model"))),
            shift_limits_for_warp_x: (!join_interface)
                .then(|| LabeledTextField::new(FieldType::Integer, "Limits to shifts in X: ")),
            shift_limits_for_warp_y: (!join_interface)
                .then(|| LabeledTextField::new(FieldType::Integer, " Y: ")),
            sobel_filter: (!join_interface).then(|| CheckBox::new_with_text("Apply Sobel filter")),
        };
        value.create_panel(join_interface);
        value
    }
    fn create_panel(&mut self, join_configuration: bool) {
        self.sigma_low_frequency.set_text("0.0");
        self.cutoff_high_frequency.set_text("0.35");
        self.sigma_high_frequency.set_text("0.05");
        self.edge_to_ignore.set_text("0.05");
        if join_configuration {
            self.reduce_by_binning.set_visible(false);
            self.skip_sections_from_1.set_visible(false);
            self.pre_cross_correlation.set_visible(false);
            self.edge_to_ignore.set_visible(false);
            self.midas_binning.set_visible(false);
        }
        self.update_display();
    }
    pub fn set_controller(&mut self) {
        self.add_listeners();
    }
    fn add_listeners(&mut self) {
        self.listener_count += 5 + usize::from(!self.join_interface) * 4;
        self.initial_auto_alignment.add_action_listener();
        self.midas.add_action_listener();
        self.refine_auto_alignment.add_action_listener();
        self.revert_to_midas.add_action_listener();
        self.revert_to_empty.add_action_listener();
    }
    pub fn get_parameters_metadata<M: AutoAlignmentMetaData>(
        &self,
        metadata: &mut M,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            metadata.set_sigma_low_frequency(
                self.sigma_low_frequency.get_text_validated(do_validation)?,
            );
            metadata.set_sigma_low_frequency_enabled(self.sigma_low_frequency.is_enabled());
            metadata.set_cutoff_high_frequency(
                self.cutoff_high_frequency
                    .get_text_validated(do_validation)?,
            );
            metadata.set_cutoff_high_frequency_enabled(self.cutoff_high_frequency.is_enabled());
            metadata.set_sigma_high_frequency(
                self.sigma_high_frequency
                    .get_text_validated(do_validation)?,
            );
            metadata.set_sigma_high_frequency_enabled(self.sigma_high_frequency.is_enabled());
            metadata.set_align_transform(self.transform);
            if let Some(find) = &self.find_warping {
                metadata.set_find_warping(find.is_selected());
                metadata.set_warp_patch_size_x(self.warp_patch_size_x.as_ref().unwrap().get_text());
                metadata.set_warp_patch_size_y(self.warp_patch_size_y.as_ref().unwrap().get_text());
                metadata.set_boundary_model(self.boundary_model.as_ref().unwrap().is_selected());
                metadata.set_shift_limits_for_warp_x(
                    self.shift_limits_for_warp_x.as_ref().unwrap().get_text(),
                );
                metadata.set_shift_limits_for_warp_y(
                    self.shift_limits_for_warp_y.as_ref().unwrap().get_text(),
                );
            }
            metadata.set_pre_cross_correlation(self.pre_cross_correlation.is_selected());
            metadata.set_skip_sections_from_1(
                self.skip_sections_from_1
                    .get_text_validated(do_validation)?,
            );
            metadata.set_edge_to_ignore(self.edge_to_ignore.get_text_validated(do_validation)?);
            metadata.set_reduce_by_binning(self.reduce_by_binning.get_value());
            metadata.set_midas_binning(self.midas_binning.get_value());
            if let Some(sobel) = &self.sobel_filter {
                metadata.set_sobel_filter(sobel.is_selected());
            }
            Ok(())
        })();
        result.is_ok()
    }
    pub fn set_parameters<M: AutoAlignmentMetaData>(&mut self, metadata: &M) {
        if let Some(v) = metadata.sigma_low_frequency() {
            self.sigma_low_frequency.set_text(&v)
        }
        if let Some(v) = metadata.cutoff_high_frequency() {
            self.cutoff_high_frequency.set_text(&v)
        }
        if let Some(v) = metadata.sigma_high_frequency() {
            self.sigma_high_frequency.set_text(&v)
        }
        self.transform = metadata.align_transform();
        if let Some(find) = &mut self.find_warping {
            find.set_selected(metadata.find_warping());
            self.warp_patch_size_x
                .as_mut()
                .unwrap()
                .set_text(&metadata.warp_patch_size_x());
            self.warp_patch_size_y
                .as_mut()
                .unwrap()
                .set_text(&metadata.warp_patch_size_y());
            self.boundary_model
                .as_mut()
                .unwrap()
                .set_selected(metadata.boundary_model());
            self.shift_limits_for_warp_x
                .as_mut()
                .unwrap()
                .set_text(&metadata.shift_limits_for_warp_x());
            self.shift_limits_for_warp_y
                .as_mut()
                .unwrap()
                .set_text(&metadata.shift_limits_for_warp_y());
        }
        self.pre_cross_correlation
            .set_selected(metadata.pre_cross_correlation());
        self.skip_sections_from_1
            .set_text(&metadata.skip_sections_from_1());
        if let Some(v) = metadata.edge_to_ignore() {
            self.edge_to_ignore.set_text(&v)
        }
        if let Some(v) = metadata.reduce_by_binning() {
            self.reduce_by_binning.set_value_int(v)
        }
        if let Some(v) = metadata.midas_binning() {
            self.midas_binning.set_value_int(v)
        }
        if let Some(sobel) = &mut self.sobel_filter {
            sobel.set_selected(metadata.sobel_filter())
        }
        self.update_display();
    }
    pub fn get_parameters_xfalign<P: XfalignParam>(
        &self,
        param: &mut P,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), FieldValidationFailedException> {
            param.set_pre_cross_correlation(if self.pre_cross_correlation.is_visible() {
                self.pre_cross_correlation.is_selected()
            } else {
                false
            });
            if self.skip_sections_from_1.is_visible() {
                param.set_skip_sections_from_1(
                    self.skip_sections_from_1
                        .get_text_validated(do_validation)?,
                )
            } else {
                param.reset_skip_sections_from_1()
            }
            if self.edge_to_ignore.is_visible() && self.edge_to_ignore.is_enabled() {
                param.set_edge_to_ignore(self.edge_to_ignore.get_text_validated(do_validation)?)
            } else {
                param.reset_edge_to_ignore()
            }
            if self.reduce_by_binning.is_visible() && self.reduce_by_binning.is_enabled() {
                param.set_reduce_by_binning(self.reduce_by_binning.get_value())
            } else {
                param.reset_reduce_by_binning()
            }
            if let Some(find) = &self.find_warping {
                if find.is_selected() {
                    param.set_warp_patch_size(
                        self.warp_patch_size_x
                            .as_ref()
                            .unwrap()
                            .get_text_validated(do_validation)?,
                        self.warp_patch_size_y
                            .as_ref()
                            .unwrap()
                            .get_text_validated(do_validation)?,
                    );
                    param.set_boundary_model(self.boundary_model.as_ref().unwrap().is_selected());
                    param.set_shift_limits_for_warp(
                        self.shift_limits_for_warp_x
                            .as_ref()
                            .unwrap()
                            .get_text_validated(do_validation)?,
                        self.shift_limits_for_warp_y
                            .as_ref()
                            .unwrap()
                            .get_text_validated(do_validation)?,
                    )
                } else {
                    param.reset_warp_patch_size();
                    param.reset_boundary_model();
                    param.reset_shift_limits_for_warp()
                }
            }
            if let Some(sobel) = &self.sobel_filter {
                if sobel.is_enabled() {
                    param.set_sobel_filter(sobel.is_selected())
                } else {
                    param.reset_sobel_filter()
                }
            } else {
                param.reset_sobel_filter()
            }
            Ok(())
        })();
        result.is_ok()
    }
    pub fn get_parameters_midas<P: MidasParam>(&self, param: &mut P) {
        param.set_binning(self.midas_binning.get_value());
    }
    pub fn equals_metadata<M: AutoAlignmentMetaData>(&self, metadata: &M) -> bool {
        metadata.sigma_low_frequency().as_deref() == Some(&self.sigma_low_frequency.get_text())
            && metadata.cutoff_high_frequency().as_deref()
                == Some(&self.cutoff_high_frequency.get_text())
            && metadata.sigma_high_frequency().as_deref()
                == Some(&self.sigma_high_frequency.get_text())
            && self.transform == metadata.align_transform()
            && self.find_warping.as_ref().is_none_or(|v| {
                v.is_selected() == metadata.find_warping()
                    && metadata.warp_patch_size_x()
                        == self.warp_patch_size_x.as_ref().unwrap().get_text()
                    && metadata.warp_patch_size_y()
                        == self.warp_patch_size_y.as_ref().unwrap().get_text()
                    && self.boundary_model.as_ref().unwrap().is_selected()
                        == metadata.boundary_model()
                    && metadata.shift_limits_for_warp_x()
                        == self.shift_limits_for_warp_x.as_ref().unwrap().get_text()
                    && metadata.shift_limits_for_warp_y()
                        == self.shift_limits_for_warp_y.as_ref().unwrap().get_text()
                    && self
                        .sobel_filter
                        .as_ref()
                        .is_none_or(|s| s.is_selected() == metadata.sobel_filter())
            })
    }
    pub fn msg_process_change(&mut self, process_ended: bool) {
        self.midas.set_enabled(process_ended);
        self.revert_to_midas.set_enabled(process_ended);
        self.revert_to_empty.set_enabled(process_ended);
    }
    pub fn action<C: AutoAlignmentController>(
        &mut self,
        command: &str,
        controller: &mut C,
        options: Option<&Run3dmodMenuOptions>,
    ) {
        if self.initial_auto_alignment.get_action_command() == Some(command) {
            self.msg_process_change(false);
            controller.xfalign_initial(self.join_interface)
        } else if self.midas.get_action_command() == Some(command) {
            controller.midas_sample(self.midas.get_quoted_label())
        } else if self.refine_auto_alignment.get_action_command() == Some(command) {
            self.msg_process_change(false);
            controller.xfalign_refine(
                self.join_interface,
                self.refine_auto_alignment.get_quoted_label(),
            )
        } else if self.revert_to_midas.get_action_command() == Some(command) {
            controller.revert_xf_file_to_midas()
        } else if self.revert_to_empty.get_action_command() == Some(command) {
            controller.revert_xf_file_to_empty()
        } else if command == "search" {
            self.update_display()
        } else if self
            .find_warping
            .as_ref()
            .is_some_and(|v| v.get_action_command() == Some(command))
            || self
                .boundary_model
                .as_ref()
                .is_some_and(|v| v.get_action_command() == Some(command))
        {
            self.update_display()
        } else if self
            .boundary_model_button
            .as_ref()
            .is_some_and(|v| v.get_action_command() == Some(command))
        {
            controller.imod_boundary_model(options)
        }
    }
    pub fn update_display(&mut self) {
        let search = self.transform == AlignTransform::Search;
        self.sigma_low_frequency.set_enabled(search);
        self.cutoff_high_frequency.set_enabled(search);
        self.sigma_high_frequency.set_enabled(search);
        if let Some(s) = &mut self.sobel_filter {
            s.set_enabled(search)
        }
        if let Some(find) = &self.find_warping {
            let warping = find.is_selected();
            self.warp_patch_size_x
                .as_mut()
                .unwrap()
                .set_enabled(warping);
            self.warp_patch_size_y
                .as_mut()
                .unwrap()
                .set_enabled(warping);
            self.boundary_model.as_mut().unwrap().set_enabled(warping);
            self.boundary_model_button
                .as_mut()
                .unwrap()
                .set_enabled(warping && self.boundary_model.as_ref().unwrap().is_selected());
            self.shift_limits_for_warp_x
                .as_mut()
                .unwrap()
                .set_enabled(warping);
            self.shift_limits_for_warp_y
                .as_mut()
                .unwrap()
                .set_enabled(warping);
        }
        self.edge_to_ignore.set_enabled(search);
        self.reduce_by_binning.set_enabled(search);
    }
    fn set_tooltips(&mut self) {
        self.tooltip_initialized = true;
        self.sigma_low_frequency.set_tool_tip_text(Some("Sigma of an inverted gaussian for filtering out low frequencies before searching for transformation.  Filter is applied to binned image."));
        self.cutoff_high_frequency.set_tool_tip_text(Some("Starting radius of a gaussian for filtering out high frequencies before searching for transformation.  Filter is applied to binned image."));
        self.sigma_high_frequency.set_tool_tip_text(Some("Sigma of gaussian for filtering out high frequencies before searching for transformation.  Filter is applied to binned image."));
        self.midas_binning
            .set_tool_tip_text(Some("Binning used by Midas."));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn join_hides_search_only_controls() {
        let panel = AutoAlignmentPanel::get_join_instance();
        assert!(!panel.reduce_by_binning.is_visible());
        assert!(panel.find_warping.is_none());
    }
    #[test]
    fn serial_warping_enablement_matches_source() {
        let mut panel = AutoAlignmentPanel::get_serial_sections_instance();
        panel.transform = AlignTransform::Existing;
        panel.find_warping.as_mut().unwrap().set_selected(true);
        panel.boundary_model.as_mut().unwrap().set_selected(true);
        panel.update_display();
        assert!(!panel.sigma_low_frequency.is_enabled());
        assert!(panel.warp_patch_size_x.as_ref().unwrap().is_enabled());
        assert!(panel.boundary_model_button.as_ref().unwrap().is_enabled());
    }
    #[test]
    fn metadata_round_trip_keeps_source_defaults() {
        let panel = AutoAlignmentPanel::get_serial_sections_instance();
        let mut metadata = AutoAlignmentPanelMetaData::default();
        assert!(panel.get_parameters_metadata(&mut metadata, true));
        assert_eq!(metadata.sigma_low_frequency.as_deref(), Some("0.0"));
        assert_eq!(metadata.reduce_by_binning, Some(2));
    }
}
