//! `IMOD/Etomo/src/etomo/ui/swing/TiltAnglePanel.java`.
//!
//! The three small Swing panels, their `ButtonGroup`, action dispatch, and
//! presentation styling are represented by the source-visible fields below.
//! `SetupDialog`, `ApplicationManager`, directive files, and flag painting are
//! direct boundaries: their values are retained in the panel context rather
//! than replaced by an unrelated policy.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::radio_button::{RadioButton, RadioButtonGroup};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::extension::{self, Extension};
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::image_filename_style::ImageFilenameStyle;
use crate::imod::etomo::ui::field_type::FieldType;

/// Java `TiltAngleType`, at the still-direct type dependency boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiltAngleType {
    Extract,
    Range,
    File,
}
impl TiltAngleType {
    pub fn descr(self) -> &'static str {
        match self {
            Self::Extract => "Extract tilt angles from image headers",
            Self::Range => "Specify the starting angle and step (degrees)",
            Self::File => "Tilt angles in existing rawtlt file",
        }
    }
}

/// Java `TiltAngleSpec` value used by `updateTemplateValues`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TiltAngleSpec {
    pub tilt_angle_type: TiltAngleType,
}

/// Direct `DirectiveFileCollection` boundary used by this panel.
#[derive(Clone, Debug, Default)]
pub struct DirectiveFileCollection {
    pub tilt_angle_spec_first: Option<TiltAngleSpec>,
    pub tilt_angle_spec_second: Option<TiltAngleSpec>,
    pub tilt_angle_spec_only: Option<TiltAngleSpec>,
}
impl DirectiveFileCollection {
    pub fn contains_tilt_angle_spec(&self, axis_id: AxisID) -> bool {
        self.tilt_angle_spec(axis_id).is_some()
    }
    pub fn tilt_angle_spec(&self, axis_id: AxisID) -> Option<TiltAngleSpec> {
        match axis_id {
            AxisID::First => self.tilt_angle_spec_first,
            AxisID::Second => self.tilt_angle_spec_second,
            AxisID::Only => self.tilt_angle_spec_only,
        }
    }
}

/// Java `SetupDialog` values reached by `updateDisplay`.
#[derive(Clone, Debug)]
pub struct SetupDialogBoundary {
    pub dataset_name: Option<String>,
    pub axis_type: AxisType,
    pub directory: PathBuf,
    pub remove_exclude_views_msg_first: bool,
    pub remove_exclude_views_msg_second: bool,
    pub remove_exclude_views_msg_only: bool,
}
impl SetupDialogBoundary {
    pub fn is_remove_exclude_views_msg(&self, axis_id: AxisID) -> bool {
        match axis_id {
            AxisID::First => self.remove_exclude_views_msg_first,
            AxisID::Second => self.remove_exclude_views_msg_second,
            AxisID::Only => self.remove_exclude_views_msg_only,
        }
    }
}

/// The two `MetaData` image naming values used by this source unit.
#[derive(Clone, Copy, Debug)]
pub struct MetaDataBoundary {
    pub image_filename_style: ImageFilenameStyle,
    pub raw_image_stack_extension: &'static Extension,
}

/// Java `RadioEbutton` state particular to its warning flag extension.
#[derive(Clone, Debug)]
pub struct RadioEbuttonBoundary {
    pub radio_button: RadioButton,
    pub warning_enabled: Option<bool>,
    pub action_listener_count: usize,
}
impl RadioEbuttonBoundary {
    pub fn get_instance(label: &str, group: Rc<RefCell<RadioButtonGroup>>) -> Self {
        Self {
            radio_button: RadioButton::new_in_group(label, group),
            warning_enabled: None,
            action_listener_count: 0,
        }
    }
    pub fn set_label(&mut self, label: &str) {
        self.radio_button.set_text(label);
    }
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }
    pub fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }
    pub fn set_selected(&mut self, selected: bool) {
        self.radio_button.set_selected(selected);
    }
    pub fn is_enabled(&self) -> bool {
        self.radio_button.is_enabled()
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.radio_button.set_enabled(enabled);
    }
    pub fn get_label(&self) -> &str {
        self.radio_button.get_text()
    }
    pub fn checkpoint(&mut self) {
        self.radio_button.checkpoint();
    }
    pub fn is_checkpoint_value(&self) -> bool {
        self.radio_button.is_checkpoint_value()
    }
    pub fn enable_warning(&mut self, value: bool) {
        self.warning_enabled = Some(value);
    }
    pub fn disable_warning(&mut self) {
        if self.warning_enabled.is_some() {
            self.warning_enabled = Some(false);
        }
    }
    pub fn set_tooltip(&mut self, tooltip: &str) {
        self.radio_button.set_tooltip(Some(tooltip));
    }
}

/// Java `TiltAnglePanelExpert` action target.
pub trait TiltAnglePanelExpert {
    fn set_radio_button_state(&mut self, event: TiltAnglePanelActionEvent);
}

/// The three Java `ActionEvent` origins which its private listener forwards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TiltAnglePanelActionEvent {
    Extract,
    File,
    Specify,
}

/// Java private `TiltAngleDialogListener`.  Its retained expert reference is
/// represented by the caller-owned Rust expert; Swing's event object is reduced
/// to the source-relevant radio-button origin above.
pub struct TiltAngleDialogListener;
impl TiltAngleDialogListener {
    pub fn action_performed<E: TiltAnglePanelExpert>(
        expert: &mut E,
        event: TiltAnglePanelActionEvent,
    ) {
        expert.set_radio_button_state(event);
    }
}

/// Source-owned JPanel state; actual layout/painting remains the Swing boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TiltAnglePanelLayout {
    pub source_enabled: bool,
    pub angle_enabled: bool,
    pub source_tooltip: Option<String>,
    pub exclude_views_message: String,
    pub angle_has_min: bool,
    pub angle_has_gap_x10: bool,
    pub angle_has_step: bool,
    pub source_children: Vec<&'static str>,
}

/// Java `TiltAnglePanel`.
pub struct TiltAnglePanel {
    pub pnl_source: TiltAnglePanelLayout,
    pub bg_source: Rc<RefCell<RadioButtonGroup>>,
    pub rb_extract: RadioButton,
    pub pnl_angle: TiltAnglePanelLayout,
    pub rb_specify: RadioEbuttonBoundary,
    pub ltf_min: LabeledTextField,
    pub ltf_step: LabeledTextField,
    pub rb_file: RadioEbuttonBoundary,
    pub expert_listener_count: usize,
    pub axis_id: AxisID,
    /// Java `parent`, set by `setParent`.
    pub parent: Option<SetupDialogBoundary>,
    /// Java `manager.getMetaData()` boundary.
    pub meta_data: Option<MetaDataBoundary>,
}

impl TiltAnglePanel {
    /// Java package-private constructor.
    pub fn new(axis_id: AxisID) -> Self {
        let bg_source = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut rb_extract =
            RadioButton::new_in_group(TiltAngleType::Extract.descr(), bg_source.clone());
        let mut rb_specify =
            RadioEbuttonBoundary::get_instance(TiltAngleType::Range.descr(), bg_source.clone());
        let mut rb_file =
            RadioEbuttonBoundary::get_instance(TiltAngleType::File.descr(), bg_source.clone());
        rb_extract.add_action_listener();
        rb_specify.add_action_listener();
        rb_file.add_action_listener();
        Self {
            pnl_source: TiltAnglePanelLayout {
                source_enabled: true,
                source_children: vec![
                    "excludeViewsMsg",
                    "extract",
                    "specify",
                    "angle",
                    "file",
                    "file",
                ],
                ..Default::default()
            },
            bg_source,
            rb_extract,
            pnl_angle: TiltAnglePanelLayout {
                source_enabled: true,
                angle_enabled: true,
                angle_has_min: true,
                angle_has_gap_x10: true,
                angle_has_step: true,
                ..Default::default()
            },
            rb_specify,
            ltf_min: LabeledTextField::new(FieldType::FloatingPoint, "Starting angle:"),
            ltf_step: LabeledTextField::new(FieldType::FloatingPoint, "Increment:"),
            rb_file,
            expert_listener_count: 3,
            axis_id,
            parent: None,
            meta_data: None,
        }
    }

    /// Java `msgExcludeViewsSucceeded`.
    pub fn msg_exclude_views_succeeded(&mut self) {
        if self.rb_specify.is_selected() {
            self.rb_file.set_selected(true);
        }
        self.update_display();
    }
    /// Java `setParent`.
    pub fn set_parent(&mut self, parent: SetupDialogBoundary) {
        self.parent = Some(parent);
    }

    /// Java `updateDisplay`, including the metadata-not-yet-created filename branch.
    pub fn update_display(&mut self) {
        let specify = self.rb_specify.is_selected() && self.rb_specify.is_enabled();
        self.ltf_min.set_enabled(specify);
        self.ltf_step.set_enabled(specify);
        let parent = self
            .parent
            .as_ref()
            .expect("TiltAnglePanel.updateDisplay requires setParent");
        let Some(dataset_name) = parent.dataset_name.as_deref() else {
            self.rb_specify.disable_warning();
            self.rb_file.set_label(TiltAngleType::File.descr());
            self.rb_file.disable_warning();
            return;
        };
        let mut cur_axis_id = self.axis_id;
        if cur_axis_id != AxisID::Second {
            cur_axis_id = if parent.axis_type == AxisType::DualAxis {
                AxisID::First
            } else {
                AxisID::Only
            };
        }
        let (image_filename_style, raw_stack_extension) = self.meta_data
            .map(|metadata| (metadata.image_filename_style, metadata.raw_image_stack_extension))
            .unwrap_or_else(|| {
                let image_file_metadata = crate::imod::etomo::r#type::image_file_meta_data::ImageFileMetaData::get_temp_instance();
                (image_file_metadata.get_image_filename_style(), image_file_metadata.get_default_raw_image_stack_extension())
            });
        let raw_tilt_angle_name = file_type::CLASS
            .raw_tilt_angles
            .derive_file_name(
                Some(dataset_name),
                Some(parent.axis_type),
                Some(cur_axis_id),
                Some(image_filename_style),
                Some(raw_stack_extension),
            )
            .expect("RAW_TILT_ANGLES derives a file name");
        let exists = parent.directory.join(raw_tilt_angle_name).exists();
        if exists || parent.is_remove_exclude_views_msg(self.axis_id) {
            self.rb_specify.enable_warning(true);
        } else {
            self.rb_specify.disable_warning();
        }
        if exists {
            self.rb_file
                .set_label(&format!("{} (File was found)", TiltAngleType::File.descr()));
            self.rb_file.disable_warning();
        } else {
            self.rb_file.set_label(&format!(
                "{} (File was not found)",
                TiltAngleType::File.descr()
            ));
            self.rb_file.enable_warning(true);
        }
    }
    /// Java `checkpoint`.
    pub fn checkpoint(&mut self) {
        self.rb_extract.checkpoint();
        self.rb_file.checkpoint();
        self.rb_specify.checkpoint();
    }
    /// Java `updateTemplateValues`.
    pub fn update_template_values(
        &mut self,
        directive_file_collection: &DirectiveFileCollection,
        axis_id: AxisID,
    ) {
        if let Some(tilt_angle_spec) = directive_file_collection.tilt_angle_spec(axis_id) {
            if tilt_angle_spec.tilt_angle_type == TiltAngleType::Extract {
                self.rb_extract.set_selected(true);
            } else if tilt_angle_spec.tilt_angle_type == TiltAngleType::File {
                self.rb_file.set_selected(true);
            }
        } else if self.rb_extract.is_checkpoint_value() {
            self.rb_extract.set_selected(true);
        } else if self.rb_file.is_checkpoint_value() {
            self.rb_file.set_selected(true);
        } else if self.rb_specify.is_checkpoint_value() {
            self.rb_specify.set_selected(true);
        } else {
            self.rb_extract.set_selected(false);
            self.rb_file.set_selected(false);
            self.rb_specify.set_selected(false);
        }
    }
    /// Java `getComponent`.
    pub fn get_component(&self) -> &TiltAnglePanelLayout {
        &self.pnl_source
    }
    pub fn set_file(&mut self, input: bool) {
        self.rb_file.set_selected(input);
    }
    pub fn set_extract(&mut self, input: bool) {
        self.rb_extract.set_selected(input);
    }
    pub fn set_specify(&mut self, input: bool) {
        self.rb_specify.set_selected(input);
    }
    pub fn set_min(&mut self, input: f64) {
        self.ltf_min.set_text_number(input);
    }
    pub fn set_step(&mut self, input: f64) {
        self.ltf_step.set_text_number(input);
    }
    pub fn set_min_enabled(&mut self, enable: bool) {
        self.ltf_min.set_enabled(enable);
    }
    pub fn set_step_enabled(&mut self, enable: bool) {
        self.ltf_step.set_enabled(enable);
    }
    pub fn is_extract_selected(&self) -> bool {
        self.rb_extract.is_selected()
    }
    pub fn is_specify_selected(&self) -> bool {
        self.rb_specify.is_selected()
    }
    pub fn is_file_selected(&self) -> bool {
        self.rb_file.is_selected()
    }
    pub fn get_min(&self) -> String {
        self.ltf_min.get_text()
    }
    pub fn get_min_validated(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_min.get_text_validated(do_validation)
    }
    pub fn get_step(&self) -> String {
        self.ltf_step.get_text()
    }
    pub fn get_step_validated(
        &self,
        do_validation: bool,
    ) -> Result<String, FieldValidationFailedException> {
        self.ltf_step.get_text_validated(do_validation)
    }
    pub fn set_source_enabled(&mut self, enable: bool) {
        self.pnl_source.source_enabled = enable;
    }
    pub fn set_angle_enabled(&mut self, enable: bool) {
        self.pnl_angle.angle_enabled = enable;
    }
    pub fn set_extract_enabled(&mut self, enable: bool) {
        self.rb_extract.set_enabled(enable);
    }
    pub fn set_file_enabled(&mut self, enable: bool) {
        self.rb_file.set_enabled(enable);
    }
    pub fn set_specify_enabled(&mut self, enable: bool) {
        self.rb_specify.set_enabled(enable);
    }
    pub fn set_source_tooltip(&mut self, tooltip: &str) {
        self.pnl_source.source_tooltip = Some(tooltip.to_owned());
    }
    pub fn set_extract_tooltip(&mut self, tooltip: &str) {
        self.rb_extract.set_tool_tip_text(Some(tooltip));
    }
    pub fn set_specify_tooltip(&mut self, tooltip: &str) {
        self.rb_specify.set_tooltip(tooltip);
    }
    pub fn set_min_tooltip(&mut self, tooltip: &str) {
        self.ltf_min.set_tool_tip_text(Some(tooltip));
    }
    pub fn set_step_tooltip(&mut self, tooltip: &str) {
        self.ltf_step.set_tool_tip_text(Some(tooltip));
    }
    pub fn set_file_tooltip(&mut self, tooltip: &str) {
        self.rb_file.set_tooltip(tooltip);
    }
    pub fn get_specify(&self) -> &str {
        self.rb_specify.get_label()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn display_derives_rawtlt_and_follows_file_warning_paths() {
        let temp =
            std::env::temp_dir().join(format!("imod-rs-tilt-angle-panel-{}", std::process::id()));
        fs::create_dir_all(&temp).unwrap();
        let mut panel = TiltAnglePanel::new(AxisID::Only);
        panel.set_parent(SetupDialogBoundary {
            dataset_name: Some("sample".into()),
            axis_type: AxisType::SingleAxis,
            directory: temp.clone(),
            remove_exclude_views_msg_first: false,
            remove_exclude_views_msg_second: false,
            remove_exclude_views_msg_only: false,
        });
        panel.set_specify(true);
        panel.update_display();
        assert!(panel.ltf_min.is_enabled());
        assert_eq!(panel.rb_file.warning_enabled, Some(true));
        let name = file_type::CLASS
            .raw_tilt_angles
            .derive_file_name(
                Some("sample"),
                Some(AxisType::SingleAxis),
                Some(AxisID::Only),
                Some(ImageFilenameStyle::Old),
                Some(&extension::CLASS.st),
            )
            .unwrap();
        fs::write(temp.join(name), []).unwrap();
        panel.update_display();
        assert!(panel.rb_file.get_label().ends_with("(File was found)"));
        assert_eq!(panel.rb_specify.warning_enabled, Some(true));
        let _ = fs::remove_dir_all(temp);
    }

    #[test]
    fn checkpoint_template_and_listener_follow_source() {
        let mut panel = TiltAnglePanel::new(AxisID::First);
        panel.set_extract(true);
        panel.checkpoint();
        panel.set_file(true);
        panel.update_template_values(&DirectiveFileCollection::default(), AxisID::First);
        assert!(panel.is_extract_selected());
        panel.update_template_values(
            &DirectiveFileCollection {
                tilt_angle_spec_first: Some(TiltAngleSpec {
                    tilt_angle_type: TiltAngleType::File,
                }),
                ..Default::default()
            },
            AxisID::First,
        );
        assert!(panel.is_file_selected());
        struct Expert(Option<TiltAnglePanelActionEvent>);
        impl TiltAnglePanelExpert for Expert {
            fn set_radio_button_state(&mut self, event: TiltAnglePanelActionEvent) {
                self.0 = Some(event);
            }
        }
        let mut expert = Expert(None);
        TiltAngleDialogListener::action_performed(&mut expert, TiltAnglePanelActionEvent::Specify);
        assert_eq!(expert.0, Some(TiltAnglePanelActionEvent::Specify));
    }
}
