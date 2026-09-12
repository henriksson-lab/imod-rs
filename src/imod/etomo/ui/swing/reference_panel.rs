//! `IMOD/Etomo/src/etomo/ui/swing/ReferencePanel.java`.
//!
//! Native widget construction, autodoc lookup, path choosing, and the manager
//! remain explicit boundaries.  This unit retains all source-owned state and
//! parameter/action rules.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::file_text_field_interface::FileTextFieldInterface;
use super::file_text_field2::FileTextField2;
use super::labeled_spinner::LabeledSpinner;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::radio_button::{RadioButton, RadioButtonGroup};
use super::reference_parent::ReferenceParent;
use crate::imod::etomo::ui::field_type::FieldType;

pub const TITLE: &str = "Reference";
pub const REFERENCE_FILE_LABEL: &str = "User supplied file: ";
pub const MULTIPARTICLE_BUTTON_LABEL: &str = "FlgFairReference with";
pub const VOLUME_LABEL: &str = "In Volume";

pub trait ReferencePanelManager {
    fn property_user_dir(&self) -> &Path;
}

/// Java `RadioTextField` state observed by `ReferencePanel`.
#[derive(Clone, Debug)]
pub struct RadioTextField {
    pub radio_button: RadioButton,
    pub field: LabeledTextField,
}
impl RadioTextField {
    pub fn set_selected(&mut self, value: bool) {
        self.radio_button.set_selected(value);
    }
    pub fn is_selected(&self) -> bool {
        self.radio_button.is_selected()
    }
    pub fn is_empty(&self) -> bool {
        self.field.is_empty()
    }
    pub fn set_text(&mut self, text: &str) {
        self.field.set_text(text);
    }
    pub fn get_text(&self, validate: bool) -> Result<String, FieldValidationFailedException> {
        self.field.get_text_validated(validate)
    }
    pub fn set_enabled(&mut self, value: bool) {
        self.radio_button.set_enabled(value);
        self.field.set_enabled(value);
    }
    pub fn action_command(&self) -> &str {
        self.radio_button.get_action_command()
    }
    pub fn label(&self) -> &str {
        self.radio_button.get_text()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ComboBox {
    pub items: Vec<i32>,
    pub selected_index: i32,
    pub enabled: bool,
    pub tooltip: Option<String>,
}

/// Calls into `MultiparticleReference.java`; retained as source names rather than a policy replacement.
pub struct MultiparticleReference;
impl MultiparticleReference {
    pub const MIN_LEVEL: i32 = 2;
    pub const MAX_LEVEL: i32 = 10;
    pub const DEFAULT_LEVEL: i32 = 5;
    pub fn get_num_entries() -> i32 {
        Self::MAX_LEVEL - Self::MIN_LEVEL + 1
    }
    pub fn get_particle_count(index: i32) -> i32 {
        2_i32.pow((index.clamp(0, Self::get_num_entries() - 1) + Self::MIN_LEVEL) as u32)
    }
    pub fn get_default_index() -> i32 {
        Self::DEFAULT_LEVEL - Self::MIN_LEVEL
    }
    pub fn convert_level_to_index(level: &str) -> (bool, i32) {
        match level.parse::<i32>() {
            Ok(value) => (
                value >= Self::MIN_LEVEL && value <= Self::MAX_LEVEL,
                value.clamp(Self::MIN_LEVEL, Self::MAX_LEVEL) - Self::MIN_LEVEL,
            ),
            Err(_) => (true, Self::get_default_index()),
        }
    }
    pub fn convert_level_to_index_int(level: i32) -> i32 {
        level.clamp(Self::MIN_LEVEL, Self::MAX_LEVEL) - Self::MIN_LEVEL
    }
    pub fn convert_index_to_level(index: i32) -> String {
        (index + Self::MIN_LEVEL)
            .clamp(Self::MIN_LEVEL, Self::MAX_LEVEL)
            .to_string()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeetMetaData {
    pub reference_volume: String,
    pub reference_particle: String,
    pub reference_file: String,
    pub reference_multiparticle_level: String,
}
pub type ConstPeetMetaData = PeetMetaData;
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatlabParam {
    pub reference_file: Option<String>,
    pub flg_fair_reference: bool,
    pub reference_level: Option<String>,
    pub reference_volume: String,
    pub reference_particle: String,
}

pub struct ReferencePanel {
    pub rtf_particle: RadioTextField,
    pub s_volume: LabeledSpinner,
    pub rb_file: RadioButton,
    pub ftf_file: FileTextField2,
    pub rb_multiparticle: RadioButton,
    pub cmb_multiparticle: ComboBox,
    pub l_multiparticle_enabled: bool,
    pub ltf_volume: LabeledTextField,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub april_fools_background: bool,
    pub component_order: Vec<String>,
}
impl ReferencePanel {
    /// Java constructor, `getInstance`, `createPanel`, `setTooltips`, and `addListeners`.
    pub fn get_instance<M: ReferencePanelManager>(manager: &M, april_fools: bool) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut result = Self {
            rtf_particle: RadioTextField {
                radio_button: RadioButton::new_in_group("Particle ", group.clone()),
                field: LabeledTextField::new(FieldType::Integer, "Particle "),
            },
            s_volume: LabeledSpinner::get_instance("In Volume: ", 1, 1, i32::MAX, 1),
            rb_file: RadioButton::new_in_group(REFERENCE_FILE_LABEL, group.clone()),
            ftf_file: FileTextField2 {
                enabled: true,
                adjusted_field_width: 225,
                use_text_as_file_chooser_dir: true,
                browsing_directory: Some(manager.property_user_dir().into()),
                ..Default::default()
            },
            rb_multiparticle: RadioButton::new_in_group(MULTIPARTICLE_BUTTON_LABEL, group),
            cmb_multiparticle: ComboBox {
                enabled: true,
                ..Default::default()
            },
            l_multiparticle_enabled: true,
            ltf_volume: LabeledTextField::new(FieldType::Integer, "In Volume: "),
            listener_count: 0,
            tooltip_initialized: false,
            april_fools_background: april_fools,
            component_order: vec![
                "border".into(),
                "particle".into(),
                "file".into(),
                "multiparticle".into(),
            ],
        };
        result.create_panel();
        result.set_tooltips();
        result.add_listeners();
        result
    }
    pub fn add_listeners(&mut self) {
        self.listener_count = 3;
    }
    pub fn create_panel(&mut self) {
        self.cmb_multiparticle.items.clear();
        for index in 0..MultiparticleReference::get_num_entries() {
            self.cmb_multiparticle
                .items
                .push(MultiparticleReference::get_particle_count(index));
        }
        self.cmb_multiparticle.selected_index = MultiparticleReference::get_default_index();
    }
    pub fn convert_copied_paths<M: ReferencePanelManager>(
        &mut self,
        manager: &M,
        orig_dataset_dir: &Path,
    ) {
        if !self.ftf_file.is_empty() && !Path::new(&self.ftf_file.text).is_absolute() {
            self.ftf_file.text = orig_dataset_dir
                .join(&self.ftf_file.text)
                .to_string_lossy()
                .into_owned();
            if let Ok(path) =
                Path::new(&self.ftf_file.text).strip_prefix(manager.property_user_dir())
            {
                self.ftf_file.text = path.to_string_lossy().into_owned();
            }
        }
    }
    pub fn is_incorrect_paths(&self) -> bool {
        self.is_reference_file_selected() && !self.ftf_file.is_empty() && !self.ftf_file.exists()
    }
    pub fn fix_incorrect_paths<P: ReferenceParent>(
        &mut self,
        parent: &mut P,
        choose: bool,
    ) -> bool {
        if self.is_incorrect_paths() {
            return parent.fix_incorrect_path(&mut self.ftf_file, choose);
        }
        true
    }
    pub fn get_peet_parameters(&self, meta: &mut PeetMetaData) {
        meta.reference_volume = if self.s_volume.is_visible() {
            self.s_volume.get_value().to_string()
        } else {
            self.ltf_volume.get_text()
        };
        meta.reference_particle = self.rtf_particle.field.get_text();
        meta.reference_file = self.ftf_file.text.clone();
        meta.reference_multiparticle_level =
            MultiparticleReference::convert_index_to_level(self.cmb_multiparticle.selected_index);
    }
    pub fn set_peet_parameters(&mut self, meta: &ConstPeetMetaData) {
        self.ftf_file.set_text(Some(&meta.reference_file));
        self.rtf_particle.set_text(&meta.reference_particle);
        self.cmb_multiparticle.selected_index = MultiparticleReference::convert_level_to_index_int(
            meta.reference_multiparticle_level
                .parse()
                .unwrap_or(MultiparticleReference::DEFAULT_LEVEL),
        );
        if self.s_volume.is_visible() {
            self.s_volume.set_text(&meta.reference_volume);
        } else {
            self.ltf_volume.set_text(&meta.reference_volume);
        }
    }
    /// Returns invalid supplied and corrected levels, the Java UIHarness problem-dialog boundary.
    pub fn set_matlab_parameters(&mut self, matlab: &MatlabParam) -> Option<(String, String)> {
        if let Some(file) = &matlab.reference_file {
            self.rb_file.set_selected(true);
            self.ftf_file.set_text(Some(&file));
            None
        } else if matlab.flg_fair_reference {
            self.rb_multiparticle.set_selected(true);
            let level = matlab.reference_level.as_deref().unwrap_or_default();
            let (valid, index) = MultiparticleReference::convert_level_to_index(level);
            self.cmb_multiparticle.selected_index = index;
            (!valid).then(|| {
                (
                    level.into(),
                    MultiparticleReference::convert_index_to_level(index),
                )
            })
        } else {
            self.rtf_particle.set_selected(true);
            self.rtf_particle.set_text(&matlab.reference_particle);
            if self.s_volume.is_visible() {
                self.s_volume.set_text(&matlab.reference_volume);
            } else {
                self.ltf_volume.set_text(&matlab.reference_volume);
            }
            None
        }
    }
    pub fn get_matlab_parameters(&self, matlab: &mut MatlabParam, validation: bool) -> bool {
        if self.rtf_particle.is_selected() {
            let Ok(particle) = self.rtf_particle.get_text(validation) else {
                return false;
            };
            matlab.reference_file = None;
            matlab.flg_fair_reference = false;
            matlab.reference_volume = if self.s_volume.is_visible() {
                self.s_volume.get_value().to_string()
            } else {
                self.ltf_volume.get_text()
            };
            matlab.reference_particle = particle;
        } else if self.rb_file.is_selected() {
            matlab.reference_file = Some(self.ftf_file.text.clone());
            matlab.flg_fair_reference = false;
        } else if self.rb_multiparticle.is_selected() {
            matlab.reference_file = None;
            matlab.flg_fair_reference = true;
            matlab.reference_level = Some(MultiparticleReference::convert_index_to_level(
                self.cmb_multiparticle.selected_index,
            ));
        }
        true
    }
    pub fn is_reference_file_selected(&self) -> bool {
        self.rb_file.is_selected()
    }
    pub fn is_reference_particle_selected(&self) -> bool {
        self.rtf_particle.is_selected()
    }
    pub fn msg_flg_vol_names_are_templates(&mut self, init: bool, on: bool) {
        if on && !init && self.ltf_volume.is_empty() {
            self.ltf_volume
                .set_text(&self.s_volume.get_value().to_string());
        }
        self.s_volume.set_visible(!on);
        self.ltf_volume.set_visible(on);
    }
    pub fn action<P: ReferenceParent>(&mut self, parent: &mut P, command: &str) {
        if command == self.rtf_particle.action_command()
            || command == self.rb_file.get_action_command()
            || command == self.rb_multiparticle.get_action_command()
        {
            parent.update_display(false);
        }
    }
    pub fn validate_run(&self) -> Option<String> {
        if self.rtf_particle.is_selected() && self.rtf_particle.is_empty() {
            return Some(format!(
                "In {TITLE}, {} is required when {} is selected.",
                self.rtf_particle.label(),
                self.s_volume.get_label()
            ));
        }
        if self.rb_file.is_selected() && self.ftf_file.is_empty() {
            return Some(format!(
                "In {TITLE}, a file is required when {} is selected.",
                self.rb_file.get_text()
            ));
        }
        None
    }
    pub fn set_defaults(&mut self) {
        self.rtf_particle.set_selected(true);
    }
    pub fn update_display<P: ReferenceParent>(&mut self, parent: &P, init: bool) {
        self.rtf_particle
            .set_enabled(parent.get_volume_table_size() > 0);
        self.s_volume.set_enabled(self.rtf_particle.is_selected());
        self.ltf_volume.set_enabled(self.rtf_particle.is_selected());
        self.s_volume.set_max(parent.get_volume_table_size());
        self.ftf_file.set_enabled(self.rb_file.is_selected());
        self.cmb_multiparticle.enabled = self.rb_multiparticle.is_selected();
        self.l_multiparticle_enabled = self.rb_multiparticle.is_selected();
        self.msg_flg_vol_names_are_templates(init, parent.is_flg_vol_names_are_templates());
    }
    pub fn set_tooltips(&mut self) {
        self.s_volume
            .set_tool_tip_text(Some("The number of the volume containing the reference."));
        self.ltf_volume
            .set_tool_tip_text(Some("The number of the volume containing the reference."));
        self.rtf_particle.radio_button.set_tool_tip_text(Some(
            "Specify the reference by volume and particle numbers.",
        ));
        self.rb_file
            .set_tool_tip_text(Some("Specify the reference by filename."));
        self.ftf_file.tooltip =
            Some("The name of the file containing the MRC volume to use as the reference.".into());
        self.rb_multiparticle
            .set_tool_tip_text(Some("Generate a fair multi-particle reference."));
        self.cmb_multiparticle.tooltip =
            Some("Number of particles to be used to generate a multi-particle reference.".into());
        self.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Manager(PathBuf);
    impl ReferencePanelManager for Manager {
        fn property_user_dir(&self) -> &Path {
            &self.0
        }
    }
    #[derive(Default)]
    struct Parent {
        rows: i32,
        templates: bool,
        updates: usize,
    }
    impl ReferenceParent for Parent {
        fn fix_incorrect_path(&mut self, _: &mut dyn FileTextFieldInterface, _: bool) -> bool {
            false
        }
        fn get_volume_table_size(&self) -> i32 {
            self.rows
        }
        fn update_display(&mut self, _: bool) {
            self.updates += 1;
        }
        fn is_flg_vol_names_are_templates(&self) -> bool {
            self.templates
        }
    }
    #[test]
    fn parameter_and_template_paths() {
        let manager = Manager(PathBuf::from("/dataset"));
        let mut panel = ReferencePanel::get_instance(&manager, false);
        let mut parent = Parent {
            rows: 3,
            templates: true,
            updates: 0,
        };
        panel.set_defaults();
        panel.update_display(&parent, false);
        assert!(!panel.s_volume.is_visible() && panel.ltf_volume.is_visible());
        panel.rtf_particle.set_text("4");
        panel.ltf_volume.set_text("2");
        let mut matlab = MatlabParam::default();
        assert!(panel.get_matlab_parameters(&mut matlab, true));
        assert_eq!(matlab.reference_particle, "4");
        let action_command = panel.rtf_particle.action_command().to_owned();
        panel.action(&mut parent, &action_command);
        assert_eq!(parent.updates, 1);
    }
    #[test]
    fn file_and_multiparticle_validation() {
        let manager = Manager(PathBuf::from("/dataset"));
        let mut panel = ReferencePanel::get_instance(&manager, false);
        panel.rb_file.set_selected(true);
        assert!(panel.validate_run().unwrap().contains("a file is required"));
        panel.rb_multiparticle.set_selected(true);
        panel.cmb_multiparticle.selected_index = 2;
        let mut matlab = MatlabParam::default();
        assert!(panel.get_matlab_parameters(&mut matlab, false));
        assert_eq!(matlab.reference_level.as_deref(), Some("4"));
    }
}
