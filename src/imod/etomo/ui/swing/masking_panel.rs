//! `IMOD/Etomo/src/etomo/ui/swing/MaskingPanel.java`.
//!
//! Swing layout, autodoc reads, file choosing, and parent-manager dispatch remain
//! explicit boundaries.  The panel's source-owned field, selection, parameter,
//! validation, and action state is retained here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::check_box::CheckBox;
use super::file_text_field_interface::FileTextFieldInterface;
use super::file_text_field2::FileTextField2;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::masking_parent::MaskingParent;
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};
use crate::imod::etomo::ui::field_type::FieldType;

pub const INSIDE_MASK_RADIUS_LABEL: &str = "Inner radius: ";
pub const OUTSIDE_MASK_RADIUS_LABEL: &str = "Outer radius: ";
pub const MAST_TYPE_LABEL: &str = "Masking";
pub const MAST_TYPE_FILE_LABEL: &str = "User supplied binary file:";

/// Java `MatlabParam.MaskType` values used by this source unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskType {
    None,
    Volume,
    Sphere,
    Cylinder,
}
impl MaskType {
    pub fn get_instance(value: &str) -> Self {
        match value {
            "none" | "None" => Self::None,
            "sphere" | "Sphere" => Self::Sphere,
            "cylinder" | "Cylinder" => Self::Cylinder,
            _ => Self::Volume,
        }
    }
    pub fn get_label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Volume => "Volume",
            Self::Sphere => "Sphere",
            Self::Cylinder => "Cylinder",
        }
    }
    pub fn value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Volume => "volume",
            Self::Sphere => "sphere",
            Self::Cylinder => "cylinder",
        }
    }
}

pub trait MaskingPanelManager {
    fn property_user_dir(&self) -> &Path;
}

/// Java `PeetMetaData` fields read and written by `MaskingPanel`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PeetMetaData {
    pub mask_model_pts_z_rotation: String,
    pub mask_model_pts_y_rotation: String,
    pub mask_type_volume: String,
    pub manual_cylinder_orientation: bool,
    pub cylinder_height: String,
    pub mask_blur_std_dev: String,
}
pub type ConstPeetMetaData = PeetMetaData;

/// Java `MatlabParam` mask fields read and written by this source unit.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatlabParam {
    pub mask_type: String,
    pub mask_blur_std_dev: String,
    pub cylinder_height: String,
    pub mask_model_pts_z_rotation: String,
    pub mask_model_pts_y_rotation: String,
    pub mask_model_pts_set: bool,
    pub inside_mask_radius: String,
    pub outside_mask_radius: String,
}
impl MatlabParam {
    pub fn is_mask_model_pts_empty(&self) -> bool {
        !self.mask_model_pts_set
    }
    pub fn clear_mask_model_pts(&mut self) {
        self.mask_model_pts_set = false;
        self.mask_model_pts_z_rotation.clear();
        self.mask_model_pts_y_rotation.clear();
    }
}

/// Java `MaskingPanel`, with Swing component hierarchy represented by ordered
/// layout names and native-widget state in the component fields.
pub struct MaskingPanel {
    pub rb_mask_type_none: RadioButton,
    pub rb_mask_type_file: RadioButton,
    pub rb_mask_type_sphere: RadioButton,
    pub rb_mask_type_cylinder: RadioButton,
    pub ftf_mask_type_file: FileTextField2,
    pub ltf_inside_mask_radius: LabeledTextField,
    pub ltf_outside_mask_radius: LabeledTextField,
    pub ltf_z_rotation: LabeledTextField,
    pub ltf_y_rotation: LabeledTextField,
    pub cb_cylinder_orientation: CheckBox,
    pub ltf_cylinder_height: LabeledTextField,
    pub ltf_mask_blur_std_dev: LabeledTextField,
    pub root_name: String,
    pub component_order: Vec<String>,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub april_fools_background: bool,
}

impl MaskingPanel {
    /// Java private constructor plus `getInstance` construction sequence.
    pub fn get_instance<M: MaskingPanelManager>(manager: &M, april_fools: bool) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut result = Self {
            rb_mask_type_none: RadioButton::new_with_enumerated_type(
                Some("None".into()),
                EnumeratedTypeBoundary {
                    label: "None".into(),
                    default: true,
                    value: Some(MaskType::None.value().into()),
                },
                Some(group.clone()),
            ),
            rb_mask_type_file: RadioButton::new_with_enumerated_type(
                Some(MAST_TYPE_FILE_LABEL.into()),
                EnumeratedTypeBoundary {
                    label: MaskType::Volume.get_label().into(),
                    default: false,
                    value: Some(MaskType::Volume.value().into()),
                },
                Some(group.clone()),
            ),
            rb_mask_type_sphere: RadioButton::new_with_enumerated_type(
                Some("Sphere".into()),
                EnumeratedTypeBoundary {
                    label: "Sphere".into(),
                    default: false,
                    value: Some(MaskType::Sphere.value().into()),
                },
                Some(group.clone()),
            ),
            rb_mask_type_cylinder: RadioButton::new_with_enumerated_type(
                Some("Cylinder".into()),
                EnumeratedTypeBoundary {
                    label: "Cylinder".into(),
                    default: false,
                    value: Some(MaskType::Cylinder.value().into()),
                },
                Some(group),
            ),
            ftf_mask_type_file: FileTextField2 {
                enabled: true,
                adjusted_field_width: 190,
                use_text_as_file_chooser_dir: true,
                browsing_directory: Some(manager.property_user_dir().to_path_buf()),
                ..Default::default()
            },
            ltf_inside_mask_radius: LabeledTextField::new(
                FieldType::FloatingPoint,
                INSIDE_MASK_RADIUS_LABEL,
            ),
            ltf_outside_mask_radius: LabeledTextField::new(
                FieldType::FloatingPoint,
                OUTSIDE_MASK_RADIUS_LABEL,
            ),
            ltf_z_rotation: LabeledTextField::new(FieldType::FloatingPoint, "Z Rotation: "),
            ltf_y_rotation: LabeledTextField::new(FieldType::FloatingPoint, "Y Rotation: "),
            cb_cylinder_orientation: CheckBox::new_with_text("Manual Cylinder Orientation"),
            ltf_cylinder_height: LabeledTextField::new(FieldType::FloatingPoint, "Height: "),
            ltf_mask_blur_std_dev: LabeledTextField::new(
                FieldType::FloatingPoint,
                "Blur mask by: ",
            ),
            root_name: MAST_TYPE_LABEL.into(),
            component_order: Vec::new(),
            listener_count: 0,
            tooltip_initialized: false,
            april_fools_background: april_fools,
        };
        result.create_panel();
        result.set_tooltips();
        result.add_listeners();
        result
    }
    pub fn add_listeners(&mut self) {
        self.listener_count = 5;
    }
    pub fn create_panel(&mut self) {
        self.ltf_cylinder_height.set_number_must_be_positive(true);
        self.ltf_mask_blur_std_dev.set_text_preferred_width(80);
        self.component_order = vec![
            "Masking".into(),
            "mask-type".into(),
            "sphere-cylinder".into(),
            "none".into(),
            "sphere".into(),
            "cylinder".into(),
            "file".into(),
            "radii".into(),
            "blur".into(),
            "cylinder-orientation".into(),
            "cylinder-rotation".into(),
        ];
    }
    pub fn get_ui_component(&self) -> &Self {
        self
    }
    pub fn get_component(&self) -> &Self {
        self
    }
    pub fn update_display(&mut self) {
        self.ftf_mask_type_file
            .set_enabled(self.rb_mask_type_file.is_selected());
        self.ltf_mask_blur_std_dev
            .set_enabled(!self.rb_mask_type_none.is_selected());
        let cylinder = self.rb_mask_type_cylinder.is_selected();
        self.cb_cylinder_orientation.set_enabled(cylinder);
        self.ltf_cylinder_height.set_enabled(cylinder);
        let cylinder_orientation = cylinder && self.cb_cylinder_orientation.is_selected();
        self.ltf_z_rotation.set_enabled(cylinder_orientation);
        self.ltf_y_rotation.set_enabled(cylinder_orientation);
        let sphere_cylinder = cylinder || self.rb_mask_type_sphere.is_selected();
        self.ltf_inside_mask_radius.set_enabled(sphere_cylinder);
        self.ltf_outside_mask_radius.set_enabled(sphere_cylinder);
    }
    pub fn convert_copied_paths<M: MaskingPanelManager>(
        &mut self,
        manager: &M,
        orig_dataset_dir: &Path,
    ) {
        if !self.ftf_mask_type_file.is_empty()
            && !Path::new(&self.ftf_mask_type_file.text).is_absolute()
        {
            let copied = orig_dataset_dir.join(&self.ftf_mask_type_file.text);
            self.ftf_mask_type_file.text = copied
                .strip_prefix(manager.property_user_dir())
                .map_or_else(
                    |_| copied.to_string_lossy().into_owned(),
                    |path| path.to_string_lossy().into_owned(),
                );
        }
    }
    pub fn is_incorrect_paths(&self) -> bool {
        !self.ftf_mask_type_file.is_empty() && !self.ftf_mask_type_file.exists()
    }
    pub fn fix_incorrect_paths<P: MaskingParent>(
        &mut self,
        parent: &mut P,
        choose_path_every_row: bool,
    ) -> bool {
        if self.is_incorrect_paths() {
            return parent.fix_incorrect_path(&mut self.ftf_mask_type_file, choose_path_every_row);
        }
        true
    }
    pub fn get_peet_parameters(&self, meta_data: &mut PeetMetaData) {
        meta_data.mask_model_pts_z_rotation = self.ltf_z_rotation.get_text();
        meta_data.mask_model_pts_y_rotation = self.ltf_y_rotation.get_text();
        meta_data.mask_type_volume = self.ftf_mask_type_file.text.clone();
        meta_data.manual_cylinder_orientation = self.cb_cylinder_orientation.is_selected();
        meta_data.cylinder_height = self.ltf_cylinder_height.get_text();
        meta_data.mask_blur_std_dev = self.ltf_mask_blur_std_dev.get_text();
    }
    pub fn set_peet_parameters(&mut self, meta_data: &ConstPeetMetaData) {
        self.ftf_mask_type_file
            .set_text(Some(&meta_data.mask_type_volume));
        self.ltf_z_rotation
            .set_text(&meta_data.mask_model_pts_z_rotation);
        self.ltf_y_rotation
            .set_text(&meta_data.mask_model_pts_y_rotation);
        self.cb_cylinder_orientation
            .set_selected(meta_data.manual_cylinder_orientation);
        self.ltf_cylinder_height
            .set_text(&meta_data.cylinder_height);
        self.ltf_mask_blur_std_dev
            .set_text(&meta_data.mask_blur_std_dev);
    }
    pub fn set_matlab_parameters(&mut self, matlab: &MatlabParam) {
        match MaskType::get_instance(&matlab.mask_type) {
            MaskType::None => self.rb_mask_type_none.set_selected(true),
            MaskType::Volume => {
                self.ltf_mask_blur_std_dev
                    .set_text(&matlab.mask_blur_std_dev);
                self.rb_mask_type_file.set_selected(true);
                self.ftf_mask_type_file.set_text(Some(&matlab.mask_type));
            }
            MaskType::Sphere => {
                self.ltf_mask_blur_std_dev
                    .set_text(&matlab.mask_blur_std_dev);
                self.rb_mask_type_sphere.set_selected(true);
            }
            MaskType::Cylinder => {
                self.ltf_mask_blur_std_dev
                    .set_text(&matlab.mask_blur_std_dev);
                self.rb_mask_type_cylinder.set_selected(true);
                self.ltf_cylinder_height.set_text(&matlab.cylinder_height);
            }
        }
        if !matlab.is_mask_model_pts_empty() {
            self.cb_cylinder_orientation.set_selected(true);
            self.ltf_z_rotation
                .set_text(&matlab.mask_model_pts_z_rotation);
            self.ltf_y_rotation
                .set_text(&matlab.mask_model_pts_y_rotation);
        }
        self.ltf_inside_mask_radius
            .set_text(&matlab.inside_mask_radius);
        self.ltf_outside_mask_radius
            .set_text(&matlab.outside_mask_radius);
    }
    pub fn get_matlab_parameters(&self, matlab: &mut MatlabParam, do_validation: bool) -> bool {
        let result: Result<(), FieldValidationFailedException> = (|| {
            if self.rb_mask_type_file.is_selected() {
                matlab.mask_type = self.ftf_mask_type_file.text.clone();
            } else {
                matlab.mask_type = if self.rb_mask_type_none.is_selected() {
                    MaskType::None.value().into()
                } else if self.rb_mask_type_sphere.is_selected() {
                    MaskType::Sphere.value().into()
                } else {
                    MaskType::Cylinder.value().into()
                };
                if self.rb_mask_type_cylinder.is_selected() {
                    matlab.cylinder_height =
                        self.ltf_cylinder_height.get_text_validated(do_validation)?;
                }
            }
            if !self.rb_mask_type_none.is_selected() {
                matlab.mask_blur_std_dev = self
                    .ltf_mask_blur_std_dev
                    .get_text_validated(do_validation)?;
            }
            if self.cb_cylinder_orientation.is_enabled()
                && self.cb_cylinder_orientation.is_selected()
            {
                matlab.mask_model_pts_z_rotation =
                    self.ltf_z_rotation.get_text_validated(do_validation)?;
                matlab.mask_model_pts_y_rotation =
                    self.ltf_y_rotation.get_text_validated(do_validation)?;
                matlab.mask_model_pts_set = true;
            } else {
                matlab.clear_mask_model_pts();
            }
            matlab.inside_mask_radius = self
                .ltf_inside_mask_radius
                .get_text_validated(do_validation)?;
            matlab.outside_mask_radius = self
                .ltf_outside_mask_radius
                .get_text_validated(do_validation)?;
            Ok(())
        })();
        result.is_ok()
    }
    pub fn set_defaults(&mut self) {
        self.rb_mask_type_none.set_selected(true);
    }
    pub fn validate_run(&self) -> Option<String> {
        if self.rb_mask_type_file.is_selected() && self.ftf_mask_type_file.is_empty() {
            return Some(format!(
                "In {MAST_TYPE_LABEL}, {} is required when {} {MAST_TYPE_LABEL} is selected. ",
                MaskType::Volume.get_label(),
                MaskType::Volume.get_label()
            ));
        }
        if (self.rb_mask_type_sphere.is_selected() || self.rb_mask_type_cylinder.is_selected())
            && self.ltf_inside_mask_radius.enabled
            && self.ltf_inside_mask_radius.is_empty()
            && self.ltf_outside_mask_radius.enabled
            && self.ltf_outside_mask_radius.is_empty()
        {
            return Some(format!(
                "In {MAST_TYPE_LABEL}, {INSIDE_MASK_RADIUS_LABEL} and/or {OUTSIDE_MASK_RADIUS_LABEL} are required when either {} or {} is selected.",
                MaskType::Sphere.get_label(),
                MaskType::Cylinder.get_label()
            ));
        }
        if self.ltf_z_rotation.enabled {
            match self.ltf_z_rotation.get_text().parse::<f64>() {
                Ok(value) if (0.0..=90.0).contains(&value.abs()) => {}
                Ok(_) => {
                    return Some(format!(
                        "Valid values for {} field in {} are 0 to 90.",
                        self.ltf_z_rotation.label, self.root_name
                    ));
                }
                Err(_) if !self.ltf_z_rotation.is_empty() => {
                    return Some(format!(
                        "{} field in {} must be numeric.",
                        self.ltf_z_rotation.label, self.root_name
                    ));
                }
                _ => {}
            }
        }
        if self.ltf_z_rotation.enabled
            && !self.ltf_y_rotation.is_empty()
            && self.ltf_y_rotation.get_text().parse::<f64>().is_err()
        {
            return Some(format!(
                "{} field in {} must be numeric.",
                self.ltf_y_rotation.label, self.root_name
            ));
        }
        None
    }
    pub fn action<P: MaskingParent>(&mut self, parent: &mut P, action_command: &str) {
        if action_command == self.rb_mask_type_none.get_action_command()
            || action_command == self.rb_mask_type_file.get_action_command()
            || action_command == self.rb_mask_type_sphere.get_action_command()
            || action_command == self.rb_mask_type_cylinder.get_action_command()
            || self
                .cb_cylinder_orientation
                .get_action_command()
                .is_some_and(|command| action_command == command)
        {
            parent.update_display(false);
        }
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<P: MaskingParent>(&mut self, parent: &mut P, action_command: &str) {
        self.action(parent, action_command);
    }
    pub fn set_tooltips(&mut self) {
        self.ltf_z_rotation
            .set_tool_tip_text(Some("Cylinder orientation from mask model points."));
        self.ltf_y_rotation
            .set_tool_tip_text(Some("Cylinder orientation from mask model points."));
        self.ltf_cylinder_height
            .set_tool_tip_text(Some("Height of the cylindrical mask."));
        self.ltf_mask_blur_std_dev
            .set_tool_tip_text(Some("Standard deviation for blurring the mask."));
        self.rb_mask_type_none
            .set_tool_tip_text(Some("No reference masking"));
        self.rb_mask_type_file
            .set_tool_tip_text(Some("Mask the reference using a specified file"));
        self.rb_mask_type_sphere.set_tool_tip_text(Some(
            "Mask the reference using inner and out spherical shells of specified radii.",
        ));
        self.rb_mask_type_cylinder.set_tool_tip_text(Some(
            "Mask the reference using inner and out cylindrical shells of specified radii.",
        ));
        self.ftf_mask_type_file.tooltip =
            Some("The name of file containing the binary mask in MRC format.".into());
        self.ltf_inside_mask_radius
            .set_tool_tip_text(Some("Inner radius of the mask region in pixels."));
        self.ltf_outside_mask_radius
            .set_tool_tip_text(Some("Inner and outer radii of the mask region in pixels."));
        self.cb_cylinder_orientation
            .set_tool_tip_text(Some("Manually specify cylindrical mask orientation."));
        self.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Manager(PathBuf);
    impl MaskingPanelManager for Manager {
        fn property_user_dir(&self) -> &Path {
            &self.0
        }
    }
    #[derive(Default)]
    struct Parent {
        updates: usize,
    }
    impl MaskingParent for Parent {
        fn is_reference_file_selected(&self) -> bool {
            false
        }
        fn get_volume_table_size(&self) -> i32 {
            0
        }
        fn fix_incorrect_path(&mut self, _: &mut dyn FileTextFieldInterface, _: bool) -> bool {
            false
        }
        fn update_display(&mut self, _: bool) {
            self.updates += 1;
        }
    }
    #[test]
    fn display_and_validation_follow_mask_selection() {
        let manager = Manager(PathBuf::from("/dataset"));
        let mut panel = MaskingPanel::get_instance(&manager, false);
        panel.set_defaults();
        panel.update_display();
        assert!(!panel.ltf_inside_mask_radius.enabled);
        panel.rb_mask_type_cylinder.set_selected(true);
        panel.update_display();
        assert!(panel.ltf_cylinder_height.enabled);
        assert!(panel.validate_run().unwrap().contains("Inner radius"));
        panel.ltf_inside_mask_radius.set_text("5");
        panel.cb_cylinder_orientation.set_selected(true);
        panel.update_display();
        panel.ltf_z_rotation.set_text("91");
        assert!(panel.validate_run().unwrap().contains("0 to 90"));
    }
    #[test]
    fn matlab_and_parent_action_are_preserved() {
        let manager = Manager(PathBuf::from("/dataset"));
        let mut panel = MaskingPanel::get_instance(&manager, false);
        panel.rb_mask_type_sphere.set_selected(true);
        panel.ltf_inside_mask_radius.set_text("2");
        panel.ltf_outside_mask_radius.set_text("4");
        panel.ltf_mask_blur_std_dev.set_text("1");
        let mut matlab = MatlabParam::default();
        assert!(panel.get_matlab_parameters(&mut matlab, true));
        assert_eq!(matlab.mask_type, "sphere");
        let mut parent = Parent::default();
        let command = panel.rb_mask_type_sphere.get_action_command().to_owned();
        panel.action(&mut parent, &command);
        assert_eq!(parent.updates, 1);
    }
}
