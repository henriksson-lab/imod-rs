//! `IMOD/Etomo/src/etomo/ui/swing/MaskingPanel.java`.
//!
//! The PEET dialog's "Masking" box: no mask, a user supplied binary file, a sphere
//! or a cylinder, with radii, blur and cylinder orientation.  An event dispatch
//! thread object, created as `Rc<Self>` by [`MaskingPanel::get_instance`]; it keeps a
//! weak reference to its parent.

use std::rc::{Rc, Weak};

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::check_box::CheckBox;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::file_text_field2::FileTextField2;
use super::labeled_text_field::LabeledTextField;
use super::masking_parent::MaskingParent;
use super::peet_dialog;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::swing_component::SwingComponent;
use crate::imod::etomo::base_manager::{BaseManager, ManagerBrowsingDirectory};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, FileFilter, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{self, MaskType, MatlabParam};
use crate::imod::etomo::storage::volume_file_filter::VolumeFileFilter;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::const_peet_meta_data::ConstPeetMetaData;
use crate::imod::etomo::r#type::enumerated_type::EnumeratedType;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::peet_meta_data::PeetMetaData;
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_displayer::FieldDisplayer;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::ui_component::UIComponent;
use crate::imod::etomo::util::file_path::FilePath;

/// Java private static final `INSIDE_MASK_RADIUS_LABEL`.
const INSIDE_MASK_RADIUS_LABEL: &str = "Inner radius: ";
/// Java private static final `OUTSIDE_MASK_RADIUS_LABEL`.
const OUTSIDE_MASK_RADIUS_LABEL: &str = "Outer radius: ";
/// Java private static final `MAST_TYPE_LABEL`.
const MAST_TYPE_LABEL: &str = "Masking";
/// Java private static final `MAST_TYPE_FILE_LABEL`.
const MAST_TYPE_FILE_LABEL: &str = "User supplied binary file:";

/// Java package-private `final class MaskingPanel implements UIComponent,
/// SwingComponent`.
pub struct MaskingPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `bgMaskType`.
    bg_mask_type: Rc<ButtonGroup>,
    /// Java private final `rbMaskTypeNone`.
    rb_mask_type_none: Rc<RadioButton>,
    /// Java private final `rbMaskTypeFile`.
    rb_mask_type_file: Rc<RadioButton>,
    /// Java private final `rbMaskTypeSphere`.
    rb_mask_type_sphere: Rc<RadioButton>,
    /// Java private final `rbMaskTypeCylinder`.
    rb_mask_type_cylinder: Rc<RadioButton>,
    /// Java private final `ftfMaskTypeFile`.
    ftf_mask_type_file: Rc<FileTextField2>,
    /// Java private final `ltfInsideMaskRadius`.
    ltf_inside_mask_radius: Rc<LabeledTextField>,
    /// Java private final `ltfOutsideMaskRadius`.
    ltf_outside_mask_radius: Rc<LabeledTextField>,
    /// Java private final `ltfZRotation`.
    ltf_z_rotation: Rc<LabeledTextField>,
    /// Java private final `ltfYRotation`.
    ltf_y_rotation: Rc<LabeledTextField>,
    /// Java private final `cbCylinderOrientation`.
    cb_cylinder_orientation: Rc<CheckBox>,
    /// Java private final `ltfCylinderHeight`.
    ltf_cylinder_height: Rc<LabeledTextField>,
    /// Java private final `ltfMaskBlurStdDev`.
    ltf_mask_blur_std_dev: Rc<LabeledTextField>,
    /// Java private final `parent`.
    parent: Weak<dyn MaskingParent>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `fieldDisplayer`.
    field_displayer: Option<Rc<dyn FieldDisplayer>>,
    /// Java `this`.
    self_ref: Weak<MaskingPanel>,
}

impl MaskingPanel {
    /// Java private `MaskingPanel(BaseManager, MaskingParent, FieldDisplayer)`, with
    /// the field initializers.
    fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn MaskingParent>,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> Rc<MaskingPanel> {
        let bg_mask_type = ButtonGroup::new();
        let text_field = |label: &str| {
            LabeledTextField::new_field_type_string_string(
                FieldType::FloatingPoint,
                Some(label),
                Some(peet_dialog::SETUP_LOCATION_DESCR),
            )
        };
        Rc::new_cyclic(|self_ref: &Weak<MaskingPanel>| MaskingPanel {
            pnl_root: EtomoPanel::new(),
            rb_mask_type_none: RadioButton::new_string_enumerated_type_button_group(
                Some("None"),
                Some(EnumeratedTypeRef::new(MaskType::None)),
                Some(&bg_mask_type),
            ),
            rb_mask_type_file: RadioButton::new_string_enumerated_type_button_group(
                Some(MAST_TYPE_FILE_LABEL),
                Some(EnumeratedTypeRef::new(MaskType::Volume)),
                Some(&bg_mask_type),
            ),
            rb_mask_type_sphere: RadioButton::new_string_enumerated_type_button_group(
                Some("Sphere"),
                Some(EnumeratedTypeRef::new(MaskType::Sphere)),
                Some(&bg_mask_type),
            ),
            rb_mask_type_cylinder: RadioButton::new_string_enumerated_type_button_group(
                Some("Cylinder"),
                Some(EnumeratedTypeRef::new(MaskType::Cylinder)),
                Some(&bg_mask_type),
            ),
            bg_mask_type,
            ftf_mask_type_file: FileTextField2::get_unlabeled_peet_instance(
                Some(manager),
                Some(MAST_TYPE_FILE_LABEL),
            ),
            ltf_inside_mask_radius: text_field(INSIDE_MASK_RADIUS_LABEL),
            ltf_outside_mask_radius: text_field(OUTSIDE_MASK_RADIUS_LABEL),
            ltf_z_rotation: text_field("Z Rotation: "),
            ltf_y_rotation: text_field("Y Rotation: "),
            cb_cylinder_orientation: CheckBox::new_string(Some("Manual Cylinder Orientation")),
            ltf_cylinder_height: text_field("Height: "),
            ltf_mask_blur_std_dev: text_field("Blur mask by: "),
            parent,
            manager,
            field_displayer,
            self_ref: self_ref.clone(),
        })
    }

    /// Java static `getInstance(BaseManager, MaskingParent, FieldDisplayer)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn MaskingParent>,
        field_displayer: Option<Rc<dyn FieldDisplayer>>,
    ) -> Rc<MaskingPanel> {
        let instance = MaskingPanel::new(manager, parent, field_displayer);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java field read `parent`.
    fn parent(&self) -> Rc<dyn MaskingParent> {
        self.parent
            .upgrade()
            .expect("the PEET dialog owns its masking panel")
    }

    /// Java private `addListeners()` with `MaskingActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(masking_panel) = adaptee.upgrade() {
                masking_panel.action(event.get_action_command().unwrap_or(""), None);
            }
        });
        self.rb_mask_type_none
            .add_action_listener(action_listener.clone());
        self.rb_mask_type_file
            .add_action_listener(action_listener.clone());
        self.rb_mask_type_sphere
            .add_action_listener(action_listener.clone());
        self.rb_mask_type_cylinder
            .add_action_listener(action_listener.clone());
        self.cb_cylinder_orientation
            .add_action_listener(Some(action_listener));
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.ftf_mask_type_file
            .set_file_filter(Some(
                Rc::new(VolumeFileFilter::get_instance(Some(self.manager))) as Rc<dyn FileFilter>,
            ));
        self.ftf_mask_type_file
            .set_use_text_as_file_chooser_dir(true);
        self.ftf_mask_type_file
            .set_browsing_directory(Some(Rc::new(ManagerBrowsingDirectory(self.manager))));
        self.ltf_cylinder_height.set_number_must_be_positive(true);
        self.ltf_mask_blur_std_dev.set_preferred_width(80);
        for field in [
            &self.ltf_outside_mask_radius,
            &self.ltf_z_rotation,
            &self.ltf_y_rotation,
            &self.ltf_cylinder_height,
            &self.ltf_mask_blur_std_dev,
        ] {
            field.set_overridable_field_displayers(None, self.field_displayer.clone());
        }
        // local panels
        let pnl_mask_type = JComponent::new_panel();
        let pnl_file = JComponent::new_panel();
        let pnl_mask_type_none = JComponent::new_panel();
        let pnl_mask_type_sphere = JComponent::new_panel();
        let pnl_mask_type_cylinder = JComponent::new_panel();
        let pnl_sphere_cylinder = JComponent::new_panel();
        let pnl_radius = JComponent::new_panel();
        let pnl_cylinder_orientation = JComponent::new_panel();
        let pnl_cylinder_rotation = JComponent::new_panel();
        let pnl_cylinder_orientation_check_box = JComponent::new_panel();
        let pnl_cylinder_orientation_x = JComponent::new_panel();
        let pnl_mask_blur_std_dev = JComponent::new_panel();
        // initalization
        self.ftf_mask_type_file.set_adjusted_field_width(190.0);
        // root panel (BoxLayout X_AXIS)
        self.pnl_root
            .set_border(&EtchedBorder::new(Some(MAST_TYPE_LABEL)).get_border());
        let root = self.pnl_root.get_component();
        root.add(&pnl_mask_type);
        root.add(&pnl_sphere_cylinder);
        // mask type (BoxLayout Y_AXIS)
        pnl_mask_type.add(&pnl_mask_type_none);
        pnl_mask_type.add(&pnl_mask_type_sphere);
        pnl_mask_type.add(&pnl_mask_type_cylinder);
        pnl_mask_type.add(&pnl_file);
        // SphereCylinder (BoxLayout Y_AXIS, rigid areas between)
        pnl_sphere_cylinder.add(&pnl_radius);
        pnl_sphere_cylinder.add(&pnl_mask_blur_std_dev);
        pnl_sphere_cylinder.add(&pnl_cylinder_orientation_x);
        // MaskTypeNone (BoxLayout X_AXIS, horizontal glue)
        pnl_mask_type_none.add(&self.rb_mask_type_none.get_component());
        // MaskTypeSphere
        pnl_mask_type_sphere.add(&self.rb_mask_type_sphere.get_component());
        // MaskTypeCylinder (rigid area x50_y0)
        pnl_mask_type_cylinder.add(&self.rb_mask_type_cylinder.get_component());
        pnl_mask_type_cylinder.add(&self.ltf_cylinder_height.get_component());
        // File
        pnl_file.add(&self.rb_mask_type_file.get_component());
        pnl_file.add(&self.ftf_mask_type_file.get_root_panel());
        // radius (rigid areas between)
        pnl_radius.add(&self.ltf_inside_mask_radius.get_container());
        pnl_radius.add(&self.ltf_outside_mask_radius.get_container());
        // MaskBlurStdDev
        pnl_mask_blur_std_dev.add(&self.ltf_mask_blur_std_dev.get_component());
        // CylinderOrientationX
        pnl_cylinder_orientation_x.add(&pnl_cylinder_orientation);
        // CylinderOrientation (BoxLayout Y_AXIS, an untitled etched border)
        pnl_cylinder_orientation.add(&pnl_cylinder_orientation_check_box);
        pnl_cylinder_orientation.add(&pnl_cylinder_rotation);
        // pnlCylinderOrientationCheckBox
        pnl_cylinder_orientation_check_box.add(&self.cb_cylinder_orientation.get_component());
        // cylinder rotation
        pnl_cylinder_rotation.add(&self.ltf_z_rotation.get_container());
        pnl_cylinder_rotation.add(&self.ltf_y_rotation.get_container());
    }

    /// Java package-private `updateDisplay()`.  Called by the parent updateDisplay().
    /// Enabled/disables fields.
    pub fn update_display(&self) {
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

    /// Java package-private `convertCopiedPaths(String)`.
    pub fn convert_copied_paths(&self, orig_dataset_dir: &str) {
        let property_user_dir = self.manager.get_property_user_dir();
        if !self.ftf_mask_type_file.is_empty() {
            self.ftf_mask_type_file.set_text_string(
                FilePath::get_rerooted_relative_path(
                    Some(orig_dataset_dir),
                    property_user_dir.as_deref(),
                    self.ftf_mask_type_file.get_text_void().as_deref(),
                )
                .as_deref(),
            );
        }
    }

    /// Java package-private `isIncorrectPaths()`.
    pub fn is_incorrect_paths(&self) -> bool {
        !self.ftf_mask_type_file.is_empty() && !self.ftf_mask_type_file.exists()
    }

    /// Java package-private `fixIncorrectPaths(boolean)`.
    pub fn fix_incorrect_paths(&self, choose_path_every_row: bool) -> bool {
        if self.is_incorrect_paths() {
            return self
                .parent()
                .fix_incorrect_path(&*self.ftf_mask_type_file, choose_path_every_row);
        }
        true
    }

    /// Java package-private `getParameters(PeetMetaData)`.
    pub fn get_parameters_meta_data(&self, meta_data: &PeetMetaData) {
        meta_data.set_mask_model_pts_z_rotation(self.ltf_z_rotation.get_text_void().as_deref());
        meta_data.set_mask_model_pts_y_rotation(self.ltf_y_rotation.get_text_void().as_deref());
        meta_data.set_mask_type_volume(self.ftf_mask_type_file.get_text_void().as_deref());
        meta_data.set_manual_cylinder_orientation(self.cb_cylinder_orientation.is_selected());
        meta_data.set_cylinder_height(self.ltf_cylinder_height.get_text_void().as_deref());
        meta_data.set_mask_blur_std_dev(self.ltf_mask_blur_std_dev.get_text_void().as_deref());
    }

    /// Java package-private `setParameters(ConstPeetMetaData)`.
    pub fn set_parameters_meta_data(&self, meta_data: &dyn ConstPeetMetaData) {
        self.ftf_mask_type_file
            .set_text_string(meta_data.get_mask_type_volume().as_deref());
        self.ltf_z_rotation
            .set_text_const_etomo_number(Some(&meta_data.get_mask_model_pts_z_rotation()));
        self.ltf_y_rotation
            .set_text_string(meta_data.get_mask_model_pts_y_rotation().as_deref());
        self.cb_cylinder_orientation
            .set_selected_boolean(meta_data.is_manual_cylinder_orientation());
        self.ltf_cylinder_height
            .set_text_string(meta_data.get_cylinder_height().as_deref());
        self.ltf_mask_blur_std_dev
            .set_text_string(meta_data.get_mask_blur_std_dev().as_deref());
    }

    /// Java package-private `setParameters(MatlabParam)`.  Load data from
    /// MatlabParamFile.
    pub fn set_parameters_matlab_param(&self, matlab_param: &MatlabParam) {
        let mask_type_value = matlab_param.get_mask_type();
        let mask_type = MaskType::get_instance(mask_type_value.as_deref());
        if mask_type == MaskType::None {
            self.rb_mask_type_none.set_selected_boolean(true);
        } else {
            self.ltf_mask_blur_std_dev
                .set_text_string(matlab_param.get_mask_blur_std_dev().as_deref());
            if mask_type == MaskType::Volume {
                self.rb_mask_type_file.set_selected_boolean(true);
                self.ftf_mask_type_file
                    .set_text_string(mask_type_value.as_deref());
            } else if mask_type == MaskType::Sphere {
                self.rb_mask_type_sphere.set_selected_boolean(true);
            } else if mask_type == MaskType::Cylinder {
                self.rb_mask_type_cylinder.set_selected_boolean(true);
                self.ltf_cylinder_height
                    .set_text_string(matlab_param.get_cylinder_height().as_deref());
            }
        }
        if !matlab_param.is_mask_model_pts_empty() {
            self.cb_cylinder_orientation.set_selected_boolean(true);
            self.ltf_z_rotation
                .set_text_string(matlab_param.get_mask_model_pts_z_rotation().as_deref());
            self.ltf_y_rotation
                .set_text_string(matlab_param.get_mask_model_pts_y_rotation().as_deref());
        }
        self.ltf_inside_mask_radius
            .set_text_string(matlab_param.get_inside_mask_radius().as_deref());
        self.ltf_outside_mask_radius
            .set_text_string(matlab_param.get_outside_mask_radius().as_deref());
    }

    /// Java package-private `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters_matlab_param(
        &self,
        matlab_param: &mut MatlabParam,
        do_validation: bool,
    ) -> bool {
        let result = (|| -> Result<(), crate::imod::etomo::ui::field_validation_failed_exception::FieldValidationFailedException> {
            if self.rb_mask_type_file.is_selected() {
                matlab_param.set_mask_type_string(self.ftf_mask_type_file.get_text_void().as_deref());
            } else {
                // ((RadioButton.RadioButtonModel) bgMaskType.getSelection())
                // .getEnumeratedType()
                let selected = self
                    .bg_mask_type
                    .get_selection()
                    .and_then(|button| button.get_model())
                    .and_then(|model| {
                        model
                            .as_any()
                            .downcast_ref::<RadioButtonModel>()
                            .and_then(|model| model.get_enumerated_type())
                    })
                    .and_then(|enumerated_type| enumerated_type.downcast_ref::<MaskType>().copied());
                if let Some(selected) = selected {
                    matlab_param.set_mask_type_enumerated_type(selected);
                }
                if self.rb_mask_type_cylinder.is_selected() {
                    let height = self.ltf_cylinder_height.get_text_boolean(do_validation)?;
                    matlab_param.set_cylinder_height(height.as_deref());
                }
            }
            if !self.rb_mask_type_none.is_selected() {
                let blur = self.ltf_mask_blur_std_dev.get_text_boolean(do_validation)?;
                matlab_param.set_mask_blur_std_dev(blur.as_deref());
            }
            if self.cb_cylinder_orientation.is_enabled() && self.cb_cylinder_orientation.is_selected() {
                let z_rotation = self.ltf_z_rotation.get_text_boolean(do_validation)?;
                let y_rotation = self.ltf_y_rotation.get_text_boolean(do_validation)?;
                matlab_param.set_mask_model_pts(z_rotation.as_deref(), y_rotation.as_deref());
            } else {
                matlab_param.clear_mask_model_pts();
            }
            let inside = self.ltf_inside_mask_radius.get_text_boolean(do_validation)?;
            matlab_param.set_inside_mask_radius(inside.as_deref());
            let outside = self.ltf_outside_mask_radius.get_text_boolean(do_validation)?;
            matlab_param.set_outside_mask_radius(outside.as_deref());
            Ok(())
        })();
        result.is_ok()
    }

    /// Java package-private `setDefaults()`.
    pub fn set_defaults(&self) {
        self.rb_mask_type_none.set_selected_boolean(true);
    }

    /// Java package-private `validateRun()`.  Returns null if valid, error message if
    /// invalid.
    ///
    /// Fixed in translation (MaskingPanel.java:476-482): the Y rotation check tests
    /// `ltfZRotation` (enabled state and text) a second time, so a non-numeric Y
    /// rotation reaches prmParser; it tests `ltfYRotation`.  (BUGS.md)
    pub fn validate_run(&self) -> Option<String> {
        // Masking
        // volume
        if self.rb_mask_type_file.is_selected() && self.ftf_mask_type_file.is_empty() {
            let volume_label = MaskType::Volume
                .get_label()
                .unwrap_or_else(|| "null".to_owned());
            return Some(format!(
                "In {MAST_TYPE_LABEL}, {volume_label} is required when {volume_label} {MAST_TYPE_LABEL} is selected. "
            ));
        }
        // validate radii
        if (self.rb_mask_type_sphere.is_selected() || self.rb_mask_type_cylinder.is_selected())
            && self.ltf_inside_mask_radius.is_enabled()
            && self.ltf_inside_mask_radius.is_empty()
            && self.ltf_outside_mask_radius.is_enabled()
            && self.ltf_outside_mask_radius.is_empty()
        {
            return Some(format!(
                "In {MAST_TYPE_LABEL}, {INSIDE_MASK_RADIUS_LABEL} and/or {OUTSIDE_MASK_RADIUS_LABEL} are required when either {} or {} is selected.",
                MaskType::Sphere
                    .get_label()
                    .unwrap_or_else(|| "null".to_owned()),
                MaskType::Cylinder
                    .get_label()
                    .unwrap_or_else(|| "null".to_owned())
            ));
        }
        // validate cylinder orientation
        let mut rotation = EtomoNumber::new_with_type(Some(Type::Double));
        let root_name = self
            .pnl_root
            .get_component()
            .get_name()
            .unwrap_or_else(|| "null".to_owned());
        // validate Z Rotation
        if self.ltf_z_rotation.is_enabled() {
            rotation.set_string(self.ltf_z_rotation.get_text_void().as_deref());
            let description = format!("{} field in {root_name}", self.ltf_z_rotation.get_label());
            if !rotation.is_valid() {
                return Some(format!(
                    "{description} must be numeric - {}.",
                    rotation.get_invalid_reason()
                ));
            }
            if !rotation.is_null() {
                let f_z_rotation = rotation.get_double().abs();
                if f_z_rotation < 0.0 || f_z_rotation > 90.0 {
                    return Some(format!("Valid values for {description} are 0 to 90."));
                }
            }
        }
        // validate Y Rotation
        if self.ltf_y_rotation.is_enabled() {
            rotation.set_string(self.ltf_y_rotation.get_text_void().as_deref());
            if !rotation.is_valid() {
                return Some(format!(
                    "{} field in {root_name} must be numeric - {}.",
                    self.ltf_y_rotation.get_label(),
                    rotation.get_invalid_reason()
                ));
            }
        }
        None
    }

    /// Java private `action(String, Run3dmodMenuOptions)`.
    fn action(&self, action_command: &str, _run_3dmod_menu_options: Option<Run3dmodMenuOptions>) {
        let action_command = Some(action_command);
        if action_command == self.rb_mask_type_none.get_action_command().as_deref()
            || action_command == self.rb_mask_type_file.get_action_command().as_deref()
            || action_command == self.rb_mask_type_sphere.get_action_command().as_deref()
            || action_command == self.rb_mask_type_cylinder.get_action_command().as_deref()
            || action_command == self.cb_cylinder_orientation.get_action_command().as_deref()
        {
            self.parent().update_display(false);
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        let autodoc = match unsafe {
            autodoc_factory::get_instance(
                Some(self.manager),
                Some(autodoc_factory::PEET_PRM),
                AxisID::Only,
                false,
            )
        } {
            Ok(autodoc) => autodoc,
            Err(LogFileError::Lock(_)) => std::ptr::null_mut(),
            Err(e) => {
                eprintln!("{e}");
                std::ptr::null_mut()
            }
        };
        let autodoc = unsafe { autodoc.as_ref() }.map(|autodoc| autodoc as &dyn ReadOnlyAutodoc);
        let tooltip =
            etomo_autodoc::get_tooltip_autodoc_add_source(autodoc, Some("maskModelPts"), false);
        self.ltf_z_rotation.set_tool_tip_text(tooltip.as_deref());
        self.ltf_y_rotation.set_tool_tip_text(tooltip.as_deref());
        self.ltf_cylinder_height.set_tool_tip_text(
            etomo_autodoc::get_tooltip_autodoc_add_source(
                autodoc,
                Some(matlab_param::CYLINDER_HEIGHT_KEY),
                false,
            )
            .as_deref(),
        );
        self.ltf_mask_blur_std_dev.set_tool_tip_text(
            etomo_autodoc::get_tooltip_autodoc_add_source(
                autodoc,
                Some(matlab_param::MASK_BLUR_STD_DEV_KEY),
                false,
            )
            .as_deref(),
        );
        self.rb_mask_type_none
            .set_tool_tip_text_string(Some("No reference masking"));
        self.rb_mask_type_file
            .set_tool_tip_text_string(Some("Mask the reference using a specified file"));
        self.rb_mask_type_sphere.set_tool_tip_text_string(Some(
            "Mask the reference using inner and out spherical shells of specified radii.",
        ));
        self.rb_mask_type_cylinder.set_tool_tip_text_string(Some(
            "Mask the reference using inner and out cylindrical shells of specified radii.",
        ));
        self.ftf_mask_type_file.set_tool_tip_text(Some(
            "The name of file containing the binary mask in MRC format.",
        ));
        self.ltf_inside_mask_radius
            .set_tool_tip_text(Some("Inner radius of the mask region in pixels."));
        self.ltf_outside_mask_radius
            .set_tool_tip_text(Some("Inner and outer radii of the mask region in pixels."));
        self.cb_cylinder_orientation
            .set_tool_tip_text_string(Some("Manually specify cylindrical mask orientation."));
    }
}

impl SwingComponent for MaskingPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }
}

impl UIComponent for MaskingPanel {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }
}
