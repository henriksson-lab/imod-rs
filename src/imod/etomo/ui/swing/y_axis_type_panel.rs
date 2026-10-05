//! `IMOD/Etomo/src/etomo/ui/swing/YAxisTypePanel.java`.
//!
//! The PEET dialog's "Particle Y Axis" box.  An event dispatch thread object,
//! created as `Rc<Self>` by [`YAxisTypePanel::get_instance`]; it keeps a weak
//! reference to its parent.

use std::rc::{Rc, Weak};

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::etched_border::EtchedBorder;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::spaced_panel::{self, SpacedPanel};
use super::y_axis_type_parent::YAxisTypeParent;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::storage::autodoc::autodoc_factory;
use crate::imod::etomo::storage::autodoc::read_only_autodoc::ReadOnlyAutodoc;
use crate::imod::etomo::storage::autodoc::read_only_section_list::ReadOnlySectionList;
use crate::imod::etomo::storage::log_file::LogFileError;
use crate::imod::etomo::storage::matlab_param::{MatlabParam, YAxisType};
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::etomo_autodoc;
use crate::imod::etomo::ui::shared_strings;

/// Java private static final `Y_AXIS_CONTOUR_LABEL` (unused in the source).
#[allow(dead_code)]
const Y_AXIS_CONTOUR_LABEL: &str = "End points of contour";

/// Java package-private `final class YAxisTypePanel`.
pub struct YAxisTypePanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<SpacedPanel>,
    /// Java private final `bgYAxisType`.
    bg_y_axis_type: Rc<ButtonGroup>,
    /// Java private final `rbYAxisTypeYAxis`.
    rb_y_axis_type_y_axis: Rc<RadioButton>,
    /// Java private final `rbYAxisTypeParticleModel`.
    rb_y_axis_type_particle_model: Rc<RadioButton>,
    /// Java private final `rbYAxisTypeContour`.
    rb_y_axis_type_contour: Rc<RadioButton>,
    /// Java private final `rbYAxisTypeCsvFiles`.
    rb_y_axis_type_csv_files: Rc<RadioButton>,
    /// Java private final `parent`.
    parent: Weak<dyn YAxisTypeParent>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java `this`.
    self_ref: Weak<YAxisTypePanel>,
}

impl YAxisTypePanel {
    /// Java private `YAxisTypePanel(BaseManager, YAxisTypeParent)`, with the field
    /// initializers.
    fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn YAxisTypeParent>,
    ) -> Rc<YAxisTypePanel> {
        let bg_y_axis_type = ButtonGroup::new();
        Rc::new_cyclic(|self_ref: &Weak<YAxisTypePanel>| YAxisTypePanel {
            pnl_root: SpacedPanel::get_instance_void(),
            rb_y_axis_type_y_axis: RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(YAxisType::YAxis),
                Some(&bg_y_axis_type),
            ),
            rb_y_axis_type_particle_model: RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(YAxisType::ParticleModel),
                Some(&bg_y_axis_type),
            ),
            rb_y_axis_type_contour: RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(YAxisType::Contour),
                Some(&bg_y_axis_type),
            ),
            rb_y_axis_type_csv_files: RadioButton::new_enumerated_type_button_group(
                EnumeratedTypeRef::new(YAxisType::CsvFiles),
                Some(&bg_y_axis_type),
            ),
            bg_y_axis_type,
            parent,
            manager,
            self_ref: self_ref.clone(),
        })
    }

    /// Java static `getInstance(BaseManager, YAxisTypeParent)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn YAxisTypeParent>,
    ) -> Rc<YAxisTypePanel> {
        let instance = YAxisTypePanel::new(manager, parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()` with `YAxisTypeActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(panel) = adaptee.upgrade() {
                panel.action(event.get_action_command().unwrap_or(""), None);
            }
        });
        self.rb_y_axis_type_y_axis
            .add_action_listener(action_listener.clone());
        self.rb_y_axis_type_particle_model
            .add_action_listener(action_listener.clone());
        self.rb_y_axis_type_contour
            .add_action_listener(action_listener.clone());
        self.rb_y_axis_type_csv_files
            .add_action_listener(action_listener);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // local panels
        let pnl_yaxis_type = SpacedPanel::get_instance_void();
        self.pnl_root.set_box_layout(spaced_panel::X_AXIS);
        self.pnl_root
            .set_border(&EtchedBorder::new(Some(shared_strings::YAXIS_TYPE_LABEL)).get_border());
        self.pnl_root.add_spaced_panel(&pnl_yaxis_type);
        self.pnl_root.add_rigid_area_dimension((197, 0));
        // YaxisType
        pnl_yaxis_type.set_box_layout(spaced_panel::Y_AXIS);
        pnl_yaxis_type.set_component_alignment_x(0.0);
        pnl_yaxis_type.add_radio_button(&self.rb_y_axis_type_y_axis);
        pnl_yaxis_type.add_radio_button(&self.rb_y_axis_type_particle_model);
        pnl_yaxis_type.add_component(&self.rb_y_axis_type_contour.get_component());
        pnl_yaxis_type.add_component(&self.rb_y_axis_type_csv_files.get_component());
        pnl_yaxis_type.add_rigid_area_dimension((0, 1));
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_container()
    }

    /// Java package-private `setParameters(MatlabParam)`.  Load data from
    /// MatlabParamFile.
    pub fn set_parameters(&self, matlab_param: &MatlabParam) {
        let yaxis_type = matlab_param.get_y_axis_type();
        if yaxis_type == YAxisType::YAxis {
            self.rb_y_axis_type_y_axis.set_selected_boolean(true);
        } else if yaxis_type == YAxisType::ParticleModel {
            self.rb_y_axis_type_particle_model
                .set_selected_boolean(true);
        } else if yaxis_type == YAxisType::Contour {
            self.rb_y_axis_type_contour.set_selected_boolean(true);
        } else if yaxis_type == YAxisType::CsvFiles {
            self.rb_y_axis_type_csv_files.set_selected_boolean(true);
        }
    }

    /// Java package-private `getParameters(MatlabParam)`.
    pub fn get_parameters(&self, matlab_param: &mut MatlabParam) {
        matlab_param.set_yaxis_type(self.get_y_axis_type());
    }

    /// Java package-private `getYAxisType()`.  (Null when no radio button is
    /// selected.)
    pub fn get_y_axis_type(&self) -> Option<YAxisType> {
        self.bg_y_axis_type
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
            .and_then(|enumerated_type| enumerated_type.downcast_ref::<YAxisType>().copied())
    }

    /// Java package-private `reset()`.
    pub fn reset(&self) {
        self.rb_y_axis_type_y_axis.set_selected_boolean(false);
        self.rb_y_axis_type_particle_model
            .set_selected_boolean(false);
        self.rb_y_axis_type_contour.set_selected_boolean(false);
        self.rb_y_axis_type_csv_files.set_selected_boolean(false);
    }

    /// Java private `action(String, Run3dmodMenuOptions)`.
    fn action(&self, action_command: &str, _run_3dmod_menu_options: Option<Run3dmodMenuOptions>) {
        let action_command = Some(action_command);
        if action_command == self.rb_y_axis_type_y_axis.get_action_command().as_deref()
            || action_command
                == self
                    .rb_y_axis_type_particle_model
                    .get_action_command()
                    .as_deref()
            || action_command == self.rb_y_axis_type_contour.get_action_command().as_deref()
            || action_command
                == self
                    .rb_y_axis_type_csv_files
                    .get_action_command()
                    .as_deref()
        {
            if let Some(parent) = self.parent.upgrade() {
                parent.update_display(false);
            }
        }
    }

    /// Java private `setTooltips()`.
    ///
    /// Fixed in translation (YAxisTypePanel.java:166): with the peetprm autodoc
    /// missing (null), Java dereferences it and the dialog fails to construct; the
    /// autodoc tooltips are left unset instead.
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
        if let Some(autodoc) = unsafe { autodoc.as_ref() } {
            let autodoc_name = ReadOnlyAutodoc::get_autodoc_name(autodoc);
            let section = unsafe {
                ReadOnlySectionList::get_section(
                    autodoc,
                    Some(etomo_autodoc::FIELD_SECTION_NAME),
                    Some(YAxisType::KEY),
                )
            };
            // Java passes a null section on to setToolTipText, which formats it as no
            // tooltip; a missing section leaves the tooltips unset.
            if let Some(section) = unsafe { section.as_ref() } {
                self.rb_y_axis_type_y_axis
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
                self.rb_y_axis_type_particle_model
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
                self.rb_y_axis_type_contour
                    .set_tool_tip_text_string_read_only_section(Some(&autodoc_name), section);
            }
        }
        self.rb_y_axis_type_csv_files.set_tool_tip_text_string(Some(
            "Read particle rotation axes from file(s) [fnOutput]_Tom[n]_RotAxes.csv",
        ));
    }
}
