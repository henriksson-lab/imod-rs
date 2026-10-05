//! `IMOD/Etomo/src/etomo/ui/swing/SphericalSamplingForThetaAndPsiPanel.java`.
//!
//! The PEET dialog's "Spherical Sampling for Theta and Psi" box.  An event dispatch
//! thread object, created as `Rc<Self>` by
//! [`SphericalSamplingForThetaAndPsiPanel::get_instance`]; it keeps a weak reference to
//! its parent.

use std::rc::{Rc, Weak};

use super::abstract_radio_button_model::AbstractRadioButtonModel;
use super::etched_border::EtchedBorder;
use super::etomo_panel::EtomoPanel;
use super::labeled_text_field::LabeledTextField;
use super::radio_button::{RadioButton, RadioButtonModel};
use super::radio_button_interface::EnumeratedTypeRef;
use super::spherical_sampling_for_theta_and_psi_parent::SphericalSamplingForThetaAndPsiParent;
use super::swing_component::SwingComponent;
use super::ui_harness;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, ButtonGroup, JComponent};
use crate::imod::etomo::storage::matlab_param::{MatlabParam, SampleSphere};
use crate::imod::etomo::ui::field::Field;
use crate::imod::etomo::ui::field_type::FieldType;
use crate::imod::etomo::ui::shared_strings;
use crate::imod::etomo::ui::ui_component::UIComponent;

/// Java private static final `SAMPLE_INTERVAL_LABEL`.
const SAMPLE_INTERVAL_LABEL: &str = "Sample interval";

/// Java package-private `final class SphericalSamplingForThetaAndPsiPanel implements
/// UIComponent, SwingComponent`.
pub struct SphericalSamplingForThetaAndPsiPanel {
    /// Java private final `pnlRoot`.
    pnl_root: Rc<EtomoPanel>,
    /// Java private final `bgSampleSphere`.
    bg_sample_sphere: Rc<ButtonGroup>,
    /// Java private final `rbSampleSphereNone`.
    rb_sample_sphere_none: Rc<RadioButton>,
    /// Java private final `rbSampleSphereFull`.
    rb_sample_sphere_full: Rc<RadioButton>,
    /// Java private final `rbSampleSphereHalf`.
    rb_sample_sphere_half: Rc<RadioButton>,
    /// Java private final `ltfSampleInterval`.
    ltf_sample_interval: Rc<LabeledTextField>,
    /// Java private final `manager`.
    manager: &'static dyn BaseManager,
    /// Java private final `parent`.
    parent: Weak<dyn SphericalSamplingForThetaAndPsiParent>,
    /// Java `this`.
    self_ref: Weak<SphericalSamplingForThetaAndPsiPanel>,
}

impl SphericalSamplingForThetaAndPsiPanel {
    /// Java private `SphericalSamplingForThetaAndPsiPanel(BaseManager,
    /// SphericalSamplingForThetaAndPsiParent)`, with the field initializers.
    fn new(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn SphericalSamplingForThetaAndPsiParent>,
    ) -> Rc<SphericalSamplingForThetaAndPsiPanel> {
        let bg_sample_sphere = ButtonGroup::new();
        Rc::new_cyclic(|self_ref: &Weak<SphericalSamplingForThetaAndPsiPanel>| {
            SphericalSamplingForThetaAndPsiPanel {
                pnl_root: EtomoPanel::new(),
                rb_sample_sphere_none: RadioButton::new_string_enumerated_type_button_group(
                    Some("None"),
                    Some(EnumeratedTypeRef::new(SampleSphere::None)),
                    Some(&bg_sample_sphere),
                ),
                rb_sample_sphere_full: RadioButton::new_string_enumerated_type_button_group(
                    Some("Full sphere"),
                    Some(EnumeratedTypeRef::new(SampleSphere::Full)),
                    Some(&bg_sample_sphere),
                ),
                rb_sample_sphere_half: RadioButton::new_string_enumerated_type_button_group(
                    Some("Half sphere"),
                    Some(EnumeratedTypeRef::new(SampleSphere::Half)),
                    Some(&bg_sample_sphere),
                ),
                bg_sample_sphere,
                ltf_sample_interval: LabeledTextField::new_field_type_string(
                    FieldType::FloatingPoint,
                    Some(&format!("{SAMPLE_INTERVAL_LABEL} (degrees) : ")),
                ),
                manager,
                parent,
                self_ref: self_ref.clone(),
            }
        })
    }

    /// Java static `getInstance(BaseManager, SphericalSamplingForThetaAndPsiParent)`.
    pub fn get_instance(
        manager: &'static dyn BaseManager,
        parent: Weak<dyn SphericalSamplingForThetaAndPsiParent>,
    ) -> Rc<SphericalSamplingForThetaAndPsiPanel> {
        let instance = SphericalSamplingForThetaAndPsiPanel::new(manager, parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners()` with
    /// `SphericalSamplingForThetaAndPsiActionListener`.
    fn add_listeners(&self) {
        let adaptee = self.self_ref.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(panel) = adaptee.upgrade() {
                panel.action(event.get_action_command().unwrap_or(""));
            }
        });
        self.rb_sample_sphere_none
            .add_action_listener(action_listener.clone());
        self.rb_sample_sphere_full
            .add_action_listener(action_listener.clone());
        self.rb_sample_sphere_half
            .add_action_listener(action_listener);
    }

    /// Java private `createPanel()`.
    fn create_panel(&self) {
        // init
        self.ltf_sample_interval.set_preferred_width(60);
        // root (BoxLayout X_AXIS, rigid areas between, horizontal glue at the end)
        self.pnl_root
            .set_border(&EtchedBorder::new(Some(shared_strings::SAMPLE_SPHERE_LABEL)).get_border());
        let root = self.pnl_root.get_component();
        root.add(&self.rb_sample_sphere_none.get_component());
        root.add(&self.rb_sample_sphere_full.get_component());
        root.add(&self.rb_sample_sphere_half.get_component());
        root.add(&self.ltf_sample_interval.get_container());
    }

    /// Java package-private `updateDisplay()`.  Called from parent updateDisplay().
    pub fn update_display(&self) {
        self.ltf_sample_interval
            .set_enabled(!self.rb_sample_sphere_none.is_selected());
    }

    /// Java package-private `isSampleSphereNoneSelected()`.
    pub fn is_sample_sphere_none_selected(&self) -> bool {
        self.rb_sample_sphere_none.is_selected()
    }

    /// Java package-private `setParameters(MatlabParam)`.  Load data from
    /// MatlabParamFile.
    pub fn set_parameters(&self, matlab_param: &MatlabParam) {
        let sample_sphere = matlab_param.get_sample_sphere(Some(self as &dyn UIComponent));
        if sample_sphere == SampleSphere::None {
            self.rb_sample_sphere_none.set_selected_boolean(true);
        } else if sample_sphere == SampleSphere::Full {
            self.rb_sample_sphere_full.set_selected_boolean(true);
        } else if sample_sphere == SampleSphere::Half {
            self.rb_sample_sphere_half.set_selected_boolean(true);
        }
        self.ltf_sample_interval
            .set_text_string(matlab_param.get_sample_interval().as_deref());
    }

    /// Java package-private `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters(&self, matlab_param: &mut MatlabParam, do_validation: bool) -> bool {
        // ((RadioButton.RadioButtonModel) bgSampleSphere.getSelection())
        // .getEnumeratedType()
        let selected = self
            .bg_sample_sphere
            .get_selection()
            .and_then(|button| button.get_model())
            .and_then(|model| {
                model
                    .as_any()
                    .downcast_ref::<RadioButtonModel>()
                    .and_then(|model| model.get_enumerated_type())
            })
            .and_then(|enumerated_type| enumerated_type.downcast_ref::<SampleSphere>().copied());
        // Fixed in translation: with no radio button selected Java throws
        // NullPointerException; the sample sphere is left as it is.
        if let Some(selected) = selected {
            matlab_param.set_sample_sphere(selected);
        }
        let Ok(sample_interval) = self.ltf_sample_interval.get_text_boolean(do_validation) else {
            // catch (FieldValidationFailedException e) { return false; }
            return false;
        };
        matlab_param.set_sample_interval(sample_interval.as_deref());
        true
    }

    /// Java package-private `reset()`.  Reset values and set defaults.
    pub fn reset(&self) {
        self.ltf_sample_interval.clear();
    }

    /// Java package-private `setDefaults()`.
    pub fn set_defaults(&self) {
        self.rb_sample_sphere_none.set_selected_boolean(true);
    }

    /// Java package-private `validateRun()`.  Validate for run.  Pops up error message
    /// if invalid.
    pub fn validate_run(&self) -> bool {
        // spherical sampling for theta and psi:
        // If full sphere or half sphere is selected, sample interval is required.
        if (self.rb_sample_sphere_full.is_selected() || self.rb_sample_sphere_half.is_selected())
            && self.ltf_sample_interval.is_enabled()
            && self.ltf_sample_interval.is_empty()
        {
            ui_harness::with(|harness| {
                harness.open_message_dialog_base_manager_string_string(
                    Some(self.manager),
                    &format!(
                        "In {}, {SAMPLE_INTERVAL_LABEL} is required when either {} or {} is selected.",
                        shared_strings::SAMPLE_SPHERE_LABEL,
                        SampleSphere::Full,
                        SampleSphere::Half
                    ),
                    "Entry Error",
                )
            });
            return false;
        }
        true
    }

    /// Java private `action(String)`.
    fn action(&self, action_command: &str) {
        let action_command = Some(action_command);
        if action_command == self.rb_sample_sphere_none.get_action_command().as_deref()
            || action_command == self.rb_sample_sphere_full.get_action_command().as_deref()
            || action_command == self.rb_sample_sphere_half.get_action_command().as_deref()
        {
            if let Some(parent) = self.parent.upgrade() {
                parent.update_display(false);
            }
        }
    }

    /// Java private `setTooltips()`.
    fn set_tooltips(&self) {
        self.rb_sample_sphere_none.set_tool_tip_text_string(Some(
            "Use the angular search parameters specified in the Iteration Table for the first iteration.",
        ));
        self.rb_sample_sphere_full.set_tool_tip_text_string(Some(
            "At the first iteration, perform an optimized search with Theta varying from -90 to 90 degrees and Psi, varying from -180 to 180 degrees. Optimization prevents over-sampling near the poles. Phi Max should be set to 180 degrees.",
        ));
        self.rb_sample_sphere_half.set_tool_tip_text_string(Some(
            "At the first iteration, perform an optimized search with Theta and Psi both varying from -90 to 90 degrees. Optimization prevents over-sampling near the poles. Phi Max should be set to 180 degrees.",
        ));
        self.ltf_sample_interval.set_tool_tip_text(Some(
            "The interval, in degrees, at which theta will be sampled when using spherical sampling. Psi will also be sampled at this interval at the equator, and with decreasing frequency near the poles.",
        ));
    }
}

impl SwingComponent for SphericalSamplingForThetaAndPsiPanel {
    /// Java `getComponent()`.
    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }
}

impl UIComponent for SphericalSamplingForThetaAndPsiPanel {
    /// Java `getUIComponent()`.
    fn get_ui_component(&self) -> &dyn SwingComponent {
        self
    }

    fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.get_component()
    }
}
