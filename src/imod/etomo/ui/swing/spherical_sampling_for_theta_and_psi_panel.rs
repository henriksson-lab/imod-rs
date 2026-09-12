//! `IMOD/Etomo/src/etomo/ui/swing/SphericalSamplingForThetaAndPsiPanel.java`.
//!
//! Swing component construction and `BaseManager` message presentation remain
//! explicit boundaries.  The source unit's selection, field, parameter, and
//! validation behavior is retained here without introducing a second PEET
//! controller.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::ui::field_type::FieldType;

use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};

pub const SAMPLE_INTERVAL_LABEL: &str = "Sample interval";
pub const SAMPLE_SPHERE_LABEL: &str = "Sample sphere";

/// Java `MatlabParam.SampleSphere`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleSphere {
    None,
    Full,
    Half,
}

impl SampleSphere {
    /// Java `SampleSphere.toString()`.
    pub const fn to_string(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Full => "full",
            Self::Half => "half",
        }
    }
}

/// Java `SphericalSamplingForThetaAndPsiParent`.
pub trait SphericalSamplingForThetaAndPsiParent {
    fn update_display(&mut self, init: bool);
}

/// The `MatlabParam` calls made by this source unit.
pub trait SphericalSamplingForThetaAndPsiMatlabParam {
    fn get_sample_sphere(&self) -> SampleSphere;
    fn get_sample_interval(&self) -> String;
    fn set_sample_sphere(&mut self, sample_sphere: SampleSphere);
    fn set_sample_interval(&mut self, sample_interval: String);
}

/// Java `UIHarness.INSTANCE.openMessageDialog(BaseManager, ...)` boundary.
pub trait SphericalSamplingForThetaAndPsiManager {
    fn open_message_dialog(&mut self, message: String, title: &str);
}

/// Source-visible Swing layout and listener attachment state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SphericalSamplingForThetaAndPsiPanelLayout {
    pub root_box_layout_x_axis: bool,
    pub root_border_title: Option<String>,
    pub root_component_order: Vec<String>,
    pub action_listener_registered: bool,
}

/// Java final `SphericalSamplingForThetaAndPsiPanel`.
pub struct SphericalSamplingForThetaAndPsiPanel<
    M: SphericalSamplingForThetaAndPsiManager,
    P: SphericalSamplingForThetaAndPsiParent,
> {
    pub pnl_root: SphericalSamplingForThetaAndPsiPanelLayout,
    pub bg_sample_sphere: Rc<RefCell<RadioButtonGroup>>,
    pub rb_sample_sphere_none: RadioButton,
    pub rb_sample_sphere_full: RadioButton,
    pub rb_sample_sphere_half: RadioButton,
    pub ltf_sample_interval: LabeledTextField,
    pub manager: M,
    pub parent: P,
}

impl<M: SphericalSamplingForThetaAndPsiManager, P: SphericalSamplingForThetaAndPsiParent>
    SphericalSamplingForThetaAndPsiPanel<M, P>
{
    /// Java private constructor.
    fn new(manager: M, parent: P) -> Self {
        let bg_sample_sphere = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root: SphericalSamplingForThetaAndPsiPanelLayout::default(),
            rb_sample_sphere_none: RadioButton::new_with_enumerated_type(
                Some("None".into()),
                EnumeratedTypeBoundary {
                    label: "None".into(),
                    default: true,
                    value: Some(SampleSphere::None.to_string().into()),
                },
                Some(bg_sample_sphere.clone()),
            ),
            rb_sample_sphere_full: RadioButton::new_with_enumerated_type(
                Some("Full sphere".into()),
                EnumeratedTypeBoundary {
                    label: "Full sphere".into(),
                    default: false,
                    value: Some(SampleSphere::Full.to_string().into()),
                },
                Some(bg_sample_sphere.clone()),
            ),
            rb_sample_sphere_half: RadioButton::new_with_enumerated_type(
                Some("Half sphere".into()),
                EnumeratedTypeBoundary {
                    label: "Half sphere".into(),
                    default: false,
                    value: Some(SampleSphere::Half.to_string().into()),
                },
                Some(bg_sample_sphere.clone()),
            ),
            bg_sample_sphere,
            ltf_sample_interval: LabeledTextField::new(
                FieldType::FloatingPoint,
                &format!("{SAMPLE_INTERVAL_LABEL} (degrees) : "),
            ),
            manager,
            parent,
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(manager: M, parent: P) -> Self {
        let mut instance = Self::new(manager, parent);
        instance.create_panel();
        instance.set_tooltips();
        instance.add_listeners();
        instance
    }

    /// Java private `addListeners`.
    fn add_listeners(&mut self) {
        let _action_listener = SphericalSamplingForThetaAndPsiActionListener::new();
        self.rb_sample_sphere_none.add_action_listener();
        self.rb_sample_sphere_full.add_action_listener();
        self.rb_sample_sphere_half.add_action_listener();
        self.pnl_root.action_listener_registered = true;
    }

    /// Java private `createPanel`.
    fn create_panel(&mut self) {
        self.ltf_sample_interval.set_preferred_width(60, None);
        self.pnl_root.root_box_layout_x_axis = true;
        self.pnl_root.root_border_title = Some(SAMPLE_SPHERE_LABEL.into());
        self.pnl_root.root_component_order = vec![
            "FixedDim.x20_y0".into(),
            "rbSampleSphereNone".into(),
            "FixedDim.x10_y0".into(),
            "rbSampleSphereFull".into(),
            "FixedDim.x10_y0".into(),
            "rbSampleSphereHalf".into(),
            "FixedDim.x70_y0".into(),
            "ltfSampleInterval".into(),
            "Box.createHorizontalGlue".into(),
        ];
    }

    /// Java `getUIComponent`.
    pub fn get_ui_component(&self) -> &Self {
        self
    }

    /// Java `getComponent`; actual Swing `Component` presentation is a GUI boundary.
    pub fn get_component(&self) -> &SphericalSamplingForThetaAndPsiPanelLayout {
        &self.pnl_root
    }

    /// Java `updateDisplay`.
    pub fn update_display(&mut self) {
        self.ltf_sample_interval
            .set_enabled(!self.rb_sample_sphere_none.is_selected());
    }

    /// Java `isSampleSphereNoneSelected`.
    pub fn is_sample_sphere_none_selected(&self) -> bool {
        self.rb_sample_sphere_none.is_selected()
    }

    /// Java `setParameters(MatlabParam)`.
    pub fn set_parameters<T: SphericalSamplingForThetaAndPsiMatlabParam>(
        &mut self,
        matlab_param: &T,
    ) {
        match matlab_param.get_sample_sphere() {
            SampleSphere::None => self.rb_sample_sphere_none.set_selected(true),
            SampleSphere::Full => self.rb_sample_sphere_full.set_selected(true),
            SampleSphere::Half => self.rb_sample_sphere_half.set_selected(true),
        }
        self.ltf_sample_interval
            .set_text(&matlab_param.get_sample_interval());
    }

    /// Java `getParameters(MatlabParam, boolean)`.
    pub fn get_parameters<T: SphericalSamplingForThetaAndPsiMatlabParam>(
        &self,
        matlab_param: &mut T,
        do_validation: bool,
    ) -> bool {
        let sample_sphere = if self.rb_sample_sphere_none.is_selected() {
            SampleSphere::None
        } else if self.rb_sample_sphere_full.is_selected() {
            SampleSphere::Full
        } else if self.rb_sample_sphere_half.is_selected() {
            SampleSphere::Half
        } else {
            // Java dereferences `bgSampleSphere.getSelection()` here.
            panic!("bgSampleSphere.getSelection() is null")
        };
        matlab_param.set_sample_sphere(sample_sphere);
        match self.ltf_sample_interval.get_text_validated(do_validation) {
            Ok(sample_interval) => {
                matlab_param.set_sample_interval(sample_interval);
                true
            }
            Err(FieldValidationFailedException(_)) => false,
        }
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.ltf_sample_interval.clear();
    }

    /// Java `setDefaults`.
    pub fn set_defaults(&mut self) {
        self.rb_sample_sphere_none.set_selected(true);
    }

    /// Java `validateRun`.
    pub fn validate_run(&mut self) -> bool {
        if (self.rb_sample_sphere_full.is_selected() || self.rb_sample_sphere_half.is_selected())
            && self.ltf_sample_interval.is_enabled()
            && self.ltf_sample_interval.is_empty()
        {
            self.manager.open_message_dialog(
                format!(
                    "In {SAMPLE_SPHERE_LABEL}, {SAMPLE_INTERVAL_LABEL} is required when either {} or {} is selected.",
                    SampleSphere::Full.to_string(),
                    SampleSphere::Half.to_string(),
                ),
                "Entry Error",
            );
            return false;
        }
        true
    }

    /// Java private `action(String)`.
    fn action(&mut self, action_command: &str) {
        if action_command == self.rb_sample_sphere_none.get_action_command()
            || action_command == self.rb_sample_sphere_full.get_action_command()
            || action_command == self.rb_sample_sphere_half.get_action_command()
        {
            self.parent.update_display(false);
        }
    }

    /// Java private `setTooltips`.
    fn set_tooltips(&mut self) {
        self.rb_sample_sphere_none.set_tool_tip_text(Some(
            "Use the angular search parameters specified in the Iteration Table for the first iteration.",
        ));
        self.rb_sample_sphere_full.set_tool_tip_text(Some(
            "At the first iteration, perform an optimized search with Theta varying from -90 to 90 degrees and Psi, varying from -180 to 180 degrees. Optimization prevents over-sampling near the poles. Phi Max should be set to 180 degrees.",
        ));
        self.rb_sample_sphere_half.set_tool_tip_text(Some(
            "At the first iteration, perform an optimized search with Theta and Psi both varying from -90 to 90 degrees. Optimization prevents over-sampling near the poles. Phi Max should be set to 180 degrees.",
        ));
        self.ltf_sample_interval.set_tool_tip_text(Some(
            "The interval, in degrees, at which theta will be sampled when using spherical sampling. Psi will also be sampled at this interval at the equator, and with decreasing frequency near the poles.",
        ));
    }
}

/// Java private `SphericalSamplingForThetaAndPsiActionListener`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SphericalSamplingForThetaAndPsiActionListener;

impl SphericalSamplingForThetaAndPsiActionListener {
    /// Java private `SphericalSamplingForThetaAndPsiActionListener(...)` constructor.
    fn new() -> Self {
        Self
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed<
        M: SphericalSamplingForThetaAndPsiManager,
        P: SphericalSamplingForThetaAndPsiParent,
    >(
        &self,
        spherical_sampling_for_theta_and_psi_panel: &mut SphericalSamplingForThetaAndPsiPanel<M, P>,
        action_command: &str,
    ) {
        spherical_sampling_for_theta_and_psi_panel.action(action_command);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Manager(Vec<(String, String)>);
    impl SphericalSamplingForThetaAndPsiManager for Manager {
        fn open_message_dialog(&mut self, message: String, title: &str) {
            self.0.push((message, title.into()));
        }
    }

    #[derive(Default)]
    struct Parent(Vec<bool>);
    impl SphericalSamplingForThetaAndPsiParent for Parent {
        fn update_display(&mut self, init: bool) {
            self.0.push(init);
        }
    }

    #[derive(Default)]
    struct Matlab {
        sample_sphere: Option<SampleSphere>,
        sample_interval: String,
    }
    impl SphericalSamplingForThetaAndPsiMatlabParam for Matlab {
        fn get_sample_sphere(&self) -> SampleSphere {
            self.sample_sphere.unwrap_or(SampleSphere::None)
        }
        fn get_sample_interval(&self) -> String {
            self.sample_interval.clone()
        }
        fn set_sample_sphere(&mut self, sample_sphere: SampleSphere) {
            self.sample_sphere = Some(sample_sphere);
        }
        fn set_sample_interval(&mut self, sample_interval: String) {
            self.sample_interval = sample_interval;
        }
    }

    #[test]
    fn creates_source_layout_and_tooltips() {
        let panel = SphericalSamplingForThetaAndPsiPanel::get_instance(
            Manager::default(),
            Parent::default(),
        );
        assert!(panel.pnl_root.root_box_layout_x_axis);
        assert_eq!(
            panel.pnl_root.root_border_title.as_deref(),
            Some(SAMPLE_SPHERE_LABEL)
        );
        assert_eq!(
            panel.ltf_sample_interval.text_preferred_size.unwrap().width,
            60
        );
        assert_eq!(
            panel.rb_sample_sphere_full.radio_button.action_command,
            None
        );
        assert_eq!(
            panel.rb_sample_sphere_half.get_tooltip(),
            Some(
                "<html>At the first iteration, perform an optimized search with Theta and Psi both varying from -90 to 90 degrees. Optimization prevents over-sampling near the poles. Phi Max should be set to 180 degrees."
            )
        );
    }

    #[test]
    fn routes_parameter_selection_and_validation() {
        let mut panel = SphericalSamplingForThetaAndPsiPanel::get_instance(
            Manager::default(),
            Parent::default(),
        );
        panel.rb_sample_sphere_half.set_selected(true);
        panel.ltf_sample_interval.set_text("4.5");
        let mut matlab = Matlab::default();
        assert!(panel.get_parameters(&mut matlab, true));
        assert_eq!(matlab.sample_sphere, Some(SampleSphere::Half));
        assert_eq!(matlab.sample_interval, "4.5");

        panel.set_parameters(&Matlab {
            sample_sphere: Some(SampleSphere::Full),
            sample_interval: "6".into(),
        });
        assert!(panel.rb_sample_sphere_full.is_selected());
        assert_eq!(panel.ltf_sample_interval.get_text(), "6");
    }

    #[test]
    fn validates_only_active_spherical_sampling_and_dispatches_action() {
        let mut panel = SphericalSamplingForThetaAndPsiPanel::get_instance(
            Manager::default(),
            Parent::default(),
        );
        panel.rb_sample_sphere_full.set_selected(true);
        panel.update_display();
        assert!(!panel.validate_run());
        assert_eq!(panel.manager.0.len(), 1);
        assert!(panel.manager.0[0].0.contains("full"));

        let listener = SphericalSamplingForThetaAndPsiActionListener;
        listener.action_performed(&mut panel, "Full sphere");
        assert_eq!(panel.parent.0, vec![false]);

        panel.rb_sample_sphere_none.set_selected(true);
        panel.update_display();
        assert!(panel.validate_run());
    }
}
