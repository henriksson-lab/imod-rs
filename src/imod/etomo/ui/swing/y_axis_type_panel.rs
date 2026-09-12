//! `IMOD/Etomo/src/etomo/ui/swing/YAxisTypePanel.java`.
//!
//! Swing construction, Autodoc lookup, and parent-display notification remain
//! explicit boundaries.  The source unit's four-way selection, parameter
//! transfer, layout order, tooltip lookup, and action filtering are retained.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::ui::shared_strings::{
    CSV_FILES_LABEL, YAXIS_TYPE_CONTOUR_LABEL, YAXIS_TYPE_LABEL, YAXIS_TYPE_PARTICLE_MODEL_LABEL,
    YAXIS_TYPE_Y_AXIS_LABEL,
};
use crate::imod::etomo::util::utilities::APRIL_FOOLS;

use super::check_box::Color;
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};

const Y_AXIS_CONTOUR_LABEL: &str = "End points of contour";

/// Java `MatlabParam.YAxisType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YAxisType {
    YAxis,
    ParticleModel,
    Contour,
    CsvFiles,
}

impl YAxisType {
    /// Java `EnumeratedType.toString()`.
    pub const fn to_string(self) -> &'static str {
        match self {
            Self::YAxis => "0",
            Self::ParticleModel => "1",
            Self::Contour => "2",
            Self::CsvFiles => "3",
        }
    }
}

/// Java `MatlabParam` calls made by this source unit.
pub trait YAxisTypeMatlabParam {
    fn get_y_axis_type(&self) -> YAxisType;
    fn set_yaxis_type(&mut self, y_axis_type: YAxisType);
}

/// Java `YAxisTypeParent`.
pub trait YAxisTypeParent {
    fn update_display(&mut self, init: bool);
}

/// Information returned by the `AutodocFactory`/`ReadOnlyAutodoc` boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct YAxisTypeAutodoc {
    pub autodoc_name: String,
    pub y_axis_type_section: String,
}

/// Java `AutodocFactory.getInstance` failure classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum YAxisTypeAutodocError {
    LogFileException,
    IOException,
    LockException,
}

/// Java `BaseManager` plus Autodoc access used by `setTooltips`.
pub trait YAxisTypeManager {
    fn get_peet_prm_autodoc(&mut self) -> Result<YAxisTypeAutodoc, YAxisTypeAutodocError>;
    fn print_stack_trace(&mut self, error: YAxisTypeAutodocError);
}

/// Source-observable `SpacedPanel`, `EtchedBorder`, and `Box` construction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct YAxisTypePanelLayout {
    pub root_box_layout_x_axis: bool,
    pub root_border_title: Option<String>,
    pub root_component_order: Vec<String>,
    pub y_axis_type_box_layout_y_axis: bool,
    pub y_axis_type_alignment_x_left: bool,
    pub y_axis_type_component_order: Vec<String>,
    pub root_background: Option<Color>,
    pub y_axis_type_background: Option<Color>,
    pub action_listener_registered: bool,
}

/// Java package-private final `YAxisTypePanel`.
pub struct YAxisTypePanel<M: YAxisTypeManager, P: YAxisTypeParent> {
    pub pnl_root: YAxisTypePanelLayout,
    pub bg_y_axis_type: Rc<RefCell<RadioButtonGroup>>,
    pub rb_y_axis_type_y_axis: RadioButton,
    pub rb_y_axis_type_particle_model: RadioButton,
    pub rb_y_axis_type_contour: RadioButton,
    pub rb_y_axis_type_csv_files: RadioButton,
    pub parent: P,
    pub manager: M,
}

impl<M: YAxisTypeManager, P: YAxisTypeParent> YAxisTypePanel<M, P> {
    /// Java private `YAxisTypePanel(BaseManager, YAxisTypeParent)`.
    fn new(manager: M, parent: P) -> Self {
        let bg_y_axis_type = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root: YAxisTypePanelLayout::default(),
            rb_y_axis_type_y_axis: RadioButton::new_with_enumerated_type(
                Some(YAXIS_TYPE_Y_AXIS_LABEL.into()),
                EnumeratedTypeBoundary {
                    label: YAXIS_TYPE_Y_AXIS_LABEL.into(),
                    default: true,
                    value: Some(YAxisType::YAxis.to_string().into()),
                },
                Some(bg_y_axis_type.clone()),
            ),
            rb_y_axis_type_particle_model: RadioButton::new_with_enumerated_type(
                Some(YAXIS_TYPE_PARTICLE_MODEL_LABEL.into()),
                EnumeratedTypeBoundary {
                    label: YAXIS_TYPE_PARTICLE_MODEL_LABEL.into(),
                    default: false,
                    value: Some(YAxisType::ParticleModel.to_string().into()),
                },
                Some(bg_y_axis_type.clone()),
            ),
            rb_y_axis_type_contour: RadioButton::new_with_enumerated_type(
                Some(Y_AXIS_CONTOUR_LABEL.into()),
                EnumeratedTypeBoundary {
                    label: Y_AXIS_CONTOUR_LABEL.into(),
                    default: false,
                    value: Some(YAxisType::Contour.to_string().into()),
                },
                Some(bg_y_axis_type.clone()),
            ),
            rb_y_axis_type_csv_files: RadioButton::new_with_enumerated_type(
                Some(CSV_FILES_LABEL.into()),
                EnumeratedTypeBoundary {
                    label: CSV_FILES_LABEL.into(),
                    default: false,
                    value: Some(YAxisType::CsvFiles.to_string().into()),
                },
                Some(bg_y_axis_type.clone()),
            ),
            bg_y_axis_type,
            parent,
            manager,
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
        let _action_listener = YAxisTypeActionListener::new();
        self.rb_y_axis_type_y_axis.add_action_listener();
        self.rb_y_axis_type_particle_model.add_action_listener();
        self.rb_y_axis_type_contour.add_action_listener();
        self.rb_y_axis_type_csv_files.add_action_listener();
        self.pnl_root.action_listener_registered = true;
    }

    /// Java private `createPanel`.
    fn create_panel(&mut self) {
        if *APRIL_FOOLS {
            let background = Color(255, 239, 148);
            self.pnl_root.root_background = Some(background);
            self.pnl_root.y_axis_type_background = Some(background);
        }
        self.pnl_root.root_box_layout_x_axis = true;
        self.pnl_root.root_border_title = Some(YAXIS_TYPE_LABEL.into());
        self.pnl_root.root_component_order = vec!["pnlYaxisType".into(), "FixedDim.x197_y0".into()];
        self.pnl_root.y_axis_type_box_layout_y_axis = true;
        self.pnl_root.y_axis_type_alignment_x_left = true;
        self.pnl_root.y_axis_type_component_order = vec![
            "rbYAxisTypeYAxis".into(),
            "rbYAxisTypeParticleModel".into(),
            "rbYAxisTypeContour.getComponent".into(),
            "rbYAxisTypeCsvFiles.getComponent".into(),
            "FixedDim.x0_y1".into(),
        ];
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &YAxisTypePanelLayout {
        &self.pnl_root
    }

    /// Java `setParameters(MatlabParam)`.
    pub fn set_parameters(&mut self, matlab_param: &impl YAxisTypeMatlabParam) {
        match matlab_param.get_y_axis_type() {
            YAxisType::YAxis => self.rb_y_axis_type_y_axis.set_selected(true),
            YAxisType::ParticleModel => self.rb_y_axis_type_particle_model.set_selected(true),
            YAxisType::Contour => self.rb_y_axis_type_contour.set_selected(true),
            YAxisType::CsvFiles => self.rb_y_axis_type_csv_files.set_selected(true),
        }
    }

    /// Java `getParameters(MatlabParam)`.
    pub fn get_parameters(&self, matlab_param: &mut impl YAxisTypeMatlabParam) {
        matlab_param.set_yaxis_type(self.get_y_axis_type());
    }

    /// Java `getYAxisType`.
    pub fn get_y_axis_type(&self) -> YAxisType {
        if self.rb_y_axis_type_y_axis.is_selected() {
            YAxisType::YAxis
        } else if self.rb_y_axis_type_particle_model.is_selected() {
            YAxisType::ParticleModel
        } else if self.rb_y_axis_type_contour.is_selected() {
            YAxisType::Contour
        } else if self.rb_y_axis_type_csv_files.is_selected() {
            YAxisType::CsvFiles
        } else {
            panic!("Java YAxisTypePanel ButtonGroup.getSelection returned null")
        }
    }

    /// Java `reset`.
    pub fn reset(&mut self) {
        self.rb_y_axis_type_y_axis.set_selected(false);
        self.rb_y_axis_type_particle_model.set_selected(false);
        self.rb_y_axis_type_contour.set_selected(false);
        self.rb_y_axis_type_csv_files.set_selected(false);
    }

    /// Java private `action(String, Run3dmodMenuOptions)`.
    fn action(&mut self, action_command: &str, _run3dmod_menu_options: Option<()>) {
        if action_command == self.rb_y_axis_type_y_axis.get_action_command()
            || action_command == self.rb_y_axis_type_particle_model.get_action_command()
            || action_command == self.rb_y_axis_type_contour.get_action_command()
            || action_command == self.rb_y_axis_type_csv_files.get_action_command()
        {
            self.parent.update_display(false);
        }
    }

    /// Java private `setTooltips`.
    fn set_tooltips(&mut self) {
        let autodoc = match self.manager.get_peet_prm_autodoc() {
            Ok(autodoc) => autodoc,
            Err(
                error @ (YAxisTypeAutodocError::LogFileException
                | YAxisTypeAutodocError::IOException),
            ) => {
                self.manager.print_stack_trace(error);
                panic!("Java YAxisTypePanel dereferences null autodoc after AutodocFactory failure")
            }
            Err(YAxisTypeAutodocError::LockException) => {
                panic!("Java YAxisTypePanel dereferences null autodoc after LockException")
            }
        };
        let tooltip = format!("{}: {}", autodoc.autodoc_name, autodoc.y_axis_type_section);
        self.rb_y_axis_type_y_axis.set_tool_tip_text(Some(&tooltip));
        self.rb_y_axis_type_particle_model
            .set_tool_tip_text(Some(&tooltip));
        self.rb_y_axis_type_contour
            .set_tool_tip_text(Some(&tooltip));
        self.rb_y_axis_type_csv_files.set_tool_tip_text(Some(
            "Read particle rotation axes from file(s) [fnOutput]_Tom[n]_RotAxes.csv",
        ));
    }
}

/// Java private `YAxisTypeActionListener`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct YAxisTypeActionListener;

impl YAxisTypeActionListener {
    /// Java private `YAxisTypeActionListener(YAxisTypePanel)` constructor.
    fn new() -> Self {
        Self
    }

    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed<M: YAxisTypeManager, P: YAxisTypeParent>(
        &self,
        y_axis_type_panel: &mut YAxisTypePanel<M, P>,
        action_command: &str,
    ) {
        y_axis_type_panel.action(action_command, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Manager;
    impl YAxisTypeManager for Manager {
        fn get_peet_prm_autodoc(&mut self) -> Result<YAxisTypeAutodoc, YAxisTypeAutodocError> {
            Ok(YAxisTypeAutodoc {
                autodoc_name: "peetprm.adoc".into(),
                y_axis_type_section: "Y-axis source selection".into(),
            })
        }
        fn print_stack_trace(&mut self, _: YAxisTypeAutodocError) {}
    }
    #[derive(Default)]
    struct Parent(Vec<bool>);
    impl YAxisTypeParent for Parent {
        fn update_display(&mut self, init: bool) {
            self.0.push(init);
        }
    }
    struct Matlab(YAxisType);
    impl YAxisTypeMatlabParam for Matlab {
        fn get_y_axis_type(&self) -> YAxisType {
            self.0
        }
        fn set_yaxis_type(&mut self, y_axis_type: YAxisType) {
            self.0 = y_axis_type;
        }
    }

    #[test]
    fn creates_source_layout_listeners_and_tooltips() {
        let panel = YAxisTypePanel::get_instance(Manager, Parent::default());
        assert!(panel.pnl_root.root_box_layout_x_axis);
        assert_eq!(
            panel.pnl_root.root_border_title.as_deref(),
            Some(YAXIS_TYPE_LABEL)
        );
        assert_eq!(
            panel.pnl_root.root_component_order,
            ["pnlYaxisType", "FixedDim.x197_y0"]
        );
        assert!(panel.pnl_root.y_axis_type_alignment_x_left);
        assert!(panel.pnl_root.action_listener_registered);
        assert_eq!(
            panel.rb_y_axis_type_y_axis.get_tooltip(),
            Some("<html>peetprm.adoc: Y-axis source selection")
        );
        assert_eq!(
            panel.rb_y_axis_type_csv_files.get_tooltip(),
            Some("<html>Read particle rotation axes from file(s) [fnOutput]_Tom[n]_RotAxes.csv")
        );
    }

    #[test]
    fn transfers_parameter_and_notifies_only_source_actions() {
        let mut panel = YAxisTypePanel::get_instance(Manager, Parent::default());
        panel.set_parameters(&Matlab(YAxisType::Contour));
        assert_eq!(panel.get_y_axis_type(), YAxisType::Contour);
        let mut matlab = Matlab(YAxisType::YAxis);
        panel.get_parameters(&mut matlab);
        assert_eq!(matlab.0, YAxisType::Contour);
        let listener = YAxisTypeActionListener::new();
        listener.action_performed(&mut panel, "unrelated");
        listener.action_performed(&mut panel, Y_AXIS_CONTOUR_LABEL);
        assert_eq!(panel.parent.0, [false]);
    }

    #[test]
    fn reset_retains_java_button_group_selection() {
        let mut panel = YAxisTypePanel::get_instance(Manager, Parent::default());
        panel.set_parameters(&Matlab(YAxisType::CsvFiles));
        panel.reset();
        assert_eq!(panel.get_y_axis_type(), YAxisType::CsvFiles);
    }
}
