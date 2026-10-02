//! `IMOD/Etomo/src/etomo/ui/swing/TiltAnglePanelExpert.java`.
//!
//! Bug# 1052 turned `TiltAnglePanel` into an extremely thin GUI and moved its
//! decisions and knowledge here.  The expert constructs its panel and hands
//! itself to it (`new TiltAnglePanel(manager, this, axisID)`), so it is built
//! with `Rc::new_cyclic`; it is an event-dispatch-thread object (`Rc`,
//! `&self`).

use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::jdk::{ActionEvent, JComponent};
use crate::imod::etomo::logic::dataset_tool;
use crate::imod::etomo::storage::directive_file_collection::DirectiveFileCollection;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::tilt_angle_spec::TiltAngleSpec;
use crate::imod::etomo::r#type::tilt_angle_type::TiltAngleType;
use crate::imod::etomo::r#type::user_configuration::UserConfiguration;
use crate::imod::etomo::ui::swing::tilt_angle_panel::TiltAnglePanel;
use std::rc::Rc;

/// Java `final class TiltAnglePanelExpert`.
pub struct TiltAnglePanelExpert {
    /// Java package-private final `manager` (declared `BaseManager`; the
    /// constructor receives the `ApplicationManager` it also gives the panel).
    pub manager: &'static ApplicationManager,
    /// Java package-private final `axisID`.
    pub axis_id: AxisID,
    /// Java private final `panel`.
    panel: Rc<TiltAnglePanel>,
}

impl TiltAnglePanelExpert {
    /// Java package-private `TiltAnglePanelExpert(ApplicationManager, AxisID)`.
    pub fn new(manager: &'static ApplicationManager, axis_id: AxisID) -> Rc<TiltAnglePanelExpert> {
        Rc::new_cyclic(|this| TiltAnglePanelExpert {
            manager,
            axis_id,
            panel: TiltAnglePanel::new(manager, this.clone(), axis_id),
        })
    }

    /// Java package-private `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.panel.get_component()
    }

    /// Java package-private `getPanel()`.
    pub fn get_panel(&self) -> Rc<TiltAnglePanel> {
        self.panel.clone()
    }

    /// Java package-private `setFields(TiltAngleSpec, UserConfiguration)`.
    pub fn set_fields(&self, tilt_angle_spec: &TiltAngleSpec, user_config: &UserConfiguration) {
        if tilt_angle_spec.get_type() == TiltAngleType::File
            || user_config.is_tilt_angles_rawtlt_file()
        {
            self.panel.set_file(true);
            self.enable_angle_fields(false);
        } else if tilt_angle_spec.get_type() == TiltAngleType::Extract {
            self.panel.set_extract(true);
            self.enable_angle_fields(false);
        } else if tilt_angle_spec.get_type() == TiltAngleType::Range {
            self.panel.set_specify(true);
            self.enable_angle_fields(true);
        }

        self.panel.set_min(tilt_angle_spec.get_range_min());
        self.panel.set_step(tilt_angle_spec.get_range_step());
    }

    /// Java package-private `enableAngleFields(boolean)`.
    pub fn enable_angle_fields(&self, enable: bool) {
        self.panel.set_min_enabled(enable);
        self.panel.set_step_enabled(enable);
    }

    /// Java package-private `getTiltAngleType()`; `None` is Java null.
    pub fn get_tilt_angle_type(&self) -> Option<TiltAngleType> {
        if self.panel.is_extract_selected() {
            return Some(TiltAngleType::Extract);
        }
        if self.panel.is_specify_selected() {
            return Some(TiltAngleType::Range);
        }
        if self.panel.is_file_selected() {
            return Some(TiltAngleType::File);
        }
        None
    }

    /// Java package-private `getFields(TiltAngleSpec, boolean)`.
    ///
    /// `Ok(false)` is the caught `FieldValidationFailedException`.  `Err`
    /// carries the message of the unchecked `NumberFormatException` that
    /// `TiltAngleSpec.setRangeMin(String)` / `setRangeStep(String)` throw for
    /// a malformed number: the Java does not catch it here, and
    /// `SetupReconUIHarness.getFields` does ("... must be numeric.").
    pub fn get_fields(
        &self,
        tilt_angle_spec: &mut TiltAngleSpec,
        do_validation: bool,
    ) -> Result<bool, String> {
        // `tiltAngleSpec.setType(getTiltAngleType())`: the Rust
        // `TiltAngleSpec` type cannot be null, so when no radio button is
        // selected (Java would store null) the type is left as it was.
        if let Some(tilt_angle_type) = self.get_tilt_angle_type() {
            tilt_angle_spec.set_type(tilt_angle_type);
        }
        let range_min = match self.panel.get_min_boolean(do_validation) {
            Ok(range_min) => range_min,
            // catch (final FieldValidationFailedException e)
            Err(_) => return Ok(false),
        };
        tilt_angle_spec.set_range_min_string(&range_min)?;
        let range_step = match self.panel.get_step_boolean(do_validation) {
            Ok(range_step) => range_step,
            // catch (final FieldValidationFailedException e)
            Err(_) => return Ok(false),
        };
        tilt_angle_spec.set_range_step_string(&range_step)?;
        Ok(true)
    }

    /// Java package-private `validate(String)`: validates and return an error
    /// messaage.  This panel does not use isValid() because it does not have
    /// enough information to display a complete error message.
    pub fn validate(&self, error_title: &str) -> bool {
        // The Java evaluates the arguments in this order.
        let specify = self.panel.is_specify_selected();
        let min = self.panel.get_min_void();
        let step = self.panel.get_step_void();
        dataset_tool::validate_tilt_angle(
            self.manager,
            AxisID::Only,
            Some(error_title),
            Some(self.axis_id),
            specify,
            Some(&min),
            Some(&step),
        )
    }

    /// Java package-private `checkpoint()`.
    pub fn checkpoint(&self) {
        self.panel.checkpoint();
    }

    /// Java package-private `updateTemplateValues(DirectiveFileCollection)`.
    pub fn update_template_values(&self, directive_file_collection: &DirectiveFileCollection) {
        self.panel
            .update_template_values(directive_file_collection, self.axis_id);
    }

    /// Java package-private `setEnabled(boolean)`: walk through all of the
    /// objects applying the appropriate state.
    pub fn set_enabled(&self, enable: bool) {
        self.panel.set_source_enabled(enable);
        self.panel.set_angle_enabled(enable);
        self.panel.set_extract_enabled(enable);
        self.panel.set_file_enabled(enable);
        self.panel.set_specify_enabled(enable);
        self.enable_angle_fields(self.panel.is_specify_selected() & enable);
    }

    /// Java package-private `setTooltips()`.
    pub fn set_tooltips(&self) {
        self.panel
            .set_source_tooltip("Specify the source of the view tilt angles");
        self.panel.set_extract_tooltip(
            "Select the Extract option if the tilt angles are contained in the extended header \
             of the raw image stack, or in an '.mdoc' file named with the full name of the \
             stack plus '.mdoc'",
        );
        self.panel.set_specify_tooltip(
            "Select the Specify option if you wish to manually \
             specify the tilt angles in the edit boxes below",
        );
        self.panel
            .set_min_tooltip("Starting tilt angle of the series");
        self.panel.set_step_tooltip("Tilt increment between views");
        self.panel.set_file_tooltip(
            "Select the File option if the tilt angles already exist in a *.rawtlt file",
        );
    }

    /// Java package-private `setRadioButtonState(ActionEvent)`: set the state
    /// of the text fields depending upon the radio button state.
    pub fn set_radio_button_state(&self, event: &ActionEvent) {
        let specify = self.panel.get_specify();
        self.enable_angle_fields(
            event
                .get_action_command()
                .is_some_and(|command| Some(command) == specify.as_deref()),
        );
        self.panel.update_display();
    }
}
