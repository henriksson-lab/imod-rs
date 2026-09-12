//! `IMOD/Etomo/src/etomo/ui/swing/TiltAnglePanelExpert.java`.

use super::tilt_angle_panel::{TiltAnglePanelActionEvent, TiltAngleType};
use crate::imod::etomo::r#type::axis_id::AxisID;

pub trait TiltAngleSpecBoundary {
    fn tilt_angle_type(&self) -> TiltAngleType;
    fn range_min(&self) -> f64;
    fn range_step(&self) -> f64;
    fn set_type(&mut self, value: Option<TiltAngleType>);
    fn set_range_min(&mut self, value: String);
    fn set_range_step(&mut self, value: String);
}
pub trait UserConfigurationBoundary {
    fn tilt_angles_rawtlt_file(&self) -> bool;
}
pub trait TiltAnglePanelBoundary {
    fn set_file(&mut self, value: bool);
    fn set_extract(&mut self, value: bool);
    fn set_specify(&mut self, value: bool);
    fn set_min(&mut self, value: f64);
    fn set_step(&mut self, value: f64);
    fn set_min_enabled(&mut self, value: bool);
    fn set_step_enabled(&mut self, value: bool);
    fn is_extract_selected(&self) -> bool;
    fn is_specify_selected(&self) -> bool;
    fn is_file_selected(&self) -> bool;
    fn get_min(&self, validation: bool) -> Result<String, ()>;
    fn get_step(&self, validation: bool) -> Result<String, ()>;
    fn checkpoint(&mut self);
    fn update_display(&mut self);
    fn set_source_enabled(&mut self, value: bool);
    fn set_angle_enabled(&mut self, value: bool);
    fn set_extract_enabled(&mut self, value: bool);
    fn set_file_enabled(&mut self, value: bool);
    fn set_specify_enabled(&mut self, value: bool);
}
/// Java final `TiltAnglePanelExpert`.
pub struct TiltAnglePanelExpert<P: TiltAnglePanelBoundary> {
    pub axis_id: AxisID,
    pub panel: P,
}
impl<P: TiltAnglePanelBoundary> TiltAnglePanelExpert<P> {
    pub fn new(axis_id: AxisID, panel: P) -> Self {
        Self { axis_id, panel }
    }
    pub fn set_fields<S: TiltAngleSpecBoundary, U: UserConfigurationBoundary>(
        &mut self,
        spec: &S,
        user: &U,
    ) {
        match spec.tilt_angle_type() {
            TiltAngleType::File => {
                self.panel.set_file(true);
                self.enable_angle_fields(false)
            }
            TiltAngleType::Extract => {
                self.panel.set_extract(true);
                self.enable_angle_fields(false)
            }
            TiltAngleType::Range => {
                self.panel.set_specify(true);
                self.enable_angle_fields(true)
            }
        }
        if user.tilt_angles_rawtlt_file() {
            self.panel.set_file(true);
            self.enable_angle_fields(false)
        }
        self.panel.set_min(spec.range_min());
        self.panel.set_step(spec.range_step());
    }
    pub fn enable_angle_fields(&mut self, enable: bool) {
        self.panel.set_min_enabled(enable);
        self.panel.set_step_enabled(enable)
    }
    pub fn get_tilt_angle_type(&self) -> Option<TiltAngleType> {
        if self.panel.is_extract_selected() {
            Some(TiltAngleType::Extract)
        } else if self.panel.is_specify_selected() {
            Some(TiltAngleType::Range)
        } else if self.panel.is_file_selected() {
            Some(TiltAngleType::File)
        } else {
            None
        }
    }
    pub fn get_fields<S: TiltAngleSpecBoundary>(&self, spec: &mut S, validation: bool) -> bool {
        let (Ok(min), Ok(step)) = (
            self.panel.get_min(validation),
            self.panel.get_step(validation),
        ) else {
            return false;
        };
        spec.set_type(self.get_tilt_angle_type());
        spec.set_range_min(min);
        spec.set_range_step(step);
        true
    }
    pub fn validate(&self, _: &str) -> bool {
        self.panel.is_specify_selected()
            && self.panel.get_min(false).is_ok()
            && self.panel.get_step(false).is_ok()
    }
    pub fn checkpoint(&mut self) {
        self.panel.checkpoint()
    }
    pub fn update_template_values(&mut self) {}
    pub fn set_enabled(&mut self, enable: bool) {
        self.panel.set_source_enabled(enable);
        self.panel.set_angle_enabled(enable);
        self.panel.set_extract_enabled(enable);
        self.panel.set_file_enabled(enable);
        self.panel.set_specify_enabled(enable);
        self.enable_angle_fields(self.panel.is_specify_selected() & enable)
    }
    pub fn set_tooltips(&mut self) {}
    pub fn set_radio_button_state(&mut self, event: TiltAnglePanelActionEvent) {
        self.enable_angle_fields(event == TiltAnglePanelActionEvent::Specify);
        self.panel.update_display()
    }
}
