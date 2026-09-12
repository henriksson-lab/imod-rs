//! `IMOD/Etomo/src/etomo/ui/swing/RadialPanel.java`.
#![allow(dead_code)]
use super::{
    filter_type::FilterType,
    labeled_text_field::{FieldValidationFailedException, LabeledTextField},
    radial_parent::RadialParent,
    radio_button::{RadioButton, RadioButtonGroup},
};
use crate::imod::etomo::{r#type::axis_id::AxisID, ui::field_type::FieldType};
use std::{cell::RefCell, rc::Rc};
pub const RADIAL_FALLOFF_OLD_LABEL: &str = " Falloff (1.4 * sigma): ";
pub const RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL: &str = " Falloff (sigma): ";
pub const FAKE_SIRT_ITERATIONS_LABEL: &str = "Use SIRT-like filter equivalent to: ";
pub const EXACT_FILTER_SIZE_LABEL: &str = "Use 'exact filter' functions with 'object size' of: ";
pub const HAMMING_LIKE_FITLER_LABEL: &str = "Hamming-like filter (as in tomo3d) starting from: ";
pub const RADIAL_BUTTON_LABEL: &str = "Standard Gaussian,";
pub const RADIAL_MAX_LABEL: &str = "cutoff: ";
pub const RADIAL_LABEL: &str = "Standard Gaussian, cutoff: ";
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PanelId {
    Tilt,
    Sirtsetup,
    Other,
}
pub trait RadialMetaData {
    fn set_radial_radius(&mut self, p: PanelId, a: AxisID, v: String);
    fn set_radial_sigma(&mut self, p: PanelId, a: AxisID, v: String);
    fn set_hamming_like_filter(&mut self, p: PanelId, a: AxisID, v: String);
    fn set_fake_sirt_iterations(&mut self, p: PanelId, a: AxisID, v: String);
    fn set_exact_filter_size(&mut self, p: PanelId, a: AxisID, v: String);
}
pub trait ConstRadialMetaData {
    fn radial_radius(&self, p: PanelId, a: AxisID) -> String;
    fn radial_sigma(&self, p: PanelId, a: AxisID) -> String;
    fn hamming_like_filter(&self, p: PanelId, a: AxisID) -> String;
    fn fake_sirt_iterations(&self, p: PanelId, a: AxisID) -> String;
    fn exact_filter_size(&self, p: PanelId, a: AxisID) -> String;
}
pub trait ConstTiltParam {
    fn has_radial_weighting_function(&self) -> bool;
    fn radial_bandwidth(&self) -> Option<String>;
    fn radial_falloff(&self) -> Option<String>;
    fn falloff_is_true_sigma(&self) -> bool;
    fn is_hamming_like_filter(&self) -> bool;
    fn hamming_like_filter(&self) -> String;
    fn is_fake_sirt_iterations(&self) -> bool;
    fn fake_sirt_iterations(&self) -> String;
    fn is_exact_filter_size(&self) -> bool;
    fn exact_filter_size(&self) -> String;
}
pub trait TiltParam {
    fn set_radial_bandwidth(&mut self, v: String);
    fn set_radial_falloff(&mut self, v: String);
    fn reset_radial_filter(&mut self);
    fn set_hamming_like_filter(&mut self, v: String);
    fn reset_hamming_like_filter(&mut self);
    fn set_fake_sirt_iterations(&mut self, v: String);
    fn reset_fake_sirt_iterations(&mut self);
    fn set_exact_filter_size(&mut self, v: String);
    fn reset_exact_filter_size(&mut self);
}
pub trait SirtsetupParam {
    fn set_radius_and_sigma(&mut self, i: usize, v: String);
    fn radius_and_sigma(&self, i: usize) -> String;
    fn falloff_is_true_sigma(&self) -> bool;
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RadialPanelLayout {
    pub root_visible: bool,
    pub form_visible: bool,
    pub exact_filter_size_visible: bool,
    pub high_frequency_filtering_enabled: bool,
    pub form_enabled: bool,
    pub listener_count: usize,
    pub packed: bool,
    pub tooltip_initialized: bool,
}
pub struct RadialPanel<P: RadialParent> {
    pub axis_id: AxisID,
    pub panel_id: PanelId,
    pub parent: P,
    pub layout: RadialPanelLayout,
    pub ltf_radial_max: LabeledTextField,
    pub ltf_radial_fall_off: LabeledTextField,
    pub tf_hamming_like_filter: Option<LabeledTextField>,
    pub tf_exact_filter_size: Option<LabeledTextField>,
    pub rb_default: Option<RadioButton>,
    pub tf_fake_sirt_iterations: Option<LabeledTextField>,
    pub rb_fake_sirt_iterations: Option<RadioButton>,
    pub rb_exact_filter_size: Option<RadioButton>,
    pub rb_radial: Option<RadioButton>,
    pub rb_hamming_like_filter: Option<RadioButton>,
    pub multifilt_filter_type: Option<Box<dyn FilterType>>,
    pub debug: bool,
    pub init: bool,
}
impl<P: RadialParent> RadialPanel<P> {
    fn new(parent: P, axis_id: AxisID, panel_id: PanelId) -> Self {
        let tilt = panel_id == PanelId::Tilt;
        let form = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let high = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            axis_id,
            panel_id,
            parent,
            layout: RadialPanelLayout::default(),
            ltf_radial_max: LabeledTextField::new(FieldType::FloatingPoint, RADIAL_LABEL),
            ltf_radial_fall_off: LabeledTextField::new(
                FieldType::FloatingPoint,
                RADIAL_FALLOFF_OLD_LABEL,
            ),
            tf_hamming_like_filter: tilt.then(|| {
                LabeledTextField::new(FieldType::FloatingPoint, HAMMING_LIKE_FITLER_LABEL)
            }),
            tf_exact_filter_size: tilt
                .then(|| LabeledTextField::new(FieldType::Integer, EXACT_FILTER_SIZE_LABEL)),
            rb_default: tilt
                .then(|| RadioButton::new_in_group("Standard ramp function", form.clone())),
            tf_fake_sirt_iterations: tilt
                .then(|| LabeledTextField::new(FieldType::Integer, FAKE_SIRT_ITERATIONS_LABEL)),
            rb_fake_sirt_iterations: tilt
                .then(|| RadioButton::new_in_group(FAKE_SIRT_ITERATIONS_LABEL, form.clone())),
            rb_exact_filter_size: tilt
                .then(|| RadioButton::new_in_group(EXACT_FILTER_SIZE_LABEL, form)),
            rb_radial: tilt.then(|| RadioButton::new_in_group(RADIAL_BUTTON_LABEL, high.clone())),
            rb_hamming_like_filter: tilt
                .then(|| RadioButton::new_in_group(HAMMING_LIKE_FITLER_LABEL, high)),
            multifilt_filter_type: None,
            debug: false,
            init: true,
        }
    }
    pub fn get_instance(parent: P, axis_id: AxisID, panel_id: PanelId) -> Self {
        let mut x = Self::new(parent, axis_id, panel_id);
        x.create_panel();
        x.add_listeners();
        x.set_tooltips();
        x
    }
    fn create_panel(&mut self) {
        if self.panel_id == PanelId::Tilt {
            self.rb_default.as_mut().unwrap().set_selected(true);
            self.rb_radial.as_mut().unwrap().set_selected(true)
        }
        self.layout.root_visible = true;
        self.layout.form_visible = self.panel_id == PanelId::Tilt;
        self.update_display();
        self.init = false
    }
    fn add_listeners(&mut self) {
        if self.panel_id == PanelId::Tilt {
            self.layout.listener_count = 5
        }
    }
    pub fn set_multifilt_filter_type(&mut self, v: Option<Box<dyn FilterType>>) {
        self.multifilt_filter_type = v;
        self.update_display()
    }
    pub fn set_visible(&mut self, v: bool) {
        self.layout.root_visible = v
    }
    pub fn action_performed(&mut self) {
        self.update_display()
    }
    pub fn set_editable(&mut self, v: bool) {
        self.ltf_radial_max.set_editable(v);
        self.ltf_radial_fall_off.set_editable(v);
        if self.panel_id == PanelId::Tilt {
            self.rb_default.as_mut().unwrap().set_editable(v);
            self.rb_fake_sirt_iterations
                .as_mut()
                .unwrap()
                .set_editable(v);
            self.tf_fake_sirt_iterations
                .as_mut()
                .unwrap()
                .set_editable(v);
            self.rb_exact_filter_size.as_mut().unwrap().set_editable(v);
            self.tf_exact_filter_size.as_mut().unwrap().set_editable(v);
            self.rb_radial.as_mut().unwrap().set_editable(v);
            self.rb_hamming_like_filter
                .as_mut()
                .unwrap()
                .set_editable(v);
            self.tf_hamming_like_filter
                .as_mut()
                .unwrap()
                .set_editable(v)
        }
    }
    pub fn msg_method_changed(&mut self) {
        self.update_display()
    }
    pub fn update_display(&mut self) {
        let advanced = self.parent.is_advanced();
        if self.panel_id == PanelId::Tilt {
            if !self.parent.is_ctf3d() {
                self.layout.form_visible = true;
                self.layout.exact_filter_size_visible = advanced || self.parent.is_multifilt()
            } else {
                self.layout.form_visible =
                    advanced || !self.rb_default.as_ref().unwrap().is_selected();
                self.layout.exact_filter_size_visible = true
            }
            self.layout.packed = true
        }
        let hi = self.init
            || !self.parent.is_multifilt()
            || self
                .multifilt_filter_type
                .as_ref()
                .is_none_or(|f| !f.is_high_frequency_filter());
        self.layout.high_frequency_filtering_enabled = hi;
        let mut radial = hi;
        if self.panel_id == PanelId::Tilt {
            let b = self.rb_radial.as_mut().unwrap();
            b.set_enabled(hi);
            radial &= b.is_selected()
        }
        self.ltf_radial_max.set_enabled(radial);
        self.ltf_radial_fall_off.set_enabled(radial);
        if self.panel_id == PanelId::Tilt {
            let h = self.rb_hamming_like_filter.as_mut().unwrap();
            h.set_enabled(hi);
            self.tf_hamming_like_filter
                .as_mut()
                .unwrap()
                .set_enabled(hi && h.is_selected());
            let filter = self.init
                || !self.parent.is_multifilt()
                || self
                    .multifilt_filter_type
                    .as_ref()
                    .is_none_or(|f| !f.is_radial_filter());
            self.layout.form_enabled = filter;
            self.rb_default.as_mut().unwrap().set_enabled(filter);
            let fake = self.rb_fake_sirt_iterations.as_mut().unwrap();
            fake.set_enabled(filter);
            self.tf_fake_sirt_iterations
                .as_mut()
                .unwrap()
                .set_enabled(filter && fake.is_selected());
            let exact = self.rb_exact_filter_size.as_mut().unwrap();
            exact.set_enabled(filter);
            self.tf_exact_filter_size
                .as_mut()
                .unwrap()
                .set_enabled(filter && exact.is_selected())
        }
    }
    pub fn get_parameters_metadata<M: RadialMetaData>(&self, m: &mut M) {
        m.set_radial_radius(self.panel_id, self.axis_id, self.ltf_radial_max.get_text());
        m.set_radial_sigma(
            self.panel_id,
            self.axis_id,
            self.ltf_radial_fall_off.get_text(),
        );
        if self.panel_id == PanelId::Tilt {
            m.set_hamming_like_filter(
                self.panel_id,
                self.axis_id,
                self.tf_hamming_like_filter.as_ref().unwrap().get_text(),
            );
            m.set_fake_sirt_iterations(
                self.panel_id,
                self.axis_id,
                self.tf_fake_sirt_iterations.as_ref().unwrap().get_text(),
            );
            m.set_exact_filter_size(
                self.panel_id,
                self.axis_id,
                self.tf_exact_filter_size.as_ref().unwrap().get_text(),
            )
        }
    }
    pub fn set_parameters_metadata<M: ConstRadialMetaData>(&mut self, m: &M) {
        self.ltf_radial_max
            .set_text(&m.radial_radius(self.panel_id, self.axis_id));
        self.ltf_radial_fall_off
            .set_text(&m.radial_sigma(self.panel_id, self.axis_id));
        if self.panel_id == PanelId::Tilt {
            self.tf_hamming_like_filter
                .as_mut()
                .unwrap()
                .set_text(&m.hamming_like_filter(self.panel_id, self.axis_id));
            self.tf_fake_sirt_iterations
                .as_mut()
                .unwrap()
                .set_text(&m.fake_sirt_iterations(self.panel_id, self.axis_id));
            self.tf_exact_filter_size
                .as_mut()
                .unwrap()
                .set_text(&m.exact_filter_size(self.panel_id, self.axis_id))
        }
        self.update_display()
    }
    pub fn set_parameters_const_tilt<T: ConstTiltParam>(&mut self, t: &T) {
        if t.has_radial_weighting_function() {
            self.ltf_radial_max
                .set_non_empty_text(t.radial_bandwidth().as_deref());
            self.ltf_radial_fall_off
                .set_non_empty_text(t.radial_falloff().as_deref())
        }
        self.ltf_radial_fall_off
            .set_label(if t.falloff_is_true_sigma() {
                RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL
            } else {
                RADIAL_FALLOFF_OLD_LABEL
            });
        if self.panel_id == PanelId::Tilt {
            if t.is_hamming_like_filter() {
                self.rb_hamming_like_filter
                    .as_mut()
                    .unwrap()
                    .set_selected(true);
                self.tf_hamming_like_filter
                    .as_mut()
                    .unwrap()
                    .set_text(&t.hamming_like_filter())
            } else {
                self.rb_radial.as_mut().unwrap().set_selected(true)
            }
            if t.is_fake_sirt_iterations() {
                self.rb_fake_sirt_iterations
                    .as_mut()
                    .unwrap()
                    .set_selected(true);
                self.tf_fake_sirt_iterations
                    .as_mut()
                    .unwrap()
                    .set_text(&t.fake_sirt_iterations())
            }
            if t.is_exact_filter_size() {
                self.rb_exact_filter_size
                    .as_mut()
                    .unwrap()
                    .set_selected(true);
                self.tf_exact_filter_size
                    .as_mut()
                    .unwrap()
                    .set_text(&t.exact_filter_size())
            }
        }
        self.update_display()
    }
    pub fn get_parameters_sirtsetup<S: SirtsetupParam>(&self, p: &mut S, v: bool) -> bool {
        let r: Result<(), FieldValidationFailedException> = (|| {
            if self.ltf_radial_max.is_editable() && self.ltf_radial_max.is_enabled() {
                p.set_radius_and_sigma(0, self.ltf_radial_max.get_text_validated(v)?)
            }
            if self.ltf_radial_fall_off.is_enabled() {
                p.set_radius_and_sigma(1, self.ltf_radial_fall_off.get_text_validated(v)?)
            }
            Ok(())
        })();
        r.is_ok()
    }
    pub fn set_parameters_sirtsetup<S: SirtsetupParam>(&mut self, p: &S) {
        self.ltf_radial_max.set_text(&p.radius_and_sigma(0));
        self.ltf_radial_fall_off.set_text(&p.radius_and_sigma(1));
        self.ltf_radial_fall_off
            .set_label(if p.falloff_is_true_sigma() {
                RADIAL_FALLOFF_IS_TRUE_SIGMA_LABEL
            } else {
                RADIAL_FALLOFF_OLD_LABEL
            })
    }
    pub fn set_debug(&mut self, v: bool) {
        self.debug = v
    }
    pub fn get_parameters_tilt<T: TiltParam>(
        &self,
        t: &mut T,
        v: bool,
    ) -> Result<bool, FieldValidationFailedException> {
        let selected =
            self.panel_id != PanelId::Tilt || self.rb_radial.as_ref().unwrap().is_selected();
        if selected && (!self.ltf_radial_max.is_empty() || !self.ltf_radial_fall_off.is_empty()) {
            t.set_radial_bandwidth(self.ltf_radial_max.get_text_validated(v)?);
            t.set_radial_falloff(self.ltf_radial_fall_off.get_text_validated(v)?)
        } else {
            t.reset_radial_filter()
        }
        if self.panel_id == PanelId::Tilt {
            if self.rb_hamming_like_filter.as_ref().unwrap().is_selected() {
                t.set_hamming_like_filter(
                    self.tf_hamming_like_filter
                        .as_ref()
                        .unwrap()
                        .get_text_validated(v)?,
                )
            } else {
                t.reset_hamming_like_filter()
            }
            if self.rb_fake_sirt_iterations.as_ref().unwrap().is_selected() {
                t.set_fake_sirt_iterations(
                    self.tf_fake_sirt_iterations
                        .as_ref()
                        .unwrap()
                        .get_text_validated(v)?,
                )
            } else {
                t.reset_fake_sirt_iterations()
            }
            if self.rb_exact_filter_size.as_ref().unwrap().is_selected() {
                t.set_exact_filter_size(
                    self.tf_exact_filter_size
                        .as_ref()
                        .unwrap()
                        .get_text_validated(v)?,
                )
            } else {
                t.reset_exact_filter_size()
            }
        } else {
            t.reset_exact_filter_size()
        }
        Ok(true)
    }
    fn set_tooltips(&mut self) {
        self.ltf_radial_max
            .set_tool_tip_text(Some("Radial filter cutoff."));
        self.ltf_radial_fall_off
            .set_tool_tip_text(Some("Radial filter falloff."));
        self.layout.tooltip_initialized = true
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone, Copy)]
    struct P;
    impl RadialParent for P {
        fn is_advanced(&self) -> bool {
            false
        }
        fn is_ctf3d(&self) -> bool {
            false
        }
        fn is_multifilt(&self) -> bool {
            false
        }
    }
    #[test]
    fn tilt_initializes() {
        let p = RadialPanel::get_instance(P, AxisID::Only, PanelId::Tilt);
        assert!(p.rb_default.as_ref().unwrap().is_selected());
        assert!(p.rb_radial.as_ref().unwrap().is_selected());
        assert_eq!(p.layout.listener_count, 5)
    }
}
