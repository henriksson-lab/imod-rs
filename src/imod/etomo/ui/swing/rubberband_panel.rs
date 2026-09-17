//! `IMOD/Etomo/src/etomo/ui/swing/RubberbandPanel.java`.
#![allow(dead_code)]

use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::{ButtonBoundary, MultiLineButton};
use crate::imod::etomo::process::imod_process::RUBBERBAND_RESULTS_STRING;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::field_type::FieldType;

pub use super::rubberband_container::RubberbandContainer;
pub trait RubberbandPanelManager {
    fn imod_get_rubberband_coordinates(&mut self, key: &str, axis: AxisID) -> Option<Vec<String>>;
}
pub trait RubberbandMetaData {
    fn set_post_trimvol_scale_x_min(&mut self, v: String);
    fn set_post_trimvol_scale_x_max(&mut self, v: String);
    fn set_post_trimvol_scale_y_min(&mut self, v: String);
    fn set_post_trimvol_scale_y_max(&mut self, v: String);
    fn set_post_trimvol_section_scale_min(&mut self, v: String);
    fn set_post_trimvol_section_scale_max(&mut self, v: String);
}
pub trait RubberbandConstMetaData {
    fn get_post_trimvol_scale_x_min(&self) -> String;
    fn get_post_trimvol_scale_x_max(&self) -> String;
    fn get_post_trimvol_scale_y_min(&self) -> String;
    fn get_post_trimvol_scale_y_max(&self) -> String;
    fn get_post_trimvol_section_scale_min(&self) -> String;
    fn get_post_trimvol_section_scale_max(&self) -> String;
}
pub trait RubberbandParallelMetaData {
    fn set_x_min(&mut self, v: String);
    fn set_x_max(&mut self, v: String);
    fn set_y_min(&mut self, v: String);
    fn set_y_max(&mut self, v: String);
    fn set_z_min(&mut self, v: String);
    fn set_z_max(&mut self, v: String);
    fn get_x_min(&self) -> String;
    fn get_x_max(&self) -> String;
    fn get_y_min(&self) -> String;
    fn get_y_max(&self) -> String;
    fn get_z_min(&self) -> String;
    fn get_z_max(&self) -> String;
    fn set_new_style_z(&mut self, min: String, max: String);
}
pub trait RubberbandXyParam {
    fn set_x_min(&mut self, v: String);
    fn set_x_max(&mut self, v: String);
    fn set_y_min(&mut self, v: String);
    fn set_y_max(&mut self, v: String);
    fn get_x_min(&self) -> String;
    fn get_x_max(&self) -> String;
    fn get_y_min(&self) -> String;
    fn get_y_max(&self) -> String;
}
pub trait RubberbandTrimvolParam {
    type ScaleXyParam: RubberbandXyParam;
    fn set_x_min(&mut self, v: String);
    fn set_x_max(&mut self, v: String);
    fn set_y_min(&mut self, v: String);
    fn set_y_max(&mut self, v: String);
    fn set_z_min(&mut self, v: String);
    fn set_z_max(&mut self, v: String);
    fn get_x_min(&self) -> String;
    fn get_x_max(&self) -> String;
    fn get_y_min(&self) -> String;
    fn get_y_max(&self) -> String;
    fn get_z_min(&self) -> String;
    fn get_z_max(&self) -> String;
    fn get_scale_xy_param(&mut self) -> &mut Self::ScaleXyParam;
    fn get_scale_xy_param_readonly(&self) -> &Self::ScaleXyParam;
    fn set_section_scale_min(&mut self, v: String);
    fn set_section_scale_max(&mut self, v: String);
    fn get_section_scale_min(&self) -> String;
    fn get_section_scale_max(&self) -> String;
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run3dmodButtonBoundary {
    pub action_command: String,
    pub enabled: bool,
}
#[derive(Clone, Debug)]
pub struct RubberbandPanel<M, C = ()> {
    pub manager: M,
    pub border_label: String,
    pub visible: bool,
    pub range_rows: i32,
    pub range_order: Vec<String>,
    pub button_order: Option<Vec<String>>,
    pub ltf_x_min: LabeledTextField,
    pub ltf_x_max: LabeledTextField,
    pub ltf_y_min: LabeledTextField,
    pub ltf_y_max: LabeledTextField,
    pub ltf_z_min: LabeledTextField,
    pub ltf_z_max: LabeledTextField,
    pub btn_rubberband: MultiLineButton,
    pub imod_key: String,
    pub btn_imod: Option<Run3dmodButtonBoundary>,
    pub container: Option<C>,
    pub place_buttons: bool,
    pub lock_panel: bool,
}
impl<M, C> RubberbandPanel<M, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manager: M,
        container: Option<C>,
        key: &str,
        border: &str,
        button: &str,
        xmin: &str,
        xmax: &str,
        ymin: &str,
        ymax: &str,
        zmin: &str,
        zmax: &str,
        imod: Option<Run3dmodButtonBoundary>,
        place: bool,
        lock: bool,
    ) -> Self {
        let has = imod.is_some();
        let mut order = vec![
            "ltfXMin".into(),
            "ltfXMax".into(),
            "ltfYMin".into(),
            "ltfYMax".into(),
        ];
        if has {
            order.extend(["ltfZMin".into(), "ltfZMax".into()])
        }
        let mut r = Self {
            manager,
            border_label: border.into(),
            visible: true,
            range_rows: if has { 3 } else { 2 },
            range_order: order,
            button_order: has.then(|| {
                if place {
                    vec![
                        "glue".into(),
                        "btnImod".into(),
                        "glue".into(),
                        "btnRubberband".into(),
                        "glue".into(),
                    ]
                } else {
                    vec![]
                }
            }),
            ltf_x_min: LabeledTextField::new(FieldType::Integer, "X min: "),
            ltf_x_max: LabeledTextField::new(FieldType::Integer, "X max: "),
            ltf_y_min: LabeledTextField::new(FieldType::Integer, "Y min: "),
            ltf_y_max: LabeledTextField::new(FieldType::Integer, "Y max: "),
            ltf_z_min: LabeledTextField::new(FieldType::Integer, "Z min: "),
            ltf_z_max: LabeledTextField::new(FieldType::Integer, "Z max: "),
            btn_rubberband: {
                let mut btn = MultiLineButton::new_with_label(Some(button));
                // Java JButton's default action command is its label.
                btn.set_action_command(Some(button));
                btn
            },
            imod_key: key.into(),
            btn_imod: imod,
            container,
            place_buttons: place,
            lock_panel: lock,
        };
        r.set_tool_tip_text(xmin, xmax, ymin, ymax, zmin, zmax);
        r
    }
    pub fn add_listeners(&mut self) {
        self.btn_rubberband.add_action_listener()
    }
    pub fn get_rubberband_button_component(&self) -> &ButtonBoundary {
        self.btn_rubberband.get_component()
    }
    pub fn get_component(&self) -> &Self {
        self
    }
    pub fn get_container(&self) -> &Self {
        self
    }
    pub fn set_x_min(&mut self, v: &str) {
        self.ltf_x_min.set_text(v)
    }
    pub fn set_x_max(&mut self, v: &str) {
        self.ltf_x_max.set_text(v)
    }
    pub fn set_y_min(&mut self, v: &str) {
        self.ltf_y_min.set_text(v)
    }
    pub fn set_y_max(&mut self, v: &str) {
        self.ltf_y_max.set_text(v)
    }
    pub fn set_enabled(&mut self, v: bool) {
        self.ltf_x_min.set_enabled(v);
        self.ltf_x_max.set_enabled(v);
        self.ltf_y_min.set_enabled(v);
        self.ltf_y_max.set_enabled(v);
        self.btn_rubberband.set_enabled(v)
    }
    pub fn set_visible(&mut self, v: bool) {
        self.visible = v
    }
    pub fn get_parameters_metadata<D: RubberbandMetaData>(&self, d: &mut D) {
        if self.lock_panel {
            return;
        }
        d.set_post_trimvol_scale_x_min(self.ltf_x_min.get_text());
        d.set_post_trimvol_scale_x_max(self.ltf_x_max.get_text());
        d.set_post_trimvol_scale_y_min(self.ltf_y_min.get_text());
        d.set_post_trimvol_scale_y_max(self.ltf_y_max.get_text());
        if self.btn_imod.is_some() {
            d.set_post_trimvol_section_scale_min(self.ltf_z_min.get_text());
            d.set_post_trimvol_section_scale_max(self.ltf_z_max.get_text())
        }
    }
    pub fn get_parameters_trimvol<D: RubberbandTrimvolParam>(&self, d: &mut D, v: bool) -> bool {
        if self.lock_panel {
            return true;
        }
        let x = (|| -> Result<_, FieldValidationFailedException> {
            Ok((
                self.ltf_x_min.get_text_validated(v)?,
                self.ltf_x_max.get_text_validated(v)?,
                self.ltf_y_min.get_text_validated(v)?,
                self.ltf_y_max.get_text_validated(v)?,
                self.ltf_z_min.get_text_validated(v)?,
                self.ltf_z_max.get_text_validated(v)?,
            ))
        })();
        let Ok((a, b, c, e, f, g)) = x else {
            return false;
        };
        d.set_x_min(a);
        d.set_x_max(b);
        d.set_y_min(c);
        d.set_y_max(e);
        if self.btn_imod.is_some() {
            d.set_z_min(f);
            d.set_z_max(g)
        }
        true
    }
    pub fn get_scale_parameters<D: RubberbandTrimvolParam>(&self, d: &mut D, v: bool) -> bool {
        if self.lock_panel {
            return true;
        }
        let x = (|| -> Result<_, FieldValidationFailedException> {
            Ok((
                self.ltf_x_min.get_text_validated(v)?,
                self.ltf_x_max.get_text_validated(v)?,
                self.ltf_y_min.get_text_validated(v)?,
                self.ltf_y_max.get_text_validated(v)?,
                self.ltf_z_min.get_text_validated(v)?,
                self.ltf_z_max.get_text_validated(v)?,
            ))
        })();
        let Ok((a, b, c, e, f, g)) = x else {
            return false;
        };
        let xy = d.get_scale_xy_param();
        xy.set_x_min(a);
        xy.set_x_max(b);
        xy.set_y_min(c);
        xy.set_y_max(e);
        if self.btn_imod.is_some() {
            d.set_section_scale_min(f);
            d.set_section_scale_max(g)
        }
        true
    }
    pub fn get_parameters_parallel_metadata<D: RubberbandParallelMetaData>(&self, d: &mut D) {
        if self.lock_panel {
            return;
        }
        d.set_x_min(self.ltf_x_min.get_text());
        d.set_x_max(self.ltf_x_max.get_text());
        d.set_y_min(self.ltf_y_min.get_text());
        d.set_y_max(self.ltf_y_max.get_text());
        if self.btn_imod.is_some() {
            d.set_z_min(self.ltf_z_min.get_text());
            d.set_z_max(self.ltf_z_max.get_text())
        }
    }
    pub fn get_parameters_for_trimvol<D: RubberbandParallelMetaData>(&self, d: &mut D) {
        if !self.lock_panel && self.btn_imod.is_some() {
            d.set_new_style_z(self.ltf_z_min.get_text(), self.ltf_z_max.get_text())
        }
    }
    pub fn set_parameters_trimvol<D: RubberbandTrimvolParam>(&mut self, d: &D) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text(&d.get_x_min());
        self.ltf_x_max.set_text(&d.get_x_max());
        self.ltf_y_min.set_text(&d.get_y_min());
        self.ltf_y_max.set_text(&d.get_y_max());
        if self.btn_imod.is_some() {
            self.ltf_z_min.set_text(&d.get_z_min());
            self.ltf_z_max.set_text(&d.get_z_max())
        }
    }
    pub fn init_scale_parameters<D: RubberbandTrimvolParam>(&mut self, d: &D) {
        if self.lock_panel {
            return;
        }
        let xy = d.get_scale_xy_param_readonly();
        self.ltf_x_min.set_text(&xy.get_x_min());
        self.ltf_x_max.set_text(&xy.get_x_max());
        self.ltf_y_min.set_text(&xy.get_y_min());
        self.ltf_y_max.set_text(&xy.get_y_max());
        if self.btn_imod.is_some() {
            self.ltf_z_min.set_text(&d.get_section_scale_min());
            self.ltf_z_max.set_text(&d.get_section_scale_max())
        }
    }
    pub fn set_parameters_const_metadata<D: RubberbandConstMetaData>(&mut self, d: &D) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text(&d.get_post_trimvol_scale_x_min());
        self.ltf_x_max.set_text(&d.get_post_trimvol_scale_x_max());
        self.ltf_y_min.set_text(&d.get_post_trimvol_scale_y_min());
        self.ltf_y_max.set_text(&d.get_post_trimvol_scale_y_max());
        if self.btn_imod.is_some() {
            self.ltf_z_min
                .set_text(&d.get_post_trimvol_section_scale_min());
            self.ltf_z_max
                .set_text(&d.get_post_trimvol_section_scale_max())
        }
    }
    pub fn set_parameters_parallel_metadata<D: RubberbandParallelMetaData>(&mut self, d: &D) {
        if self.lock_panel {
            return;
        }
        self.ltf_x_min.set_text(&d.get_x_min());
        self.ltf_x_max.set_text(&d.get_x_max());
        self.ltf_y_min.set_text(&d.get_y_min());
        self.ltf_y_max.set_text(&d.get_y_max());
        if self.btn_imod.is_some() {
            self.ltf_z_min.set_text(&d.get_z_min());
            self.ltf_z_max.set_text(&d.get_z_max())
        }
    }
    pub fn set_tool_tip_text(
        &mut self,
        xmin: &str,
        xmax: &str,
        ymin: &str,
        ymax: &str,
        zmin: &str,
        zmax: &str,
    ) {
        self.ltf_x_min.set_tool_tip_text(Some(xmin));
        self.ltf_x_max.set_tool_tip_text(Some(xmax));
        self.ltf_y_min.set_tool_tip_text(Some(ymin));
        self.ltf_y_max.set_tool_tip_text(Some(ymax));
        self.ltf_z_min.set_tool_tip_text(Some(zmin));
        self.ltf_z_max.set_tool_tip_text(Some(zmax));
        let axes = if self.btn_imod.is_none() {
            " and Y"
        } else {
            ", Y, and Z"
        };
        self.btn_rubberband.set_tool_tip_text(Some(&format!("After opening the volume in 3dmod, press shift-B in the ZaP window.  Create a rubberband around the contrast range.  Then press this button to retrieve the X{axes} coordinates.")))
    }
}
impl<M: RubberbandPanelManager, C: RubberbandContainer> RubberbandPanel<M, C> {
    pub fn button_action(&mut self, c: Option<&str>) {
        if c != self.btn_rubberband.get_action_command() {
            return;
        }
        let v = self
            .manager
            .imod_get_rubberband_coordinates(&self.imod_key, AxisID::Only);
        self.set_min_and_max(v)
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for `RubberbandActionListener::actionPerformed`.
    pub fn actionPerformed(&mut self, command: Option<&str>) {
        self.button_action(command);
    }
    pub fn set_min_and_max(&mut self, input: Option<Vec<String>>) {
        let Some(v) = input else { return };
        let mut i = 0;
        while i < v.len() {
            if v[i] == RUBBERBAND_RESULTS_STRING {
                i += 1;
                for field in [
                    &mut self.ltf_x_min,
                    &mut self.ltf_y_min,
                    &mut self.ltf_x_max,
                    &mut self.ltf_y_max,
                ] {
                    let Some(x) = v.get(i) else { return };
                    field.set_text(x);
                    i += 1
                }
                if self.btn_imod.is_none() && self.container.is_none() {
                    return;
                }
                let Some(z) = v.get(i) else { return };
                if self.btn_imod.is_some() {
                    self.ltf_z_min.set_text(z)
                }
                if let Some(c) = &mut self.container {
                    c.set_rubberband_container_z_min(z)
                }
                i += 1;
                let Some(z) = v.get(i) else { return };
                if self.btn_imod.is_some() {
                    self.ltf_z_max.set_text(z)
                }
                if let Some(c) = &mut self.container {
                    c.set_rubberband_container_z_max(z)
                }
                return;
            }
            i += 1
        }
    }
}
impl<M, C> RubberbandPanel<M, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_with_container(
        m: M,
        c: C,
        key: &str,
        border: &str,
        button: &str,
        xmin: &str,
        xmax: &str,
        ymin: &str,
        ymax: &str,
    ) -> Self {
        let mut v = Self::new(
            m,
            Some(c),
            key,
            border,
            button,
            xmin,
            xmax,
            ymin,
            ymax,
            "",
            "",
            None,
            true,
            false,
        );
        v.add_listeners();
        v
    }
    #[allow(clippy::too_many_arguments)]
    pub fn get_no_button_instance(
        m: M,
        c: C,
        key: &str,
        border: &str,
        button: &str,
        xmin: &str,
        xmax: &str,
        ymin: &str,
        ymax: &str,
        lock: bool,
    ) -> Self {
        let mut v = Self::new(
            m,
            Some(c),
            key,
            border,
            button,
            xmin,
            xmax,
            ymin,
            ymax,
            "",
            "",
            None,
            false,
            lock,
        );
        v.add_listeners();
        v
    }
}
impl<M> RubberbandPanel<M, ()> {
    #[allow(clippy::too_many_arguments)]
    pub fn get_instance_with_imod(
        m: M,
        key: &str,
        border: &str,
        button: &str,
        xmin: &str,
        xmax: &str,
        ymin: &str,
        ymax: &str,
        zmin: &str,
        zmax: &str,
        imod: Run3dmodButtonBoundary,
    ) -> Self {
        let mut v = Self::new(
            m,
            None,
            key,
            border,
            button,
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            Some(imod),
            true,
            false,
        );
        v.add_listeners();
        v
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct M {
        v: Option<Vec<String>>,
    }
    impl RubberbandPanelManager for M {
        fn imod_get_rubberband_coordinates(&mut self, _: &str, _: AxisID) -> Option<Vec<String>> {
            self.v.take()
        }
    }
    #[derive(Default)]
    struct C {
        max: String,
    }
    impl RubberbandContainer for C {
        fn set_rubberband_container_z_min(&mut self, _: &str) {}
        fn set_rubberband_container_z_max(&mut self, v: &str) {
            self.max = v.into()
        }
    }
    #[test]
    fn parses_coordinates() {
        let mut p = RubberbandPanel::get_instance_with_container(
            M {
                v: Some(vec![
                    RUBBERBAND_RESULTS_STRING.into(),
                    "1".into(),
                    "2".into(),
                    "3".into(),
                    "4".into(),
                    "5".into(),
                    "6".into(),
                ]),
            },
            C::default(),
            "k",
            "r",
            "get",
            "",
            "",
            "",
            "",
        );
        let a = p.btn_rubberband.get_action_command().unwrap().to_owned();
        p.button_action(Some(&a));
        assert_eq!(p.ltf_x_min.get_text(), "1");
        assert_eq!(p.container.unwrap().max, "6")
    }
}
