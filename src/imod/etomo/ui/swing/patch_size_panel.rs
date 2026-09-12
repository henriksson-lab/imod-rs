//! `IMOD/Etomo/src/etomo/ui/swing/PatchSizePanel.java`.
//!
//! Swing `JPanel`, `BoxLayout`, `EtchedBorder`, and event-listener dispatch are
//! retained as direct GUI boundaries.  Patch-size selection, field enabling,
//! validation, and parameter handoff retain the Java source behavior here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::r#type::combine_patch_size::CombinePatchSize;
use crate::imod::etomo::ui::field_type::FieldType;

use super::labeled_text_field::LabeledTextField;
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};

pub const DEFAULT_FINAL_SIZE: CombinePatchSize = CombinePatchSize::ExtraLarge;
pub const X_INDEX: usize = 0;
pub const Y_INDEX: usize = 1;
pub const Z_INDEX: usize = 2;

/// The `ConstCombineParams` calls directly made by this Java source unit.
pub trait ConstCombineParams {
    fn get_patch_size(&self, auto_final: bool) -> Option<CombinePatchSize>;
    fn get_patch_size_xyz_array(&self, auto_final: bool) -> Option<Vec<String>>;
}

/// The `CombineParams` writes/reset directly made by this Java source unit.
pub trait CombineParams: ConstCombineParams {
    fn set_patch_size(&mut self, auto_final: bool, patch_size: CombinePatchSize);
    fn set_patch_size_xyz(&mut self, auto_final: bool, xyz: Vec<String>);
    fn reset_patch_size(&mut self, auto_final: bool);
}

/// The `ConstPatchcrawl3DParam` reads directly made by this Java source unit.
pub trait ConstPatchcrawl3dParam {
    fn get_x_patch_size(&self) -> i32;
    fn get_y_patch_size(&self) -> i32;
    fn get_z_patch_size(&self) -> i32;
}

/// Source-visible state of Java's root/type/XYZ `JPanel`s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchSizePanelRoot {
    pub title: String,
    pub max_size: bool,
    pub root_box_layout_x_axis: bool,
    pub root_alignment_center: bool,
    pub type_box_layout_y_axis: bool,
    pub xyz_box_layout_y_axis: bool,
    pub xyz_border_title: String,
    pub horizontal_glue_between_panels: bool,
    pub vertical_glue_between_fields: usize,
}

/// Java package-private final `PatchSizePanel`.
pub struct PatchSizePanel {
    pub pnl_root: PatchSizePanelRoot,
    pub bg_type: Rc<RefCell<RadioButtonGroup>>,
    pub rb_type_small: Option<RadioButton>,
    pub rb_type_medium: RadioButton,
    pub rb_type_large: RadioButton,
    pub rb_type_extra_large: Option<RadioButton>,
    pub rb_type_custom: RadioButton,
    pub xyz_labels: [&'static str; 3],
    pub ltf_xyz: [LabeledTextField; 3],
    pub title: String,
    pub final_size: bool,
}

impl PatchSizePanel {
    /// Java private `PatchSizePanel(boolean)`.
    fn new(final_size: bool) -> Self {
        let bg_type = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut ltf_xyz = [
            LabeledTextField::new(FieldType::Integer, "X: "),
            LabeledTextField::new(FieldType::Integer, "Y: "),
            LabeledTextField::new(FieldType::Integer, "Z: "),
        ];
        for field in &mut ltf_xyz {
            field.set_required(true);
        }
        let mut medium = RadioButton::new_with_enumerated_type(
            Some("Medium patches".into()),
            EnumeratedTypeBoundary {
                label: CombinePatchSize::Medium.get_label().into(),
                default: CombinePatchSize::Medium.is_default(),
                value: Some(CombinePatchSize::Medium.get_option().into()),
            },
            Some(bg_type.clone()),
        );
        medium.set_selected(true);
        let title = if final_size {
            "Max Patch Size"
        } else {
            "Patch Size"
        }
        .to_string();
        Self {
            pnl_root: PatchSizePanelRoot {
                title: title.clone(),
                max_size: final_size,
                root_box_layout_x_axis: false,
                root_alignment_center: false,
                type_box_layout_y_axis: false,
                xyz_box_layout_y_axis: false,
                xyz_border_title: String::new(),
                horizontal_glue_between_panels: false,
                vertical_glue_between_fields: 0,
            },
            rb_type_small: (!final_size).then(|| {
                RadioButton::new_with_enumerated_type(
                    Some("Small patches".into()),
                    EnumeratedTypeBoundary {
                        label: CombinePatchSize::Small.get_label().into(),
                        default: CombinePatchSize::Small.is_default(),
                        value: Some(CombinePatchSize::Small.get_option().into()),
                    },
                    Some(bg_type.clone()),
                )
            }),
            rb_type_medium: medium,
            rb_type_large: RadioButton::new_with_enumerated_type(
                Some("Large patches".into()),
                EnumeratedTypeBoundary {
                    label: CombinePatchSize::Large.get_label().into(),
                    default: CombinePatchSize::Large.is_default(),
                    value: Some(CombinePatchSize::Large.get_option().into()),
                },
                Some(bg_type.clone()),
            ),
            rb_type_extra_large: final_size.then(|| {
                RadioButton::new_with_enumerated_type(
                    Some("Extra Large patches".into()),
                    EnumeratedTypeBoundary {
                        label: CombinePatchSize::ExtraLarge.get_label().into(),
                        default: CombinePatchSize::ExtraLarge.is_default(),
                        value: Some(CombinePatchSize::ExtraLarge.get_option().into()),
                    },
                    Some(bg_type.clone()),
                )
            }),
            rb_type_custom: RadioButton::new_with_enumerated_type(
                Some("Custom".into()),
                EnumeratedTypeBoundary {
                    label: CombinePatchSize::Custom.get_label().into(),
                    default: CombinePatchSize::Custom.is_default(),
                    value: Some(CombinePatchSize::Custom.get_option().into()),
                },
                Some(bg_type.clone()),
            ),
            bg_type,
            xyz_labels: ["X: ", "Y: ", "Z: "],
            ltf_xyz,
            title,
            final_size,
        }
    }

    /// Java static `getInstance(boolean)`.
    pub fn get_instance(final_size: bool) -> Self {
        let mut instance = Self::new(final_size);
        instance.create_panel(final_size);
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel(boolean)`.
    fn create_panel(&mut self, _max_size: bool) {
        if let Some(small) = &mut self.rb_type_small {
            small.radio_button.alignment_x = 0.0;
        }
        self.rb_type_medium.radio_button.alignment_x = 0.0;
        self.rb_type_large.radio_button.alignment_x = 0.0;
        if let Some(extra_large) = &mut self.rb_type_extra_large {
            extra_large.radio_button.alignment_x = 0.0;
        }
        self.rb_type_custom.radio_button.alignment_x = 0.0;
        self.pnl_root.root_box_layout_x_axis = true;
        self.pnl_root.root_alignment_center = true;
        self.pnl_root.type_box_layout_y_axis = true;
        self.pnl_root.xyz_box_layout_y_axis = true;
        self.pnl_root.xyz_border_title = "In Pixels".into();
        self.pnl_root.horizontal_glue_between_panels = true;
        self.pnl_root.vertical_glue_between_fields = self.ltf_xyz.len() - 1;
        self.action_performed();
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &PatchSizePanelRoot {
        &self.pnl_root
    }

    /// Java private `addListeners`.
    fn add_listeners(&mut self) {
        if let Some(small) = &mut self.rb_type_small {
            small.add_action_listener();
        }
        self.rb_type_medium.add_action_listener();
        self.rb_type_large.add_action_listener();
        if let Some(extra_large) = &mut self.rb_type_extra_large {
            extra_large.add_action_listener();
        }
        self.rb_type_custom.add_action_listener();
    }

    /// Java `actionPerformed(ActionEvent)`; native action-event dispatch calls this method.
    pub fn action_performed(&mut self) {
        let selected_type = if self.rb_type_custom.is_selected() {
            CombinePatchSize::Custom
        } else if self
            .rb_type_small
            .as_ref()
            .is_some_and(RadioButton::is_selected)
        {
            CombinePatchSize::Small
        } else if self.rb_type_medium.is_selected() {
            CombinePatchSize::Medium
        } else if self.rb_type_large.is_selected() {
            CombinePatchSize::Large
        } else if self
            .rb_type_extra_large
            .as_ref()
            .is_some_and(RadioButton::is_selected)
        {
            CombinePatchSize::ExtraLarge
        } else {
            panic!("Java ButtonGroup has no selected RadioButton")
        };
        if selected_type != CombinePatchSize::Custom {
            let len = self.ltf_xyz.len().min(selected_type.get_xyz_len());
            for index in 0..len {
                self.ltf_xyz[index].set_text_number(selected_type.get_xyz(index));
            }
        }
        self.update_display();
    }

    /// Java private `updateDisplay`.
    fn update_display(&mut self) {
        let enabled = self.rb_type_custom.is_enabled();
        let selected = self.rb_type_custom.is_selected();
        for field in &mut self.ltf_xyz {
            field.set_enabled(enabled);
            field.set_editable(selected);
        }
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, enabled: bool) {
        if let Some(small) = &mut self.rb_type_small {
            small.set_enabled(enabled);
        }
        self.rb_type_medium.set_enabled(enabled);
        self.rb_type_large.set_enabled(enabled);
        if let Some(extra_large) = &mut self.rb_type_extra_large {
            extra_large.set_enabled(enabled);
        }
        self.rb_type_custom.set_enabled(enabled);
        self.update_display();
    }

    /// Java private `isEnabled`.
    fn is_enabled(&self) -> bool {
        self.rb_type_medium.is_enabled()
    }

    /// Java `setParameters(ConstCombineParams)`.
    pub fn set_parameters_combine_params<P: ConstCombineParams>(&mut self, combine_params: &P) {
        let mut combine_patch_size = combine_params.get_patch_size(self.final_size);
        if combine_patch_size.is_none() && self.final_size {
            combine_patch_size = Some(DEFAULT_FINAL_SIZE);
        }
        if combine_patch_size == Some(CombinePatchSize::Custom) {
            self.rb_type_custom.set_selected(true);
            if let Some(xyz) = combine_params.get_patch_size_xyz_array(self.final_size) {
                let len = self.ltf_xyz.len().min(xyz.len());
                for (index, value) in xyz.into_iter().take(len).enumerate() {
                    self.ltf_xyz[index].set_text(&value);
                }
            }
        } else {
            self.set_fixed_type(combine_patch_size);
        }
        self.action_performed();
    }

    /// Java `setParameters(ConstPatchcrawl3DParam)`.
    pub fn set_parameters_patchcrawl_3d_param<P: ConstPatchcrawl3dParam>(
        &mut self,
        patchcrawl_param: &P,
    ) {
        let xyz = [
            patchcrawl_param.get_x_patch_size(),
            patchcrawl_param.get_z_patch_size(),
            patchcrawl_param.get_y_patch_size(),
        ];
        let combine_patch_size =
            CombinePatchSize::get_instance_xyz_ints(Some(&xyz)).unwrap_or(CombinePatchSize::Custom);
        if combine_patch_size == CombinePatchSize::Custom {
            self.rb_type_custom.set_selected(true);
            let len = self.ltf_xyz.len().min(xyz.len());
            for (index, value) in xyz.into_iter().take(len).enumerate() {
                self.ltf_xyz[index].set_text_number(value);
            }
        } else {
            self.set_fixed_type(Some(combine_patch_size));
        }
        self.action_performed();
    }

    /// Java `setFixedType(CombinePatchSize)`.
    pub fn set_fixed_type(&mut self, combine_patch_size: Option<CombinePatchSize>) {
        match combine_patch_size {
            Some(CombinePatchSize::Small) if self.rb_type_small.is_some() => {
                self.rb_type_small.as_mut().unwrap().set_selected(true);
            }
            Some(CombinePatchSize::Small) => self.select_type_xyz(CombinePatchSize::Small),
            Some(CombinePatchSize::Medium) => {
                self.rb_type_medium.set_selected(true);
            }
            Some(CombinePatchSize::Large) => {
                self.rb_type_large.set_selected(true);
            }
            Some(CombinePatchSize::ExtraLarge) if self.rb_type_extra_large.is_some() => {
                self.rb_type_extra_large
                    .as_mut()
                    .unwrap()
                    .set_selected(true);
            }
            Some(CombinePatchSize::ExtraLarge) => {
                self.select_type_xyz(CombinePatchSize::ExtraLarge)
            }
            Some(CombinePatchSize::Custom) | None => {}
        }
    }

    /// Java `getParameters(CombineParams, boolean)`.
    pub fn get_parameters<P: CombineParams>(
        &self,
        combine_params: &mut P,
        do_validation: bool,
    ) -> bool {
        if self.is_enabled() {
            let selected_type = if self.rb_type_custom.is_selected() {
                CombinePatchSize::Custom
            } else if self
                .rb_type_small
                .as_ref()
                .is_some_and(RadioButton::is_selected)
            {
                CombinePatchSize::Small
            } else if self.rb_type_medium.is_selected() {
                CombinePatchSize::Medium
            } else if self.rb_type_large.is_selected() {
                CombinePatchSize::Large
            } else if self
                .rb_type_extra_large
                .as_ref()
                .is_some_and(RadioButton::is_selected)
            {
                CombinePatchSize::ExtraLarge
            } else {
                panic!("Java ButtonGroup has no selected RadioButton")
            };
            combine_params.set_patch_size(self.final_size, selected_type);
            if selected_type == CombinePatchSize::Custom {
                let mut xyz = Vec::with_capacity(self.ltf_xyz.len());
                for field in &self.ltf_xyz {
                    match field.get_text_validated(do_validation) {
                        Ok(value) => xyz.push(value),
                        Err(_) => return false,
                    }
                }
                combine_params.set_patch_size_xyz(self.final_size, xyz);
            }
        } else {
            combine_params.reset_patch_size(self.final_size);
        }
        true
    }

    /// Java private `selectTypeXyz(CombinePatchSize)`.
    fn select_type_xyz(&mut self, combine_patch_size: CombinePatchSize) {
        self.rb_type_custom.set_selected(true);
        let len = self.ltf_xyz.len().min(combine_patch_size.get_xyz_len());
        for index in 0..len {
            self.ltf_xyz[index].set_text_number(combine_patch_size.get_xyz(index));
        }
    }

    /// Java `setSmallTooltip`.
    pub fn set_small_tooltip(&mut self, tooltip: Option<&str>) {
        if let Some(small) = &mut self.rb_type_small {
            small.set_tool_tip_text(tooltip);
        }
    }

    /// Java `setMediumTooltip`.
    pub fn set_medium_tooltip(&mut self, tooltip: Option<&str>) {
        self.rb_type_medium.set_tool_tip_text(tooltip);
    }

    /// Java `setLargeTooltip`.
    pub fn set_large_tooltip(&mut self, tooltip: Option<&str>) {
        self.rb_type_large.set_tool_tip_text(tooltip);
    }

    /// Java `setExtraLargeTooltip`.
    pub fn set_extra_large_tooltip(&mut self, tooltip: Option<&str>) {
        if let Some(extra_large) = &mut self.rb_type_extra_large {
            extra_large.set_tool_tip_text(tooltip);
        }
    }

    /// Java `setCustomTooltip`.
    pub fn set_custom_tooltip(&mut self, tooltip: Option<&str>) {
        self.rb_type_custom.set_tool_tip_text(tooltip);
    }

    /// Java `setXTooltip`.
    pub fn set_x_tooltip(&mut self, tooltip: Option<&str>) {
        self.ltf_xyz[X_INDEX].set_tool_tip_text(tooltip);
    }

    /// Java `setYTooltip`.
    pub fn set_y_tooltip(&mut self, tooltip: Option<&str>) {
        self.ltf_xyz[Y_INDEX].set_tool_tip_text(tooltip);
    }

    /// Java `setZTooltip`.
    pub fn set_z_tooltip(&mut self, tooltip: Option<&str>) {
        self.ltf_xyz[Z_INDEX].set_tool_tip_text(tooltip);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::etomo::r#type::combine_patch_size::PatchSizeArray;

    #[derive(Default)]
    struct Params {
        initial: Option<CombinePatchSize>,
        initial_xyz: Option<Vec<String>>,
        set: Option<CombinePatchSize>,
        set_xyz: Option<Vec<String>>,
        reset: bool,
    }
    impl ConstCombineParams for Params {
        fn get_patch_size(&self, _auto_final: bool) -> Option<CombinePatchSize> {
            self.initial
        }
        fn get_patch_size_xyz_array(&self, _auto_final: bool) -> Option<Vec<String>> {
            self.initial_xyz.clone()
        }
    }
    impl CombineParams for Params {
        fn set_patch_size(&mut self, _auto_final: bool, value: CombinePatchSize) {
            self.set = Some(value);
        }
        fn set_patch_size_xyz(&mut self, _auto_final: bool, xyz: Vec<String>) {
            self.set_xyz = Some(xyz);
        }
        fn reset_patch_size(&mut self, _auto_final: bool) {
            self.reset = true;
        }
    }

    #[test]
    fn custom_size_is_editable_and_validated_into_parameters() {
        let mut panel = PatchSizePanel::get_instance(false);
        panel.set_parameters_combine_params(&Params {
            initial: Some(CombinePatchSize::Custom),
            initial_xyz: Some(vec!["80".into(), "82".into(), "44".into()]),
            ..Default::default()
        });
        let mut params = Params::default();
        assert!(panel.get_parameters(&mut params, true));
        assert_eq!(params.set, Some(CombinePatchSize::Custom));
        assert_eq!(
            params.set_xyz,
            Some(vec!["80".into(), "82".into(), "44".into()])
        );
        assert!(panel.ltf_xyz.iter().all(LabeledTextField::is_editable));
    }

    #[test]
    fn unavailable_fixed_type_becomes_custom_with_xyz() {
        CombinePatchSize::set_patch_size_array(PatchSizeArray {
            extra_large: [128, 96, 64],
            loaded: true,
            ..Default::default()
        });
        let mut panel = PatchSizePanel::get_instance(false);
        panel.set_fixed_type(Some(CombinePatchSize::ExtraLarge));
        panel.action_performed();
        assert!(panel.rb_type_custom.is_selected());
        assert_eq!(panel.ltf_xyz[0].get_text(), "128");
        assert!(panel.ltf_xyz[0].is_editable());
    }

    #[test]
    fn disabled_panel_resets_parameters() {
        let mut panel = PatchSizePanel::get_instance(true);
        panel.set_enabled(false);
        let mut params = Params::default();
        assert!(panel.get_parameters(&mut params, true));
        assert!(params.reset);
    }
}
