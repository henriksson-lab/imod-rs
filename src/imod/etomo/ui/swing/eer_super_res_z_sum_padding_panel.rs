//! `IMOD/Etomo/src/etomo/ui/swing/EERSuperResZSumPaddingPanel.java`.
//!
//! Swing construction and autodoc I/O remain explicit presentation/storage
//! boundaries.  This source unit retains the EER super-resolution and mutually
//! exclusive z-summing choice, exact bounds, parameter transfer, and header
//! expansion routing.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::r#type::{axis_id::AxisID, dialog_type::DialogType};

use super::{
    panel_header::{ExpandButton, Expandable, PanelHeader},
    radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup},
    spinner::Spinner,
};

pub const EER_Z_SUM_FRAMES_MIN: i32 = 2;
pub const EER_Z_SUM_FRAMES_MAX: i32 = 100;
pub const EER_Z_SUM_SETS_MIN: i32 = 1;
pub const EER_Z_SUM_SETS_MAX: i32 = 1000;
pub const EER_Z_SUM_SETS_DEFAULT: i32 = 1;
pub const EER_SUPER_RES_Z_SUM_PADDING_KEY: &str = "EERSuperResZSumPadding";

/// Java `EERSuperRes` values used by this panel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EerSuperRes {
    None,
    TwoX,
    FourX,
}

impl EerSuperRes {
    pub const DEFAULT: Self = Self::TwoX;

    pub fn get_instance(value: Option<i32>) -> Option<Self> {
        match value {
            Some(0) => Some(Self::None),
            Some(1) => Some(Self::TwoX),
            Some(2) => Some(Self::FourX),
            _ => None,
        }
    }
    pub fn get_value(self) -> i32 {
        match self {
            Self::None => 0,
            Self::TwoX => 1,
            Self::FourX => 2,
        }
    }
    pub fn get_label(self) -> &'static str {
        match self {
            Self::None => "4K",
            Self::TwoX => "8K",
            Self::FourX => "16K",
        }
    }
    pub fn get_tooltip(self) -> &'static str {
        match self {
            Self::None => "Read in frames for processing with anti-aliased reduction to 4K by 4K.",
            Self::TwoX => "Read in frames for processing with anti-aliased reduction to 8K by 8K.",
            Self::FourX => "Read in 16K frames at full 4x super-resolution.",
        }
    }
}

/// Java private static `EERZSum`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EerZSum {
    Frames,
    Sets,
}

impl EerZSum {
    pub const DEFAULT: Self = Self::Frames;
    /// Java `isDefault()`.
    pub fn is_default(self) -> bool {
        self == Self::DEFAULT
    }
    pub fn radio_enum_value_name(self) -> &'static str {
        match self {
            Self::Frames => "framesradio",
            Self::Sets => "setsradio",
        }
    }
    pub fn spinner_enum_value_name(self) -> &'static str {
        match self {
            Self::Frames => "framesspin",
            Self::Sets => "setsspin",
        }
    }
    pub fn get_label(self) -> &'static str {
        match self {
            Self::Frames => "Sum frames to make",
            Self::Sets => "Sum successive sets of",
        }
    }
}

/// Java `toString()`.
impl std::fmt::Display for EerZSum {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.radio_enum_value_name())
    }
}

/// Java `AlignFramesParam` calls made by this source unit.
pub trait EerSuperResZSumPaddingParameters {
    fn get_eer_super_res(&self) -> Option<i32>;
    fn is_eer_z_sum_frames_set(&self) -> bool;
    fn get_eer_z_sum_frames(&self) -> i32;
    fn get_eer_z_sum_sets(&self) -> i32;
    fn set_eer_super_res(&mut self, value: i32);
    fn set_eer_z_sum_frames(&mut self, value: i32);
    fn set_eer_z_sum_sets(&mut self, value: i32);
}

/// Small source-facing form of the `AlignFramesParam` EER fields.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AlignFramesEerParameters {
    pub eer_super_res: Option<i32>,
    pub eer_z_sum_frames: Option<i32>,
    pub eer_z_sum_sets: i32,
}

impl EerSuperResZSumPaddingParameters for AlignFramesEerParameters {
    fn get_eer_super_res(&self) -> Option<i32> {
        self.eer_super_res
    }
    fn is_eer_z_sum_frames_set(&self) -> bool {
        self.eer_z_sum_frames.is_some()
    }
    fn get_eer_z_sum_frames(&self) -> i32 {
        self.eer_z_sum_frames.unwrap_or_default()
    }
    fn get_eer_z_sum_sets(&self) -> i32 {
        self.eer_z_sum_sets
    }
    fn set_eer_super_res(&mut self, value: i32) {
        self.eer_super_res = Some(value);
    }
    fn set_eer_z_sum_frames(&mut self, value: i32) {
        self.eer_z_sum_frames = Some(value);
    }
    fn set_eer_z_sum_sets(&mut self, value: i32) {
        self.eer_z_sum_sets = value;
        self.eer_z_sum_frames = None;
    }
}

/// Direct Java `UIHarness.INSTANCE.pack(axisID, manager)` boundary.
pub trait EerSuperResZSumPaddingManager {
    fn pack(&mut self, axis_id: AxisID);
}

/// Swing label state kept at the presentation boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Label {
    pub text: String,
    pub enabled: bool,
    pub tooltip: Option<String>,
}
impl Label {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            enabled: true,
            tooltip: None,
        }
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    pub fn set_tooltip_text(&mut self, tooltip: Option<&str>) {
        self.tooltip = tooltip.map(str::to_owned);
    }
}

/// Java `JPanel` construction and ordered `BoxLayout` children.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EerSuperResZSumPaddingPanelLayout {
    pub root_box_layout_y_axis: bool,
    pub root_etched_border: bool,
    pub body_box_layout_y_axis: bool,
    pub body_visible: bool,
    pub root_component_order: Vec<String>,
    pub body_component_order: Vec<String>,
    pub super_res_component_order: Vec<String>,
    pub z_sum_frames_component_order: Vec<String>,
    pub z_sum_sets_component_order: Vec<String>,
}

/// Java final `EERSuperResZSumPaddingPanel`.
pub struct EerSuperResZSumPaddingPanel {
    pub pnl_root: EerSuperResZSumPaddingPanelLayout,
    pub l_super_res: Label,
    pub rb_super_res_none: RadioButton,
    pub rb_super_res_2x: RadioButton,
    pub rb_super_res_4x: RadioButton,
    pub rb_z_sum_frames: RadioButton,
    pub sp_z_sum_frames: Spinner,
    pub l_z_sum_frames: Label,
    pub rb_z_sum_sets: RadioButton,
    pub sp_z_sum_sets: Spinner,
    pub l_z_sum_sets: Label,
    pub axis_id: AxisID,
    pub header: PanelHeader,
    pub listeners_added: bool,
    pub tooltip_autodoc_available: bool,
}

impl EerSuperResZSumPaddingPanel {
    /// Java private constructor.
    pub fn new(axis_id: AxisID, dialog_type: DialogType, eer_z_sum_frames_default: i32) -> Self {
        let super_res_group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let z_sum_group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        Self {
            pnl_root: EerSuperResZSumPaddingPanelLayout {
                body_visible: true,
                ..Default::default()
            },
            l_super_res: Label::new("Frame size to read in:"),
            rb_super_res_none: RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: EerSuperRes::None.get_label().into(),
                    default: false,
                    value: Some(EerSuperRes::None.get_value().to_string()),
                },
                Some(super_res_group.clone()),
            ),
            rb_super_res_2x: RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: EerSuperRes::TwoX.get_label().into(),
                    default: true,
                    value: Some(EerSuperRes::TwoX.get_value().to_string()),
                },
                Some(super_res_group.clone()),
            ),
            rb_super_res_4x: RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: EerSuperRes::FourX.get_label().into(),
                    default: false,
                    value: Some(EerSuperRes::FourX.get_value().to_string()),
                },
                Some(super_res_group),
            ),
            rb_z_sum_frames: RadioButton::new_with_enumerated_type(
                Some(EerZSum::Frames.get_label().into()),
                EnumeratedTypeBoundary {
                    label: EerZSum::Frames.get_label().into(),
                    default: true,
                    value: None,
                },
                Some(z_sum_group.clone()),
            ),
            sp_z_sum_frames: Spinner::get_instance(
                EerZSum::Frames.get_label(),
                eer_z_sum_frames_default,
                EER_Z_SUM_FRAMES_MIN,
                EER_Z_SUM_FRAMES_MAX,
                1,
            ),
            l_z_sum_frames: Label::new("images to align"),
            rb_z_sum_sets: RadioButton::new_with_enumerated_type(
                Some(EerZSum::Sets.get_label().into()),
                EnumeratedTypeBoundary {
                    label: EerZSum::Sets.get_label().into(),
                    default: false,
                    value: None,
                },
                Some(z_sum_group),
            ),
            sp_z_sum_sets: Spinner::get_instance(
                EerZSum::Sets.get_label(),
                EER_Z_SUM_SETS_DEFAULT,
                EER_Z_SUM_SETS_MIN,
                EER_Z_SUM_SETS_MAX,
                1,
            ),
            l_z_sum_sets: Label::new("frames"),
            axis_id,
            header: PanelHeader::new(
                "Reading EER Files",
                false,
                false,
                dialog_type,
                true,
                false,
                true,
                false,
                true,
            ),
            listeners_added: false,
            tooltip_autodoc_available: false,
        }
    }

    /// Java static `getInstance`; the autodoc acquisition is retained as an
    /// explicit caller-provided storage boundary.
    pub fn get_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        eer_z_sum_frames_default: i32,
        autodoc_available: bool,
    ) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, eer_z_sum_frames_default);
        instance.create_panel();
        instance.set_tooltips(autodoc_available, None, None);
        instance.add_listeners();
        instance
    }

    /// Java private `createPanel`.
    pub fn create_panel(&mut self) {
        self.header.btn_open_close.as_mut().unwrap().update(false);
        self.pnl_root.root_box_layout_y_axis = true;
        self.pnl_root.root_etched_border = true;
        self.pnl_root.body_box_layout_y_axis = true;
        self.pnl_root.root_component_order = vec!["header".into(), "pnlBody".into()];
        self.pnl_root.body_component_order = vec![
            "pnlSuperRes".into(),
            "pnlZSumFrames".into(),
            "verticalStrut(2)".into(),
            "pnlZSumSets".into(),
            "verticalStrut(2)".into(),
        ];
        self.pnl_root.super_res_component_order = vec![
            "horizontalStrut(2)".into(),
            "lSuperRes".into(),
            "rbSuperResNone".into(),
            "rbSuperRes2x".into(),
            "rbSuperRes4x".into(),
        ];
        self.pnl_root.z_sum_frames_component_order = vec![
            "rbZSumFrames".into(),
            "spZSumFrames".into(),
            "horizontalStrut(3)".into(),
            "lZSumFrames".into(),
            "horizontalStrut(2)".into(),
        ];
        self.pnl_root.z_sum_sets_component_order = vec![
            "rbZSumSets".into(),
            "spZSumSets".into(),
            "horizontalStrut(3)".into(),
            "lZSumSets".into(),
            "horizontalStrut(2)".into(),
        ];
        self.update_display();
    }
    /// Java `getComponent` boundary.
    pub fn get_component(&self) -> &EerSuperResZSumPaddingPanelLayout {
        &self.pnl_root
    }
    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.rb_z_sum_frames.add_action_listener();
        self.rb_z_sum_sets.add_action_listener();
        self.listeners_added = true;
    }
    /// Java `actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self) {
        self.update_display();
    }
    /// Java private `updateDisplay`.
    pub fn update_display(&mut self) {
        let z_sum_frames = self.rb_z_sum_frames.is_selected();
        self.sp_z_sum_frames.set_enabled(z_sum_frames);
        self.l_z_sum_frames.set_enabled(z_sum_frames);
        let z_sum_sets = self.rb_z_sum_sets.is_selected();
        self.sp_z_sum_sets.set_enabled(z_sum_sets);
        self.l_z_sum_sets.set_enabled(z_sum_sets);
    }
    /// Java `setParameters(AlignFramesParam)`.
    pub fn set_parameters<P: EerSuperResZSumPaddingParameters>(&mut self, param: &P) {
        match EerSuperRes::get_instance(param.get_eer_super_res()) {
            Some(EerSuperRes::None) => self.rb_super_res_none.set_selected(true),
            Some(EerSuperRes::TwoX) => self.rb_super_res_2x.set_selected(true),
            Some(EerSuperRes::FourX) => self.rb_super_res_4x.set_selected(true),
            None => {}
        }
        if param.is_eer_z_sum_frames_set() {
            self.rb_z_sum_frames.set_selected(true);
            self.sp_z_sum_frames.set_value(param.get_eer_z_sum_frames());
        } else {
            self.rb_z_sum_sets.set_selected(true);
            self.sp_z_sum_sets.set_value(param.get_eer_z_sum_sets());
        }
        self.update_display();
    }
    /// Java `getParameters(AlignFramesParam)`.
    pub fn get_parameters<P: EerSuperResZSumPaddingParameters>(&self, param: &mut P) {
        for (button, value) in [
            (&self.rb_super_res_none, EerSuperRes::None),
            (&self.rb_super_res_2x, EerSuperRes::TwoX),
            (&self.rb_super_res_4x, EerSuperRes::FourX),
        ] {
            if button.is_selected() {
                param.set_eer_super_res(value.get_value());
                break;
            }
        }
        if self.rb_z_sum_frames.is_selected() {
            param.set_eer_z_sum_frames(self.sp_z_sum_frames.get_int_value());
        } else if self.rb_z_sum_sets.is_selected() {
            param.set_eer_z_sum_sets(self.sp_z_sum_sets.get_int_value());
        }
    }
    /// Java `Expandable.expand(ExpandButton)`.
    pub fn expand<M: EerSuperResZSumPaddingManager>(
        &mut self,
        button: &ExpandButton,
        manager: &mut M,
    ) {
        if self.header.equals_open_close(button) {
            self.pnl_root.body_visible = button.is_expanded();
        }
        manager.pack(self.axis_id);
    }
    /// Java `Expandable.expand(GlobalExpandButton)`, intentionally empty.
    pub fn expand_global_button(&mut self) {}
    /// Java private `setTootips`, with concrete autodoc extraction at the
    /// storage boundary and resulting strings passed directly here.
    pub fn set_tooltips(
        &mut self,
        autodoc_available: bool,
        frames_tooltip: Option<&str>,
        sets_tooltip: Option<&str>,
    ) {
        self.tooltip_autodoc_available = autodoc_available;
        if autodoc_available {
            self.rb_super_res_none
                .set_tool_tip_text(Some(EerSuperRes::None.get_tooltip()));
            self.rb_super_res_2x
                .set_tool_tip_text(Some(EerSuperRes::TwoX.get_tooltip()));
            self.rb_super_res_4x
                .set_tool_tip_text(Some(EerSuperRes::FourX.get_tooltip()));
            self.rb_z_sum_frames.set_tool_tip_text(frames_tooltip);
            self.sp_z_sum_frames.set_tool_tip_text(frames_tooltip);
            self.l_z_sum_frames.set_tooltip_text(frames_tooltip);
            self.rb_z_sum_sets.set_tool_tip_text(sets_tooltip);
            self.sp_z_sum_sets.set_tool_tip_text(sets_tooltip);
            self.l_z_sum_sets.set_tooltip_text(sets_tooltip);
        }
    }
}

impl Expandable for EerSuperResZSumPaddingPanel {
    fn expand_expand_button(&mut self, button: &ExpandButton) {
        if self.header.equals_open_close(button) {
            self.pnl_root.body_visible = button.is_expanded();
        }
    }
    fn expand_global_button(&mut self, _: &super::process_dialog::GlobalExpandButton) {
        self.expand_global_button();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Manager {
        packed: Vec<AxisID>,
    }
    impl EerSuperResZSumPaddingManager for Manager {
        fn pack(&mut self, axis_id: AxisID) {
            self.packed.push(axis_id);
        }
    }
    #[test]
    fn source_default_and_z_sum_choice_drive_dependent_controls() {
        let mut panel =
            EerSuperResZSumPaddingPanel::get_instance(AxisID::Only, DialogType::Tools, 7, false);
        assert!(EerZSum::Frames.is_default());
        assert!(!EerZSum::Sets.is_default());
        assert_eq!(EerZSum::Sets.to_string(), "setsradio");
        assert!(panel.rb_super_res_2x.is_selected());
        assert!(panel.sp_z_sum_frames.is_enabled());
        panel.rb_z_sum_sets.set_selected(true);
        panel.action_performed();
        assert!(!panel.sp_z_sum_frames.is_enabled());
        assert!(panel.sp_z_sum_sets.is_enabled());
    }
    #[test]
    fn source_parameter_round_trip_keeps_super_res_and_selected_z_sum() {
        let mut panel =
            EerSuperResZSumPaddingPanel::get_instance(AxisID::Only, DialogType::Tools, 7, false);
        let input = AlignFramesEerParameters {
            eer_super_res: Some(2),
            eer_z_sum_frames: None,
            eer_z_sum_sets: 42,
        };
        panel.set_parameters(&input);
        let mut output = AlignFramesEerParameters::default();
        panel.get_parameters(&mut output);
        assert_eq!(output.eer_super_res, Some(2));
        assert_eq!(output.eer_z_sum_frames, None);
        assert_eq!(output.eer_z_sum_sets, 42);
    }
    #[test]
    fn source_open_close_expansion_changes_body_and_packs() {
        let mut panel =
            EerSuperResZSumPaddingPanel::get_instance(AxisID::Only, DialogType::Tools, 7, false);
        let mut button = panel.header.btn_open_close.clone().unwrap();
        button.update(true);
        let mut manager = Manager::default();
        panel.expand(&button, &mut manager);
        assert!(panel.pnl_root.body_visible);
        assert_eq!(manager.packed, vec![AxisID::Only]);
    }
}
