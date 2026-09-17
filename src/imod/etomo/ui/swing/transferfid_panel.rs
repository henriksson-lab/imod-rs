//! `IMOD/Etomo/src/etomo/ui/swing/TransferfidPanel.java`.
//!
//! Swing containers and the concrete `ApplicationManager` are presentation
//! boundaries.  The source-owned widget state, field validation, metadata
//! transfer, expand behaviour, and action arguments are retained here.
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::ui::field_type::FieldType;

use super::check_box::CheckBox;
use super::labeled_text_field::{FieldValidationFailedException, LabeledTextField};
use super::multi_line_button::MultiLineButton;
use super::panel_header::{ExpandButton, PanelHeader};
use super::radio_button::{EnumeratedTypeBoundary, RadioButton, RadioButtonGroup};

/// Java `ImodManager.COARSE_ALIGNED_KEY`.
pub const COARSE_ALIGNED_KEY: &str = "coarseAligned";

/// Java `MirrorInX` values selected through the three radio buttons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MirrorInX {
    AssessBoth,
    Always,
    Never,
}

impl MirrorInX {
    /// Java `MirrorInX.getValue()`.
    pub fn get_value(self) -> i32 {
        match self {
            Self::AssessBoth => 0,
            Self::Always => 1,
            Self::Never => -1,
        }
    }

    /// Java `MirrorInX.getLabel()`.
    pub fn get_label(self) -> &'static str {
        match self {
            Self::AssessBoth => "Try with and without",
            Self::Always => "Use mirroring",
            Self::Never => "Do not use mirroring",
        }
    }
}

/// The `TransferfidParam` operations made by this source unit.
pub trait TransferfidParam {
    fn initialize(&mut self);
    fn run_midas(&self) -> bool;
    fn set_run_midas(&mut self, value: bool);
    fn center_view_a(&self) -> String;
    fn set_center_view_a(&mut self, value: String);
    fn center_view_b(&self) -> String;
    fn set_center_view_b(&mut self, value: String);
    fn number_views(&self) -> String;
    fn set_number_views(&mut self, value: String);
    fn search_direction(&self) -> Option<i32>;
    fn reset_search_direction(&mut self);
    fn set_search_direction(&mut self, value: i32);
    fn mirror_xaxis(&self) -> MirrorInX;
    fn set_mirror_xaxis(&mut self, value: MirrorInX);
}

/// The direct `ApplicationManager` / `MetaData` calls from this Java unit.
pub trait TransferfidPanelApplicationManager<P: TransferfidParam> {
    fn new_transferfid_param(&self, axis_id: AxisID) -> P;
    fn get_transferfid_a_fields(&self, params: &mut P);
    fn get_transferfid_b_fields(&self, params: &mut P);
    fn set_transferfid_a_fields(&mut self, params: &P);
    fn set_transferfid_b_fields(&mut self, params: &P);
    fn transferfid(
        &mut self,
        axis_id: AxisID,
        button: &MultiLineButton,
        deferred_3dmod_button: Option<&MultiLineButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
    );
    fn imod_seed_model(
        &mut self,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        button: &MultiLineButton,
        imod_key: &str,
        seed_file: String,
        raw_tilt_file: String,
        dialog_type: DialogType,
    );
    fn seed_file_name(&self, axis_id: AxisID) -> String;
    fn raw_tilt_file(&self, axis_id: AxisID) -> String;
    fn pack(&mut self, axis_id: AxisID);
}

/// Source-visible Swing layout state initialized by the constructor.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TransferfidPanelLayout {
    pub root_visible: bool,
    pub body_visible: bool,
    pub minimum_body_width: i32,
    pub search_direction_border: Option<String>,
    pub mirror_xaxis_border: Option<String>,
    pub body_component_order: Vec<String>,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
}

/// Java `TransferfidPanel`.
pub struct TransferfidPanel {
    pub panel_transferfid: TransferfidPanelLayout,
    pub cb_run_midas: CheckBox,
    pub ltf_center_view_a: LabeledTextField,
    pub ltf_center_view_b: LabeledTextField,
    pub ltf_number_views: LabeledTextField,
    pub bg_search_direction: Rc<RefCell<RadioButtonGroup>>,
    pub rb_search_both: RadioButton,
    pub rb_search_plus_90: RadioButton,
    pub rb_search_minus_90: RadioButton,
    pub btn_3dmod_seed: MultiLineButton,
    pub bg_mirror_in_x: Rc<RefCell<RadioButtonGroup>>,
    pub rb_mirror_in_x_assess_both: RadioButton,
    pub rb_mirror_in_x_always: RadioButton,
    pub rb_mirror_in_x_never: RadioButton,
    pub header: PanelHeader,
    pub button_transferfid: MultiLineButton,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
}

impl TransferfidPanel {
    /// Java private constructor `TransferfidPanel(...)`.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let bg_search_direction = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let bg_mirror_in_x = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut result = Self {
            panel_transferfid: TransferfidPanelLayout {
                root_visible: true,
                body_visible: true,
                minimum_body_width: 300,
                search_direction_border: Some("Search Direction".into()),
                mirror_xaxis_border: Some("Mirroring around X axis".into()),
                body_component_order: vec![
                    "Run Midas".into(),
                    "centerViewA".into(),
                    "centerViewB".into(),
                    "numberViews".into(),
                    "searchDirection".into(),
                    "mirrorXaxis".into(),
                    "transferfid".into(),
                ],
                ..Default::default()
            },
            cb_run_midas: CheckBox::new_with_text("Run midas"),
            ltf_center_view_a: LabeledTextField::new(FieldType::Integer, "Center view A: "),
            ltf_center_view_b: LabeledTextField::new(FieldType::Integer, "Center view B: "),
            ltf_number_views: LabeledTextField::new(
                FieldType::Integer,
                "Number of views in the search: ",
            ),
            bg_search_direction: bg_search_direction.clone(),
            rb_search_both: RadioButton::new_in_group(
                "Both directions",
                bg_search_direction.clone(),
            ),
            rb_search_plus_90: RadioButton::new_in_group(
                "+90 (CCW) only",
                bg_search_direction.clone(),
            ),
            rb_search_minus_90: RadioButton::new_in_group("-90 (CW) only", bg_search_direction),
            btn_3dmod_seed: MultiLineButton::new_with_label(Some("Open Seed Model")),
            bg_mirror_in_x: bg_mirror_in_x.clone(),
            rb_mirror_in_x_assess_both: RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: MirrorInX::AssessBoth.get_label().into(),
                    default: true,
                    value: Some("0".into()),
                },
                Some(bg_mirror_in_x.clone()),
            ),
            rb_mirror_in_x_always: RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: MirrorInX::Always.get_label().into(),
                    default: false,
                    value: Some("1".into()),
                },
                Some(bg_mirror_in_x.clone()),
            ),
            rb_mirror_in_x_never: RadioButton::new_with_enumerated_type(
                None,
                EnumeratedTypeBoundary {
                    label: MirrorInX::Never.get_label().into(),
                    default: false,
                    value: Some("-1".into()),
                },
                Some(bg_mirror_in_x),
            ),
            header: PanelHeader::new(
                "Transfer Fiducials",
                true,
                false,
                dialog_type,
                true,
                true,
                true,
                false,
                true,
            ),
            button_transferfid: MultiLineButton::new_with_label(Some(
                "Transfer Fiducials From Other Axis",
            )),
            axis_id,
            dialog_type,
        };
        // Java `Run3dmodButton` supplies the wrapped button's default command
        // from its label at the Swing boundary.
        result
            .btn_3dmod_seed
            .set_action_command(Some("Open Seed Model"));
        result
            .button_transferfid
            .set_action_command(Some("Transfer Fiducials From Other Axis"));
        result.button_transferfid.set_alignment_x(0.5);
        result.set_tool_tip_text();
        result
    }

    /// Java static `getInstance(...)`.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.add_listeners();
        instance
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<P: TransferfidParam, M: TransferfidPanelApplicationManager<P>>(
        &self,
        manager: &mut M,
        command: &str,
        deferred_3dmod_button: Option<&MultiLineButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if Some(command) == self.button_transferfid.get_action_command() {
            manager.transferfid(
                self.axis_id,
                &self.button_transferfid,
                deferred_3dmod_button,
                run_3dmod_menu_options,
                self.dialog_type,
            );
        } else if Some(command) == self.btn_3dmod_seed.get_action_command() {
            manager.imod_seed_model(
                self.axis_id,
                run_3dmod_menu_options,
                &self.btn_3dmod_seed,
                COARSE_ALIGNED_KEY,
                manager.seed_file_name(self.axis_id),
                manager.raw_tilt_file(self.axis_id),
                self.dialog_type,
            );
        }
    }

    #[allow(non_snake_case)]
    /// Native-name adapter for the Java listener's `actionPerformed`.
    pub fn actionPerformed<P: TransferfidParam, M: TransferfidPanelApplicationManager<P>>(
        &self,
        manager: &mut M,
        command: &str,
    ) {
        self.action(manager, command, None, None);
    }

    /// Java `expand(GlobalExpandButton)`.
    pub fn expand_global(&mut self) {}

    /// Java `expand(ExpandButton)`.
    pub fn expand<P: TransferfidParam, M: TransferfidPanelApplicationManager<P>>(
        &mut self,
        manager: &mut M,
        button: &ExpandButton,
    ) {
        if self.header.equals_open_close(button) {
            self.panel_transferfid.body_visible = button.is_expanded();
        } else if self.header.equals_advanced_basic(button) {
            self.update_advanced(button.is_expanded());
        }
        manager.pack(self.axis_id);
    }

    /// Java private `setup`.
    pub fn setup(&mut self) {}

    /// Java `setVisible(boolean)`.
    pub fn set_visible(&mut self, visible: bool) {
        self.panel_transferfid.root_visible = visible;
    }

    /// Java `setParameters()`.
    pub fn set_parameters<P: TransferfidParam, M: TransferfidPanelApplicationManager<P>>(
        &mut self,
        manager: &M,
    ) {
        let mut params = manager.new_transferfid_param(self.axis_id);
        params.initialize();
        if self.axis_id == AxisID::Second {
            manager.get_transferfid_b_fields(&mut params);
        } else {
            manager.get_transferfid_a_fields(&mut params);
        }
        self.cb_run_midas.set_selected(params.run_midas());
        self.ltf_center_view_a.set_text(&params.center_view_a());
        self.ltf_center_view_b.set_text(&params.center_view_b());
        self.ltf_number_views.set_text(&params.number_views());
        if params.search_direction().is_none() {
            self.rb_search_both.set_selected(true);
        }
        if params.search_direction() == Some(-1) {
            self.rb_search_minus_90.set_selected(true);
        }
        if params.search_direction() == Some(1) {
            self.rb_search_plus_90.set_selected(true);
        }
        match params.mirror_xaxis() {
            MirrorInX::AssessBoth => self.rb_mirror_in_x_assess_both.set_selected(true),
            MirrorInX::Always => self.rb_mirror_in_x_always.set_selected(true),
            MirrorInX::Never => self.rb_mirror_in_x_never.set_selected(true),
        }
    }

    /// Java `getParameters(boolean)`.
    pub fn get_parameters<P: TransferfidParam, M: TransferfidPanelApplicationManager<P>>(
        &self,
        manager: &mut M,
        do_validation: bool,
    ) -> bool {
        let mut params = manager.new_transferfid_param(self.axis_id);
        self.get_parameters_into(manager, &mut params, do_validation)
    }

    /// Java `getParameters(TransferfidParam, boolean)`.
    pub fn get_parameters_into<P: TransferfidParam, M: TransferfidPanelApplicationManager<P>>(
        &self,
        manager: &mut M,
        params: &mut P,
        do_validation: bool,
    ) -> bool {
        let fields = (|| -> Result<(), FieldValidationFailedException> {
            params.set_run_midas(self.cb_run_midas.is_selected());
            params.set_center_view_a(self.ltf_center_view_a.get_text_validated(do_validation)?);
            params.set_center_view_b(self.ltf_center_view_b.get_text_validated(do_validation)?);
            if self.rb_search_both.is_selected() {
                params.reset_search_direction();
            }
            if self.rb_search_plus_90.is_selected() {
                params.set_search_direction(1);
            }
            if self.rb_search_minus_90.is_selected() {
                params.set_search_direction(-1);
            }
            params.set_number_views(self.ltf_number_views.get_text_validated(do_validation)?);
            if self.rb_mirror_in_x_assess_both.is_selected() {
                params.set_mirror_xaxis(MirrorInX::AssessBoth);
            }
            if self.rb_mirror_in_x_always.is_selected() {
                params.set_mirror_xaxis(MirrorInX::Always);
            }
            if self.rb_mirror_in_x_never.is_selected() {
                params.set_mirror_xaxis(MirrorInX::Never);
            }
            Ok(())
        })();
        if fields.is_err() {
            return false;
        }
        if self.axis_id == AxisID::Second {
            manager.set_transferfid_b_fields(params);
        } else {
            manager.set_transferfid_a_fields(params);
        }
        true
    }

    /// Java private `addListeners`.
    pub fn add_listeners(&mut self) {
        self.button_transferfid.add_action_listener();
        self.btn_3dmod_seed.add_action_listener();
        self.panel_transferfid.listener_count += 2;
    }

    /// Java `done`.
    pub fn done(&mut self) {
        self.button_transferfid.remove_action_listener();
        self.panel_transferfid.listener_count =
            self.panel_transferfid.listener_count.saturating_sub(1);
    }

    /// Java `updateAdvanced(boolean)`.
    pub fn update_advanced(&mut self, is_advanced: bool) {
        self.ltf_center_view_a.set_visible(is_advanced);
        self.ltf_center_view_b.set_visible(is_advanced);
        self.ltf_number_views.set_visible(is_advanced);
        self.rb_search_both.set_visible(is_advanced);
        self.rb_search_plus_90.set_visible(is_advanced);
        self.rb_search_minus_90.set_visible(is_advanced);
        self.rb_mirror_in_x_assess_both.set_visible(is_advanced);
        self.rb_mirror_in_x_always.set_visible(is_advanced);
        self.rb_mirror_in_x_never.set_visible(is_advanced);
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&mut self, is_enabled: bool) {
        self.button_transferfid.set_enabled(is_enabled);
        self.cb_run_midas.set_enabled(is_enabled);
        self.ltf_center_view_a.set_enabled(is_enabled);
        self.ltf_center_view_b.set_enabled(is_enabled);
        self.ltf_number_views.set_enabled(is_enabled);
        self.rb_search_both.set_enabled(is_enabled);
        self.rb_search_plus_90.set_enabled(is_enabled);
        self.rb_search_minus_90.set_enabled(is_enabled);
        self.rb_mirror_in_x_assess_both.set_enabled(is_enabled);
        self.rb_mirror_in_x_always.set_enabled(is_enabled);
        self.rb_mirror_in_x_never.set_enabled(is_enabled);
    }

    /// Java private `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.cb_run_midas
            .set_tool_tip_text(Some("Run Midas to adjust initial alignment manually."));
        self.ltf_center_view_a.set_tool_tip_text(Some(
            "View from A around which to search for the best pair of views.",
        ));
        self.ltf_center_view_b.set_tool_tip_text(Some(
            "View from B around which to search for the best pair of views.",
        ));
        self.ltf_number_views.set_tool_tip_text(Some(
            "Number of views from each axis to consider in searching for best pair.",
        ));
        self.rb_search_both.set_tool_tip_text(Some(
            "Try both +90 and -90 degree rotations in searching for best pair of views.",
        ));
        self.rb_search_plus_90.set_tool_tip_text(Some(
            "Try only +90 degree rotations in searching for best pair of views.",
        ));
        self.rb_search_minus_90.set_tool_tip_text(Some(
            "Try only -90 degree rotations in searching for best pair of views.",
        ));
        self.button_transferfid.set_tool_tip_text(Some("Run Transferfid to make a seed model for this axis from fiducial model for the other axis."));
        self.rb_mirror_in_x_always.set_tool_tip_text(Some(
            "Mirror one image around the X axis before rotating by 90 degrees.",
        ));
        self.rb_mirror_in_x_assess_both.set_tool_tip_text(Some("Assess both mirroring one image around the X axis before rotating by 90 degrees, and not mirroring.  Use the best method."));
        self.rb_mirror_in_x_never.set_tool_tip_text(Some(
            "Do not mirror one image around the X axis before rotating by 90 degrees.",
        ));
        self.panel_transferfid.tooltip_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Default)]
    struct Param {
        run: bool,
        a: String,
        b: String,
        number: String,
        direction: Option<i32>,
        mirror: Option<MirrorInX>,
    }
    impl TransferfidParam for Param {
        fn initialize(&mut self) {}
        fn run_midas(&self) -> bool {
            self.run
        }
        fn set_run_midas(&mut self, value: bool) {
            self.run = value;
        }
        fn center_view_a(&self) -> String {
            self.a.clone()
        }
        fn set_center_view_a(&mut self, value: String) {
            self.a = value;
        }
        fn center_view_b(&self) -> String {
            self.b.clone()
        }
        fn set_center_view_b(&mut self, value: String) {
            self.b = value;
        }
        fn number_views(&self) -> String {
            self.number.clone()
        }
        fn set_number_views(&mut self, value: String) {
            self.number = value;
        }
        fn search_direction(&self) -> Option<i32> {
            self.direction
        }
        fn reset_search_direction(&mut self) {
            self.direction = None;
        }
        fn set_search_direction(&mut self, value: i32) {
            self.direction = Some(value);
        }
        fn mirror_xaxis(&self) -> MirrorInX {
            self.mirror.unwrap_or(MirrorInX::AssessBoth)
        }
        fn set_mirror_xaxis(&mut self, value: MirrorInX) {
            self.mirror = Some(value);
        }
    }

    #[derive(Default)]
    struct Manager {
        a: Param,
        b: Param,
        action: Option<String>,
        packed: Option<AxisID>,
    }
    impl TransferfidPanelApplicationManager<Param> for Manager {
        fn new_transferfid_param(&self, _: AxisID) -> Param {
            Param::default()
        }
        fn get_transferfid_a_fields(&self, p: &mut Param) {
            *p = self.a.clone();
        }
        fn get_transferfid_b_fields(&self, p: &mut Param) {
            *p = self.b.clone();
        }
        fn set_transferfid_a_fields(&mut self, p: &Param) {
            self.a = p.clone();
        }
        fn set_transferfid_b_fields(&mut self, p: &Param) {
            self.b = p.clone();
        }
        fn transferfid(
            &mut self,
            _: AxisID,
            _: &MultiLineButton,
            _: Option<&MultiLineButton>,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
        ) {
            self.action = Some("transferfid".into());
        }
        fn imod_seed_model(
            &mut self,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: &MultiLineButton,
            _: &str,
            _: String,
            _: String,
            _: DialogType,
        ) {
            self.action = Some("seed".into());
        }
        fn seed_file_name(&self, _: AxisID) -> String {
            "seed.mod".into()
        }
        fn raw_tilt_file(&self, _: AxisID) -> String {
            "raw.st".into()
        }
        fn pack(&mut self, axis_id: AxisID) {
            self.packed = Some(axis_id);
        }
    }

    #[test]
    fn parameters_follow_source_direction_and_metadata_order() {
        let mut manager = Manager {
            b: Param {
                run: true,
                a: "11".into(),
                b: "12".into(),
                number: "5".into(),
                direction: Some(-1),
                mirror: Some(MirrorInX::Never),
            },
            ..Default::default()
        };
        let mut panel = TransferfidPanel::get_instance(AxisID::Second, DialogType::FiducialModel);
        panel.set_parameters(&manager);
        assert!(panel.rb_search_minus_90.is_selected());
        assert!(panel.rb_mirror_in_x_never.is_selected());
        panel.rb_search_plus_90.set_selected(true);
        panel.rb_mirror_in_x_always.set_selected(true);
        assert!(panel.get_parameters(&mut manager, true));
        assert_eq!(manager.b.direction, Some(1));
        assert_eq!(manager.b.mirror, Some(MirrorInX::Always));
    }

    #[test]
    fn actions_expand_and_advanced_visibility_keep_java_routing() {
        let mut panel = TransferfidPanel::get_instance(AxisID::First, DialogType::FiducialModel);
        let mut manager = Manager::default();
        let command = panel
            .button_transferfid
            .get_action_command()
            .unwrap()
            .to_owned();
        panel.action(&mut manager, &command, None, None);
        assert_eq!(manager.action.as_deref(), Some("transferfid"));
        let advanced = panel.header.btn_advanced_basic.as_ref().unwrap().clone();
        panel.expand(&mut manager, &advanced);
        assert!(!panel.ltf_center_view_a.is_visible());
        assert_eq!(manager.packed, Some(AxisID::First));
    }
}
