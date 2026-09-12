//! `IMOD/Etomo/src/etomo/ui/swing/Blendmont3dFindPanel.java`.
//!
//! The inherited `NewstackOrBlendmont3dFindPanel` Swing controls and the
//! `ApplicationManager` remain explicit boundaries.  This source unit owns no
//! extra parameter widgets: it sets the inherited binning, selects
//! `BLEND_3DFIND`, converts the saved output size, and routes its two manager
//! calls exactly as the Java class does.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

use super::beads3d_find_panel::{Deferred3dmodButton, ProcessResultDisplay, ProcessSeries};

/// Java `BlendmontParam.Mode.BLEND_3DFIND`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Blendmont3dFindMode {
    Blend3dFind,
}

/// Java `BlendmontParam` calls made by this source unit.
pub trait Blendmont3dFindParam {
    fn set_bin_by_factor(&mut self, binning: i32);
    fn set_mode(&mut self, mode: Blendmont3dFindMode);
    fn convert_to_starting_and_ending_x_and_y(
        &mut self,
        stack_user_size_to_output_in_x_and_y: &str,
        image_rotation: f64,
    );
}

/// The inherited panel values observed or changed by this subclass.  The
/// actual `JPanel`, `LabeledSpinner`, and `Run3dmodButton` are retained at the
/// GUI boundary until `NewstackOrBlendmont3dFindPanel.java` is translated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NewstackOrBlendmont3dFindPanelState {
    pub binning: i32,
    pub full_3dmod_action_command: String,
    pub panel_created: bool,
    pub listeners_added: bool,
    pub tool_tip_text_set: bool,
}

impl Default for NewstackOrBlendmont3dFindPanelState {
    fn default() -> Self {
        Self {
            binning: 1,
            full_3dmod_action_command: "View Full Aligned Stack".into(),
            panel_created: false,
            listeners_added: false,
            tool_tip_text_set: false,
        }
    }
}

/// Direct `ApplicationManager` calls made by `Blendmont3dFindPanel.java`.
pub trait Blendmont3dFindPanelApplicationManager {
    fn stack_user_size_to_output_in_x_and_y(&self, axis_id: AxisID) -> String;
    fn image_rotation(&self, axis_id: AxisID) -> f64;
    fn blend_3d_find(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        process_series: &ProcessSeries,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        display: &Blendmont3dFindPanel,
    );
    fn imod_fine_align_3d_find(
        &mut self,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// Java `BlendmontDisplay`, implemented by `Blendmont3dFindPanel`.
pub trait Blendmont3dFindDisplay {}

/// Java final `Blendmont3dFindPanel`, composed with the values of its direct
/// superclass boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Blendmont3dFindPanel {
    pub newstack_or_blendmont_3d_find_panel: NewstackOrBlendmont3dFindPanelState,
    pub axis_id: AxisID,
    pub dialog_type: DialogType,
}

impl Blendmont3dFindDisplay for Blendmont3dFindPanel {}

impl Blendmont3dFindPanel {
    /// Java private constructor `Blendmont3dFindPanel(ApplicationManager,
    /// AxisID, DialogType, NewstackOrBlendmont3dFindParent)`.  Manager and
    /// parent are call-time owner boundaries: this Java constructor only passes
    /// them to its superclass and stores the latter's values represented above.
    pub fn new(axis_id: AxisID, dialog_type: DialogType) -> Self {
        Self {
            newstack_or_blendmont_3d_find_panel: NewstackOrBlendmont3dFindPanelState::default(),
            axis_id,
            dialog_type,
        }
    }

    /// Java static `getInstance`.  The three inherited setup calls retain their
    /// Java order, following construction.
    pub fn get_instance(axis_id: AxisID, dialog_type: DialogType) -> Self {
        let mut instance = Self::new(axis_id, dialog_type);
        instance.newstack_or_blendmont_3d_find_panel.panel_created = true;
        instance.newstack_or_blendmont_3d_find_panel.listeners_added = true;
        instance
            .newstack_or_blendmont_3d_find_panel
            .tool_tip_text_set = true;
        instance
    }

    /// Java override `setParameters(BlendmontParam)`, intentionally empty.
    pub fn set_parameters<P: Blendmont3dFindParam>(&mut self, _param: &P) {}

    /// Java override `getParameters(BlendmontParam, boolean)`.  As in Java,
    /// `do_validation` is unused because the values originate in a process
    /// that has already run.
    pub fn get_parameters<M: Blendmont3dFindPanelApplicationManager, P: Blendmont3dFindParam>(
        &self,
        manager: &M,
        param: &mut P,
        _do_validation: bool,
    ) -> bool {
        param.set_bin_by_factor(self.newstack_or_blendmont_3d_find_panel.binning);
        param.set_mode(Blendmont3dFindMode::Blend3dFind);
        param.convert_to_starting_and_ending_x_and_y(
            &manager.stack_user_size_to_output_in_x_and_y(self.axis_id),
            manager.image_rotation(self.axis_id),
        );
        true
    }

    /// Java override `runProcess(ProcessResultDisplay, ProcessSeries,
    /// Run3dmodMenuOptions)`.
    pub fn run_process<M: Blendmont3dFindPanelApplicationManager>(
        &self,
        manager: &mut M,
        process_result_display: &ProcessResultDisplay,
        process_series: &ProcessSeries,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        manager.blend_3d_find(
            process_result_display,
            process_series,
            None,
            self.axis_id,
            run_3dmod_menu_options,
            self.dialog_type,
            self,
        );
    }

    /// Java override `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: Blendmont3dFindPanelApplicationManager>(
        &self,
        manager: &mut M,
        command: &str,
        _deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if command
            == self
                .newstack_or_blendmont_3d_find_panel
                .full_3dmod_action_command
        {
            manager.imod_fine_align_3d_find(self.axis_id, run_3dmod_menu_options);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Param {
        binning: Option<i32>,
        mode: Option<Blendmont3dFindMode>,
        conversion: Option<(String, f64)>,
    }

    impl Blendmont3dFindParam for Param {
        fn set_bin_by_factor(&mut self, binning: i32) {
            self.binning = Some(binning);
        }
        fn set_mode(&mut self, mode: Blendmont3dFindMode) {
            self.mode = Some(mode);
        }
        fn convert_to_starting_and_ending_x_and_y(&mut self, size: &str, rotation: f64) {
            self.conversion = Some((size.into(), rotation));
        }
    }

    #[derive(Default)]
    struct Manager {
        blend: Option<(AxisID, DialogType, Option<Run3dmodMenuOptions>)>,
        fine_align: Option<(AxisID, Option<Run3dmodMenuOptions>)>,
    }

    impl Blendmont3dFindPanelApplicationManager for Manager {
        fn stack_user_size_to_output_in_x_and_y(&self, _: AxisID) -> String {
            "1200,900".into()
        }
        fn image_rotation(&self, _: AxisID) -> f64 {
            17.25
        }
        fn blend_3d_find(
            &mut self,
            _: &ProcessResultDisplay,
            _: &ProcessSeries,
            deferred: Option<&Deferred3dmodButton>,
            axis: AxisID,
            options: Option<Run3dmodMenuOptions>,
            dialog: DialogType,
            _: &Blendmont3dFindPanel,
        ) {
            assert!(deferred.is_none());
            self.blend = Some((axis, dialog, options));
        }
        fn imod_fine_align_3d_find(&mut self, axis: AxisID, options: Option<Run3dmodMenuOptions>) {
            self.fine_align = Some((axis, options));
        }
    }

    #[test]
    fn get_instance_performs_inherited_setup_in_source_order() {
        let panel =
            Blendmont3dFindPanel::get_instance(AxisID::First, DialogType::TomogramGeneration);
        assert!(panel.newstack_or_blendmont_3d_find_panel.panel_created);
        assert!(panel.newstack_or_blendmont_3d_find_panel.listeners_added);
        assert!(panel.newstack_or_blendmont_3d_find_panel.tool_tip_text_set);
    }

    #[test]
    fn get_parameters_sets_bin_mode_and_converts_saved_output_size() {
        let mut panel =
            Blendmont3dFindPanel::get_instance(AxisID::Second, DialogType::TomogramGeneration);
        panel.newstack_or_blendmont_3d_find_panel.binning = 4;
        let mut param = Param::default();
        assert!(panel.get_parameters(&Manager::default(), &mut param, true));
        assert_eq!(param.binning, Some(4));
        assert_eq!(param.mode, Some(Blendmont3dFindMode::Blend3dFind));
        assert_eq!(param.conversion, Some(("1200,900".into(), 17.25)));
    }

    #[test]
    fn process_and_full_stack_action_preserve_source_manager_arguments() {
        let panel =
            Blendmont3dFindPanel::get_instance(AxisID::Only, DialogType::TomogramGeneration);
        let options = Run3dmodMenuOptions {
            bin_by_2: true,
            ..Default::default()
        };
        let mut manager = Manager::default();
        panel.run_process(
            &mut manager,
            &ProcessResultDisplay,
            &ProcessSeries::new(AxisID::Only, DialogType::TomogramGeneration),
            Some(options),
        );
        assert_eq!(
            manager.blend,
            Some((AxisID::Only, DialogType::TomogramGeneration, Some(options)))
        );
        panel.action(
            &mut manager,
            "View Full Aligned Stack",
            Some(&Deferred3dmodButton),
            Some(options),
        );
        assert_eq!(manager.fine_align, Some((AxisID::Only, Some(options))));
    }
}
