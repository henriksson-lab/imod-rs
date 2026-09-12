//! `IMOD/Etomo/src/etomo/ui/swing/NewstackPanel.java`.
//!
//! Java inheritance is represented by the owned `newstack_or_blendmont_panel`
//! field.  The manager remains a direct source boundary: this unit preserves
//! both mutually-exclusive calls and every source argument.
#![allow(dead_code)]

use super::multi_line_button::MultiLineButton;
use super::newstack_or_blendmont_panel::{
    FiducialessParams, GlobalExpandButton, NewstackOrBlendmontPanel,
};
use super::tilt_panel::Deferred3dmodButton;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;

pub const HEADER_TITLE: &str = "Newstack";

/// Direct `ApplicationManager` methods called from Java `NewstackPanel`.
pub trait NewstackPanelApplicationManager {
    fn newst(
        &mut self,
        process_result_display: &MultiLineButton,
        process_series: Option<()>,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
        fiducialess_params: &FiducialessParams,
        display: &NewstackPanel,
        process_name: ProcessName,
    );

    fn imod_fine_align(
        &mut self,
        axis_id: AxisID,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// Java final `NewstackPanel`.
#[derive(Clone, Debug, PartialEq)]
pub struct NewstackPanel {
    pub newstack_or_blendmont_panel: NewstackOrBlendmontPanel,
}

impl NewstackPanel {
    /// Java private constructor `NewstackPanel(ApplicationManager, AxisID,
    /// DialogType, GlobalExpandButton)`.
    pub fn new(
        axis_id: AxisID,
        dialog_type: DialogType,
        _global_advanced_button: &GlobalExpandButton,
    ) -> Self {
        Self {
            newstack_or_blendmont_panel: NewstackOrBlendmontPanel::new(
                axis_id,
                dialog_type,
                HEADER_TITLE,
            ),
        }
    }

    /// Java static `getInstance`.
    pub fn get_instance(
        axis_id: AxisID,
        dialog_type: DialogType,
        global_advanced_button: &GlobalExpandButton,
    ) -> Self {
        let mut instance = Self::new(axis_id, dialog_type, global_advanced_button);
        instance.newstack_or_blendmont_panel.create_panel();
        instance.newstack_or_blendmont_panel.add_listeners();
        instance.newstack_or_blendmont_panel.set_tool_tip_text();
        instance
    }

    /// Java override `getHeaderTitle`.
    pub fn get_header_title(&self) -> &'static str {
        HEADER_TITLE
    }

    /// Java override `action(String, Deferred3dmodButton,
    /// Run3dmodMenuOptions)`.
    pub fn action<M: NewstackPanelApplicationManager>(
        &self,
        manager: &mut M,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    ) {
        if command
            == self
                .newstack_or_blendmont_panel
                .get_run_process_button_action_command()
        {
            manager.newst(
                self.newstack_or_blendmont_panel
                    .get_run_process_result_display(),
                None,
                deferred_3dmod_button,
                self.newstack_or_blendmont_panel.axis_id,
                run_3dmod_menu_options,
                self.newstack_or_blendmont_panel.dialog_type,
                self.newstack_or_blendmont_panel.get_fiducialess_params(),
                self,
                ProcessName::NEWST,
            );
        } else if command
            == self
                .newstack_or_blendmont_panel
                .get_3dmod_full_button_action_command()
        {
            manager.imod_fine_align(
                self.newstack_or_blendmont_panel.axis_id,
                run_3dmod_menu_options,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Manager {
        newst: Option<(AxisID, DialogType, ProcessName, bool)>,
        fine_align: Option<(AxisID, Option<Run3dmodMenuOptions>)>,
    }

    impl NewstackPanelApplicationManager for Manager {
        fn newst(
            &mut self,
            _display: &MultiLineButton,
            _series: Option<()>,
            _deferred: Option<&Deferred3dmodButton>,
            axis_id: AxisID,
            _options: Option<Run3dmodMenuOptions>,
            dialog_type: DialogType,
            fiducialess: &FiducialessParams,
            _panel: &NewstackPanel,
            process_name: ProcessName,
        ) {
            self.newst = Some((axis_id, dialog_type, process_name, fiducialess.fiducialess));
        }

        fn imod_fine_align(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>) {
            self.fine_align = Some((axis_id, options));
        }
    }

    #[test]
    fn get_instance_follows_java_creation_order() {
        let panel = NewstackPanel::get_instance(
            AxisID::First,
            DialogType::FinalAlignedStack,
            &GlobalExpandButton::get_instance("Advanced", "Basic"),
        );
        let base = &panel.newstack_or_blendmont_panel;
        assert_eq!(panel.get_header_title(), HEADER_TITLE);
        assert!(base.pnl_root.header_added && base.pnl_root.body_added);
        assert!(base.deferred_3dmod_button_set);
        assert_eq!(base.btn_run_process.button.action_listener_count, 1);
        assert_eq!(base.btn_3dmod_full.button.action_listener_count, 1);
        assert_eq!(
            base.btn_run_process.button.tooltip.as_deref(),
            Some(super::super::newstack_or_blendmont_panel::CREATE_FULL_ALIGNED_STACK_TOOLTIP)
        );
    }

    #[test]
    fn action_routes_newst_and_fine_align_with_exact_source_arguments() {
        let mut panel = NewstackPanel::get_instance(
            AxisID::Second,
            DialogType::FinalAlignedStack,
            &GlobalExpandButton::get_instance("Advanced", "Basic"),
        );
        panel
            .newstack_or_blendmont_panel
            .set_fiducialess_alignment(true);
        let run = panel
            .newstack_or_blendmont_panel
            .get_run_process_button_action_command()
            .to_string();
        let full = panel
            .newstack_or_blendmont_panel
            .get_3dmod_full_button_action_command()
            .to_string();
        let options = Run3dmodMenuOptions {
            bin_by_2: true,
            ..Default::default()
        };
        let mut manager = Manager::default();
        panel.action(
            &mut manager,
            &run,
            Some(&Deferred3dmodButton),
            Some(options),
        );
        assert_eq!(
            manager.newst,
            Some((
                AxisID::Second,
                DialogType::FinalAlignedStack,
                ProcessName::NEWST,
                true
            ))
        );
        panel.action(&mut manager, &full, None, Some(options));
        assert_eq!(manager.fine_align, Some((AxisID::Second, Some(options))));
    }
}
