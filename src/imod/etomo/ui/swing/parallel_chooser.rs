//! `IMOD/Etomo/src/etomo/ui/swing/ParallelChooser.java`.
#![allow(dead_code)]

use super::etomo_panel::TitledBorder;
use super::multi_line_button::MultiLineButton;
use super::spaced_panel::{SpacedPanel, SpacedPanelChild, X_AXIS};
use crate::imod::etomo::parallel_manager::ParallelManager;

pub const GENERIC_PROCESS_LABEL: &str = "Generic Parallel Process";
pub const ANISOTROPIC_DIFFUSION_LABEL: &str = "Nonlinear Anisotropic Diffusion";

/// Java final `ParallelChooser`.
pub struct ParallelChooser {
    pub root_panel: SpacedPanel,
    pub btn_generic: MultiLineButton,
    pub btn_anisotropic_diffusion: MultiLineButton,
    pub manager: &'static ParallelManager,
}

impl ParallelChooser {
    /// Java private `ParallelChooser(ParallelManager)`.
    pub fn new(manager: &'static ParallelManager) -> Self {
        let mut root_panel = SpacedPanel::get_instance();
        root_panel.set_box_layout(X_AXIS);
        // Java passes the `BeveledBorder`'s titled Swing border directly to
        // this panel.  SpacedPanel's existing native boundary records the
        // source-observable title here; painting remains its Swing boundary.
        root_panel.set_titled_border(TitledBorder {
            title: "Choose a process".into(),
        });
        let btn_generic = MultiLineButton::new_with_label(Some(GENERIC_PROCESS_LABEL));
        let btn_anisotropic_diffusion =
            MultiLineButton::new_with_label(Some(ANISOTROPIC_DIFFUSION_LABEL));
        root_panel
            .panel_children
            .push(SpacedPanelChild::MultiLineButton(btn_generic.clone()));
        root_panel
            .panel_children
            .push(SpacedPanelChild::MultiLineButton(
                btn_anisotropic_diffusion.clone(),
            ));
        Self {
            root_panel,
            btn_generic,
            btn_anisotropic_diffusion,
            manager,
        }
    }

    /// Java `getInstance(ParallelManager)`.
    pub fn get_instance(manager: &'static ParallelManager) -> Self {
        let mut instance = Self::new(manager);
        instance.add_listeners();
        instance
    }

    /// Java `getContainer()`; the native panel remains the Swing boundary.
    pub fn get_container(&self) -> &super::spaced_panel::JPanel {
        self.root_panel.get_container()
    }

    /// Java private `addListeners()`.
    pub fn add_listeners(&mut self) {
        self.btn_generic.add_action_listener();
        self.btn_anisotropic_diffusion.add_action_listener();
    }

    /// Java private `action(ActionEvent)`.  Java compares action-command
    /// references; the two commands originate from the two final buttons, so
    /// command content selects the same stable source identity here.
    pub fn action(&mut self, command: Option<&str>) {
        if command == self.btn_generic.get_action_command() {
            self.root_panel.set_visible(false);
            self.manager.open_parallel_dialog();
        } else if command == self.btn_anisotropic_diffusion.get_action_command() {
            self.root_panel.set_visible(false);
            self.manager.open_anisotropic_diffusion_dialog();
        }
    }

    /// Java private inner `PCActionListener.actionPerformed(ActionEvent)`.
    pub fn action_performed(&mut self, command: Option<&str>) {
        self.action(command);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_adds_source_buttons_and_listeners() {
        let chooser = ParallelChooser::get_instance(ParallelManager::new());
        assert_eq!(chooser.root_panel.panel_layout_axis, Some(X_AXIS));
        assert_eq!(
            chooser.root_panel.titled_border.as_ref().unwrap().title,
            "Choose a process"
        );
        assert_eq!(chooser.btn_generic.button.action_listener_count, 1);
        assert_eq!(
            chooser
                .btn_anisotropic_diffusion
                .button
                .action_listener_count,
            1
        );
    }

    #[test]
    fn known_action_hides_chooser() {
        let mut chooser = ParallelChooser::get_instance(ParallelManager::new());
        chooser.action_performed(Some(GENERIC_PROCESS_LABEL));
        assert!(!chooser.root_panel.outer_panel.visible);
    }
}
