//! `IMOD/Etomo/src/etomo/ui/swing/ParallelChooser.java`.
//!
//! The `ParallelManager`'s process chooser, shown when a parallel manager is
//! opened with no dialog type: "Generic Parallel Process" or "Nonlinear
//! Anisotropic Diffusion".  An event dispatch thread object.

use std::rc::{Rc, Weak};

use super::beveled_border::BeveledBorder;
use super::multi_line_button::MultiLineButton;
use super::spaced_panel::{self, SpacedPanel};
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::parallel_manager::ParallelManager;

/// Java `public static final String rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `public final class ParallelChooser`.
pub struct ParallelChooser {
    /// Java private final `rootPanel = SpacedPanel.getInstance()`.
    root_panel: Rc<SpacedPanel>,
    /// Java private final `btnGeneric`.
    btn_generic: Rc<MultiLineButton>,
    /// Java private final `btnAnisotropicDiffusion`.
    btn_anisotropic_diffusion: Rc<MultiLineButton>,
    /// Java private final `manager`.
    manager: &'static ParallelManager,
}

impl ParallelChooser {
    /// Java private `ParallelChooser(ParallelManager)`.
    fn new(manager: &'static ParallelManager) -> Rc<ParallelChooser> {
        let instance = Rc::new(ParallelChooser {
            root_panel: SpacedPanel::get_instance_void(),
            btn_generic: MultiLineButton::new_string(Some("Generic Parallel Process")),
            btn_anisotropic_diffusion: MultiLineButton::new_string(Some(
                "Nonlinear Anisotropic Diffusion",
            )),
            manager,
        });
        instance.root_panel.set_box_layout(spaced_panel::X_AXIS);
        instance
            .root_panel
            .set_border(&BeveledBorder::new(Some("Choose a process")).get_border());
        instance
            .root_panel
            .add_multi_line_button(&instance.btn_generic);
        instance
            .root_panel
            .add_multi_line_button(&instance.btn_anisotropic_diffusion);
        instance
    }

    /// Java static `getInstance(ParallelManager)`.
    pub fn get_instance(manager: &'static ParallelManager) -> Rc<ParallelChooser> {
        let instance = ParallelChooser::new(manager);
        instance.add_listeners(&instance);
        instance
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        self.root_panel.get_container()
    }

    /// Java private `addListeners()`.
    fn add_listeners(&self, this: &Rc<ParallelChooser>) {
        // new PCActionListener(this)
        let adaptee: Weak<ParallelChooser> = Rc::downgrade(this);
        let listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            if let Some(adaptee) = adaptee.upgrade() {
                adaptee.action(event);
            }
        });
        self.btn_generic.add_action_listener(listener.clone());
        self.btn_anisotropic_diffusion.add_action_listener(listener);
    }

    /// Java private `action(ActionEvent)`.
    fn action(&self, event: &ActionEvent) {
        let command = event.get_action_command();
        if command == self.btn_generic.get_action_command().as_deref() {
            self.root_panel.set_visible(false);
            self.manager.open_parallel_dialog();
        } else if command
            == self
                .btn_anisotropic_diffusion
                .get_action_command()
                .as_deref()
        {
            self.root_panel.set_visible(false);
            self.manager.open_anisotropic_diffusion_dialog();
        }
    }
}
