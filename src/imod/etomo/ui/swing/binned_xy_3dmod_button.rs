//! `IMOD/Etomo/src/etomo/ui/swing/BinnedXY3dmodButton.java`.
//!
//! A 3dmod button with a spinner choosing the X/Y binning to open with.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::imod::etomo::jdk::{ActionListener, JComponent};

use super::deferred_3dmod_button::Deferred3dmodButton;
use super::labeled_spinner::LabeledSpinner;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::spaced_panel::SpacedPanel;
use super::tooltip_formatter::TooltipFormatter;

/// Java package-private final `BinnedXY3dmodButton`.
pub struct BinnedXY3dmodButton {
    /// Java final `button`.
    button: Rc<Run3dmodButton>,
    /// Java final `spBinningXY`.
    sp_binning_xy: Rc<LabeledSpinner>,
    /// Java final `label`.
    label: Rc<JComponent>,
    /// Java `panel`.
    panel: RefCell<Option<Rc<JComponent>>>,
}

impl BinnedXY3dmodButton {
    /// Java public static final `rcsid`.
    pub const RCSID: &'static str = "$Id$";

    /// Java `BinnedXY3dmodButton(String, Run3dmodButtonContainer)`.
    pub fn new(
        label: Option<&str>,
        container: Option<Weak<dyn Run3dmodButtonContainer>>,
    ) -> Rc<BinnedXY3dmodButton> {
        let sp_binning_xy =
            LabeledSpinner::get_instance_string_int_int_int_int(Some("Open binned by "), 1, 1, 50, 1);
        let label_component = JComponent::new_label(" in X and Y");
        let button =
            Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(label, container);
        Rc::new(BinnedXY3dmodButton {
            button,
            sp_binning_xy,
            label: label_component,
            panel: RefCell::new(None),
        })
    }

    /// Java `getContainer()`.
    pub fn get_container(&self) -> Rc<JComponent> {
        if self.panel.borrow().is_none() {
            let panel = JComponent::new_panel();
            // Swing layout: X_AXIS BoxLayout, CENTER_ALIGNMENT, three
            // horizontal glues before and after the bordered panel.
            let border_panel = SpacedPanel::get_instance_void();
            // Swing layout: borderPanel Y_AXIS box layout, etched border,
            // CENTER_ALIGNMENT.
            let spinner_panel = JComponent::new_panel();
            // Swing layout: spinnerPanel X_AXIS BoxLayout, CENTER_ALIGNMENT.
            spinner_panel.add(&self.sp_binning_xy.get_container());
            spinner_panel.add(&self.label);
            border_panel.add_j_panel(&spinner_panel);
            let pnl_buttons = JComponent::new_panel();
            // Swing layout: pnlButtons X_AXIS BoxLayout, CENTER_ALIGNMENT.
            pnl_buttons.add(&self.button.get_component());
            border_panel.add_j_panel(&pnl_buttons);
            panel.add(&border_panel.get_container());
            *self.panel.borrow_mut() = Some(panel);
        }
        self.panel.borrow().clone().unwrap()
    }

    /// Java `getButton()`.
    pub fn get_button(&self) -> Rc<dyn Deferred3dmodButton> {
        self.button.clone()
    }

    /// Java `setSpinnerToolTipText(String)`.
    pub fn set_spinner_tool_tip_text(&self, text: Option<&str>) {
        self.sp_binning_xy.set_tool_tip_text(text);
        self.label
            .set_tool_tip_text(super::tooltip_formatter::INSTANCE.format(text).as_deref());
    }

    /// Java `setButtonToolTipText(String)`.
    pub fn set_button_tool_tip_text(&self, text: Option<&str>) {
        self.button.set_tool_tip_text(text);
    }

    /// Java `addActionListener(ActionListener)`.
    pub fn add_action_listener(&self, action_listener: ActionListener) {
        self.button.add_action_listener(action_listener);
    }

    /// Java `getActionCommand()`.
    pub fn get_action_command(&self) -> Option<String> {
        self.button.get_action_command()
    }

    /// Java `getBinningInXandY()`: `((Integer) spBinningXY.getValue()).intValue()`.
    /// The spinner is an integer spinner, so the value is integral.
    pub fn get_binning_in_xand_y(&self) -> i32 {
        self.sp_binning_xy.get_value().int_value()
    }

    /// Java `setEnabled(boolean)`.
    pub fn set_enabled(&self, enabled: bool) {
        self.sp_binning_xy.set_enabled(enabled);
        self.label.set_enabled(enabled);
        self.button.set_enabled(enabled);
    }
}
