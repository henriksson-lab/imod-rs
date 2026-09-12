//! `IMOD/Etomo/src/etomo/ui/swing/NewstackOrBlendmont3dFindPanel.java`.
//!
//! The native widget implementation and `ApplicationManager` are direct GUI and
//! application boundaries.  This abstract base keeps the source-owned root
//! layout, spinner, 3dmod button, metadata transfer, initialization, validation,
//! and listener dispatch.  The two source subclasses own `runProcess` and
//! `action`, expressed by the explicit abstract boundary trait below.
#![allow(dead_code)]

use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;

use super::beads3d_find_panel::{Deferred3dmodButton, ProcessResultDisplay, ProcessSeries};
use super::find_beads3d_panel::NewstackOrBlendmont3dFindParent;
use super::labeled_spinner::LabeledSpinner;
use super::newstack_or_blendmont_panel::MetaData;

pub const BINNING_LABEL: &str = "Binning";
pub const VIEW_FULL_ALIGNED_STACK_LABEL: &str = "View Full Aligned Stack";
pub const BINNING_TOOLTIP: &str =
    "Set the binning for the aligned image stack and tomogram to use with findbeads3d.";
pub const VIEW_FULL_ALIGNED_STACK_TOOLTIP: &str = "Open the complete aligned stack in 3dmod";
pub const SMALL_BINNED_FIDUCIAL_WARNING: &str =
    "The binned fiducial diameter will be less then 4 pixels.  Do you want to continue?";

/// Java `JPanel pnlRoot` state.  Native Swing construction remains a GUI boundary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NewstackOrBlendmont3dFindPanelRoot {
    pub component_order: Vec<String>,
}

/// Java `Run3dmodButton btn3dmodFull` state at the GUI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run3dmodButton {
    pub label: String,
    pub action_command: String,
    pub action_listener_count: usize,
    pub tooltip: Option<String>,
}

impl Run3dmodButton {
    /// Java `Run3dmodButton.get3dmodInstance`.
    pub fn get_3dmod_instance(label: &str) -> Self {
        Self {
            label: label.into(),
            action_command: label.into(),
            action_listener_count: 0,
            tooltip: None,
        }
    }

    /// Java `addActionListener`.
    pub fn add_action_listener(&mut self) {
        self.action_listener_count += 1;
    }

    /// Java `getComponent`; the native button is represented by its direct state.
    pub fn get_component(&self) -> Deferred3dmodButton {
        Deferred3dmodButton
    }

    /// Java `getActionCommand`.
    pub fn get_action_command(&self) -> &str {
        &self.action_command
    }

    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self, text: &str) {
        self.tooltip = Some(text.into());
    }
}

/// Direct `ApplicationManager` and `UIHarness` calls made by this source unit.
pub trait NewstackOrBlendmont3dFindPanelApplicationManager {
    fn calc_unbinned_bead_diameter_pixels(&self) -> String;
    fn open_yes_no_warning_dialog(&self, message: &str, axis_id: AxisID) -> bool;
}

/// Java abstract `runProcess` and `action` contracts supplied by
/// `Newstack3dFindPanel` and `Blendmont3dFindPanel`.
pub trait NewstackOrBlendmont3dFindPanelAction {
    fn run_process(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        process_series: &mut ProcessSeries,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );

    fn action(
        &mut self,
        command: &str,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// Java abstract `NewstackOrBlendmont3dFindPanel`.
pub struct NewstackOrBlendmont3dFindPanel<M, P> {
    pub pnl_root: NewstackOrBlendmont3dFindPanelRoot,
    pub action_listener_present: bool,
    pub spin_binning: LabeledSpinner,
    pub btn_3dmod_full: Run3dmodButton,
    pub parent: P,
    pub axis_id: AxisID,
    pub manager: M,
    pub dialog_type: DialogType,
}

impl<M, P> NewstackOrBlendmont3dFindPanel<M, P>
where
    M: NewstackOrBlendmont3dFindPanelApplicationManager,
    P: NewstackOrBlendmont3dFindParent,
{
    /// Java package-private constructor.
    pub fn new(manager: M, axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        Self {
            pnl_root: NewstackOrBlendmont3dFindPanelRoot::default(),
            action_listener_present: false,
            spin_binning: LabeledSpinner::get_instance(BINNING_LABEL, 1, 1, 12, 1),
            btn_3dmod_full: Run3dmodButton::get_3dmod_instance(VIEW_FULL_ALIGNED_STACK_LABEL),
            parent,
            axis_id,
            manager,
            dialog_type,
        }
    }

    /// Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_3dmod_full.add_action_listener();
        self.action_listener_present = true;
    }

    /// Java `getComponent`.
    pub fn get_component(&self) -> &NewstackOrBlendmont3dFindPanelRoot {
        &self.pnl_root
    }

    /// Java `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root.component_order.push("spinBinning".into());
    }

    /// Java `get3dmodButton`.
    pub fn get_3dmod_button(&self) -> Deferred3dmodButton {
        self.btn_3dmod_full.get_component()
    }

    /// Java `get3dmodFullButtonActionCommand`.
    pub fn get_3dmod_full_button_action_command(&self) -> &str {
        self.btn_3dmod_full.get_action_command()
    }

    /// Java `getBinning`.
    pub fn get_binning(&self) -> i32 {
        self.spin_binning.get_value()
    }

    /// Java `isFiducialess`.
    pub fn is_fiducialess(&self) -> bool {
        self.parent.is_fiducialess()
    }

    /// Java `setBinning`.
    pub fn set_binning(&mut self, input: i32) {
        self.spin_binning.set_value_int(input);
    }

    /// Java `getParameters(MetaData)`.
    pub fn get_parameters(&self, meta_data: &mut MetaData) {
        meta_data.set_stack_3d_find_binning(self.axis_id, self.spin_binning.get_value());
    }

    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters(&mut self, meta_data: &MetaData) {
        if meta_data.is_stack_3d_find_binning_set(self.axis_id) {
            self.spin_binning
                .set_value_int(meta_data.get_stack_3d_find_binning(self.axis_id));
        }
    }

    /// Java `initialize`.
    pub fn initialize(&mut self) {
        let bead_size = self
            .manager
            .calc_unbinned_bead_diameter_pixels()
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite());
        if let Some(bead_size) = bead_size {
            let mut binning = ((bead_size / 5.0).round() as i32).max(1);
            if binning > 1 && bead_size / f64::from(binning) < 4.0 {
                binning -= 1;
            }
            self.spin_binning.set_value_int(binning.min(12));
        }
    }

    /// Java `validate`.
    pub fn validate(&self) -> bool {
        let binning = self.spin_binning.get_value();
        if binning > 1 {
            let bead_size = self
                .parent
                .get_bead_size()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite());
            if bead_size.is_some_and(|value| value / f64::from(binning) < 4.0)
                && !self
                    .manager
                    .open_yes_no_warning_dialog(SMALL_BINNED_FIDUCIAL_WARNING, self.axis_id)
            {
                return false;
            }
        }
        true
    }

    /// Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.spin_binning.set_tool_tip_text(Some(BINNING_TOOLTIP));
        self.btn_3dmod_full
            .set_tool_tip_text(VIEW_FULL_ALIGNED_STACK_TOOLTIP);
    }

    /// Java nested `NewstackOrBlendmont3dFindPanelActionListener.actionPerformed`.
    pub fn action_performed<A: NewstackOrBlendmont3dFindPanelAction>(
        &self,
        adaptee: &mut A,
        action_command: &str,
    ) {
        adaptee.action(action_command, None, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Manager {
        diameter: String,
        allow: bool,
    }
    impl NewstackOrBlendmont3dFindPanelApplicationManager for Manager {
        fn calc_unbinned_bead_diameter_pixels(&self) -> String {
            self.diameter.clone()
        }
        fn open_yes_no_warning_dialog(&self, _: &str, _: AxisID) -> bool {
            self.allow
        }
    }
    #[derive(Clone)]
    struct Parent {
        bead_size: String,
        fiducialess: bool,
    }
    impl NewstackOrBlendmont3dFindParent for Parent {
        fn get_bead_size(&self) -> String {
            self.bead_size.clone()
        }
        fn is_fiducialess(&self) -> bool {
            self.fiducialess
        }
    }
    fn panel(
        diameter: &str,
        bead_size: &str,
        allow: bool,
    ) -> NewstackOrBlendmont3dFindPanel<Manager, Parent> {
        NewstackOrBlendmont3dFindPanel::new(
            Manager {
                diameter: diameter.into(),
                allow,
            },
            AxisID::First,
            DialogType::FinalAlignedStack,
            Parent {
                bead_size: bead_size.into(),
                fiducialess: true,
            },
        )
    }
    #[test]
    fn source_construction_layout_listener_and_tooltips_are_preserved() {
        let mut panel = panel("10", "10", true);
        panel.create_panel();
        panel.add_listeners();
        panel.set_tool_tip_text();
        assert_eq!(panel.pnl_root.component_order, ["spinBinning"]);
        assert_eq!(panel.btn_3dmod_full.action_listener_count, 1);
        assert_eq!(panel.spin_binning.tooltip.as_deref(), Some(BINNING_TOOLTIP));
        assert!(panel.is_fiducialess());
    }
    #[test]
    fn initialization_and_metadata_keep_source_binning_rules() {
        let mut panel = panel("24", "10", true);
        panel.initialize();
        assert_eq!(panel.get_binning(), 5);
        let mut meta_data = MetaData::default();
        panel.get_parameters(&mut meta_data);
        assert_eq!(meta_data.get_stack_3d_find_binning(AxisID::First), 5);
        panel.set_binning(1);
        panel.set_parameters(&meta_data);
        assert_eq!(panel.get_binning(), 5);
    }
    #[test]
    fn validation_only_warns_for_small_valid_binned_beads() {
        let mut panel = panel("10", "7", false);
        panel.set_binning(2);
        assert!(!panel.validate());
        panel.parent.bead_size = "not a number".into();
        assert!(panel.validate());
    }
}
