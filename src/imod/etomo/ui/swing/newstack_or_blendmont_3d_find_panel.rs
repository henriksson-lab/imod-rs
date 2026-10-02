//! `IMOD/Etomo/src/etomo/ui/swing/NewstackOrBlendmont3dFindPanel.java`.
//!
//! Java `abstract class NewstackOrBlendmont3dFindPanel implements
//! Run3dmodButtonContainer`: the binning / "View Full Aligned Stack" part of
//! the erase-gold "3d find" stack panels.  Following `ui.md`, this struct is
//! the superclass part; `Newstack3dFindPanel` and `Blendmont3dFindPanel`
//! embed it as `base` (reached through `Deref`) and implement
//! [`NewstackOrBlendmont3dFindPanelVirtual`] for the abstract members.  The
//! subclass passes its own `Weak` (Java `this`) to [`NewstackOrBlendmont3dFindPanel::new`],
//! which is used as the `Run3dmodButtonContainer` of `btn3dmodFull` and by the
//! action listener (Java `adaptee.action(...)` dispatches to the override).

use std::rc::{Rc, Weak};

use super::labeled_spinner::LabeledSpinner;
use super::newstack_and_blendmont_param_panel::BINNING_LABEL;
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;
use super::run_3dmod_button::Run3dmodButton;
use super::run_3dmod_button_container::Run3dmodButtonContainer;
use super::ui_harness;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::jdk::{ActionEvent, ActionListener, JComponent};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::process_series::ProcessSeriesHandle;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::const_meta_data::ConstMetaData;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::etomo_number::EtomoNumber;
use crate::imod::etomo::r#type::meta_data::MetaData;
use crate::imod::etomo::r#type::process_result_display::ProcessResultDisplayHandle;
use crate::imod::etomo::util::utilities;

/// The abstract members of Java `NewstackOrBlendmont3dFindPanel`:
/// `runProcess(ProcessResultDisplay, ProcessSeries, Run3dmodMenuOptions)` and
/// `action(String, Deferred3dmodButton, Run3dmodMenuOptions)` (the latter is
/// `Run3dmodButtonContainer::action`, hence the supertrait).
pub trait NewstackOrBlendmont3dFindPanelVirtual: Run3dmodButtonContainer {
    /// Java `abstract void runProcess(ProcessResultDisplay, ProcessSeries,
    /// Run3dmodMenuOptions)`.
    fn run_process(
        &self,
        process_result_display: Option<ProcessResultDisplayHandle>,
        process_series: Option<ProcessSeriesHandle>,
        run_3dmod_menu_options: Option<Run3dmodMenuOptions>,
    );
}

/// Java `abstract class NewstackOrBlendmont3dFindPanel`.
pub struct NewstackOrBlendmont3dFindPanel {
    /// Java private final `pnlRoot = new JPanel()`.
    pnl_root: Rc<JComponent>,
    /// Java private final `actionListener`
    /// (`NewstackOrBlendmont3dFindPanelActionListener`).
    action_listener: ActionListener,
    /// Java private final `spinBinning`.
    spin_binning: Rc<LabeledSpinner>,
    /// Java private final `btn3dmodFull`.
    btn_3dmod_full: Rc<Run3dmodButton>,

    /// Java private final `parent`.
    parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
    /// Java package-private final `axisID`.
    pub axis_id: AxisID,
    /// Java package-private final `manager`.
    pub manager: &'static ApplicationManager,
    /// Java package-private final `dialogType`.
    pub dialog_type: DialogType,
}

impl NewstackOrBlendmont3dFindPanel {
    /// Java constructor `NewstackOrBlendmont3dFindPanel(ApplicationManager,
    /// AxisID, DialogType, NewstackOrBlendmont3dFindParent)`.  `this` is the
    /// subclass being built (Java `this`, used by the field initializers).
    pub fn new(
        manager: &'static ApplicationManager,
        axis_id: AxisID,
        dialog_type: DialogType,
        parent: Weak<dyn NewstackOrBlendmont3dFindParent>,
        this: Weak<dyn NewstackOrBlendmont3dFindPanelVirtual>,
    ) -> NewstackOrBlendmont3dFindPanel {
        // Field initializers.
        let pnl_root = JComponent::new_panel();
        // NewstackOrBlendmont3dFindPanelActionListener: actionPerformed calls
        // adaptee.action(event.getActionCommand(), null, null).
        let adaptee = this.clone();
        let action_listener: ActionListener = Rc::new(move |event: &ActionEvent| {
            let Some(adaptee) = adaptee.upgrade() else {
                return;
            };
            adaptee.action(event.get_action_command().unwrap_or(""), None, None);
        });
        let spin_binning = LabeledSpinner::get_instance_string_int_int_int_int(
            Some(&format!("{BINNING_LABEL}: ")),
            1,
            1,
            12,
            1,
        );
        let container: Weak<dyn Run3dmodButtonContainer> = this;
        let btn_3dmod_full = Run3dmodButton::get_3dmod_instance_string_run_3dmod_button_container(
            Some("View Full Aligned Stack"),
            Some(container),
        );
        NewstackOrBlendmont3dFindPanel {
            pnl_root,
            action_listener,
            spin_binning,
            btn_3dmod_full,
            parent,
            axis_id,
            manager,
            dialog_type,
        }
    }

    /// Java final `addListeners()`.
    pub fn add_listeners(&self) {
        self.btn_3dmod_full
            .add_action_listener(self.action_listener.clone());
    }

    /// Java final `getComponent()`.
    pub fn get_component(&self) -> Rc<JComponent> {
        self.pnl_root.clone()
    }

    /// Java final `createPanel()`.
    pub fn create_panel(&self) {
        // Initialize
        self.pnl_root.add(&self.spin_binning.get_container());
    }

    /// Java final `get3dmodButton()`.
    pub fn get3dmod_button(&self) -> Rc<JComponent> {
        self.btn_3dmod_full.get_component()
    }

    /// Java final `get3dmodFullButtonActionCommand()`.
    pub fn get3dmod_full_button_action_command(&self) -> Option<String> {
        self.btn_3dmod_full.get_action_command()
    }

    /// Java final `getBinning()`.
    pub fn get_binning(&self) -> i32 {
        self.spin_binning.get_value().int_value()
    }

    /// Java public `isFiducialess()`.  The parent owns this panel and outlives
    /// it; a dropped parent reads as not fiducialess.
    pub fn is_fiducialess(&self) -> bool {
        self.parent
            .upgrade()
            .is_some_and(|parent| parent.is_fiducialess())
    }

    /// Java final `setBinning(int)`.
    pub fn set_binning(&self, input: i32) {
        self.spin_binning.set_value_int(input);
    }

    /// Java final `getParameters(MetaData)`.  The Metadata values that are
    /// from the setup dialog should not be overrided by this dialog unless the
    /// Metadata values are empty.  Must save data from the two instances under
    /// separate keys.
    pub fn get_parameters(&self, meta_data: &MetaData) {
        meta_data
            .set_stack_3d_find_binning_int(self.axis_id, self.spin_binning.get_value().int_value());
    }

    /// Java final `setParameters(ConstMetaData)`.
    pub fn set_parameters(&self, meta_data: &dyn ConstMetaData) {
        if meta_data.is_stack_3d_find_binning_set(self.axis_id) {
            self.spin_binning
                .set_value_int(meta_data.get_stack_3d_find_binning(self.axis_id));
        }
    }

    /// Java `initialize()`.
    pub fn initialize(&self) {
        let mut bead_size = EtomoNumber::new_with_type(Some(Type::Double));
        bead_size.set_double(self.manager.calc_unbinned_bead_diameter_pixels());
        if !bead_size.is_null() && bead_size.is_valid() {
            // (int) Math.round((beadSize.getDouble() / 5f)): double / float is
            // a double division; Math.round(double) is a long.
            let mut binning = std::cmp::max(
                utilities::java_lang_math_round(bead_size.get_double() / 5.0) as i32,
                1,
            );
            // Adjust the binning if the bead size is too small.
            if binning > 1 && bead_size.get_double() / (binning as f64) < 4.0 {
                binning -= 1;
            }
            binning = std::cmp::min(binning, 12);
            self.spin_binning.set_value_int(binning);
        }
    }

    /// Java public final `validate()` (the `NewstackDisplay` /
    /// `BlendmontDisplay` `validate()` of the subclasses).
    pub fn validate(&self) -> bool {
        let binning = self.spin_binning.get_value().int_value();
        // Warn if the pixel size is too small
        if binning > 1 {
            let mut bead_size = EtomoNumber::new_with_type(Some(Type::Double));
            // Java dereferences `parent` unconditionally; the parent owns this
            // panel, so it is alive whenever the panel is used.
            let parent_bead_size = self.parent.upgrade().map(|parent| parent.get_bead_size());
            bead_size.set_string(parent_bead_size.as_deref());
            if !bead_size.is_null()
                && bead_size.is_valid()
                && bead_size.get_double() / (binning as f64) < 4.0
            {
                let manager: &'static dyn BaseManager = self.manager;
                let axis_id = self.axis_id;
                if !ui_harness::with(|harness| {
                    harness.open_yes_no_warning_dialog(
                        Some(manager),
                        &("The binned fiducial diameter will be less then 4 pixels.  Do you "
                            .to_string()
                            + "want to continue?"),
                        Some(axis_id),
                    )
                }) {
                    return false;
                }
            }
        }
        true
    }

    /// Java `setToolTipText()`.
    pub fn set_tool_tip_text(&self) {
        self.spin_binning.set_tool_tip_text(Some(
            &("Set the binning for the aligned image stack and ".to_string()
                + "tomogram to use with findbeads3d."),
        ));
        self.btn_3dmod_full
            .set_tool_tip_text(Some("Open the complete aligned stack in 3dmod"));
    }
}
