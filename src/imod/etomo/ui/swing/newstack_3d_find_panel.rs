//! `IMOD/Etomo/src/etomo/ui/swing/Newstack3dFindPanel.java`.
//!
//! The inherited `NewstackOrBlendmont3dFindPanel` widget state is represented
//! directly in this unit until that Java source unit is translated.  The
//! application manager, metadata/state, and com-script parameter objects stay
//! explicit boundaries: they own file-name lookup, persistent values, header
//! reads, and process launch respectively.
#![allow(dead_code)]

use super::beads3d_find_panel::{Deferred3dmodButton, ProcessResultDisplay, ProcessSeries};
use super::labeled_spinner::LabeledSpinner;
use super::newstack_or_blendmont_3d_find_parent::NewstackOrBlendmont3dFindParent;
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_name::ProcessName;

/// Java `NewstParam.Mode.FULL_ALIGNED_STACK`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NewstMode {
    FullAlignedStack,
}

/// Java `ConstNewstParam` is intentionally empty here: `setParameters` is
/// empty in the original source unit.
pub trait ConstNewstParam {}

/// Every `NewstParam` write performed by `Newstack3dFindPanel.getParameters`.
pub trait Newstack3dFindNewstParam: ConstNewstParam {
    fn set_command_mode(&mut self, mode: NewstMode);
    fn set_fiducialess_alignment(&mut self, value: bool);
    fn set_bin_by_factor(&mut self, value: i32);
    fn set_linear_interpolation(&mut self, value: bool);
    fn set_size_to_output_in_x_and_y(
        &mut self,
        size: Option<(i32, i32)>,
        binning: i32,
        image_rotation: f64,
    ) -> Result<(), String>;
    fn set_output_file(&mut self, file_name: String);
    fn set_process_name(&mut self, process_name: ProcessName);
}

/// Java `TomogramState` reads performed here.
pub trait Newstack3dFindTomogramState {
    fn is_stack_use_linear_interpolation(&self, axis_id: AxisID) -> bool;
    fn stack_user_size_to_output_in_x_and_y(&self, axis_id: AxisID) -> Option<(i32, i32)>;
}

/// Java `MetaData` reads performed here.
pub trait Newstack3dFindMetaData {
    fn is_fiducialess_alignment(&self, axis_id: AxisID) -> bool;
    fn image_rotation(&self, axis_id: AxisID) -> f64;
}

/// Direct `ApplicationManager` calls from the Java source unit.
pub trait Newstack3dFindApplicationManager {
    type MetaData: Newstack3dFindMetaData;
    type TomogramState: Newstack3dFindTomogramState;

    fn meta_data(&self) -> &Self::MetaData;
    fn state(&self) -> &Self::TomogramState;
    fn newst_or_blend_3d_find_output_file_name(&self, axis_id: AxisID) -> String;
    fn newst_3d_find(
        &mut self,
        process_result_display: &ProcessResultDisplay,
        process_series: &ProcessSeries,
        deferred_3dmod_button: Option<&Deferred3dmodButton>,
        axis_id: AxisID,
        options: Option<Run3dmodMenuOptions>,
        dialog_type: DialogType,
    );
    fn imod_fine_align_3d_find(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn calc_unbinned_bead_diameter_pixels(&self) -> Option<f64>;
    fn open_yes_no_warning_dialog(&mut self, message: &str, axis_id: AxisID) -> bool;
}

/// Source-visible state of the inherited `NewstackOrBlendmont3dFindPanel`.
#[derive(Clone, Debug)]
pub struct Newstack3dFindPanel<M, P> {
    pub spin_binning: LabeledSpinner,
    pub btn_3dmod_full_action_command: String,
    pub btn_3dmod_full_tooltip: Option<String>,
    pub pnl_root_contains_binning: bool,
    pub btn_3dmod_full_action_listener_count: usize,
    pub parent: P,
    pub axis_id: AxisID,
    pub manager: M,
    pub dialog_type: DialogType,
}

impl<M: Newstack3dFindApplicationManager, P: NewstackOrBlendmont3dFindParent>
    Newstack3dFindPanel<M, P>
{
    /// Java private constructor.
    pub fn new(manager: M, axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        Self {
            spin_binning: LabeledSpinner::get_instance("Binning: ", 1, 1, 12, 1),
            btn_3dmod_full_action_command: "View Full Aligned Stack".into(),
            btn_3dmod_full_tooltip: None,
            pnl_root_contains_binning: false,
            btn_3dmod_full_action_listener_count: 0,
            parent,
            axis_id,
            manager,
            dialog_type,
        }
    }

    /// Java static `getInstance` and its exact creation order.
    pub fn get_instance(manager: M, axis_id: AxisID, dialog_type: DialogType, parent: P) -> Self {
        let mut instance = Self::new(manager, axis_id, dialog_type, parent);
        instance.create_panel();
        instance.add_listeners();
        instance.set_tool_tip_text();
        instance
    }

    /// Inherited Java `createPanel`.
    pub fn create_panel(&mut self) {
        self.pnl_root_contains_binning = true;
    }
    /// Inherited Java `addListeners`.
    pub fn add_listeners(&mut self) {
        self.btn_3dmod_full_action_listener_count += 1;
    }
    /// Inherited Java `setToolTipText`.
    pub fn set_tool_tip_text(&mut self) {
        self.spin_binning.set_tool_tip_text(Some(
            "Set the binning for the aligned image stack and tomogram to use with findbeads3d.",
        ));
        self.btn_3dmod_full_tooltip = Some("Open the complete aligned stack in 3dmod".into());
    }
    /// Inherited Java `getBinning`.
    pub fn get_binning(&self) -> i32 {
        self.spin_binning.get_value()
    }
    /// Inherited Java `setBinning`.
    pub fn set_binning(&mut self, input: i32) {
        self.spin_binning.set_value_int(input);
    }
    /// Inherited Java `isFiducialess`.
    pub fn is_fiducialess(&self) -> bool {
        self.parent.is_fiducialess()
    }
    /// Inherited Java `validate`.
    pub fn validate(&mut self) -> bool {
        let binning = self.get_binning();
        if binning > 1 {
            if let Ok(bead_size) = self.parent.get_bead_size().parse::<f64>() {
                if bead_size / f64::from(binning) < 4.0 && !self.manager.open_yes_no_warning_dialog("The binned fiducial diameter will be less then 4 pixels.  Do you want to continue?", self.axis_id) {
                    return false;
                }
            }
        }
        true
    }
    /// Inherited Java `initialize`.
    pub fn initialize(&mut self) {
        if let Some(bead_size) = self.manager.calc_unbinned_bead_diameter_pixels() {
            let mut binning = (bead_size / 5.0).round() as i32;
            binning = binning.max(1);
            if binning > 1 && bead_size / f64::from(binning) < 4.0 {
                binning -= 1;
            }
            self.set_binning(binning.min(12));
        }
    }
    /// Java override `setParameters(ConstNewstParam)`, intentionally empty.
    pub fn set_parameters<T: ConstNewstParam>(&mut self, _param: &T) {}
    /// Java override `getParameters(NewstParam, boolean)`.
    pub fn get_parameters<T: Newstack3dFindNewstParam>(
        &self,
        param: &mut T,
        _do_validation: bool,
    ) -> Result<bool, String> {
        param.set_command_mode(NewstMode::FullAlignedStack);
        param.set_fiducialess_alignment(
            self.manager
                .meta_data()
                .is_fiducialess_alignment(self.axis_id),
        );
        let binning = self.get_binning();
        param.set_bin_by_factor(if binning > 1 { binning } else { i32::MIN });
        let state = self.manager.state();
        param.set_linear_interpolation(state.is_stack_use_linear_interpolation(self.axis_id));
        param.set_size_to_output_in_x_and_y(
            state.stack_user_size_to_output_in_x_and_y(self.axis_id),
            binning,
            self.manager.meta_data().image_rotation(self.axis_id),
        )?;
        param.set_output_file(
            self.manager
                .newst_or_blend_3d_find_output_file_name(self.axis_id),
        );
        param.set_process_name(ProcessName::NEWST_3D_FIND);
        Ok(true)
    }
    /// Java override `runProcess`.
    pub fn run_process(
        &mut self,
        display: &ProcessResultDisplay,
        series: &ProcessSeries,
        options: Option<Run3dmodMenuOptions>,
    ) {
        self.manager.newst_3d_find(
            display,
            series,
            None,
            self.axis_id,
            options,
            self.dialog_type,
        );
    }
    /// Java override `action`.
    pub fn action(
        &mut self,
        command: &str,
        _deferred: Option<&Deferred3dmodButton>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if command == self.btn_3dmod_full_action_command {
            self.manager.imod_fine_align_3d_find(self.axis_id, options);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Parent;
    impl NewstackOrBlendmont3dFindParent for Parent {
        fn get_bead_size(&self) -> String {
            "8".into()
        }
        fn is_fiducialess(&self) -> bool {
            false
        }
    }
    #[derive(Default)]
    struct Meta;
    impl Newstack3dFindMetaData for Meta {
        fn is_fiducialess_alignment(&self, _: AxisID) -> bool {
            true
        }
        fn image_rotation(&self, _: AxisID) -> f64 {
            12.5
        }
    }
    #[derive(Default)]
    struct State;
    impl Newstack3dFindTomogramState for State {
        fn is_stack_use_linear_interpolation(&self, _: AxisID) -> bool {
            true
        }
        fn stack_user_size_to_output_in_x_and_y(&self, _: AxisID) -> Option<(i32, i32)> {
            Some((100, 200))
        }
    }
    #[derive(Default)]
    struct Manager {
        meta: Meta,
        state: State,
        run: bool,
        view: bool,
    }
    impl Newstack3dFindApplicationManager for Manager {
        type MetaData = Meta;
        type TomogramState = State;
        fn meta_data(&self) -> &Meta {
            &self.meta
        }
        fn state(&self) -> &State {
            &self.state
        }
        fn newst_or_blend_3d_find_output_file_name(&self, _: AxisID) -> String {
            "x_3dfind.ali".into()
        }
        fn newst_3d_find(
            &mut self,
            _: &ProcessResultDisplay,
            _: &ProcessSeries,
            _: Option<&Deferred3dmodButton>,
            _: AxisID,
            _: Option<Run3dmodMenuOptions>,
            _: DialogType,
        ) {
            self.run = true
        }
        fn imod_fine_align_3d_find(&mut self, _: AxisID, _: Option<Run3dmodMenuOptions>) {
            self.view = true
        }
        fn calc_unbinned_bead_diameter_pixels(&self) -> Option<f64> {
            Some(26.0)
        }
        fn open_yes_no_warning_dialog(&mut self, _: &str, _: AxisID) -> bool {
            true
        }
    }
    #[derive(Default)]
    struct Param {
        bin: i32,
        fid: bool,
        interpolation: bool,
        size: Option<(i32, i32)>,
        rotation: f64,
        file: String,
        process: Option<ProcessName>,
    }
    impl ConstNewstParam for Param {}
    impl Newstack3dFindNewstParam for Param {
        fn set_command_mode(&mut self, _: NewstMode) {}
        fn set_fiducialess_alignment(&mut self, v: bool) {
            self.fid = v
        }
        fn set_bin_by_factor(&mut self, v: i32) {
            self.bin = v
        }
        fn set_linear_interpolation(&mut self, v: bool) {
            self.interpolation = v
        }
        fn set_size_to_output_in_x_and_y(
            &mut self,
            s: Option<(i32, i32)>,
            _: i32,
            r: f64,
        ) -> Result<(), String> {
            self.size = s;
            self.rotation = r;
            Ok(())
        }
        fn set_output_file(&mut self, v: String) {
            self.file = v
        }
        fn set_process_name(&mut self, v: ProcessName) {
            self.process = Some(v)
        }
    }
    #[test]
    fn parameters_preserve_full_aligned_stack_rules() {
        let mut p = Newstack3dFindPanel::get_instance(
            Manager::default(),
            AxisID::First,
            DialogType::FinalAlignedStack,
            Parent,
        );
        p.set_binning(1);
        let mut out = Param::default();
        assert_eq!(p.get_parameters(&mut out, true), Ok(true));
        assert_eq!(out.bin, i32::MIN);
        assert!(out.fid && out.interpolation);
        assert_eq!(out.size, Some((100, 200)));
        assert_eq!(out.rotation, 12.5);
        assert_eq!(out.file, "x_3dfind.ali");
        assert_eq!(out.process, Some(ProcessName::NEWST_3D_FIND));
    }
    #[test]
    fn factory_and_actions_keep_base_creation_and_manager_routes() {
        let mut p = Newstack3dFindPanel::get_instance(
            Manager::default(),
            AxisID::First,
            DialogType::FinalAlignedStack,
            Parent,
        );
        assert!(p.pnl_root_contains_binning);
        assert_eq!(p.btn_3dmod_full_action_listener_count, 1);
        assert!(p.btn_3dmod_full_tooltip.is_some());
        p.initialize();
        assert_eq!(p.get_binning(), 5);
        p.run_process(
            &ProcessResultDisplay,
            &ProcessSeries::new(AxisID::First, DialogType::FinalAlignedStack),
            None,
        );
        assert!(p.manager.run);
        p.action("View Full Aligned Stack", None, None);
        assert!(p.manager.view);
    }
}
