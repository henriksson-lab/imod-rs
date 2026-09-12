//! `IMOD/Etomo/src/etomo/ui/swing/Ctf3dPanel.java`.
//!
//! Swing construction, autodoc acquisition, and process invocation remain GUI
//! boundaries.  This source unit keeps the Java state, parameter paths,
//! validation, visibility rules, and action dispatch explicit.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use super::check_box::CheckBox;
use super::labeled_text_field::LabeledTextField;
use super::multi_line_button::MultiLineButton;
use super::radio_button::{RadioButton, RadioButtonGroup};
use crate::imod::etomo::process::imod_process::Run3dmodMenuOptions;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::ui::field_type::FieldType;

pub const RUN_BUTTON_LABEL: &str = "Generate CTF-corrected Tomogram";
pub const USE_BUTTON_LABEL: &str = "Use CTF-corrected Tomogram";
pub const SLAB_THICKNESS_IN_NM_LABEL: &str = "Thickness of slab for each CTF correction (nm): ";
pub const NUMBER_OF_SLABS_MIN: i64 = 2;

/// Java `Ctf3dSetupParam` calls used by this panel.
pub trait Ctf3dSetupParam {
    fn slab_thickness_in_nm(&self) -> Option<String>;
    fn set_slab_thickness_in_nm(&mut self, value: String) -> Result<(), String>;
    fn temporary_directory(&self) -> Option<String>;
    fn set_temporary_directory(&mut self, value: String) -> Result<(), String>;
    fn run_slabs_in_parallel(&self) -> bool;
    fn set_run_slabs_in_parallel(&mut self, value: bool);
    fn erase_fiducials(&self) -> bool;
    fn set_erase_fiducials(&mut self, value: bool);
    fn filter_in_2d(&self) -> bool;
    fn set_filter_in_2d(&mut self, value: bool);
    fn use_unaligned_images(&self) -> bool;
    fn set_use_unaligned_images(&mut self, value: bool);
    fn adjust_for_align_z_shift(&self) -> bool;
    fn set_adjust_for_align_z_shift(&mut self, value: bool);
    fn fourier_reduce_by_factor(&self) -> i32;
    fn set_fourier_reduce_by_factor(&mut self, value: i32);
    fn vertical_slices(&self) -> bool;
    fn set_vertical_slices(&mut self, value: bool);
    fn old_style_x_tilting(&self) -> bool;
    fn set_old_style_x_tilting(&mut self, value: bool);
}

/// Java `ConstMetaData`/`MetaData` calls made by overloaded `set/getParameters`.
pub trait Ctf3dMetaData {
    fn gen_ctf3d_fourier_reduce_by_factor(&self, axis_id: AxisID) -> i32;
    fn is_gen_ctf3d_vertical_slices(&self, axis_id: AxisID) -> bool;
    fn is_gen_ctf3d_old_style_x_tilting(&self, axis_id: AxisID) -> bool;
    fn set_gen_ctf3d_fourier_reduce_by_factor(&mut self, axis_id: AxisID, value: i32);
    fn set_gen_ctf3d_vertical_slices(&mut self, axis_id: AxisID, value: bool);
    fn set_gen_ctf3d_old_style_x_tilting(&mut self, axis_id: AxisID, value: bool);
}

/// Calls to `TomogramGenerationDialog` owned outside this source unit.
pub trait Ctf3dPanelParent {
    fn tomo_thickness(&self) -> Option<i64>;
    fn x_axis_tilt(&self) -> Option<String>;
    fn is_use_local_alignment(&self) -> bool;
    fn is_use_z_factors(&self) -> bool;
    fn is_ctf3d(&self) -> bool;
}

/// Direct manager calls dispatched by the Java panel.
pub trait Ctf3dPanelApplicationManager {
    fn ctf3d_setup(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn open_ctf3d(&mut self, axis_id: AxisID, options: Option<Run3dmodMenuOptions>);
    fn use_ctf3d(&mut self, axis_id: AxisID);
    fn open_message_dialog(&mut self, message: String, title: &str, axis_id: AxisID);
    fn pack(&mut self, axis_id: AxisID);
}

/// Java `ReconScreenState` button-state access used by this panel.
pub trait Ctf3dReconScreenState {
    fn get_button_state(&self, key: &str) -> bool;
    fn set_button_state(&mut self, key: &str, state: bool);
}

/// In-memory `ReconScreenState` boundary for native frontends and tests.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Ctf3dPanelScreenState {
    pub button_states: BTreeMap<String, bool>,
}
impl Ctf3dReconScreenState for Ctf3dPanelScreenState {
    fn get_button_state(&self, key: &str) -> bool {
        self.button_states.get(key).copied().unwrap_or(false)
    }
    fn set_button_state(&mut self, key: &str, state: bool) {
        self.button_states.insert(key.into(), state);
    }
}

/// Java `Spinner` state, retained at the widget dependency boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FourierReduceByFactorSpinner {
    pub value: i32,
    pub enabled: bool,
    pub tooltip: Option<String>,
}

/// Swing state directly set in Java `createPanel` and `updateDisplay`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ctf3dPanelLayout {
    pub root_visible: bool,
    pub x_axis_tilted_slices_visible: bool,
    pub x_axis_tilted_slices_enabled: bool,
    pub adjust_for_align_z_shift_visible: bool,
    pub erase_fiducials_label_enabled: bool,
    pub filter_in_2d_label_enabled: bool,
    pub listener_count: usize,
    pub tooltip_initialized: bool,
    pub number_of_slabs_label: String,
    pub component_order: Vec<String>,
}

/// Java `Ctf3dPanel` field state.
pub struct Ctf3dPanel {
    pub axis_id: AxisID,
    pub header_advanced: bool,
    pub layout: Ctf3dPanelLayout,
    pub tf_slab_thickness_in_nm: LabeledTextField,
    pub cb_run_slabs_in_parallel: CheckBox,
    pub cb_erase_fiducials: CheckBox,
    pub cb_filter_in_2d: CheckBox,
    pub cb_use_unaligned_images: CheckBox,
    pub sp_fourier_reduce_by_factor: FourierReduceByFactorSpinner,
    pub rb_x_axis_tilted_slices: RadioButton,
    pub rb_vertical_slices: RadioButton,
    pub rb_old_style_x_tilting: RadioButton,
    pub temporary_directory: LabeledTextField,
    pub btn_ctf3d_setup: MultiLineButton,
    pub btn_3dmod_ctf3d: MultiLineButton,
    pub btn_use_ctf3d: MultiLineButton,
    pub cb_adjust_for_align_z_shift: CheckBox,
    pub number_of_slabs: Option<i64>,
}

impl Ctf3dPanel {
    /// Java constructor and `getInstance` initialization sequence.
    pub fn get_instance(axis_id: AxisID) -> Self {
        let group = Rc::new(RefCell::new(RadioButtonGroup::new()));
        let mut result = Self {
            axis_id,
            header_advanced: false,
            layout: Ctf3dPanelLayout {
                root_visible: true,
                x_axis_tilted_slices_visible: false,
                x_axis_tilted_slices_enabled: false,
                adjust_for_align_z_shift_visible: false,
                erase_fiducials_label_enabled: false,
                filter_in_2d_label_enabled: false,
                listener_count: 9,
                tooltip_initialized: true,
                number_of_slabs_label: " (?) slabs) ".into(),
                component_order: vec![
                    "ctf3dSetupHeader".into(),
                    "ctf3dSetupBody".into(),
                    "slabThicknessInNm".into(),
                    "runSlabsInParallel".into(),
                    "eraseFiducials".into(),
                    "filterIn2D".into(),
                    "useUnalignedImages".into(),
                    "adjustForAlignZShift".into(),
                    "xAxisTiltedSlices".into(),
                    "temporaryDirectory".into(),
                    "buttons".into(),
                ],
            },
            tf_slab_thickness_in_nm: LabeledTextField::new(
                FieldType::Integer,
                SLAB_THICKNESS_IN_NM_LABEL,
            ),
            cb_run_slabs_in_parallel: CheckBox::new_with_text("Compute slabs in parallel"),
            cb_erase_fiducials: CheckBox::new_with_text("Erase gold"),
            cb_filter_in_2d: CheckBox::new_with_text("Apply 2D filter"),
            cb_use_unaligned_images: CheckBox::new_with_text("Reconstruct from raw images"),
            sp_fourier_reduce_by_factor: FourierReduceByFactorSpinner {
                value: 1,
                enabled: false,
                tooltip: None,
            },
            rb_x_axis_tilted_slices: RadioButton::new_in_group(
                "Let programs decide whether to interpolate from vertical slices",
                group.clone(),
            ),
            rb_vertical_slices: RadioButton::new_in_group(
                "Always use direct backprojection into X-tilted output slice",
                group.clone(),
            ),
            rb_old_style_x_tilting: RadioButton::new_in_group(
                "Always make vertical slices and interpolate to get X-tilted output",
                group,
            ),
            temporary_directory: LabeledTextField::new(FieldType::String, "Temporary directory: "),
            btn_ctf3d_setup: MultiLineButton::new_with_label(Some(RUN_BUTTON_LABEL)),
            btn_3dmod_ctf3d: MultiLineButton::new_with_label(Some("View Tomogram In 3dmod")),
            btn_use_ctf3d: MultiLineButton::new_with_label(Some(USE_BUTTON_LABEL)),
            cb_adjust_for_align_z_shift: CheckBox::new_with_text(
                "Adjust for Z shift in fine alignment and positioning",
            ),
            number_of_slabs: None,
        };
        result.tf_slab_thickness_in_nm.set_text("100");
        result.rb_x_axis_tilted_slices.set_selected(true);
        result.set_tooltips();
        result
    }

    /// Java `focusLost`; caller supplies the read from `MetaData.getPixelSize`.
    pub fn focus_lost<P: Ctf3dPanelParent>(&mut self, parent: &P, pixel_size: Option<f64>) {
        let slab = self
            .tf_slab_thickness_in_nm
            .get_text_validated(false)
            .ok()
            .and_then(|value| value.parse::<i64>().ok());
        self.number_of_slabs = match (parent.tomo_thickness(), pixel_size, slab) {
            (Some(t), Some(p), Some(s)) if p > 0.0 && s > 0 => {
                Some(((t as f64 * p) / s as f64).ceil() as i64)
            }
            _ => None,
        };
        self.layout.number_of_slabs_label = self.number_of_slabs.map_or_else(
            || " (?) slabs) ".into(),
            |number| format!(" ({number} slabs) "),
        );
        self.update_display(parent);
    }

    /// Java `action(String, Deferred3dmodButton, Run3dmodMenuOptions)`.
    pub fn action<M: Ctf3dPanelApplicationManager, P: Ctf3dPanelParent>(
        &mut self,
        manager: &mut M,
        parent: &P,
        action_command: Option<&str>,
        options: Option<Run3dmodMenuOptions>,
    ) {
        if action_command == self.btn_ctf3d_setup.get_action_command().as_deref() {
            manager.ctf3d_setup(self.axis_id, options);
        } else if action_command == self.btn_3dmod_ctf3d.get_action_command().as_deref() {
            manager.open_ctf3d(self.axis_id, options);
        } else if action_command == self.btn_use_ctf3d.get_action_command().as_deref() {
            manager.use_ctf3d(self.axis_id);
        } else {
            self.update_display(parent);
        }
    }

    /// Java `updateDisplay`.
    pub fn update_display<P: Ctf3dPanelParent>(&mut self, parent: &P) {
        self.layout.x_axis_tilted_slices_visible = self.header_advanced;
        self.sp_fourier_reduce_by_factor.enabled = self.cb_use_unaligned_images.is_selected();
        self.layout.adjust_for_align_z_shift_visible = self.header_advanced;
        self.layout.erase_fiducials_label_enabled = self.cb_erase_fiducials.is_selected();
        self.layout.filter_in_2d_label_enabled = self.cb_filter_in_2d.is_selected();
        let enabled = parent
            .x_axis_tilt()
            .and_then(|value| value.parse::<f64>().ok())
            .is_some_and(|value| value != 0.0)
            && !parent.is_use_local_alignment()
            && !parent.is_use_z_factors();
        self.layout.x_axis_tilted_slices_enabled = enabled;
        self.rb_x_axis_tilted_slices.set_enabled(enabled);
        self.rb_vertical_slices.set_enabled(enabled);
        self.rb_old_style_x_tilting.set_enabled(enabled);
    }
    pub fn msg_method_changed<P: Ctf3dPanelParent>(&mut self, parent: &P) {
        self.layout.root_visible = parent.is_ctf3d();
    }
    pub fn done(&mut self) {
        self.layout.listener_count = 0;
    }
    pub fn is_run_slabs_in_parallel(&self) -> bool {
        self.cb_run_slabs_in_parallel.is_selected()
    }
    pub fn is_erase_fiducials(&self) -> bool {
        self.cb_erase_fiducials.is_selected()
    }
    pub fn is_filter_in_2d(&self) -> bool {
        self.cb_filter_in_2d.is_selected()
    }
    pub fn is_use_unaligned_images(&self) -> bool {
        self.cb_use_unaligned_images.is_selected()
    }

    /// Java `getParameters(Ctf3dSetupParam, boolean)`.
    pub fn get_parameters<
        P: Ctf3dSetupParam,
        T: Ctf3dPanelParent,
        M: Ctf3dPanelApplicationManager,
    >(
        &mut self,
        param: &mut P,
        validation: bool,
        parent: &T,
        pixel_size: Option<f64>,
        manager: &mut M,
    ) -> bool {
        let (Ok(slab), Ok(directory)) = (
            self.tf_slab_thickness_in_nm.get_text_validated(validation),
            self.temporary_directory.get_text_validated(validation),
        ) else {
            return false;
        };
        if param.set_slab_thickness_in_nm(slab).is_err()
            || param.set_temporary_directory(directory).is_err()
        {
            return false;
        }
        param.set_run_slabs_in_parallel(self.cb_run_slabs_in_parallel.is_selected());
        param.set_erase_fiducials(self.cb_erase_fiducials.is_selected());
        param.set_filter_in_2d(self.cb_filter_in_2d.is_selected());
        param.set_use_unaligned_images(self.cb_use_unaligned_images.is_selected());
        param.set_adjust_for_align_z_shift(self.cb_adjust_for_align_z_shift.is_selected());
        param.set_fourier_reduce_by_factor(self.sp_fourier_reduce_by_factor.value);
        param.set_vertical_slices(self.rb_vertical_slices.is_selected());
        param.set_old_style_x_tilting(self.rb_old_style_x_tilting.is_selected());
        if validation {
            self.focus_lost(parent, pixel_size);
            if self
                .number_of_slabs
                .is_none_or(|value| value < NUMBER_OF_SLABS_MIN)
            {
                manager.open_message_dialog(
                    format!("Requires at least {NUMBER_OF_SLABS_MIN} slabs:  decrease thickness."),
                    "Not Enough Slabs",
                    self.axis_id,
                );
                return false;
            }
        }
        true
    }
    /// Java `setParameters(ConstMetaData)`.
    pub fn set_parameters_meta_data<M: Ctf3dMetaData>(&mut self, meta_data: &M) {
        self.sp_fourier_reduce_by_factor.value =
            meta_data.gen_ctf3d_fourier_reduce_by_factor(self.axis_id);
        if meta_data.is_gen_ctf3d_vertical_slices(self.axis_id) {
            self.rb_vertical_slices.set_selected(true);
        } else if meta_data.is_gen_ctf3d_old_style_x_tilting(self.axis_id) {
            self.rb_old_style_x_tilting.set_selected(true);
        } else {
            self.rb_x_axis_tilted_slices.set_selected(true);
        }
    }
    /// Java `getParameters(MetaData)`.
    pub fn get_parameters_meta_data<M: Ctf3dMetaData>(&self, meta_data: &mut M) {
        meta_data.set_gen_ctf3d_fourier_reduce_by_factor(
            self.axis_id,
            self.sp_fourier_reduce_by_factor.value,
        );
        meta_data
            .set_gen_ctf3d_vertical_slices(self.axis_id, self.rb_vertical_slices.is_selected());
        meta_data.set_gen_ctf3d_old_style_x_tilting(
            self.axis_id,
            self.rb_old_style_x_tilting.is_selected(),
        );
    }
    /// Java `setParameters(Ctf3dSetupParam)`.
    pub fn set_parameters_param<P: Ctf3dSetupParam, T: Ctf3dPanelParent>(
        &mut self,
        param: &P,
        parent: &T,
        pixel_size: Option<f64>,
    ) {
        if let Some(value) = param.slab_thickness_in_nm() {
            self.tf_slab_thickness_in_nm.set_text(&value)
        }
        self.cb_run_slabs_in_parallel
            .set_selected(param.run_slabs_in_parallel());
        self.cb_erase_fiducials
            .set_selected(param.erase_fiducials());
        self.cb_filter_in_2d.set_selected(param.filter_in_2d());
        self.cb_use_unaligned_images
            .set_selected(param.use_unaligned_images());
        self.cb_adjust_for_align_z_shift
            .set_selected(param.adjust_for_align_z_shift());
        self.sp_fourier_reduce_by_factor.value = param.fourier_reduce_by_factor();
        if param.vertical_slices() {
            self.rb_vertical_slices.set_selected(true)
        } else if param.old_style_x_tilting() {
            self.rb_old_style_x_tilting.set_selected(true)
        } else {
            self.rb_x_axis_tilted_slices.set_selected(true)
        }
        if let Some(value) = param.temporary_directory() {
            self.temporary_directory.set_text(&value)
        }
        self.focus_lost(parent, pixel_size);
    }
    /// Java `getParameters(ReconScreenState)`.
    pub fn get_parameters_screen_state<S: Ctf3dReconScreenState>(&mut self, screen_state: &mut S) {
        let key = self
            .btn_ctf3d_setup
            .get_button_state_key()
            .unwrap_or("ctf3dSetup".into());
        screen_state.set_button_state(&key, self.btn_ctf3d_setup.get_button_state());
    }
    /// Java `setParameters(ReconScreenState)`.
    pub fn set_parameters_screen_state<S: Ctf3dReconScreenState>(&mut self, screen_state: &S) {
        let key = self
            .btn_ctf3d_setup
            .get_button_state_key()
            .unwrap_or("ctf3dSetup".into());
        self.btn_ctf3d_setup
            .set_button_state(screen_state.get_button_state(&key));
    }
    /// Java `expand` effects.
    pub fn expand<P: Ctf3dPanelParent, M: Ctf3dPanelApplicationManager>(
        &mut self,
        parent: &P,
        manager: &mut M,
    ) {
        self.update_display(parent);
        manager.pack(self.axis_id);
    }
    /// Java `setTooltips`; autodoc text lookup remains a storage boundary.
    fn set_tooltips(&mut self) {
        self.tf_slab_thickness_in_nm
            .set_tooltip(Some("SlabThicknessInNm"));
        self.cb_run_slabs_in_parallel
            .set_tooltip(Some("RunSlabsInParallel"));
        self.cb_erase_fiducials.set_tooltip(Some("EraseFiducials"));
        self.cb_filter_in_2d.set_tooltip(Some("FilterIn2D"));
        self.cb_use_unaligned_images
            .set_tooltip(Some("UseUnalignedImages"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Parent;
    impl Ctf3dPanelParent for Parent {
        fn tomo_thickness(&self) -> Option<i64> {
            Some(1000)
        }
        fn x_axis_tilt(&self) -> Option<String> {
            Some("3".into())
        }
        fn is_use_local_alignment(&self) -> bool {
            false
        }
        fn is_use_z_factors(&self) -> bool {
            false
        }
        fn is_ctf3d(&self) -> bool {
            true
        }
    }
    #[test]
    fn focus_lost_calculates_slabs() {
        let mut panel = Ctf3dPanel::get_instance(AxisID::Only);
        panel.tf_slab_thickness_in_nm.set_text("100");
        panel.focus_lost(&Parent, Some(1.0));
        assert_eq!(panel.number_of_slabs, Some(10));
        assert_eq!(panel.layout.number_of_slabs_label, " (10 slabs) ");
        assert!(panel.layout.x_axis_tilted_slices_enabled)
    }
    #[test]
    fn display_follows_source_rules() {
        let mut panel = Ctf3dPanel::get_instance(AxisID::Only);
        panel.header_advanced = true;
        panel.cb_use_unaligned_images.set_selected(true);
        panel.cb_erase_fiducials.set_selected(true);
        panel.update_display(&Parent);
        assert!(panel.sp_fourier_reduce_by_factor.enabled);
        assert!(panel.layout.adjust_for_align_z_shift_visible);
        assert!(panel.layout.erase_fiducials_label_enabled)
    }
}
