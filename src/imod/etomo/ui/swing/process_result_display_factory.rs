//! `IMOD/Etomo/src/etomo/ui/swing/ProcessResultDisplayFactory.java`.
//!
//! Swing `Run3dmodButton` is represented by the same process-display state as
//! `MultiLineButton`; the deferred 3dmod launch itself remains at the frontend
//! boundary.  This unit retains the factory's permanent process-display graph.
#![allow(dead_code)]

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::dialog_type::DialogType;

use super::multi_line_button::{BaseScreenState, MultiLineButton};

/// Java `ProcessResultDisplayFactory`.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessResultDisplayFactory {
    pub factory_id: String,
    pub screen_state: BaseScreenState,
    pub dependency_order: Vec<i32>,
    pub find_x_rays: MultiLineButton,
    pub create_fixed_stack: MultiLineButton,
    pub use_fixed_stack: MultiLineButton,
    pub coarse_tiltxcorr: MultiLineButton,
    pub distortion_corrected_stack: MultiLineButton,
    pub fix_edges_midas: MultiLineButton,
    pub coarse_align: MultiLineButton,
    pub midas: MultiLineButton,
    pub transfer_fiducials: MultiLineButton,
    pub raptor: MultiLineButton,
    pub use_raptor: MultiLineButton,
    pub seed_fiducial_model: MultiLineButton,
    pub track_fiducials: MultiLineButton,
    pub fix_fiducial_model: MultiLineButton,
    pub track_tiltxcorr: MultiLineButton,
    pub autofidseed: MultiLineButton,
    pub just_find_shifts_near_zero: MultiLineButton,
    pub imodchopconts: MultiLineButton,
    pub use_adjusted_track_com: MultiLineButton,
    pub compute_alignment: MultiLineButton,
    pub sample_tomogram: MultiLineButton,
    pub compute_pitch: MultiLineButton,
    pub final_alignment: MultiLineButton,
    pub full_aligned_stack: MultiLineButton,
    pub ctf_correction: MultiLineButton,
    pub use_ctf_correction: MultiLineButton,
    pub xf_model: MultiLineButton,
    pub stack_tilt: MultiLineButton,
    pub find_beads3d: MultiLineButton,
    pub reproject_model: MultiLineButton,
    pub ccd_eraser_beads: MultiLineButton,
    pub use_ccd_eraser_beads: MultiLineButton,
    pub filter: MultiLineButton,
    pub use_filtered_stack: MultiLineButton,
    pub use_trial_tomogram: MultiLineButton,
    pub gen_tilt: MultiLineButton,
    pub delete_aligned_stack: MultiLineButton,
    pub multifilt_setup: MultiLineButton,
    pub sirtsetup: MultiLineButton,
    pub use_sirt: MultiLineButton,
    pub ctf3d_setup: MultiLineButton,
    pub use_ctf3d: MultiLineButton,
    pub create_combine: MultiLineButton,
    pub combine: MultiLineButton,
    pub restart_combine: MultiLineButton,
    pub restart_matchvol1: MultiLineButton,
    pub restart_patchcorr: MultiLineButton,
    pub restart_matchorwarp: MultiLineButton,
    pub restart_volcombine: MultiLineButton,
    pub trim_volume: MultiLineButton,
    pub flatten: MultiLineButton,
    pub flatten_warp: MultiLineButton,
    pub squeeze_volume: MultiLineButton,
    pub smoothing_assessment: MultiLineButton,
}

impl ProcessResultDisplayFactory {
    /// Java private constructor `ProcessResultDisplayFactory(BaseScreenState, AxisID, AxisType)`.
    pub fn new(screen_state: BaseScreenState, axis_id: AxisID, axis_type: AxisType) -> Self {
        let factory_id = format!(
            "etomo.ui.swing.ProcessResultDisplayFactory{}",
            if axis_type == AxisType::DualAxis {
                axis_id.to_string()
            } else {
                String::new()
            }
        );
        Self {
            factory_id,
            screen_state,
            dependency_order: Vec::new(),
            find_x_rays: Self::deferred("Find X-rays (Trial Mode)", DialogType::PreProcessing),
            create_fixed_stack: Self::deferred("Create Fixed Stack", DialogType::PreProcessing),
            use_fixed_stack: Self::toggle("Use Fixed Stack", DialogType::PreProcessing),
            coarse_tiltxcorr: Self::toggle(
                "Calculate Cross-Correlation",
                DialogType::CoarseAlignment,
            ),
            distortion_corrected_stack: Self::toggle(
                "Make Distortion Corrected Stack",
                DialogType::CoarseAlignment,
            ),
            fix_edges_midas: Self::toggle("Fix Edges With Midas", DialogType::CoarseAlignment),
            coarse_align: Self::deferred(
                "Generate Coarse Aligned Stack",
                DialogType::CoarseAlignment,
            ),
            midas: Self::toggle("Fix Alignment With Midas", DialogType::CoarseAlignment),
            transfer_fiducials: Self::deferred(
                "Transfer Fiducials From Other Axis",
                DialogType::FiducialModel,
            ),
            raptor: Self::deferred("Run RAPTOR", DialogType::FiducialModel),
            use_raptor: Self::toggle(
                "Use RAPTOR Result as Fiducial Model",
                DialogType::FiducialModel,
            ),
            seed_fiducial_model: Self::toggle("Seed Fiducial Model", DialogType::FiducialModel),
            track_fiducials: Self::toggle("Track Fiducials", DialogType::FiducialModel),
            fix_fiducial_model: Self::toggle("Fix Fiducial Model", DialogType::FiducialModel),
            track_tiltxcorr: Self::deferred("Track Patches", DialogType::FiducialModel),
            autofidseed: Self::deferred("Generate Seed Model", DialogType::FiducialModel),
            just_find_shifts_near_zero: Self::toggle(
                "Run Autofidseed to Find Shifts",
                DialogType::FiducialModel,
            ),
            imodchopconts: Self::deferred("Recut or Restore Contours", DialogType::FiducialModel),
            use_adjusted_track_com: Self::toggle(
                "Use Adjusted Track Com File",
                DialogType::FiducialModel,
            ),
            compute_alignment: Self::toggle("Compute Alignment", DialogType::FineAlignment),
            sample_tomogram: Self::deferred(
                "Create Sample Tomograms",
                DialogType::TomogramPositioning,
            ),
            compute_pitch: Self::toggle(
                "Compute Z Shift & Pitch Angles",
                DialogType::TomogramPositioning,
            ),
            final_alignment: Self::toggle(
                "Create Final Alignment",
                DialogType::TomogramPositioning,
            ),
            full_aligned_stack: Self::deferred(
                "Create Full Aligned Stack",
                DialogType::FinalAlignedStack,
            ),
            ctf_correction: Self::deferred("Correct CTF", DialogType::FinalAlignedStack),
            use_ctf_correction: Self::toggle("Use CTF Correction", DialogType::FinalAlignedStack),
            xf_model: Self::deferred("Transform Fiducial Model", DialogType::FinalAlignedStack),
            stack_tilt: Self::deferred("Align and Build Tomogram", DialogType::FinalAlignedStack),
            find_beads3d: Self::deferred("Run Findbeads3d", DialogType::FinalAlignedStack),
            reproject_model: Self::deferred("Reproject Model", DialogType::FinalAlignedStack),
            ccd_eraser_beads: Self::deferred("Erase Beads", DialogType::FinalAlignedStack),
            use_ccd_eraser_beads: Self::deferred("Use Erased Stack", DialogType::FinalAlignedStack),
            filter: Self::deferred("Filter", DialogType::FinalAlignedStack),
            use_filtered_stack: Self::toggle("Use Filtered Stack", DialogType::FinalAlignedStack),
            use_trial_tomogram: Self::toggle(
                "Use Current Trial Tomogram",
                DialogType::TomogramGeneration,
            ),
            gen_tilt: Self::deferred("Generate Tomogram", DialogType::TomogramGeneration),
            delete_aligned_stack: Self::toggle(
                "Delete Intermediate Image Stacks",
                DialogType::TomogramGeneration,
            ),
            multifilt_setup: Self::deferred("Run Filter Trials", DialogType::TomogramGeneration),
            sirtsetup: Self::deferred("Run SIRT", DialogType::TomogramGeneration),
            use_sirt: Self::toggle("Use SIRT Output File", DialogType::TomogramGeneration),
            ctf3d_setup: Self::deferred(
                "Generate CTF-corrected Tomogram",
                DialogType::TomogramGeneration,
            ),
            use_ctf3d: Self::toggle("Use CTF-corrected Tomogram", DialogType::TomogramGeneration),
            create_combine: Self::toggle("Create Combine Scripts", DialogType::TomogramCombination),
            combine: Self::deferred("Start Combine", DialogType::TomogramCombination),
            restart_combine: Self::deferred("Restart Combine", DialogType::TomogramCombination),
            restart_matchvol1: Self::deferred(
                "Restart at Matchvol1",
                DialogType::TomogramCombination,
            ),
            restart_patchcorr: Self::deferred(
                "Restart at Patchcorr",
                DialogType::TomogramCombination,
            ),
            restart_matchorwarp: Self::deferred(
                "Restart at Matchorwarp",
                DialogType::TomogramCombination,
            ),
            restart_volcombine: Self::deferred(
                "Restart at Volcombine",
                DialogType::TomogramCombination,
            ),
            trim_volume: Self::deferred("Trim Volume", DialogType::PostProcessing),
            flatten: Self::deferred("Flatten", DialogType::PostProcessing),
            flatten_warp: MultiLineButton::new_with_label(Some("Run Flattenwarp")),
            squeeze_volume: Self::deferred("Reduce/Filter Volume", DialogType::PostProcessing),
            smoothing_assessment: Self::deferred(
                "Run Flattenwarp to Assess Smoothing",
                DialogType::PostProcessing,
            ),
        }
    }
    /// Java `Run3dmodButton.getDeferredToggle3dmodInstance`; the launch component is a boundary.
    fn deferred(label: &str, dialog_type: DialogType) -> MultiLineButton {
        MultiLineButton::get_toggle_button_instance_with_dialog(Some(label), Some(dialog_type))
    }
    /// Java `MultiLineButton.getToggleButtonInstance(String, DialogType)`.
    fn toggle(label: &str, dialog_type: DialogType) -> MultiLineButton {
        MultiLineButton::get_toggle_button_instance_with_dialog(Some(label), Some(dialog_type))
    }
    /// Java static `getInstance`.
    pub fn get_instance(
        screen_state: BaseScreenState,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> Self {
        let mut instance = Self::new(screen_state, axis_id, axis_type);
        instance.initialize();
        instance
    }
    /// Java private `initialize`.
    pub fn initialize(&mut self) {
        macro_rules! dependency {
            ($display:ident) => {{
                let id = self.dependency_order.len() as i32;
                self.$display.set_id(id, Some(&self.factory_id));
                self.dependency_order.push(id);
            }};
        }
        dependency!(find_x_rays);
        dependency!(create_fixed_stack);
        dependency!(use_fixed_stack);
        dependency!(coarse_tiltxcorr);
        dependency!(distortion_corrected_stack);
        dependency!(fix_edges_midas);
        dependency!(coarse_align);
        dependency!(midas);
        dependency!(transfer_fiducials);
        dependency!(seed_fiducial_model);
        dependency!(autofidseed);
        dependency!(just_find_shifts_near_zero);
        dependency!(track_tiltxcorr);
        dependency!(imodchopconts);
        dependency!(raptor);
        dependency!(use_raptor);
        dependency!(track_fiducials);
        dependency!(fix_fiducial_model);
        dependency!(compute_alignment);
        dependency!(sample_tomogram);
        dependency!(compute_pitch);
        dependency!(final_alignment);
        dependency!(full_aligned_stack);
        dependency!(ctf_correction);
        dependency!(use_ctf_correction);
        dependency!(xf_model);
        dependency!(stack_tilt);
        dependency!(find_beads3d);
        dependency!(reproject_model);
        dependency!(ccd_eraser_beads);
        dependency!(use_ccd_eraser_beads);
        dependency!(filter);
        dependency!(use_filtered_stack);
        dependency!(gen_tilt);
        dependency!(use_trial_tomogram);
        dependency!(delete_aligned_stack);
        dependency!(multifilt_setup);
        dependency!(sirtsetup);
        dependency!(ctf3d_setup);
        dependency!(use_sirt);
        dependency!(use_ctf3d);
        dependency!(create_combine);
        dependency!(combine);
        dependency!(restart_combine);
        dependency!(restart_matchvol1);
        dependency!(restart_patchcorr);
        dependency!(restart_matchorwarp);
        dependency!(restart_volcombine);
        dependency!(trim_volume);
        dependency!(smoothing_assessment);
        dependency!(flatten_warp);
        dependency!(flatten);
        dependency!(squeeze_volume);
        // `AbstractProcessResultDisplayFactory.addDependency` links every prior
        // display to its successor and clears the tail link.
        macro_rules! has_next {
            ($display:ident) => {
                self.$display.set_next(true);
            };
        }
        has_next!(find_x_rays);
        has_next!(create_fixed_stack);
        has_next!(use_fixed_stack);
        has_next!(coarse_tiltxcorr);
        has_next!(distortion_corrected_stack);
        has_next!(fix_edges_midas);
        has_next!(coarse_align);
        has_next!(midas);
        has_next!(transfer_fiducials);
        has_next!(seed_fiducial_model);
        has_next!(autofidseed);
        has_next!(just_find_shifts_near_zero);
        has_next!(track_tiltxcorr);
        has_next!(imodchopconts);
        has_next!(raptor);
        has_next!(use_raptor);
        has_next!(track_fiducials);
        has_next!(fix_fiducial_model);
        has_next!(compute_alignment);
        has_next!(sample_tomogram);
        has_next!(compute_pitch);
        has_next!(final_alignment);
        has_next!(full_aligned_stack);
        has_next!(ctf_correction);
        has_next!(use_ctf_correction);
        has_next!(xf_model);
        has_next!(stack_tilt);
        has_next!(find_beads3d);
        has_next!(reproject_model);
        has_next!(ccd_eraser_beads);
        has_next!(use_ccd_eraser_beads);
        has_next!(filter);
        has_next!(use_filtered_stack);
        has_next!(gen_tilt);
        has_next!(use_trial_tomogram);
        has_next!(delete_aligned_stack);
        has_next!(multifilt_setup);
        has_next!(sirtsetup);
        has_next!(ctf3d_setup);
        has_next!(use_sirt);
        has_next!(use_ctf3d);
        has_next!(create_combine);
        has_next!(combine);
        has_next!(restart_combine);
        has_next!(restart_matchvol1);
        has_next!(restart_patchcorr);
        has_next!(restart_matchorwarp);
        has_next!(restart_volcombine);
        has_next!(trim_volume);
        has_next!(smoothing_assessment);
        has_next!(flatten_warp);
        has_next!(flatten);
        self.midas.set_use_global_dependency_list(false);
        self.just_find_shifts_near_zero
            .set_use_global_dependency_list(false);
        self.raptor.set_use_global_dependency_list(false);
        self.ctf_correction.set_use_global_dependency_list(false);
        self.xf_model.set_use_global_dependency_list(false);
        self.stack_tilt.set_use_global_dependency_list(false);
        self.find_beads3d.set_use_global_dependency_list(false);
        self.reproject_model.set_use_global_dependency_list(false);
        self.ccd_eraser_beads.set_use_global_dependency_list(false);
        self.filter.set_use_global_dependency_list(false);
        self.delete_aligned_stack
            .set_use_global_dependency_list(false);
        self.smoothing_assessment
            .set_use_global_dependency_list(false);
        self.flatten_warp.set_use_global_dependency_list(false);
        self.flatten.set_use_global_dependency_list(false);
        self.squeeze_volume.set_use_global_dependency_list(false);
        self.coarse_align.add_dependent_display();
        self.raptor.add_dependent_display();
        self.ctf_correction.add_dependent_display();
        self.xf_model.add_dependent_display();
        self.xf_model.add_dependent_display();
        self.stack_tilt.add_dependent_display();
        self.stack_tilt.add_dependent_display();
        self.stack_tilt.add_dependent_display();
        self.stack_tilt.add_dependent_display();
        self.find_beads3d.add_dependent_display();
        self.find_beads3d.add_dependent_display();
        self.find_beads3d.add_dependent_display();
        self.reproject_model.add_dependent_display();
        self.reproject_model.add_dependent_display();
        self.ccd_eraser_beads.add_dependent_display();
        self.filter.add_dependent_display();
        self.sirtsetup.add_dependent_display();
        self.delete_aligned_stack.add_dependent_display();
        self.flatten_warp.add_dependent_display();
        self.use_trial_tomogram.add_success_display();
        self.combine.add_failure_display();
        self.combine.add_success_display();
        self.restart_combine.add_failure_display();
        self.restart_combine.add_success_display();
        macro_rules! state {
            ($display:ident) => {
                self.$display.set_screen_state(self.screen_state.clone());
            };
        }
        state!(find_x_rays);
        state!(create_fixed_stack);
        state!(use_fixed_stack);
        state!(coarse_tiltxcorr);
        state!(distortion_corrected_stack);
        state!(fix_edges_midas);
        state!(coarse_align);
        state!(midas);
        state!(transfer_fiducials);
        state!(track_tiltxcorr);
        state!(imodchopconts);
        state!(raptor);
        state!(use_raptor);
        state!(seed_fiducial_model);
        state!(autofidseed);
        state!(just_find_shifts_near_zero);
        state!(use_adjusted_track_com);
        state!(track_fiducials);
        state!(fix_fiducial_model);
        state!(compute_alignment);
        state!(sample_tomogram);
        state!(compute_pitch);
        state!(final_alignment);
        state!(full_aligned_stack);
        state!(ctf_correction);
        state!(use_ctf_correction);
        state!(xf_model);
        state!(stack_tilt);
        state!(find_beads3d);
        state!(reproject_model);
        state!(ccd_eraser_beads);
        state!(use_ccd_eraser_beads);
        state!(filter);
        state!(use_filtered_stack);
        state!(use_trial_tomogram);
        state!(gen_tilt);
        state!(delete_aligned_stack);
        state!(multifilt_setup);
        state!(sirtsetup);
        state!(ctf3d_setup);
        state!(use_sirt);
        state!(use_ctf3d);
        state!(create_combine);
        state!(combine);
        state!(restart_combine);
        state!(restart_matchvol1);
        state!(restart_patchcorr);
        state!(restart_matchorwarp);
        state!(restart_volcombine);
        state!(trim_volume);
        state!(smoothing_assessment);
        state!(flatten_warp);
        state!(flatten);
        state!(squeeze_volume);
    }
    /// Java `getProcessResultDisplay` from `AbstractProcessResultDisplayFactory`.
    pub fn get_process_result_display(
        &self,
        display_id: i32,
        factory_id: Option<&str>,
    ) -> Option<&MultiLineButton> {
        self.all_displays()
            .into_iter()
            .find(|display| display.equals_id(display_id, factory_id))
    }
    /// Java `getTiltxcorr`.
    pub fn get_tiltxcorr(&self, dialog_type: DialogType) -> Option<&MultiLineButton> {
        match dialog_type {
            DialogType::CoarseAlignment => Some(&self.coarse_tiltxcorr),
            DialogType::FiducialModel => Some(&self.track_tiltxcorr),
            _ => None,
        }
    }
    /// Java `getTilt`.
    pub fn get_tilt(&self, dialog_type: DialogType) -> &MultiLineButton {
        if dialog_type == DialogType::FinalAlignedStack {
            &self.stack_tilt
        } else {
            &self.gen_tilt
        }
    }
    // Java's package-private display accessors.  The Rust module keeps them public so
    // translated managers outside this package retain the same wiring surface.
    pub fn get_find_x_rays(&self) -> &MultiLineButton {
        &self.find_x_rays
    }
    pub fn get_create_fixed_stack(&self) -> &MultiLineButton {
        &self.create_fixed_stack
    }
    pub fn get_use_fixed_stack(&self) -> &MultiLineButton {
        &self.use_fixed_stack
    }
    pub fn get_imodchopconts(&self) -> &MultiLineButton {
        &self.imodchopconts
    }
    pub fn get_autofidseed(&self) -> &MultiLineButton {
        &self.autofidseed
    }
    pub fn get_just_find_shifts_near_zero(&self) -> &MultiLineButton {
        &self.just_find_shifts_near_zero
    }
    pub fn get_use_adjusted_track_com(&self) -> &MultiLineButton {
        &self.use_adjusted_track_com
    }
    pub fn get_distortion_corrected_stack(&self) -> &MultiLineButton {
        &self.distortion_corrected_stack
    }
    pub fn get_fix_edges_midas(&self) -> &MultiLineButton {
        &self.fix_edges_midas
    }
    pub fn get_coarse_align(&self) -> &MultiLineButton {
        &self.coarse_align
    }
    pub fn get_midas(&self) -> &MultiLineButton {
        &self.midas
    }
    pub fn get_transfer_fiducials(&self) -> &MultiLineButton {
        &self.transfer_fiducials
    }
    pub fn get_raptor(&self) -> &MultiLineButton {
        &self.raptor
    }
    pub fn get_use_raptor(&self) -> &MultiLineButton {
        &self.use_raptor
    }
    pub fn get_seed_fiducial_model(&self) -> &MultiLineButton {
        &self.seed_fiducial_model
    }
    pub fn get_track_fiducials(&self) -> &MultiLineButton {
        &self.track_fiducials
    }
    pub fn get_fix_fiducial_model(&self) -> &MultiLineButton {
        &self.fix_fiducial_model
    }
    pub fn get_compute_alignment(&self) -> &MultiLineButton {
        &self.compute_alignment
    }
    pub fn get_sample_tomogram(&self) -> &MultiLineButton {
        &self.sample_tomogram
    }
    pub fn get_compute_pitch(&self) -> &MultiLineButton {
        &self.compute_pitch
    }
    pub fn get_final_alignment(&self) -> &MultiLineButton {
        &self.final_alignment
    }
    pub fn get_full_aligned_stack(&self) -> &MultiLineButton {
        &self.full_aligned_stack
    }
    pub fn get_ctf_correction(&self) -> &MultiLineButton {
        &self.ctf_correction
    }
    pub fn get_use_ctf_correction(&self) -> &MultiLineButton {
        &self.use_ctf_correction
    }
    pub fn get_xf_model(&self) -> &MultiLineButton {
        &self.xf_model
    }
    pub fn get_find_beads3d(&self) -> &MultiLineButton {
        &self.find_beads3d
    }
    pub fn get_multifilt_setup(&self) -> &MultiLineButton {
        &self.multifilt_setup
    }
    pub fn get_sirtsetup(&self) -> &MultiLineButton {
        &self.sirtsetup
    }
    pub fn get_use_sirt(&self) -> &MultiLineButton {
        &self.use_sirt
    }
    pub fn get_ctf3d_setup(&self) -> &MultiLineButton {
        &self.ctf3d_setup
    }
    pub fn get_use_ctf3d(&self) -> &MultiLineButton {
        &self.use_ctf3d
    }
    pub fn get_reproject_model(&self) -> &MultiLineButton {
        &self.reproject_model
    }
    pub fn get_ccd_eraser_beads(&self) -> &MultiLineButton {
        &self.ccd_eraser_beads
    }
    pub fn get_use_ccd_eraser_beads(&self) -> &MultiLineButton {
        &self.use_ccd_eraser_beads
    }
    pub fn get_filter(&self) -> &MultiLineButton {
        &self.filter
    }
    pub fn get_use_filtered_stack(&self) -> &MultiLineButton {
        &self.use_filtered_stack
    }
    pub fn get_use_trial_tomogram(&self) -> &MultiLineButton {
        &self.use_trial_tomogram
    }
    pub fn get_delete_aligned_stack(&self) -> &MultiLineButton {
        &self.delete_aligned_stack
    }
    pub fn get_create_combine(&self) -> &MultiLineButton {
        &self.create_combine
    }
    pub fn get_combine(&self) -> &MultiLineButton {
        &self.combine
    }
    pub fn get_restart_combine(&self) -> &MultiLineButton {
        &self.restart_combine
    }
    pub fn get_restart_matchvol1(&self) -> &MultiLineButton {
        &self.restart_matchvol1
    }
    pub fn get_restart_patchcorr(&self) -> &MultiLineButton {
        &self.restart_patchcorr
    }
    pub fn get_restart_matchorwarp(&self) -> &MultiLineButton {
        &self.restart_matchorwarp
    }
    pub fn get_restart_volcombine(&self) -> &MultiLineButton {
        &self.restart_volcombine
    }
    pub fn get_trim_volume(&self) -> &MultiLineButton {
        &self.trim_volume
    }
    pub fn get_flatten(&self) -> &MultiLineButton {
        &self.flatten
    }
    pub fn get_flatten_warp(&self) -> &MultiLineButton {
        &self.flatten_warp
    }
    pub fn get_smoothing_assessment(&self) -> &MultiLineButton {
        &self.smoothing_assessment
    }
    pub fn get_squeeze_volume(&self) -> &MultiLineButton {
        &self.squeeze_volume
    }
    /// All fixed source fields, in declaration order; frontend iteration boundary.
    pub fn all_displays(&self) -> Vec<&MultiLineButton> {
        vec![
            &self.find_x_rays,
            &self.create_fixed_stack,
            &self.use_fixed_stack,
            &self.coarse_tiltxcorr,
            &self.distortion_corrected_stack,
            &self.fix_edges_midas,
            &self.coarse_align,
            &self.midas,
            &self.transfer_fiducials,
            &self.raptor,
            &self.use_raptor,
            &self.seed_fiducial_model,
            &self.track_fiducials,
            &self.fix_fiducial_model,
            &self.track_tiltxcorr,
            &self.autofidseed,
            &self.just_find_shifts_near_zero,
            &self.imodchopconts,
            &self.use_adjusted_track_com,
            &self.compute_alignment,
            &self.sample_tomogram,
            &self.compute_pitch,
            &self.final_alignment,
            &self.full_aligned_stack,
            &self.ctf_correction,
            &self.use_ctf_correction,
            &self.xf_model,
            &self.stack_tilt,
            &self.find_beads3d,
            &self.reproject_model,
            &self.ccd_eraser_beads,
            &self.use_ccd_eraser_beads,
            &self.filter,
            &self.use_filtered_stack,
            &self.use_trial_tomogram,
            &self.gen_tilt,
            &self.delete_aligned_stack,
            &self.multifilt_setup,
            &self.sirtsetup,
            &self.use_sirt,
            &self.ctf3d_setup,
            &self.use_ctf3d,
            &self.create_combine,
            &self.combine,
            &self.restart_combine,
            &self.restart_matchvol1,
            &self.restart_patchcorr,
            &self.restart_matchorwarp,
            &self.restart_volcombine,
            &self.trim_volume,
            &self.flatten,
            &self.flatten_warp,
            &self.squeeze_volume,
            &self.smoothing_assessment,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialize_preserves_global_dependency_order_and_opt_outs() {
        let f = ProcessResultDisplayFactory::get_instance(
            BaseScreenState::default(),
            AxisID::First,
            AxisType::DualAxis,
        );
        assert_eq!(
            f.factory_id,
            "etomo.ui.swing.ProcessResultDisplayFactoryFirst"
        );
        assert_eq!(f.dependency_order.len(), 53);
        assert_eq!(
            f.get_process_result_display(0, Some(&f.factory_id))
                .unwrap()
                .get_text(),
            Some("Find X-rays (Trial Mode)")
        );
        assert!(
            !f.midas
                .process_result_display_state
                .use_global_dependency_list
        );
        assert!(
            !f.flatten
                .process_result_display_state
                .use_global_dependency_list
        );
    }
    #[test]
    fn source_display_edges_are_present() {
        let f = ProcessResultDisplayFactory::get_instance(
            BaseScreenState::default(),
            AxisID::Only,
            AxisType::SingleAxis,
        );
        assert_eq!(
            f.stack_tilt
                .process_result_display_state
                .dependent_display_count,
            4
        );
        assert_eq!(
            f.combine.process_result_display_state.failure_display_count,
            1
        );
        assert_eq!(
            f.restart_combine
                .process_result_display_state
                .success_display_count,
            1
        );
        assert!(f.get_tiltxcorr(DialogType::TomogramGeneration).is_none());
        assert_eq!(
            f.get_tilt(DialogType::TomogramGeneration).get_text(),
            Some("Generate Tomogram")
        );
    }
}
