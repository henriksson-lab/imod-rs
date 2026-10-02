//! `IMOD/Etomo/src/etomo/ui/swing/ProcessResultDisplayFactory.java`.
//!
//! Description: Updates toggle buttons based on other buttons.  If a button is
//! part the standard process of running up a tomogram, then it has a global
//! dependency.  If it the process it runs relies on a previous process, then it
//! has an individual dependency.
//!
//! The factory owns the toggle buttons that outlive their dialogs; the dialogs
//! fetch them with the `get*` methods and put them in their panels.  Each
//! button is held by its concrete type (`Rc<MultiLineButton>` /
//! `Rc<Run3dmodButton>`, as the Java instance was created) so a dialog gets the
//! widget it lays out, while the dependency lists hold the same `Rc` as a
//! `ProcessResultDisplayHandle` (`Rc<dyn ProcessResultDisplay>`).  Where the
//! Java getter's declared type is `ProcessResultDisplay` and the dialog casts it
//! back to its class, the Rust getter returns that class directly; the one
//! getter whose result can be either class (`getTiltxcorr`) returns the
//! interface as the Java does.
//!
//! The factory is an event-dispatch-thread object (`Rc`, `&self`); the manager
//! keeps one per axis.

use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::base_screen_state::BaseScreenState;
use crate::imod::etomo::r#type::dialog_type::DialogType;
use crate::imod::etomo::r#type::process_result_display::{
    ProcessResultDisplay, ProcessResultDisplayHandle,
};
use crate::imod::etomo::ui::swing::abstract_process_result_display_factory::AbstractProcessResultDisplayFactory;
use crate::imod::etomo::ui::swing::multi_line_button::MultiLineButton;
use crate::imod::etomo::ui::swing::run_3dmod_button::Run3dmodButton;
use crate::imod::etomo::ui::swing::{
    beadtrack_panel, ccd_eraser_beads_panel, ccd_eraser_xrays_panel, ctf3d_panel,
    fiducial_model_dialog, final_aligned_stack_dialog, flatten_volume_panel,
    newstack_or_blendmont_panel, raptor_panel, reproject_model_panel, smoothing_assessment_panel,
    tilt3d_find_panel, tomogram_positioning_dialog,
};
use std::ops::Deref;
use std::rc::Rc;

/// Java `public final class ProcessResultDisplayFactory extends
/// AbstractProcessResultDisplayFactory`.
pub struct ProcessResultDisplayFactory {
    /// Java superclass `AbstractProcessResultDisplayFactory`.
    base: AbstractProcessResultDisplayFactory,
    /// Java private final `screenState`.
    screen_state: &'static BaseScreenState,

    // Recon
    // preprocessing
    /// Java private final `findXRays`.
    find_x_rays: Rc<Run3dmodButton>,
    /// Java private final `createFixedStack`.
    create_fixed_stack: Rc<Run3dmodButton>,
    /// Java private final `useFixedStack`.
    use_fixed_stack: Rc<MultiLineButton>,
    // coarse alignment
    /// Java private final `coarseTiltxcorr`.
    coarse_tiltxcorr: Rc<MultiLineButton>,
    /// Java private final `distortionCorrectedStack`.
    distortion_corrected_stack: Rc<MultiLineButton>,
    /// Java private final `fixEdgesMidas`.
    fix_edges_midas: Rc<MultiLineButton>,
    /// Java private final `coarseAlign`.
    coarse_align: Rc<Run3dmodButton>,
    /// Java private final `midas`.
    midas: Rc<MultiLineButton>,
    // fiducial model
    /// Java private final `transferFiducials`.
    transfer_fiducials: Rc<Run3dmodButton>,
    /// Java private final `raptor`.
    raptor: Rc<Run3dmodButton>,
    /// Java private final `useRaptor`.
    use_raptor: Rc<MultiLineButton>,
    /// Java private final `seedFiducialModel`.
    seed_fiducial_model: Rc<Run3dmodButton>,
    /// Java private final `trackFiducials`.
    track_fiducials: Rc<MultiLineButton>,
    /// Java private final `fixFiducialModel`.
    fix_fiducial_model: Rc<Run3dmodButton>,
    /// Java private final `trackTiltxcorr`.
    track_tiltxcorr: Rc<Run3dmodButton>,
    /// Java private final `autofidseed`.
    autofidseed: Rc<Run3dmodButton>,
    /// Java private final `justFindShiftsNearZero`.
    just_find_shifts_near_zero: Rc<MultiLineButton>,
    /// Java private final `imodchopconts`.
    imodchopconts: Rc<Run3dmodButton>,
    /// Java private final `useAdjustedTrackCom`.
    use_adjusted_track_com: Rc<MultiLineButton>,
    // fine alignment
    /// Java private final `computeAlignment`.
    compute_alignment: Rc<MultiLineButton>,
    // positioning
    /// Java private final `sampleTomogram`.
    sample_tomogram: Rc<Run3dmodButton>,
    /// Java private final `computePitch`.
    compute_pitch: Rc<MultiLineButton>,
    /// Java private final `finalAlignment`.
    final_alignment: Rc<MultiLineButton>,
    // final aligned stack
    /// Java private final `fullAlignedStack`.
    full_aligned_stack: Rc<Run3dmodButton>,
    /// Java private final `ctfCorrection`.
    ctf_correction: Rc<Run3dmodButton>,
    /// Java private final `useCtfCorrection`.
    use_ctf_correction: Rc<MultiLineButton>,
    /// Java private final `xfModel`.
    xf_model: Rc<Run3dmodButton>,
    /// Java private final `stackTilt`.
    stack_tilt: Rc<Run3dmodButton>,
    /// Java private final `findBeads3d`.
    find_beads3d: Rc<Run3dmodButton>,
    /// Java private final `reprojectModel`.
    reproject_model: Rc<Run3dmodButton>,
    /// Java private final `ccdEraserBeads`.
    ccd_eraser_beads: Rc<Run3dmodButton>,
    /// Java private final `useCcdEraserBeads`.
    use_ccd_eraser_beads: Rc<Run3dmodButton>,
    /// Java private final `filter`.
    filter: Rc<Run3dmodButton>,
    /// Java private final `useFilteredStack`.
    use_filtered_stack: Rc<MultiLineButton>,
    // generation
    /// Java private final `useTrialTomogram`.
    use_trial_tomogram: Rc<MultiLineButton>,
    /// Java private final `genTilt`.
    gen_tilt: Rc<Run3dmodButton>,
    /// Java private final `deleteAlignedStack`.
    delete_aligned_stack: Rc<MultiLineButton>,
    /// Java private final `multifiltSetup`.
    multifilt_setup: Rc<Run3dmodButton>,
    /// Java private final `sirtsetup`.
    sirtsetup: Rc<Run3dmodButton>,
    /// Java private final `useSirt`.
    use_sirt: Rc<MultiLineButton>,
    /// Java private final `ctf3dSetup`.
    ctf3d_setup: Rc<Run3dmodButton>,
    /// Java private final `useCtf3d`.
    use_ctf3d: Rc<MultiLineButton>,
    // combination
    /// Java private final `createCombine`.
    create_combine: Rc<MultiLineButton>,
    /// Java private final `combine`.
    combine: Rc<Run3dmodButton>,
    /// Java private final `restartCombine`.
    restart_combine: Rc<Run3dmodButton>,
    /// Java private final `restartMatchvol1`.
    restart_matchvol1: Rc<Run3dmodButton>,
    /// Java private final `restartPatchcorr`.
    restart_patchcorr: Rc<Run3dmodButton>,
    /// Java private final `restartMatchorwarp`.
    restart_matchorwarp: Rc<Run3dmodButton>,
    /// Java private final `restartVolcombine`.
    restart_volcombine: Rc<Run3dmodButton>,
    // post processing
    /// Java private final `trimVolume`.
    trim_volume: Rc<Run3dmodButton>,
    /// Java private final `flatten`.
    flatten: Rc<Run3dmodButton>,
    /// Java private final `flattenWarp`.
    flatten_warp: Rc<MultiLineButton>,
    /// Java private final `squeezeVolume`.
    squeeze_volume: Rc<Run3dmodButton>,
    /// Java private final `smoothingAssessment`.
    smoothing_assessment: Rc<Run3dmodButton>,
}

impl Deref for ProcessResultDisplayFactory {
    type Target = AbstractProcessResultDisplayFactory;

    fn deref(&self) -> &AbstractProcessResultDisplayFactory {
        &self.base
    }
}

impl ProcessResultDisplayFactory {
    /// Java private `ProcessResultDisplayFactory(BaseScreenState, AxisID,
    /// AxisType)`, with the field initialisers Java runs after `super(...)`
    /// and before the body, in declaration order.
    fn new(
        screen_state: &'static BaseScreenState,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> ProcessResultDisplayFactory {
        // The factory instance's ID must be unique to the dataset.
        let base = AbstractProcessResultDisplayFactory::new(format!(
            "etomo.ui.swing.ProcessResultDisplayFactory{}",
            if axis_type == AxisType::DualAxis {
                axis_id.to_string()
            } else {
                String::new()
            }
        ));
        let find_x_rays = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Find X-rays (Trial Mode)"),
            Some(DialogType::PreProcessing),
        );
        let create_fixed_stack =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some(ccd_eraser_xrays_panel::ERASE_LABEL),
                Some(DialogType::PreProcessing),
            );
        let use_fixed_stack = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some(ccd_eraser_xrays_panel::USE_FIXED_STACK_LABEL),
            Some(DialogType::PreProcessing),
        );
        let coarse_tiltxcorr = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Calculate Cross-Correlation"),
            Some(DialogType::CoarseAlignment),
        );
        let distortion_corrected_stack =
            MultiLineButton::get_toggle_button_instance_string_dialog_type(
                Some("Make Distortion Corrected Stack"),
                Some(DialogType::CoarseAlignment),
            );
        let fix_edges_midas = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Fix Edges With Midas"),
            Some(DialogType::CoarseAlignment),
        );
        let coarse_align = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Generate Coarse Aligned Stack"),
            Some(DialogType::CoarseAlignment),
        );
        let midas = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Fix Alignment With Midas"),
            Some(DialogType::CoarseAlignment),
        );
        let transfer_fiducials =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some("Transfer Fiducials From Other Axis"),
                Some(DialogType::FiducialModel),
            );
        let raptor = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(raptor_panel::RUN_RAPTOR_LABEL),
            Some(DialogType::FiducialModel),
        );
        let use_raptor = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some(raptor_panel::USE_RAPTOR_RESULT_LABEL),
            Some(DialogType::FiducialModel),
        );
        let seed_fiducial_model = Run3dmodButton::get_toggle_3dmod_instance(
            Some(fiducial_model_dialog::SEEDING_NOT_DONE_LABEL),
            Some(DialogType::FiducialModel),
        );
        let track_fiducials = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some(beadtrack_panel::TRACK_LABEL),
            Some(DialogType::FiducialModel),
        );
        let fix_fiducial_model = Run3dmodButton::get_toggle_3dmod_instance(
            Some("Fix Fiducial Model"),
            Some(DialogType::FiducialModel),
        );
        let track_tiltxcorr = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Track Patches"),
            Some(DialogType::FiducialModel),
        );
        let autofidseed = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(fiducial_model_dialog::AUTOFIDSEED_NEW_MODEL_LABEL),
            Some(DialogType::FiducialModel),
        );
        let just_find_shifts_near_zero =
            MultiLineButton::get_toggle_button_instance_string_dialog_type(
                Some("Run Autofidseed to Find Shifts"),
                Some(DialogType::FiducialModel),
            );
        let imodchopconts = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Recut or Restore Contours"),
            Some(DialogType::FiducialModel),
        );
        let use_adjusted_track_com = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Use Adjusted Track Com File"),
            Some(DialogType::FiducialModel),
        );
        let compute_alignment = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Compute Alignment"),
            Some(DialogType::FineAlignment),
        );
        let sample_tomogram = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(tomogram_positioning_dialog::SAMPLE_TOMOGRAMS_LABEL),
            Some(DialogType::TomogramPositioning),
        );
        let compute_pitch = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Compute Z Shift & Pitch Angles"),
            Some(DialogType::TomogramPositioning),
        );
        let final_alignment = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Create Final Alignment"),
            Some(DialogType::TomogramPositioning),
        );
        let full_aligned_stack =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some(newstack_or_blendmont_panel::RUN_BUTTON_LABEL),
                Some(DialogType::FinalAlignedStack),
            );
        let ctf_correction = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(final_aligned_stack_dialog::CTF_CORRECTION_LABEL),
            Some(DialogType::FinalAlignedStack),
        );
        let use_ctf_correction = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some(final_aligned_stack_dialog::USE_CTF_CORRECTION_LABEL),
            Some(DialogType::FinalAlignedStack),
        );
        let xf_model = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Transform Fiducial Model"),
            Some(DialogType::FinalAlignedStack),
        );
        let stack_tilt = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(tilt3d_find_panel::TILT_3D_FIND_LABEL),
            Some(DialogType::FinalAlignedStack),
        );
        let find_beads3d = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Run Findbeads3d"),
            Some(DialogType::FinalAlignedStack),
        );
        let reproject_model = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(reproject_model_panel::REPROJECT_MODEL_LABEL),
            Some(DialogType::FinalAlignedStack),
        );
        let ccd_eraser_beads =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some(ccd_eraser_beads_panel::CCD_ERASER_LABEL),
                Some(DialogType::FinalAlignedStack),
            );
        let use_ccd_eraser_beads =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some(ccd_eraser_beads_panel::USE_ERASED_STACK_LABEL),
                Some(DialogType::FinalAlignedStack),
            );
        let filter = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Filter"),
            Some(DialogType::FinalAlignedStack),
        );
        let use_filtered_stack = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some(final_aligned_stack_dialog::USE_FILTERED_STACK_LABEL),
            Some(DialogType::FinalAlignedStack),
        );
        let use_trial_tomogram = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Use Current Trial Tomogram"),
            Some(DialogType::TomogramGeneration),
        );
        let gen_tilt = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Generate Tomogram"),
            Some(DialogType::TomogramGeneration),
        );
        let delete_aligned_stack = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Delete Intermediate Image Stacks"),
            Some(DialogType::TomogramGeneration),
        );
        let multifilt_setup = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Run Filter Trials"),
            Some(DialogType::TomogramGeneration),
        );
        let sirtsetup = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Run SIRT"),
            Some(DialogType::TomogramGeneration),
        );
        let use_sirt = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Use SIRT Output File"),
            Some(DialogType::TomogramGeneration),
        );
        let ctf3d_setup = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(ctf3d_panel::RUN_BUTTON_LABEL),
            Some(DialogType::TomogramGeneration),
        );
        let use_ctf3d = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some(ctf3d_panel::USE_BUTTON_LABEL),
            Some(DialogType::TomogramGeneration),
        );
        let create_combine = MultiLineButton::get_toggle_button_instance_string_dialog_type(
            Some("Create Combine Scripts"),
            Some(DialogType::TomogramCombination),
        );
        let combine = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Start Combine"),
            Some(DialogType::TomogramCombination),
        );
        let restart_combine = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Restart Combine"),
            Some(DialogType::TomogramCombination),
        );
        let restart_matchvol1 =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some("Restart at Matchvol1"),
                Some(DialogType::TomogramCombination),
            );
        let restart_patchcorr =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some("Restart at Patchcorr"),
                Some(DialogType::TomogramCombination),
            );
        let restart_matchorwarp =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some("Restart at Matchorwarp"),
                Some(DialogType::TomogramCombination),
            );
        let restart_volcombine =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some("Restart at Volcombine"),
                Some(DialogType::TomogramCombination),
            );
        let trim_volume = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Trim Volume"),
            Some(DialogType::PostProcessing),
        );
        let flatten = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some(flatten_volume_panel::FLATTEN_LABEL),
            Some(DialogType::PostProcessing),
        );
        let flatten_warp =
            MultiLineButton::new_string(Some(flatten_volume_panel::FLATTEN_WARP_LABEL));
        let squeeze_volume = Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
            Some("Reduce/Filter Volume"),
            Some(DialogType::PostProcessing),
        );
        let smoothing_assessment =
            Run3dmodButton::get_deferred_toggle_3dmod_instance_string_dialog_type(
                Some(smoothing_assessment_panel::FLATTEN_WARP_LABEL),
                Some(DialogType::PostProcessing),
            );
        ProcessResultDisplayFactory {
            base,
            screen_state,
            find_x_rays,
            create_fixed_stack,
            use_fixed_stack,
            coarse_tiltxcorr,
            distortion_corrected_stack,
            fix_edges_midas,
            coarse_align,
            midas,
            transfer_fiducials,
            raptor,
            use_raptor,
            seed_fiducial_model,
            track_fiducials,
            fix_fiducial_model,
            track_tiltxcorr,
            autofidseed,
            just_find_shifts_near_zero,
            imodchopconts,
            use_adjusted_track_com,
            compute_alignment,
            sample_tomogram,
            compute_pitch,
            final_alignment,
            full_aligned_stack,
            ctf_correction,
            use_ctf_correction,
            xf_model,
            stack_tilt,
            find_beads3d,
            reproject_model,
            ccd_eraser_beads,
            use_ccd_eraser_beads,
            filter,
            use_filtered_stack,
            use_trial_tomogram,
            gen_tilt,
            delete_aligned_stack,
            multifilt_setup,
            sirtsetup,
            use_sirt,
            ctf3d_setup,
            use_ctf3d,
            create_combine,
            combine,
            restart_combine,
            restart_matchvol1,
            restart_patchcorr,
            restart_matchorwarp,
            restart_volcombine,
            trim_volume,
            flatten,
            flatten_warp,
            squeeze_volume,
            smoothing_assessment,
        }
    }

    /// Java public static `getInstance(BaseScreenState, AxisID, AxisType)`.
    pub fn get_instance(
        screen_state: &'static BaseScreenState,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> Rc<ProcessResultDisplayFactory> {
        let instance = Rc::new(ProcessResultDisplayFactory::new(
            screen_state,
            axis_id,
            axis_type,
        ));
        instance.initialize();
        instance
    }

    /// Java private `initialize()`.
    fn initialize(&self) {
        // The purpose of dependency list is to give every toggle button the
        // correct setting when another button is pressed. The two situations
        // currently handled are:
        // - going back to an earlier step and invalidating the output of a
        //   later step, and
        // - button mirroring.

        // The global dependency list:
        // All displays should be added to this list. The order in which the
        // displays are added affects the behavior.

        // The function of the global dependency list: when a display's process
        // succeeds or fails, the displays that follow it in the list will be
        // unselected.

        // The display ID (second parameter) must be unique to each dependency
        // within this factory. The display ID will be saved to files as
        // process data, so it must be the same each time etomo runs. However,
        // because process data is short-lived, this ID does not have to be
        // stable from version to version.
        let mut id = 0;
        // preprocessing
        self.add_dependency(
            Some(&(self.find_x_rays.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.create_fixed_stack.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_fixed_stack.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // coarse alignment
        self.add_dependency(
            Some(&(self.coarse_tiltxcorr.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.distortion_corrected_stack.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.fix_edges_midas.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.coarse_align.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.midas.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // fiducial model
        self.add_dependency(
            Some(&(self.transfer_fiducials.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.seed_fiducial_model.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.autofidseed.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.just_find_shifts_near_zero.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // Dependency depends on whether the adjusted track comfile exists.
        // addDependency(useAdjustedTrackCom, id++);
        self.add_dependency(
            Some(&(self.track_tiltxcorr.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.imodchopconts.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.raptor.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_raptor.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.track_fiducials.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.fix_fiducial_model.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // fine alignment
        self.add_dependency(
            Some(&(self.compute_alignment.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // positioning
        self.add_dependency(
            Some(&(self.sample_tomogram.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.compute_pitch.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.final_alignment.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // stack
        self.add_dependency(
            Some(&(self.full_aligned_stack.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.ctf_correction.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_ctf_correction.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.xf_model.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.stack_tilt.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.find_beads3d.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.reproject_model.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.ccd_eraser_beads.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_ccd_eraser_beads.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.filter.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_filtered_stack.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // generation
        self.add_dependency(
            Some(&(self.gen_tilt.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_trial_tomogram.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.delete_aligned_stack.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.multifilt_setup.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.sirtsetup.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.ctf3d_setup.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_sirt.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.use_ctf3d.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // combination
        self.add_dependency(
            Some(&(self.create_combine.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.combine.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.restart_combine.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.restart_matchvol1.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.restart_patchcorr.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.restart_matchorwarp.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.restart_volcombine.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        // post processing
        self.add_dependency(
            Some(&(self.trim_volume.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.smoothing_assessment.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.flatten_warp.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.flatten.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;
        self.add_dependency(
            Some(&(self.squeeze_volume.clone() as ProcessResultDisplayHandle)),
            id,
        );
        id += 1;

        // Turning off the use of the global dependency list for individual
        // displays.  This is mostly used for buttons that create temporary or
        // optional files. Temporary file creaters are usually paired with a
        // "Use..." button, which will still use the global list.

        // These displays are still unselected by previous displays, since they
        // are dependent on the output of these displays. This is why they
        // where added to the global list.

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

        //
        // The display level lists:
        // For these lists, order does not count.
        //

        // The dependency list:
        // Works like the global list: when a display's process succeeds or
        // fails, all displays on the display's list are unselected. This can
        // be used to allow temporary and optional file creaters to affect only
        // the buttons that they are teamed with.

        // coarse alignment
        self.coarse_align.add_dependent_display(
            self.just_find_shifts_near_zero.clone() as ProcessResultDisplayHandle
        );
        // fiducial model
        self.raptor
            .add_dependent_display(self.use_raptor.clone() as ProcessResultDisplayHandle);
        // stack
        self.ctf_correction
            .add_dependent_display(self.use_ctf_correction.clone() as ProcessResultDisplayHandle);
        self.xf_model
            .add_dependent_display(self.ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.xf_model
            .add_dependent_display(self.use_ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.stack_tilt
            .add_dependent_display(self.find_beads3d.clone() as ProcessResultDisplayHandle);
        self.stack_tilt
            .add_dependent_display(self.reproject_model.clone() as ProcessResultDisplayHandle);
        self.stack_tilt
            .add_dependent_display(self.ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.stack_tilt
            .add_dependent_display(self.use_ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.find_beads3d
            .add_dependent_display(self.reproject_model.clone() as ProcessResultDisplayHandle);
        self.find_beads3d
            .add_dependent_display(self.ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.find_beads3d
            .add_dependent_display(self.use_ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.reproject_model
            .add_dependent_display(self.ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.reproject_model
            .add_dependent_display(self.use_ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.ccd_eraser_beads
            .add_dependent_display(self.use_ccd_eraser_beads.clone() as ProcessResultDisplayHandle);
        self.filter
            .add_dependent_display(self.use_filtered_stack.clone() as ProcessResultDisplayHandle);
        // gen
        self.sirtsetup
            .add_dependent_display(self.use_sirt.clone() as ProcessResultDisplayHandle);
        self.delete_aligned_stack.add_dependent_display(Some(
            self.full_aligned_stack.clone() as ProcessResultDisplayHandle
        ));
        // post processing
        self.flatten_warp
            .add_dependent_display(Some(self.flatten.clone() as ProcessResultDisplayHandle));

        // Display mirroring:

        // The failure list: when the process fails (and is therefore
        // unselected), all displays on this list are unselected.

        // The success list: when the process succeeds (and is selected), all
        // displays on this list are selected.

        // gen
        // Creating a trial tomogram and then using it is equivalent to running
        // genTilt.
        self.use_trial_tomogram
            .add_success_display(Some(self.gen_tilt.clone() as ProcessResultDisplayHandle));
        // combination
        // Completely mirror combine and restartCombine
        self.combine
            .add_failure_display(self.restart_combine.clone() as ProcessResultDisplayHandle);
        self.combine
            .add_success_display(self.restart_combine.clone() as ProcessResultDisplayHandle);
        self.restart_combine
            .add_failure_display(self.combine.clone() as ProcessResultDisplayHandle);
        self.restart_combine
            .add_success_display(self.combine.clone() as ProcessResultDisplayHandle);

        // Setting display states:
        // Display states are generically saved and loaded from the .edf file
        // using screenState.

        // preprocessing
        self.find_x_rays.set_screen_state(self.screen_state);
        self.create_fixed_stack.set_screen_state(self.screen_state);
        self.use_fixed_stack.set_screen_state(self.screen_state);
        // coarse alignment
        self.coarse_tiltxcorr.set_screen_state(self.screen_state);
        self.distortion_corrected_stack
            .set_screen_state(self.screen_state);
        self.fix_edges_midas.set_screen_state(self.screen_state);
        self.coarse_align.set_screen_state(self.screen_state);
        self.midas.set_screen_state(self.screen_state);
        // fiducial model
        self.transfer_fiducials.set_screen_state(self.screen_state);
        self.track_tiltxcorr.set_screen_state(self.screen_state);
        self.imodchopconts.set_screen_state(self.screen_state);
        self.raptor.set_screen_state(self.screen_state);
        self.use_raptor.set_screen_state(self.screen_state);
        self.seed_fiducial_model.set_screen_state(self.screen_state);
        self.autofidseed.set_screen_state(self.screen_state);
        self.just_find_shifts_near_zero
            .set_screen_state(self.screen_state);
        self.use_adjusted_track_com
            .set_screen_state(self.screen_state);
        self.track_fiducials.set_screen_state(self.screen_state);
        self.fix_fiducial_model.set_screen_state(self.screen_state);
        // fine alignment
        self.compute_alignment.set_screen_state(self.screen_state);
        // positioning
        self.sample_tomogram.set_screen_state(self.screen_state);
        self.compute_pitch.set_screen_state(self.screen_state);
        self.final_alignment.set_screen_state(self.screen_state);
        // stack
        self.full_aligned_stack.set_screen_state(self.screen_state);
        self.ctf_correction.set_screen_state(self.screen_state);
        self.use_ctf_correction.set_screen_state(self.screen_state);
        self.xf_model.set_screen_state(self.screen_state);
        self.stack_tilt.set_screen_state(self.screen_state);
        self.find_beads3d.set_screen_state(self.screen_state);
        self.reproject_model.set_screen_state(self.screen_state);
        self.ccd_eraser_beads.set_screen_state(self.screen_state);
        self.use_ccd_eraser_beads
            .set_screen_state(self.screen_state);
        self.filter.set_screen_state(self.screen_state);
        self.use_filtered_stack.set_screen_state(self.screen_state);
        // generation
        self.use_trial_tomogram.set_screen_state(self.screen_state);
        self.gen_tilt.set_screen_state(self.screen_state);
        self.delete_aligned_stack
            .set_screen_state(self.screen_state);
        self.multifilt_setup.set_screen_state(self.screen_state);
        self.sirtsetup.set_screen_state(self.screen_state);
        self.ctf3d_setup.set_screen_state(self.screen_state);
        self.use_sirt.set_screen_state(self.screen_state);
        self.use_ctf3d.set_screen_state(self.screen_state);
        // combination
        self.create_combine.set_screen_state(self.screen_state);
        self.combine.set_screen_state(self.screen_state);
        self.restart_combine.set_screen_state(self.screen_state);
        self.restart_matchvol1.set_screen_state(self.screen_state);
        self.restart_patchcorr.set_screen_state(self.screen_state);
        self.restart_matchorwarp.set_screen_state(self.screen_state);
        self.restart_volcombine.set_screen_state(self.screen_state);
        // post processing
        self.trim_volume.set_screen_state(self.screen_state);
        self.smoothing_assessment
            .set_screen_state(self.screen_state);
        self.flatten_warp.set_screen_state(self.screen_state);
        self.flatten.set_screen_state(self.screen_state);
        self.squeeze_volume.set_screen_state(self.screen_state);
    }

    // preprocessing

    /// Java package-private `getFindXRays()`.
    pub fn get_find_xrays(&self) -> Rc<Run3dmodButton> {
        self.find_x_rays.clone()
    }

    /// Java package-private `getCreateFixedStack()`.
    pub fn get_create_fixed_stack(&self) -> Rc<Run3dmodButton> {
        self.create_fixed_stack.clone()
    }

    /// Java package-private `getUseFixedStack()`.
    pub fn get_use_fixed_stack(&self) -> Rc<MultiLineButton> {
        self.use_fixed_stack.clone()
    }

    // coarse alignment

    /// Java package-private `getTiltxcorr(DialogType)`.  The two candidates
    /// are of different classes (`coarseTiltxcorr` a `MultiLineButton`,
    /// `trackTiltxcorr` a `Run3dmodButton`), so the Java's declared
    /// `ProcessResultDisplay` is returned.
    pub fn get_tiltxcorr(&self, dialog_type: DialogType) -> Option<ProcessResultDisplayHandle> {
        if dialog_type == DialogType::CoarseAlignment {
            return Some(self.coarse_tiltxcorr.clone() as ProcessResultDisplayHandle);
        } else if dialog_type == DialogType::FiducialModel {
            return Some(self.track_tiltxcorr.clone() as ProcessResultDisplayHandle);
        }
        None
    }

    /// Java package-private `getImodchopconts()`.
    pub fn get_imodchopconts(&self) -> Rc<Run3dmodButton> {
        self.imodchopconts.clone()
    }

    /// Java package-private `getAutofidseed()`.
    pub fn get_autofidseed(&self) -> Rc<Run3dmodButton> {
        self.autofidseed.clone()
    }

    /// Java package-private `getJustFindShiftsNearZero()`.
    pub fn get_just_find_shifts_near_zero(&self) -> Rc<MultiLineButton> {
        self.just_find_shifts_near_zero.clone()
    }

    /// Java package-private `getUseAdjustedTrackCom()`.
    pub fn get_use_adjusted_track_com(&self) -> Rc<MultiLineButton> {
        self.use_adjusted_track_com.clone()
    }

    /// Java package-private `getDistortionCorrectedStack()`.
    pub fn get_distortion_corrected_stack(&self) -> Rc<MultiLineButton> {
        self.distortion_corrected_stack.clone()
    }

    /// Java package-private `getFixEdgesMidas()`.
    pub fn get_fix_edges_midas(&self) -> Rc<MultiLineButton> {
        self.fix_edges_midas.clone()
    }

    /// Java package-private `getCoarseAlign()`.
    pub fn get_coarse_align(&self) -> Rc<Run3dmodButton> {
        self.coarse_align.clone()
    }

    /// Java package-private `getMidas()`.
    pub fn get_midas(&self) -> Rc<MultiLineButton> {
        self.midas.clone()
    }

    // fiducial model

    /// Java package-private `getTransferFiducials()`.
    pub fn get_transfer_fiducials(&self) -> Rc<Run3dmodButton> {
        self.transfer_fiducials.clone()
    }

    /// Java package-private `getRaptor()`.
    pub fn get_raptor(&self) -> Rc<Run3dmodButton> {
        self.raptor.clone()
    }

    /// Java package-private `getUseRaptor()`.
    pub fn get_use_raptor(&self) -> Rc<MultiLineButton> {
        self.use_raptor.clone()
    }

    /// Java package-private `getSeedFiducialModel()`.
    pub fn get_seed_fiducial_model(&self) -> Rc<Run3dmodButton> {
        self.seed_fiducial_model.clone()
    }

    /// Java package-private `getTrackFiducials()`.
    pub fn get_track_fiducials(&self) -> Rc<MultiLineButton> {
        self.track_fiducials.clone()
    }

    /// Java package-private `getFixFiducialModel()`.
    pub fn get_fix_fiducial_model(&self) -> Rc<Run3dmodButton> {
        self.fix_fiducial_model.clone()
    }

    // fine alignment

    /// Java public `getComputeAlignment()`.
    pub fn get_compute_alignment(&self) -> Rc<MultiLineButton> {
        self.compute_alignment.clone()
    }

    // positioning

    /// Java package-private `getSampleTomogram()`.
    pub fn get_sample_tomogram(&self) -> Rc<Run3dmodButton> {
        self.sample_tomogram.clone()
    }

    /// Java package-private `getComputePitch()`.
    pub fn get_compute_pitch(&self) -> Rc<MultiLineButton> {
        self.compute_pitch.clone()
    }

    /// Java package-private `getFinalAlignment()`.
    pub fn get_final_alignment(&self) -> Rc<MultiLineButton> {
        self.final_alignment.clone()
    }

    // stack

    /// Java package-private `getFullAlignedStack()`.
    pub fn get_full_aligned_stack(&self) -> Rc<Run3dmodButton> {
        self.full_aligned_stack.clone()
    }

    /// Java public `getCtfCorrection()`.
    pub fn get_ctf_correction(&self) -> Rc<Run3dmodButton> {
        self.ctf_correction.clone()
    }

    /// Java package-private `getUseCtfCorrection()`.
    pub fn get_use_ctf_correction(&self) -> Rc<MultiLineButton> {
        self.use_ctf_correction.clone()
    }

    /// Java package-private `getXfModel()`.
    pub fn get_xf_model(&self) -> Rc<Run3dmodButton> {
        self.xf_model.clone()
    }

    /// Java package-private `getFindBeads3d()`.
    pub fn get_find_beads3d(&self) -> Rc<Run3dmodButton> {
        self.find_beads3d.clone()
    }

    /// Java public `getTilt(DialogType)`.
    pub fn get_tilt(&self, dialog_type: DialogType) -> Rc<Run3dmodButton> {
        if dialog_type == DialogType::FinalAlignedStack {
            return self.stack_tilt.clone();
        }
        self.gen_tilt.clone()
    }

    /// Java package-private `getMultifiltSetup()`.
    pub fn get_multifilt_setup(&self) -> Rc<Run3dmodButton> {
        self.multifilt_setup.clone()
    }

    /// Java package-private `getSirtsetup()`.
    pub fn get_sirtsetup(&self) -> Rc<Run3dmodButton> {
        self.sirtsetup.clone()
    }

    /// Java public `getUseSirt()`.
    pub fn get_use_sirt(&self) -> Rc<MultiLineButton> {
        self.use_sirt.clone()
    }

    /// Java package-private `getCtf3dSetup()`.
    pub fn get_ctf3d_setup(&self) -> Rc<Run3dmodButton> {
        self.ctf3d_setup.clone()
    }

    /// Java package-private `getUseCtf3d()`.
    pub fn get_use_ctf3d(&self) -> Rc<MultiLineButton> {
        self.use_ctf3d.clone()
    }

    /// Java package-private `getReprojectModel()`.
    pub fn get_reproject_model(&self) -> Rc<Run3dmodButton> {
        self.reproject_model.clone()
    }

    /// Java public `getCcdEraserBeads()`.
    pub fn get_ccd_eraser_beads(&self) -> Rc<Run3dmodButton> {
        self.ccd_eraser_beads.clone()
    }

    /// Java package-private `getUseCcdEraserBeads()`.
    pub fn get_use_ccd_eraser_beads(&self) -> Rc<Run3dmodButton> {
        self.use_ccd_eraser_beads.clone()
    }

    /// Java public `getFilter()`.
    pub fn get_filter(&self) -> Rc<Run3dmodButton> {
        self.filter.clone()
    }

    /// Java public `getUseFilteredStack()`.
    pub fn get_use_filtered_stack(&self) -> Rc<MultiLineButton> {
        self.use_filtered_stack.clone()
    }

    // generation

    /// Java package-private `getUseTrialTomogram()`.
    pub fn get_use_trial_tomogram(&self) -> Rc<MultiLineButton> {
        self.use_trial_tomogram.clone()
    }

    /// Java package-private `getDeleteAlignedStack()`.
    pub fn get_delete_aligned_stack(&self) -> Rc<MultiLineButton> {
        self.delete_aligned_stack.clone()
    }

    // combination

    /// Java package-private `getCreateCombine()`.
    pub fn get_create_combine(&self) -> Rc<MultiLineButton> {
        self.create_combine.clone()
    }

    /// Java package-private `getCombine()`.
    pub fn get_combine(&self) -> Rc<Run3dmodButton> {
        self.combine.clone()
    }

    /// Java package-private `getRestartCombine()`.
    pub fn get_restart_combine(&self) -> Rc<Run3dmodButton> {
        self.restart_combine.clone()
    }

    /// Java public `getRestartMatchvol1()`.
    pub fn get_restart_matchvol1(&self) -> Rc<Run3dmodButton> {
        self.restart_matchvol1.clone()
    }

    /// Java public `getRestartPatchcorr()`.
    pub fn get_restart_patchcorr(&self) -> Rc<Run3dmodButton> {
        self.restart_patchcorr.clone()
    }

    /// Java public `getRestartMatchorwarp()`.
    pub fn get_restart_matchorwarp(&self) -> Rc<Run3dmodButton> {
        self.restart_matchorwarp.clone()
    }

    /// Java public `getRestartVolcombine()`.
    pub fn get_restart_volcombine(&self) -> Rc<Run3dmodButton> {
        self.restart_volcombine.clone()
    }

    // post processing

    /// Java package-private `getTrimVolume()`.
    pub fn get_trim_volume(&self) -> Rc<Run3dmodButton> {
        self.trim_volume.clone()
    }

    /// Java package-private `getFlatten()`.
    pub fn get_flatten(&self) -> Rc<Run3dmodButton> {
        self.flatten.clone()
    }

    /// Java package-private `getFlattenWarp()`.
    pub fn get_flatten_warp(&self) -> Rc<MultiLineButton> {
        self.flatten_warp.clone()
    }

    /// Java package-private `getSmoothingAssessment()`.
    pub fn get_smoothing_assessment(&self) -> Rc<Run3dmodButton> {
        self.smoothing_assessment.clone()
    }

    /// Java package-private `getSqueezeVolume()`.
    pub fn get_squeeze_volume(&self) -> Rc<Run3dmodButton> {
        self.squeeze_volume.clone()
    }
}
