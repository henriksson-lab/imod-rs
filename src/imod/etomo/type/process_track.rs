//! `IMOD/Etomo/src/etomo/type/ProcessTrack.java`.
//!
//! The per-dialog progress of a reconstruction (not started, in progress, complete),
//! stored in the `.edf` under the `ProcessTrack` group.
//!
//! Java hands the one `ProcessTrack` to the event dispatch thread and to process
//! threads, so each field carries its own lock and every method takes `&self`, as in
//! `base_meta_data.rs`.
//!
//! **`prepend == ""`.**  `store`/`load` test `prepend == ""`, a reference comparison
//! true for the interned literal that `store(Properties)`/`load(Properties)` pass; it is
//! translated as `prepend.is_empty()`.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_process_track::BaseProcessTrack;
use super::const_etomo_number::java_lang_double_value_of;
use super::dialog_type::DialogType;
use crate::imod::etomo::process::process_state::ProcessState;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};
use crate::imod::etomo::ui::swing::abstract_parallel_dialog::AbstractParallelDialog;

/// Java `rcsid`.
pub const RCSID: &str = "$Id$";

/// Java `ProcessTrack`.
pub struct ProcessTrack {
    /// Java field `setup`, initialised to `ProcessState.NOTSTARTED`.
    setup: Mutex<ProcessState>,
    /// Java field `preProcessingA`, initialised to `ProcessState.NOTSTARTED`.
    pre_processing_a: Mutex<ProcessState>,
    /// Java field `coarseAlignmentA`, initialised to `ProcessState.NOTSTARTED`.
    coarse_alignment_a: Mutex<ProcessState>,
    /// Java field `fiducialModelA`, initialised to `ProcessState.NOTSTARTED`.
    fiducial_model_a: Mutex<ProcessState>,
    /// Java field `fineAlignmentA`, initialised to `ProcessState.NOTSTARTED`.
    fine_alignment_a: Mutex<ProcessState>,
    /// Java field `tomogramPositioningA`, initialised to `ProcessState.NOTSTARTED`.
    tomogram_positioning_a: Mutex<ProcessState>,
    /// Java field `finalAlignedStackA`, initialised to `ProcessState.NOTSTARTED`.
    final_aligned_stack_a: Mutex<ProcessState>,
    /// Java field `tomogramGenerationA`, initialised to `ProcessState.NOTSTARTED`.
    tomogram_generation_a: Mutex<ProcessState>,
    /// Java field `tomogramCombination`, initialised to `ProcessState.NOTSTARTED`.
    tomogram_combination: Mutex<ProcessState>,
    /// Java field `postProcessing`, initialised to `ProcessState.NOTSTARTED`.
    post_processing: Mutex<ProcessState>,
    /// Java field `cleanUp`, initialised to `ProcessState.NOTSTARTED`.
    clean_up: Mutex<ProcessState>,
    /// Java field `preProcessingB`, initialised to `ProcessState.NOTSTARTED`.
    pre_processing_b: Mutex<ProcessState>,
    /// Java field `coarseAlignmentB`, initialised to `ProcessState.NOTSTARTED`.
    coarse_alignment_b: Mutex<ProcessState>,
    /// Java field `fiducialModelB`, initialised to `ProcessState.NOTSTARTED`.
    fiducial_model_b: Mutex<ProcessState>,
    /// Java field `fineAlignmentB`, initialised to `ProcessState.NOTSTARTED`.
    fine_alignment_b: Mutex<ProcessState>,
    /// Java field `tomogramPositioningB`, initialised to `ProcessState.NOTSTARTED`.
    tomogram_positioning_b: Mutex<ProcessState>,
    /// Java field `finalAlignedStackB`, initialised to `ProcessState.NOTSTARTED`.
    final_aligned_stack_b: Mutex<ProcessState>,
    /// Java field `tomogramGenerationB`, initialised to `ProcessState.NOTSTARTED`.
    tomogram_generation_b: Mutex<ProcessState>,
    /// Java package-private field `revisionNumber`.
    revision_number: Mutex<String>,
    /// Java package-private field `isModified`, initialised to false.
    is_modified: Mutex<bool>,
}

impl ProcessTrack {
    /// Java `ProcessTrack()`.
    pub fn new() -> ProcessTrack {
        ProcessTrack {
            setup: Mutex::new(ProcessState::NotStarted),
            pre_processing_a: Mutex::new(ProcessState::NotStarted),
            coarse_alignment_a: Mutex::new(ProcessState::NotStarted),
            fiducial_model_a: Mutex::new(ProcessState::NotStarted),
            fine_alignment_a: Mutex::new(ProcessState::NotStarted),
            tomogram_positioning_a: Mutex::new(ProcessState::NotStarted),
            final_aligned_stack_a: Mutex::new(ProcessState::NotStarted),
            tomogram_generation_a: Mutex::new(ProcessState::NotStarted),
            tomogram_combination: Mutex::new(ProcessState::NotStarted),
            post_processing: Mutex::new(ProcessState::NotStarted),
            clean_up: Mutex::new(ProcessState::NotStarted),
            pre_processing_b: Mutex::new(ProcessState::NotStarted),
            coarse_alignment_b: Mutex::new(ProcessState::NotStarted),
            fiducial_model_b: Mutex::new(ProcessState::NotStarted),
            fine_alignment_b: Mutex::new(ProcessState::NotStarted),
            tomogram_positioning_b: Mutex::new(ProcessState::NotStarted),
            final_aligned_stack_b: Mutex::new(ProcessState::NotStarted),
            tomogram_generation_b: Mutex::new(ProcessState::NotStarted),
            revision_number: Mutex::new("2.0".to_string()),
            is_modified: Mutex::new(false),
        }
    }

    /// Java `store(Properties)`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `load(Properties)`.
    pub fn load(&self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.  Insert the objects attributes into the
    /// properties object.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let group;
        if prepend.is_empty() {
            group = "ProcessTrack.".to_string();
        } else {
            group = format!("{}.ProcessTrack.", prepend);
        }
        let revision_number = self.revision_number.lock().unwrap().clone();
        props.insert(format!("{}RevisionNumber", group), revision_number);
        props.insert(
            format!("{}Setup", group),
            self.setup.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}PreProcessing-A", group),
            self.pre_processing_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}PreProcessing-B", group),
            self.pre_processing_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}CoarseAlignment-A", group),
            self.coarse_alignment_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}CoarseAlignment-B", group),
            self.coarse_alignment_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}FiducialModel-A", group),
            self.fiducial_model_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}FiducialModel-B", group),
            self.fiducial_model_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}FineAlignment-A", group),
            self.fine_alignment_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}FineAlignment-B", group),
            self.fine_alignment_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}TomogramPositioning-A", group),
            self.tomogram_positioning_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}TomogramPositioning-B", group),
            self.tomogram_positioning_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}FinalAlignedStack-A", group),
            self.final_aligned_stack_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}FinalAlignedStack-B", group),
            self.final_aligned_stack_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}TomogramGeneration-A", group),
            self.tomogram_generation_a.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}TomogramGeneration-B", group),
            self.tomogram_generation_b.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}TomogramCombination", group),
            self.tomogram_combination.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}PostProcessing", group),
            self.post_processing.lock().unwrap().to_string(),
        );
        props.insert(
            format!("{}CleanUp", group),
            self.clean_up.lock().unwrap().to_string(),
        );
    }

    /// Java `load(Properties, String)`.  Get the objects attributes from the properties
    /// object.
    ///
    /// Upstream bugs fixed in translation:
    /// - ProcessTrack.java:137-...: `ProcessState.fromString` returns null for an
    ///   unrecognised value, and the next `store` throws `NullPointerException` on
    ///   `toString()`.  An unrecognised value now loads as the property's own default,
    ///   "Not started".
    /// - ProcessTrack.java:147: `Double.parseDouble(revisionNumber)` throws
    ///   `NumberFormatException` out of the whole data-file load for a malformed
    ///   `RevisionNumber`.  A malformed value is now treated as the default revision
    ///   "1.0" (the pre-2.0 layout).
    pub fn load_with_prepend(&self, props: &BTreeMap<String, String>, prepend: &str) {
        let group;
        if prepend.is_empty() {
            group = "ProcessTrack.".to_string();
        } else {
            group = format!("{}.ProcessTrack.", prepend);
        }

        *self.revision_number.lock().unwrap() = props
            .get(&format!("{}RevisionNumber", group))
            .cloned()
            .unwrap_or("1.0".to_string());
        *self.setup.lock().unwrap() = ProcessState::from_string(Some(
            props
                .get(&format!("{}Setup", group))
                .map(|s| s.as_str())
                .unwrap_or("Not started"),
        ))
        .unwrap_or(ProcessState::NotStarted);
        *self.tomogram_combination.lock().unwrap() = ProcessState::from_string(Some(
            props
                .get(&format!("{}TomogramCombination", group))
                .map(|s| s.as_str())
                .unwrap_or("Not started"),
        ))
        .unwrap_or(ProcessState::NotStarted);
        *self.post_processing.lock().unwrap() = ProcessState::from_string(Some(
            props
                .get(&format!("{}PostProcessing", group))
                .map(|s| s.as_str())
                .unwrap_or("Not started"),
        ))
        .unwrap_or(ProcessState::NotStarted);
        *self.clean_up.lock().unwrap() = ProcessState::from_string(Some(
            props
                .get(&format!("{}CleanUp", group))
                .map(|s| s.as_str())
                .unwrap_or("Not started"),
        ))
        .unwrap_or(ProcessState::NotStarted);

        // Added separate process for A and B axis for 2.0 layout
        let revision_number = self.revision_number.lock().unwrap().clone();
        if java_lang_double_value_of(&revision_number).unwrap_or(1.0) < 2.0 {
            *self.pre_processing_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}PreProcessing", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.pre_processing_b.lock().unwrap() = *self.pre_processing_a.lock().unwrap();
            *self.coarse_alignment_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}CoarseAlignment", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.coarse_alignment_b.lock().unwrap() = *self.coarse_alignment_a.lock().unwrap();
            *self.fiducial_model_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FiducialModel", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.fiducial_model_b.lock().unwrap() = *self.fiducial_model_a.lock().unwrap();
            *self.fine_alignment_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FineAlignment", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.fine_alignment_b.lock().unwrap() = *self.fine_alignment_a.lock().unwrap();
            *self.tomogram_positioning_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}TomogramPositioning", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.tomogram_positioning_b.lock().unwrap() =
                *self.tomogram_positioning_a.lock().unwrap();
            *self.final_aligned_stack_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FinalAlignedStack", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.final_aligned_stack_b.lock().unwrap() =
                *self.final_aligned_stack_a.lock().unwrap();
            *self.tomogram_generation_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}TomogramGeneration", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.tomogram_generation_b.lock().unwrap() =
                *self.tomogram_generation_a.lock().unwrap();
        } else {
            *self.pre_processing_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}PreProcessing-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.pre_processing_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}PreProcessing-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.coarse_alignment_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}CoarseAlignment-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.coarse_alignment_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}CoarseAlignment-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.fiducial_model_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FiducialModel-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.fiducial_model_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FiducialModel-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.fine_alignment_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FineAlignment-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.fine_alignment_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FineAlignment-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.tomogram_positioning_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}TomogramPositioning-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.tomogram_positioning_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}TomogramPositioning-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.final_aligned_stack_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FinalAlignedStack-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.final_aligned_stack_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}FinalAlignedStack-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.tomogram_generation_a.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}TomogramGeneration-A", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
            *self.tomogram_generation_b.lock().unwrap() = ProcessState::from_string(Some(
                props
                    .get(&format!("{}TomogramGeneration-B", group))
                    .map(|s| s.as_str())
                    .unwrap_or("Not started"),
            ))
            .unwrap_or(ProcessState::NotStarted);
        }
    }

    /// Java `setAll`.  Set all processes to the specified state.
    pub fn set_all(&self, state: ProcessState) {
        *self.setup.lock().unwrap() = state;
        *self.tomogram_combination.lock().unwrap() = state;
        *self.post_processing.lock().unwrap() = state;
        *self.clean_up.lock().unwrap() = state;
        *self.pre_processing_a.lock().unwrap() = state;
        *self.coarse_alignment_a.lock().unwrap() = state;
        *self.fiducial_model_a.lock().unwrap() = state;
        *self.fine_alignment_a.lock().unwrap() = state;
        *self.tomogram_positioning_a.lock().unwrap() = state;
        *self.final_aligned_stack_a.lock().unwrap() = state;
        *self.tomogram_generation_a.lock().unwrap() = state;
        *self.pre_processing_b.lock().unwrap() = state;
        *self.coarse_alignment_b.lock().unwrap() = state;
        *self.fiducial_model_b.lock().unwrap() = state;
        *self.fine_alignment_b.lock().unwrap() = state;
        *self.tomogram_positioning_b.lock().unwrap() = state;
        *self.final_aligned_stack_b.lock().unwrap() = state;
        *self.tomogram_generation_b.lock().unwrap() = state;
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `setSetupState`.  Set the setup state.
    pub fn set_setup_state(&self, state: ProcessState) {
        *self.setup.lock().unwrap() = state;
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getSetupState`.
    pub fn get_setup_state(&self) -> ProcessState {
        *self.setup.lock().unwrap()
    }

    /// Java final `setState(ProcessState, AxisID, DialogType)`.
    pub fn set_state_dialog_type(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        dialog_type: DialogType,
    ) {
        if dialog_type == DialogType::CleanUp {
            self.set_clean_up_state(process_state);
        } else if dialog_type == DialogType::CoarseAlignment {
            self.set_coarse_alignment_state(process_state, axis_id);
        } else if dialog_type == DialogType::FiducialModel {
            self.set_fiducial_model_state(process_state, axis_id);
        } else if dialog_type == DialogType::FineAlignment {
            self.set_fine_alignment_state(process_state, axis_id);
        } else if dialog_type == DialogType::PostProcessing {
            self.set_post_processing_state(process_state);
        } else if dialog_type == DialogType::PreProcessing {
            self.set_pre_processing_state(process_state, axis_id);
        } else if dialog_type == DialogType::SetupRecon {
            self.set_setup_state(process_state);
        } else if dialog_type == DialogType::TomogramCombination {
            self.set_tomogram_combination_state(process_state);
        } else if dialog_type == DialogType::FinalAlignedStack {
            self.set_final_aligned_stack_state(process_state, axis_id);
        } else if dialog_type == DialogType::TomogramGeneration {
            self.set_tomogram_generation_state(process_state, axis_id);
        } else if dialog_type == DialogType::TomogramPositioning {
            self.set_tomogram_positioning_state(process_state, axis_id);
        }
    }

    /// Java `setPreProcessingState`.
    pub fn set_pre_processing_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.pre_processing_b.lock().unwrap() = state;
        } else {
            *self.pre_processing_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getPreProcessingState`.
    pub fn get_pre_processing_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.pre_processing_b.lock().unwrap();
        }
        *self.pre_processing_a.lock().unwrap()
    }

    /// Java `setCoarseAlignmentState`.
    pub fn set_coarse_alignment_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.coarse_alignment_b.lock().unwrap() = state;
        } else {
            *self.coarse_alignment_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getCoarseAlignmentState`.
    pub fn get_coarse_alignment_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.coarse_alignment_b.lock().unwrap();
        }
        *self.coarse_alignment_a.lock().unwrap()
    }

    /// Java `setFiducialModelState`.
    pub fn set_fiducial_model_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.fiducial_model_b.lock().unwrap() = state;
        } else {
            *self.fiducial_model_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getFiducialModelState`.
    pub fn get_fiducial_model_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.fiducial_model_b.lock().unwrap();
        }
        *self.fiducial_model_a.lock().unwrap()
    }

    /// Java `setFineAlignmentState`.
    pub fn set_fine_alignment_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.fine_alignment_b.lock().unwrap() = state;
        } else {
            *self.fine_alignment_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getFineAlignmentState`.
    pub fn get_fine_alignment_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.fine_alignment_b.lock().unwrap();
        }
        *self.fine_alignment_a.lock().unwrap()
    }

    /// Java `setTomogramPositioningState`.
    pub fn set_tomogram_positioning_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.tomogram_positioning_b.lock().unwrap() = state;
        } else {
            *self.tomogram_positioning_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getTomogramPositioningState`.
    pub fn get_tomogram_positioning_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.tomogram_positioning_b.lock().unwrap();
        }
        *self.tomogram_positioning_a.lock().unwrap()
    }

    /// Java `setFinalAlignedStackState`.
    pub fn set_final_aligned_stack_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.final_aligned_stack_b.lock().unwrap() = state;
        } else {
            *self.final_aligned_stack_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getFinalAlignedStackState`.
    pub fn get_final_aligned_stack_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.final_aligned_stack_b.lock().unwrap();
        }
        *self.final_aligned_stack_a.lock().unwrap()
    }

    /// Java `setTomogramGenerationState`.
    pub fn set_tomogram_generation_state(&self, state: ProcessState, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.tomogram_generation_b.lock().unwrap() = state;
        } else {
            *self.tomogram_generation_a.lock().unwrap() = state;
        }
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getTomogramGenerationState`.
    pub fn get_tomogram_generation_state(&self, axis_id: AxisID) -> ProcessState {
        if axis_id == AxisID::Second {
            return *self.tomogram_generation_b.lock().unwrap();
        }
        *self.tomogram_generation_a.lock().unwrap()
    }

    /// Java `setTomogramCombinationState`.
    pub fn set_tomogram_combination_state(&self, state: ProcessState) {
        *self.tomogram_combination.lock().unwrap() = state;
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getTomogramCombinationState`.
    pub fn get_tomogram_combination_state(&self) -> ProcessState {
        *self.tomogram_combination.lock().unwrap()
    }

    /// Java `setPostProcessingState`.
    pub fn set_post_processing_state(&self, state: ProcessState) {
        *self.post_processing.lock().unwrap() = state;
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getPostProcessingState`.
    pub fn get_post_processing_state(&self) -> ProcessState {
        *self.post_processing.lock().unwrap()
    }

    /// Java `setCleanUpState`.
    pub fn set_clean_up_state(&self, state: ProcessState) {
        *self.clean_up.lock().unwrap() = state;
        *self.is_modified.lock().unwrap() = true;
    }

    /// Java `getCleanUpState`.
    pub fn get_clean_up_state(&self) -> ProcessState {
        *self.clean_up.lock().unwrap()
    }

    /// Java private `mapAxis`.
    fn map_axis(
        &self,
        axis_id: AxisID,
        process_state_a: ProcessState,
        process_state_b: ProcessState,
    ) -> ProcessState {
        if axis_id == AxisID::Second {
            return process_state_b;
        }
        process_state_a
    }

    /// Java `printState`.
    pub fn print_state(&self, r#type: AxisType) {
        if r#type == AxisType::SingleAxis {
            println!("setup: {}", self.setup.lock().unwrap());
            println!("preProcessingA: {}", self.pre_processing_a.lock().unwrap());
            println!(
                "coarseAlignmentA: {}",
                self.coarse_alignment_a.lock().unwrap()
            );
            println!("fineAlignmentA: {}", self.fine_alignment_a.lock().unwrap());
            println!(
                "tomogramPositioningA: {}",
                self.tomogram_positioning_a.lock().unwrap()
            );
            println!(
                "finalAlignedStackA: {}",
                self.final_aligned_stack_a.lock().unwrap()
            );
            println!(
                "tomogramGenerationA: {}",
                self.tomogram_generation_a.lock().unwrap()
            );
            println!(
                "tomogramCombination: {}",
                self.tomogram_combination.lock().unwrap()
            );
            println!("postProcessing: {}", self.post_processing.lock().unwrap());
            println!("cleanUp: {}", self.clean_up.lock().unwrap());
        } else {
            println!("setup: {}", self.setup.lock().unwrap());
            println!("preProcessingA: {}", self.pre_processing_a.lock().unwrap());
            println!(
                "coarseAlignmentA: {}",
                self.coarse_alignment_a.lock().unwrap()
            );
            println!("fineAlignmentA: {}", self.fine_alignment_a.lock().unwrap());
            println!(
                "tomogramPositioningA: {}",
                self.tomogram_positioning_a.lock().unwrap()
            );
            println!(
                "finalAlignedStackA: {}",
                self.final_aligned_stack_a.lock().unwrap()
            );
            println!(
                "tomogramGenerationA: {}",
                self.tomogram_generation_a.lock().unwrap()
            );
            println!("preProcessingB: {}", self.pre_processing_b.lock().unwrap());
            println!(
                "coarseAlignmentB: {}",
                self.coarse_alignment_b.lock().unwrap()
            );
            println!("fiducialModelB: {}", self.fiducial_model_b.lock().unwrap());
            println!("fineAlignmentB: {}", self.fine_alignment_b.lock().unwrap());
            println!(
                "tomogramPositioningB: {}",
                self.tomogram_positioning_b.lock().unwrap()
            );
            println!(
                "finalAlignesStackB: {}",
                self.final_aligned_stack_b.lock().unwrap()
            );
            println!(
                "tomogramGenerationB: {}",
                self.tomogram_generation_b.lock().unwrap()
            );
            println!(
                "tomogramCombination: {}",
                self.tomogram_combination.lock().unwrap()
            );
            println!("postProcessing: {}", self.post_processing.lock().unwrap());
            println!("cleanUp: {}", self.clean_up.lock().unwrap());
        }
    }
}

impl Default for ProcessTrack {
    fn default() -> ProcessTrack {
        ProcessTrack::new()
    }
}

impl BaseProcessTrack for ProcessTrack {
    /// Java `getRevisionNumber`.
    fn get_revision_number(&self) -> String {
        self.revision_number.lock().unwrap().clone()
    }

    /// Java `isModified`.
    fn is_modified(&self) -> bool {
        *self.is_modified.lock().unwrap()
    }

    /// Java `resetModified`.
    fn reset_modified(&self) {
        *self.is_modified.lock().unwrap() = false;
    }

    /// Java final `setState(ProcessState, AxisID, AbstractParallelDialog)`.
    fn set_state(
        &self,
        process_state: ProcessState,
        axis_id: AxisID,
        parallel_dialog: &dyn AbstractParallelDialog,
    ) {
        self.set_state_dialog_type(process_state, axis_id, parallel_dialog.get_dialog_type());
    }
}

impl Storable for ProcessTrack {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        ProcessTrack::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        ProcessTrack::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &mut BTreeMap<String, String>) {
        ProcessTrack::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        ProcessTrack::load_with_prepend(self, properties, prepend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ProcessTrack.store` of the state set below, from the Java reference (the classes
    /// compiled from the vendored source, run headless).
    const JAVA_STORE: &str = r#"ProcessTrack.CleanUp=Not started
ProcessTrack.CoarseAlignment-A=Not started
ProcessTrack.CoarseAlignment-B=Not started
ProcessTrack.FiducialModel-A=Not started
ProcessTrack.FiducialModel-B=Not started
ProcessTrack.FinalAlignedStack-A=Not started
ProcessTrack.FinalAlignedStack-B=Not started
ProcessTrack.FineAlignment-A=Not started
ProcessTrack.FineAlignment-B=In progress
ProcessTrack.PostProcessing=Not started
ProcessTrack.PreProcessing-A=Not started
ProcessTrack.PreProcessing-B=Not started
ProcessTrack.RevisionNumber=2.0
ProcessTrack.Setup=Complete
ProcessTrack.TomogramCombination=Not started
ProcessTrack.TomogramGeneration-A=Complete
ProcessTrack.TomogramGeneration-B=Not started
ProcessTrack.TomogramPositioning-A=Not started
ProcessTrack.TomogramPositioning-B=Not started"#;

    fn java_map() -> BTreeMap<String, String> {
        JAVA_STORE
            .lines()
            .map(|line| {
                let (key, value) = line.split_once('=').unwrap();
                (key.to_string(), value.to_string())
            })
            .collect()
    }

    #[test]
    fn store_matches_java_and_round_trips() {
        let track = ProcessTrack::new();
        track.set_setup_state(ProcessState::Complete);
        track.set_fine_alignment_state(ProcessState::InProgress, AxisID::Second);
        track.set_state_dialog_type(
            ProcessState::Complete,
            AxisID::First,
            DialogType::TomogramGeneration,
        );
        assert!(track.is_modified());
        let mut props = BTreeMap::new();
        track.store(&mut props);
        assert_eq!(props, java_map());

        let loaded = ProcessTrack::new();
        loaded.load(&props);
        assert_eq!(
            loaded.get_fine_alignment_state(AxisID::Second),
            ProcessState::InProgress
        );
        assert_eq!(
            loaded.get_tomogram_generation_state(AxisID::First),
            ProcessState::Complete
        );
        assert!(!loaded.is_modified());
        let mut again = BTreeMap::new();
        loaded.store(&mut again);
        assert_eq!(again, props);
    }

    #[test]
    fn old_layout_copies_a_into_b_and_bad_values_default() {
        // setup_revision_1_5.edf (IMOD/Etomo/unitTestData) has no RevisionNumber 2.0
        // per-axis keys in its oldest form; a pre-2.0 revision reads the shared keys.
        let mut props = BTreeMap::new();
        props.insert("ProcessTrack.RevisionNumber".to_string(), "1.0".to_string());
        props.insert(
            "ProcessTrack.FineAlignment".to_string(),
            "Complete".to_string(),
        );
        props.insert("ProcessTrack.Setup".to_string(), "Bogus".to_string());
        let track = ProcessTrack::new();
        track.load(&props);
        assert_eq!(
            track.get_fine_alignment_state(AxisID::First),
            ProcessState::Complete
        );
        assert_eq!(
            track.get_fine_alignment_state(AxisID::Second),
            ProcessState::Complete
        );
        assert_eq!(track.get_setup_state(), ProcessState::NotStarted);
        props.insert("ProcessTrack.RevisionNumber".to_string(), "x".to_string());
        track.load(&props);
        assert_eq!(track.get_revision_number(), "x");
        assert_eq!(
            track.get_fine_alignment_state(AxisID::Second),
            ProcessState::Complete
        );
    }

    #[test]
    fn process_track_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProcessTrack>();
    }
}
