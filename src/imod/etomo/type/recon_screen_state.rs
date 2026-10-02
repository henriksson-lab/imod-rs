//! `IMOD/Etomo/src/etomo/type/ReconScreenState.java`.
//!
//! The reconstruction's per-axis screen state: the panel headers of the processing
//! dialogs and the patchcorr kernel sigma, on top of `BaseScreenState`.
//!
//! Java's `ReconScreenState extends BaseScreenState`.  The superclass is the `base`
//! field, reached through `Deref`.  Java's `store(Properties)`/`load(Properties)`
//! overrides call `super.store(props)`/`super.load(props)`, which call the two-argument
//! form with `""` - and that dispatches back to this class's override; here they call
//! this class's two-argument methods directly.  The panel header states lock their own
//! fields, so the getters hand out references to them, as the Java getters hand out the
//! objects.
//!
//! **`prepend == ""`.**  See `base_screen_state.rs`.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

use super::axis_id::AxisID;
use super::axis_type::AxisType;
use super::base_screen_state::BaseScreenState;
use super::const_etomo_number::{ConstEtomoNumber, Type};
use super::dialog_type::DialogType;
use super::etomo_number::EtomoNumber;
use super::etomo_version::EtomoVersion;
use super::panel_header_state::PanelHeaderState;
use super::process_name::ProcessName;
use crate::imod::etomo::storage::storable::{Storable, StorableValue};

/// Java `HEADER_GROUP`.
pub const HEADER_GROUP: &str = ".Header";

/// Java `TOMO_GEN_NEWST_HEADER_GROUP` (deprecated).
pub static TOMO_GEN_NEWST_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Newst{}",
        DialogType::TomogramGeneration.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java `TOMO_GEN_MTFFILTER_HEADER_GROUP` (deprecated).
pub static TOMO_GEN_MTFFILTER_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Mtffilter{}",
        DialogType::TomogramGeneration.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java `STACK_NEWST_HEADER_GROUP`.
pub static STACK_NEWST_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Newst{}",
        DialogType::FinalAlignedStack.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java `STACK_MTFFILTER_HEADER_GROUP`.
pub static STACK_MTFFILTER_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Mtffilter{}",
        DialogType::FinalAlignedStack.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java `STACK_CTF_CORRECTION_HEADER_GROUP`.
pub static STACK_CTF_CORRECTION_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.CtfCorrection{}",
        DialogType::FinalAlignedStack.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java `TOMO_GEN_TILT_HEADER_GROUP`.
pub static TOMO_GEN_TILT_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Tilt{}",
        DialogType::TomogramGeneration.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java `TOMO_GEN_TRIAL_TILT_HEADER_GROUP`.
pub static TOMO_GEN_TRIAL_TILT_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.TrialTilt{}",
        DialogType::TomogramGeneration.get_storable_name(),
        HEADER_GROUP
    )
});

/// Java private static `SETUP_GROUP`.
static SETUP_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Setup.",
        DialogType::TomogramCombination.get_storable_name()
    )
});

/// Java private static `SOLVEMATCH_GROUP`.
const SOLVEMATCH_GROUP: &str = "Solvematch";
/// Java private static `PATCHCORR_GROUP`.
const PATCHCORR_GROUP: &str = "Patchcorr";
/// Java private static `VOLCOMBINE_GROUP`.
const VOLCOMBINE_GROUP: &str = "Volcombine";

/// Java `Patchcrawl3DParam.KERNEL_SIGMA_KEY` = "KernelSigma"
/// (ConstPatchcrawl3DParam.java:48); the param class has no Rust module.
const PATCHCRAWL_3D_PARAM_KERNEL_SIGMA_KEY: &str = "KernelSigma";

/// Java private static `PATCHCORR_KERNEL_SIGMA_KEY`.
static PATCHCORR_KERNEL_SIGMA_KEY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.{}.{}",
        DialogType::TomogramCombination.get_storable_name(),
        ProcessName::PATCHCORR,
        PATCHCRAWL_3D_PARAM_KERNEL_SIGMA_KEY
    )
});

/// Java `COMBINE_SETUP_TO_SELECTOR_HEADER_GROUP`.
pub static COMBINE_SETUP_TO_SELECTOR_HEADER_GROUP: LazyLock<String> =
    LazyLock::new(|| format!("{}ToSelector.{}", SETUP_GROUP.as_str(), HEADER_GROUP));

/// Java `COMBINE_SETUP_SOLVEMATCH_HEADER_GROUP`.
pub static COMBINE_SETUP_SOLVEMATCH_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}{}",
        SETUP_GROUP.as_str(),
        SOLVEMATCH_GROUP,
        HEADER_GROUP
    )
});

/// Java `COMBINE_SETUP_PATCHCORR_HEADER_GROUP`.
pub static COMBINE_SETUP_PATCHCORR_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}{}",
        SETUP_GROUP.as_str(),
        PATCHCORR_GROUP,
        HEADER_GROUP
    )
});

/// Java `COMBINE_SETUP_VOLCOMBINE_HEADER_GROUP`.
pub static COMBINE_SETUP_VOLCOMBINE_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}{}",
        SETUP_GROUP.as_str(),
        VOLCOMBINE_GROUP,
        HEADER_GROUP
    )
});

/// Java `COMBINE_SETUP_TEMP_DIR_HEADER_GROUP`.
pub static COMBINE_SETUP_TEMP_DIR_HEADER_GROUP: LazyLock<String> =
    LazyLock::new(|| format!("{}TempDir{}", SETUP_GROUP.as_str(), HEADER_GROUP));

/// Java `COMBINE_INITIAL_SOLVEMATCH_HEADER_GROUP`.
pub static COMBINE_INITIAL_SOLVEMATCH_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Initial.{}{}",
        DialogType::TomogramCombination.get_storable_name(),
        SOLVEMATCH_GROUP,
        HEADER_GROUP
    )
});

/// Java private static `FINAL_GROUP`.
static FINAL_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}.Final.",
        DialogType::TomogramCombination.get_storable_name()
    )
});

/// Java `COMBINE_FINAL_PATCH_REGION_HEADER_GROUP`.
pub static COMBINE_FINAL_PATCH_REGION_HEADER_GROUP: LazyLock<String> =
    LazyLock::new(|| format!("{}PatchRegion{}", FINAL_GROUP.as_str(), HEADER_GROUP));

/// Java `COMBINE_FINAL_PATCHCORR_HEADER_GROUP`.
pub static COMBINE_FINAL_PATCHCORR_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}{}",
        FINAL_GROUP.as_str(),
        PATCHCORR_GROUP,
        HEADER_GROUP
    )
});

/// Java `COMBINE_FINAL_MATCHORWARP_HEADER_GROUP`.
pub static COMBINE_FINAL_MATCHORWARP_HEADER_GROUP: LazyLock<String> =
    LazyLock::new(|| format!("{}Matchorwarp{}", FINAL_GROUP.as_str(), HEADER_GROUP));

/// Java `COMBINE_FINAL_VOLCOMBINE_HEADER_GROUP`.
pub static COMBINE_FINAL_VOLCOMBINE_HEADER_GROUP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{}{}{}",
        FINAL_GROUP.as_str(),
        VOLCOMBINE_GROUP,
        HEADER_GROUP
    )
});

// Dialog keys
/// Java private static `STACK_KEY`.
const STACK_KEY: &str = "Stack";

// Panel keys
/// Java private static `ERASE_GOLD_KEY`.
const ERASE_GOLD_KEY: &str = "EraseGold";
/// Java private static `NEWSTACK_OR_BLENDMONT_KEY`.
const NEWSTACK_OR_BLENDMONT_KEY: &str = "NewstackOrBlendmont";

// Header keys
/// Java private static `HEADER_KEY`.
const HEADER_KEY: &str = "Header";
/// Java private static `NEWST_KEY`.
const NEWST_KEY: &str = "Newst";

/// Java `ReconScreenState`.
pub struct ReconScreenState {
    /// Java superclass `BaseScreenState` state.
    pub base: BaseScreenState,
    /// Java field `tomoGenMtffilterHeaderState` (deprecated).
    tomo_gen_mtffilter_header_state: PanelHeaderState,
    /// Java field `tomoGenNewstHeaderState` (deprecated).
    tomo_gen_newst_header_state: PanelHeaderState,
    /// Java field `stackNewstHeaderState`.
    stack_newst_header_state: PanelHeaderState,
    /// Java field `stackMtffilterHeaderState`.
    stack_mtffilter_header_state: PanelHeaderState,
    /// Java field `stackCtfCorrectionHeaderState`.
    stack_ctf_correction_header_state: PanelHeaderState,
    /// Java field `tomoGenTiltHeaderState`.
    tomo_gen_tilt_header_state: PanelHeaderState,
    /// Java field `tomoGenTrialTiltHeaderState`.
    tomo_gen_trial_tilt_header_state: PanelHeaderState,
    /// Java field `tomoGenSirtHeaderState`.
    tomo_gen_sirt_header_state: PanelHeaderState,
    /// Java field `combineSetupToSelectorHeaderState`.
    combine_setup_to_selector_header_state: PanelHeaderState,
    /// Java field `combineSetupSolvematchHeaderState`.
    combine_setup_solvematch_header_state: PanelHeaderState,
    /// Java field `combineSetupPatchcorrHeaderState`.
    combine_setup_patchcorr_header_state: PanelHeaderState,
    /// Java field `combineSetupVolcombineHeaderState`.
    combine_setup_volcombine_header_state: PanelHeaderState,
    /// Java field `combineSetupTempDirHeaderState`.
    combine_setup_temp_dir_header_state: PanelHeaderState,
    /// Java field `combineInitialSolvematchHeaderState`.
    combine_initial_solvematch_header_state: PanelHeaderState,
    /// Java field `combineFinalPatchRegionHeaderState`.
    combine_final_patch_region_header_state: PanelHeaderState,
    /// Java field `combineFinalPatchcorrHeaderState`.
    combine_final_patchcorr_header_state: PanelHeaderState,
    /// Java field `combineFinalMatchorwarpHeaderState`.
    combine_final_matchorwarp_header_state: PanelHeaderState,
    /// Java field `combineFinalVolcombineHeaderState`.
    combine_final_volcombine_header_state: PanelHeaderState,
    /// Java field `patchcorrKernelSigma`, initialised to null.
    patchcorr_kernel_sigma: Mutex<Option<EtomoNumber>>,
    /// Java field `version = EtomoVersion.getDefaultInstance("1.1")`.
    version: EtomoVersion,
    /// Java field `stackEraseGoldNewstHeaderState`.
    stack_erase_gold_newst_header_state: PanelHeaderState,
    /// Java field `stackFindBeads3dHeaderState`.
    stack_find_beads3d_header_state: PanelHeaderState,
    /// Java field `stackAlignAndTiltHeaderState`.
    stack_align_and_tilt_header_state: PanelHeaderState,
}

/// Java inheritance: every `BaseScreenState` member is reachable on a
/// `ReconScreenState`.
impl std::ops::Deref for ReconScreenState {
    type Target = BaseScreenState;

    fn deref(&self) -> &BaseScreenState {
        &self.base
    }
}

impl ReconScreenState {
    /// Java `ReconScreenState(AxisID, AxisType)`.
    pub fn new(axis_id: AxisID, axis_type: AxisType) -> ReconScreenState {
        ReconScreenState {
            base: BaseScreenState::new(axis_id, axis_type),
            tomo_gen_mtffilter_header_state: PanelHeaderState::new(
                TOMO_GEN_MTFFILTER_HEADER_GROUP.as_str(),
            ),
            tomo_gen_newst_header_state: PanelHeaderState::new(
                TOMO_GEN_NEWST_HEADER_GROUP.as_str(),
            ),
            stack_newst_header_state: PanelHeaderState::new(STACK_NEWST_HEADER_GROUP.as_str()),
            stack_mtffilter_header_state: PanelHeaderState::new(
                STACK_MTFFILTER_HEADER_GROUP.as_str(),
            ),
            stack_ctf_correction_header_state: PanelHeaderState::new(
                STACK_CTF_CORRECTION_HEADER_GROUP.as_str(),
            ),
            tomo_gen_tilt_header_state: PanelHeaderState::new(TOMO_GEN_TILT_HEADER_GROUP.as_str()),
            tomo_gen_trial_tilt_header_state: PanelHeaderState::new(
                TOMO_GEN_TRIAL_TILT_HEADER_GROUP.as_str(),
            ),
            tomo_gen_sirt_header_state: PanelHeaderState::new(&format!(
                "{}.Sirt{}",
                DialogType::TomogramGeneration.get_storable_name(),
                HEADER_GROUP
            )),
            combine_setup_to_selector_header_state: PanelHeaderState::new(
                COMBINE_SETUP_TO_SELECTOR_HEADER_GROUP.as_str(),
            ),
            combine_setup_solvematch_header_state: PanelHeaderState::new(
                COMBINE_SETUP_SOLVEMATCH_HEADER_GROUP.as_str(),
            ),
            combine_setup_patchcorr_header_state: PanelHeaderState::new(
                COMBINE_SETUP_PATCHCORR_HEADER_GROUP.as_str(),
            ),
            combine_setup_volcombine_header_state: PanelHeaderState::new(
                COMBINE_SETUP_VOLCOMBINE_HEADER_GROUP.as_str(),
            ),
            combine_setup_temp_dir_header_state: PanelHeaderState::new(
                COMBINE_SETUP_TEMP_DIR_HEADER_GROUP.as_str(),
            ),
            combine_initial_solvematch_header_state: PanelHeaderState::new(
                COMBINE_INITIAL_SOLVEMATCH_HEADER_GROUP.as_str(),
            ),
            combine_final_patch_region_header_state: PanelHeaderState::new(
                COMBINE_FINAL_PATCH_REGION_HEADER_GROUP.as_str(),
            ),
            combine_final_patchcorr_header_state: PanelHeaderState::new(
                COMBINE_FINAL_PATCHCORR_HEADER_GROUP.as_str(),
            ),
            combine_final_matchorwarp_header_state: PanelHeaderState::new(
                COMBINE_FINAL_MATCHORWARP_HEADER_GROUP.as_str(),
            ),
            combine_final_volcombine_header_state: PanelHeaderState::new(
                COMBINE_FINAL_VOLCOMBINE_HEADER_GROUP.as_str(),
            ),
            patchcorr_kernel_sigma: Mutex::new(None),
            version: EtomoVersion::get_default_instance_with_version(Some("1.1")),
            stack_erase_gold_newst_header_state: PanelHeaderState::new(&format!(
                "{}.{}.{}.{}.{}",
                STACK_KEY, ERASE_GOLD_KEY, NEWSTACK_OR_BLENDMONT_KEY, NEWST_KEY, HEADER_KEY
            )),
            stack_find_beads3d_header_state: PanelHeaderState::new(&format!(
                "{}.FindBeads3d{}",
                STACK_KEY, HEADER_KEY
            )),
            stack_align_and_tilt_header_state: PanelHeaderState::new(&format!(
                "{}.AlignAndTilt{}",
                STACK_KEY, HEADER_KEY
            )),
        }
    }

    /// Java `store(Properties)`: `super.store(props)`, which dispatches to this class's
    /// `store(props, "")`.
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }

    /// Java `store(Properties, String)`.
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        self.base.store_with_prepend(props, prepend);
        let prepend = self.base.get_prepend(prepend);
        self.stack_newst_header_state
            .store_with_prepend(props, &prepend);
        self.stack_mtffilter_header_state
            .store_with_prepend(props, &prepend);
        self.stack_ctf_correction_header_state
            .store_with_prepend(props, &prepend);
        self.tomo_gen_tilt_header_state
            .store_with_prepend(props, &prepend);
        self.tomo_gen_trial_tilt_header_state
            .store_with_prepend(props, &prepend);
        self.stack_erase_gold_newst_header_state
            .store_with_prepend(props, &prepend);
        self.stack_find_beads3d_header_state
            .store_with_prepend(props, &prepend);
        self.stack_align_and_tilt_header_state
            .store_with_prepend(props, &prepend);
        self.tomo_gen_sirt_header_state
            .store_with_prepend(props, &prepend);
        if self.base.axis_id == AxisID::First {
            self.combine_setup_to_selector_header_state
                .store_with_prepend(props, &prepend);
            self.combine_setup_solvematch_header_state
                .store_with_prepend(props, &prepend);
            self.combine_setup_patchcorr_header_state
                .store_with_prepend(props, &prepend);
            self.combine_setup_volcombine_header_state
                .store_with_prepend(props, &prepend);
            self.combine_setup_temp_dir_header_state
                .store_with_prepend(props, &prepend);
            self.combine_initial_solvematch_header_state
                .store_with_prepend(props, &prepend);
            self.combine_final_patch_region_header_state
                .store_with_prepend(props, &prepend);
            self.combine_final_patchcorr_header_state
                .store_with_prepend(props, &prepend);
            self.combine_final_matchorwarp_header_state
                .store_with_prepend(props, &prepend);
            self.combine_final_volcombine_header_state
                .store_with_prepend(props, &prepend);
            ConstEtomoNumber::store_etomo_number(
                self.patchcorr_kernel_sigma.lock().unwrap().as_ref(),
                PATCHCORR_KERNEL_SIGMA_KEY.as_str(),
                props,
                Some(&prepend),
            );
        }
    }

    /// Java `load(Properties)`: `super.load(props)`, which dispatches to this class's
    /// `load(props, "")`.
    pub fn load(&self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "");
    }

    /// Java `load(Properties, String)`.
    pub fn load_with_prepend(&self, props: &BTreeMap<String, String>, prepend: &str) {
        self.base.load_with_prepend(props, prepend);
        let prepend = self.base.get_prepend(prepend);
        // backwards compatibility
        // Moved newst and mtffilter to final aligned stack dialog at version 1.1.
        if self
            .version
            .lt(Some(&EtomoVersion::get_default_instance_with_version(
                Some("1.1"),
            )))
        {
            self.tomo_gen_mtffilter_header_state
                .load_with_prepend(props, &prepend);
            self.stack_mtffilter_header_state
                .set(&self.tomo_gen_mtffilter_header_state);
            self.tomo_gen_newst_header_state
                .load_with_prepend(props, &prepend);
            self.stack_newst_header_state
                .set(&self.tomo_gen_newst_header_state);
        } else {
            self.stack_mtffilter_header_state
                .load_with_prepend(props, &prepend);
            self.stack_newst_header_state
                .load_with_prepend(props, &prepend);
        }
        self.stack_ctf_correction_header_state
            .load_with_prepend(props, &prepend);
        self.tomo_gen_tilt_header_state
            .load_with_prepend(props, &prepend);
        self.tomo_gen_sirt_header_state
            .load_with_prepend(props, &prepend);
        self.tomo_gen_trial_tilt_header_state
            .load_with_prepend(props, &prepend);
        self.stack_erase_gold_newst_header_state
            .load_with_prepend(props, &prepend);
        self.stack_find_beads3d_header_state
            .load_with_prepend(props, &prepend);
        self.stack_align_and_tilt_header_state
            .load_with_prepend(props, &prepend);
        if self.base.axis_id == AxisID::First {
            self.combine_setup_to_selector_header_state
                .load_with_prepend(props, &prepend);
            self.combine_setup_solvematch_header_state
                .load_with_prepend(props, &prepend);
            self.combine_setup_patchcorr_header_state
                .load_with_prepend(props, &prepend);
            self.combine_setup_volcombine_header_state
                .load_with_prepend(props, &prepend);
            self.combine_setup_temp_dir_header_state
                .load_with_prepend(props, &prepend);
            self.combine_initial_solvematch_header_state
                .load_with_prepend(props, &prepend);
            self.combine_final_patch_region_header_state
                .load_with_prepend(props, &prepend);
            self.combine_final_patchcorr_header_state
                .load_with_prepend(props, &prepend);
            self.combine_final_matchorwarp_header_state
                .load_with_prepend(props, &prepend);
            self.combine_final_volcombine_header_state
                .load_with_prepend(props, &prepend);
            let current = self.patchcorr_kernel_sigma.lock().unwrap().take();
            *self.patchcorr_kernel_sigma.lock().unwrap() = EtomoNumber::load_instance_with_type(
                current,
                Type::Double,
                PATCHCORR_KERNEL_SIGMA_KEY.as_str(),
                props,
                Some(&prepend),
            );
        }
    }

    /// Java private static `getAxisExtension`.
    fn get_axis_extension(axis_id: AxisID) -> String {
        let mut axis_id = axis_id;
        if axis_id == AxisID::Only {
            axis_id = AxisID::First;
        }
        axis_id.get_extension().to_uppercase()
    }

    /// Java `getNewstHeaderState`.
    pub fn get_newst_header_state(&self) -> &PanelHeaderState {
        &self.stack_newst_header_state
    }

    /// Java `getStackMtffilterHeaderState`.
    pub fn get_stack_mtffilter_header_state(&self) -> &PanelHeaderState {
        &self.stack_mtffilter_header_state
    }

    /// Java `getStackCtfCorrectionHeaderState`.
    pub fn get_stack_ctf_correction_header_state(&self) -> &PanelHeaderState {
        &self.stack_ctf_correction_header_state
    }

    /// Java `getStackFindBeads3dHeaderState`.
    pub fn get_stack_find_beads3d_header_state(&self) -> &PanelHeaderState {
        &self.stack_find_beads3d_header_state
    }

    /// Java `getStackAlignAndTiltHeaderState`.
    pub fn get_stack_align_and_tilt_header_state(&self) -> &PanelHeaderState {
        &self.stack_align_and_tilt_header_state
    }

    /// Java `getTomoGenTiltHeaderState`.
    pub fn get_tomo_gen_tilt_header_state(&self) -> &PanelHeaderState {
        &self.tomo_gen_tilt_header_state
    }

    /// Java `getTomoGenSirtHeaderState`.
    pub fn get_tomo_gen_sirt_header_state(&self) -> &PanelHeaderState {
        &self.tomo_gen_sirt_header_state
    }

    /// Java `getTomoGenTrialTiltHeaderState`.
    pub fn get_tomo_gen_trial_tilt_header_state(&self) -> &PanelHeaderState {
        &self.tomo_gen_trial_tilt_header_state
    }

    /// Java `getCombineSetupToSelectorHeaderState`.
    pub fn get_combine_setup_to_selector_header_state(&self) -> &PanelHeaderState {
        &self.combine_setup_to_selector_header_state
    }

    /// Java `getCombineSetupSolvematchHeaderState`.
    pub fn get_combine_setup_solvematch_header_state(&self) -> &PanelHeaderState {
        &self.combine_setup_solvematch_header_state
    }

    /// Java `getCombineSetupPatchcorrHeaderState`.
    pub fn get_combine_setup_patchcorr_header_state(&self) -> &PanelHeaderState {
        &self.combine_setup_patchcorr_header_state
    }

    /// Java `getCombineSetupTempDirHeaderState`.
    pub fn get_combine_setup_temp_dir_header_state(&self) -> &PanelHeaderState {
        &self.combine_setup_temp_dir_header_state
    }

    /// Java `getCombineInitialSolvematchHeaderState`.
    pub fn get_combine_initial_solvematch_header_state(&self) -> &PanelHeaderState {
        &self.combine_initial_solvematch_header_state
    }

    /// Java `getCombineFinalPatchRegionHeaderState`.
    pub fn get_combine_final_patch_region_header_state(&self) -> &PanelHeaderState {
        &self.combine_final_patch_region_header_state
    }

    /// Java `getCombineFinalPatchcorrHeaderState`.
    pub fn get_combine_final_patchcorr_header_state(&self) -> &PanelHeaderState {
        &self.combine_final_patchcorr_header_state
    }

    /// Java `getCombineFinalMatchorwarpHeaderState`.
    pub fn get_combine_final_matchorwarp_header_state(&self) -> &PanelHeaderState {
        &self.combine_final_matchorwarp_header_state
    }

    /// Java `getCombineFinalVolcombineHeaderState`.
    pub fn get_combine_final_volcombine_header_state(&self) -> &PanelHeaderState {
        &self.combine_final_volcombine_header_state
    }

    /// Java `getCombineSetupVolcombineHeaderState`.
    pub fn get_combine_setup_volcombine_header_state(&self) -> &PanelHeaderState {
        &self.combine_setup_volcombine_header_state
    }

    /// Java `getPatchcorrKernelSigma`.  Java returns the field (null when unset); a copy
    /// is returned here.
    pub fn get_patchcorr_kernel_sigma(&self) -> Option<EtomoNumber> {
        self.patchcorr_kernel_sigma.lock().unwrap().clone()
    }

    /// Java `setPatchcorrKernelSigma`.
    pub fn set_patchcorr_kernel_sigma(&self, patchcorr_kernel_sigma: Option<&str>) {
        let mut field = self.patchcorr_kernel_sigma.lock().unwrap();
        if field.is_none() {
            *field = Some(EtomoNumber::new_with_type_and_name(
                Type::Double,
                PATCHCORR_KERNEL_SIGMA_KEY.as_str(),
            ));
        }
        field.as_mut().unwrap().set_string(patchcorr_kernel_sigma);
    }
}

/// Java `Storable`.
impl Storable for ReconScreenState {
    fn store(&self, properties: &mut BTreeMap<String, String>) {
        ReconScreenState::store(self, properties);
    }

    fn store_with_prepend(&self, properties: &mut BTreeMap<String, String>, prepend: &str) {
        ReconScreenState::store_with_prepend(self, properties, prepend);
    }

    fn load(&self, properties: &BTreeMap<String, String>) {
        ReconScreenState::load(self, properties);
    }

    fn load_with_prepend(&self, properties: &BTreeMap<String, String>, prepend: &str) {
        ReconScreenState::load_with_prepend(self, properties, prepend);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ReconScreenState.store` for the state set below, from the Java reference (the
    /// classes compiled from the vendored source, run headless), with an empty prepend
    /// and with the prepend "Pre".
    const JAVA_STORE: &str = r#"ScreenStateA.Button1=true
ScreenStateA.Combine.Final.Patchcorr.Header.MoreLess=more
ScreenStateA.Combine.patchcorr.KernelSigma=1.5
ScreenStateA.FinalStack.Mtffilter.Header.OpenClose=open
ScreenStateA.ParallelProcess.Header.OpenClose=closed
ScreenStateA.TomoGen.Tilt.Header.AdvancedBasic=advanced"#;
    const JAVA_STORE_PRE: &str = r#"Pre.ScreenStateA.Button1=true
Pre.ScreenStateA.Combine.Final.Patchcorr.Header.MoreLess=more
Pre.ScreenStateA.Combine.patchcorr.KernelSigma=1.5
Pre.ScreenStateA.FinalStack.Mtffilter.Header.OpenClose=open
Pre.ScreenStateA.ParallelProcess.Header.OpenClose=closed
Pre.ScreenStateA.TomoGen.Tilt.Header.AdvancedBasic=advanced"#;

    fn parse(text: &str) -> BTreeMap<String, String> {
        text.lines()
            .map(|line| {
                let (key, value) = line.split_once('=').unwrap();
                (key.to_string(), value.to_string())
            })
            .collect()
    }

    fn configured() -> ReconScreenState {
        let rs = ReconScreenState::new(AxisID::First, AxisType::DualAxis);
        rs.get_stack_mtffilter_header_state()
            .set_open_close_state(Some("open"));
        rs.get_tomo_gen_tilt_header_state()
            .set_advanced_basic_state(Some("advanced"));
        rs.get_combine_final_patchcorr_header_state()
            .set_more_less_state(Some("more"));
        rs.get_parallel_header_state()
            .set_open_close_state(Some("closed"));
        rs.set_patchcorr_kernel_sigma(Some("1.5"));
        rs.set_button_state(Some("Button1"), true);
        rs.get_button_state(Some("Button1"));
        rs.get_button_state(Some("Button2"));
        rs
    }

    #[test]
    fn store_matches_java() {
        let rs = configured();
        let mut props = BTreeMap::new();
        rs.store(&mut props);
        assert_eq!(props, parse(JAVA_STORE));
        let mut props = BTreeMap::new();
        rs.store_with_prepend(&mut props, "Pre");
        assert_eq!(props, parse(JAVA_STORE_PRE));
    }

    #[test]
    fn load_store_round_trip() {
        let java = parse(JAVA_STORE);
        let rs = ReconScreenState::new(AxisID::First, AxisType::DualAxis);
        rs.load(&java);
        assert_eq!(rs.get_patchcorr_kernel_sigma().unwrap().get_double(), 1.5);
        assert!(rs.get_button_state(Some("Button1")));
        let mut props = BTreeMap::new();
        rs.store(&mut props);
        assert_eq!(props, java);
        // A single-axis state reads the "ScreenState" group, and has no combine state.
        let single = ReconScreenState::new(AxisID::Only, AxisType::SingleAxis);
        single.load(&java);
        assert!(single.get_patchcorr_kernel_sigma().is_none());
    }
}
