//! `IMOD/Etomo/src/etomo/comscript/ComScriptManager.java`.
//!
//! Description: This class provides a high level manager for loading and saving
//! particlar com scripts and extracting the parameter sets for the commands within
//! those scripts.
//!
//! Copyright: Copyright 2002 - 2025 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! **Threading and mutability.**  A `ComScript` holds `Rc<RefCell<ComScriptCommand>>`
//! elements and is not `Send`; the manager lives on the event dispatch thread
//! (`ApplicationManager` calls it from dialog actions), so it is `!Sync` and every
//! `ComScript` field is a `RefCell<Option<ComScript>>`.  Methods take `&self`, as the
//! Java object is shared by reference (`CombineComscriptState.initialize(this)` calls
//! back into it).  No `RefCell` borrow is held across a call that can re-enter the
//! manager.
//!
//! **Overloads.**  Java overloads that differ only in the parameter class get the
//! parameter class as a suffix (`saveXcorr(TiltxcorrParam, AxisID)` is
//! `save_xcorr_tiltxcorr`).

use super::alt_tomo_setup_param::AltTomoSetupParam;
use super::autofidseed_param::AutofidseedParam;
use super::beadtrack_param::BeadtrackParam;
use super::blendmont_param::{self, BlendmontParam};
use super::ccd_eraser_param::{self, CCDEraserParam};
use super::com_script::ComScript;
use super::com_script_util::ComScriptUtil;
use super::combine_comscript_state::{self, CombineComscriptState};
use super::cryo_position_param::CryoPositionParam;
use super::ctf_phase_flip_param::{self, CtfPhaseFlipParam};
use super::ctf_plotter_param::{self, CtfPlotterParam};
use super::ctf3d_setup_param::Ctf3dSetupParam;
use super::dualvolmatch_param::DualvolmatchParam;
use super::echo_param::{self, EchoParam};
use super::exit_param::{self, ExitParam};
use super::find_beads3d_param::FindBeads3dParam;
use super::goto_param::{self, GotoParam};
use super::imodchopconts_param::ImodchopcontsParam;
use super::label_param::LabelParam;
use super::matchorwarp_param::MatchorwarpParam;
use super::matchshifts_param::MatchshiftsParam;
use super::matchvol_param::{self, MatchvolParam};
use super::mrc_taper_param::{self, MrcTaperParam};
use super::mtf_filter_param::MTFFilterParam;
use super::multifilt_setup_param::MultifiltSetupParam;
use super::newst_param::NewstParam;
use super::patchcrawl3d_param::{self, Patchcrawl3DParam};
use super::patchcrawl3d_pre_pip_param;
use super::reduce_filt_vol_param::ReduceFiltVolParam;
use super::restrictalign_param::RestrictalignParam;
use super::set_env_param::{self, SetEnvParam};
use super::set_param::{self, SetParam};
use super::sirtsetup_param::SirtsetupParam;
use super::solvematch_param::SolvematchParam;
use super::solvematchmod_param::SolvematchmodParam;
use super::solvematchshift_param::SolvematchshiftParam;
use super::subtomo_setup_param::SubtomoSetupParam;
use super::tilt_param::TiltParam;
use super::tiltalign_param::TiltalignParam;
use super::tiltxcorr_param::TiltxcorrParam;
use super::tomopitch_param::TomopitchParam;
use super::warp_vol_param::{self, WarpVolParam};
use super::xfproduct_param::XfproductParam;
use crate::imod::etomo::application_manager::ApplicationManager;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::axis_type::AxisType;
use crate::imod::etomo::r#type::const_etomo_number::Type;
use crate::imod::etomo::r#type::file_type;
use crate::imod::etomo::r#type::process_name::ProcessName;
use crate::imod::etomo::ui::swing::ui_harness::{self, UIHarness};
use std::cell::RefCell;
use std::thread::LocalKey;

/// Java package-private `PARAM_KEY`.
pub const PARAM_KEY: &str = "Param";

/// Java package-private `MTFFILTER_COMMAND`.
pub const MTFFILTER_COMMAND: &str = "mtffilter";

/// Java final `ComScriptManager`.
pub struct ComScriptManager {
    /// Java private final field `appManager`.
    app_manager: &'static ApplicationManager,
    /// Java package-private field `uiHarness`, initialised to `UIHarness.INSTANCE`
    /// (a thread-local singleton here).
    pub ui_harness: &'static LocalKey<std::rc::Rc<UIHarness>>,
    /// Java field `scriptEraserA`, initialised to null.
    script_eraser_a: RefCell<Option<ComScript>>,
    /// Java field `scriptEraserB`, initialised to null.
    script_eraser_b: RefCell<Option<ComScript>>,
    /// Java field `scriptXcorrA`, initialised to null.
    script_xcorr_a: RefCell<Option<ComScript>>,
    /// Java field `scriptXcorrB`, initialised to null.
    script_xcorr_b: RefCell<Option<ComScript>>,
    /// Java field `scriptPrenewstA`, initialised to null.
    script_prenewst_a: RefCell<Option<ComScript>>,
    /// Java field `scriptPrenewstB`, initialised to null.
    script_prenewst_b: RefCell<Option<ComScript>>,
    /// Java field `scriptTrackA`, initialised to null.
    script_track_a: RefCell<Option<ComScript>>,
    /// Java field `scriptTrackB`, initialised to null.
    script_track_b: RefCell<Option<ComScript>>,
    /// Java field `scriptAlignA`, initialised to null.
    script_align_a: RefCell<Option<ComScript>>,
    /// Java field `scriptAlignB`, initialised to null.
    script_align_b: RefCell<Option<ComScript>>,
    /// Java field `scriptNewstA`, initialised to null.
    script_newst_a: RefCell<Option<ComScript>>,
    /// Java field `scriptNewstB`, initialised to null.
    script_newst_b: RefCell<Option<ComScript>>,
    /// Java field `scriptTiltA`, initialised to null.
    script_tilt_a: RefCell<Option<ComScript>>,
    /// Java field `scriptTiltB`, initialised to null.
    script_tilt_b: RefCell<Option<ComScript>>,
    /// Java field `scriptMTFFilterA`, initialised to null.
    script_mtf_filter_a: RefCell<Option<ComScript>>,
    /// Java field `scriptMTFFilterB`, initialised to null.
    script_mtf_filter_b: RefCell<Option<ComScript>>,
    /// Java field `scriptPreblendA`, initialised to null.
    script_preblend_a: RefCell<Option<ComScript>>,
    /// Java field `scriptPreblendB`, initialised to null.
    script_preblend_b: RefCell<Option<ComScript>>,
    /// Java field `scriptBlendA`, initialised to null.
    script_blend_a: RefCell<Option<ComScript>>,
    /// Java field `scriptBlendB`, initialised to null.
    script_blend_b: RefCell<Option<ComScript>>,
    /// Java field `scriptUndistortA`, initialised to null.
    script_undistort_a: RefCell<Option<ComScript>>,
    /// Java field `scriptUndistortB`, initialised to null.
    script_undistort_b: RefCell<Option<ComScript>>,
    /// Java field `scriptMatchvol1`, initialised to null.
    script_matchvol1: RefCell<Option<ComScript>>,
    /// Java field `scriptSolvematch`, initialised to null.
    script_solvematch: RefCell<Option<ComScript>>,
    /// Java field `scriptDualvolmatch`, initialised to null.
    script_dualvolmatch: RefCell<Option<ComScript>>,
    /// Java field `scriptSolvematchshift`, initialised to null.
    script_solvematchshift: RefCell<Option<ComScript>>,
    /// Java field `scriptSolvematchmod`, initialised to null.
    script_solvematchmod: RefCell<Option<ComScript>>,
    /// Java field `scriptPatchcorr`, initialised to null.
    script_patchcorr: RefCell<Option<ComScript>>,
    /// Java field `scriptMatchorwarp`, initialised to null.
    script_matchorwarp: RefCell<Option<ComScript>>,
    /// Java field `scriptTomopitchA`, initialised to null.
    script_tomopitch_a: RefCell<Option<ComScript>>,
    /// Java field `scriptTomopitchB`, initialised to null.
    script_tomopitch_b: RefCell<Option<ComScript>>,
    /// Java field `scriptCombine`, initialised to null.
    script_combine: RefCell<Option<ComScript>>,
    /// Java field `scriptVolcombine`, initialised to null.
    script_volcombine: RefCell<Option<ComScript>>,
    /// Java field `scriptCtfCorrectionA`, initialised to null.
    script_ctf_correction_a: RefCell<Option<ComScript>>,
    /// Java field `scriptCtfCorrectionB`, initialised to null.
    script_ctf_correction_b: RefCell<Option<ComScript>>,
    /// Java field `scriptCtfPlotterA`, initialised to null.
    script_ctf_plotter_a: RefCell<Option<ComScript>>,
    /// Java field `scriptCtfPlotterB`, initialised to null.
    script_ctf_plotter_b: RefCell<Option<ComScript>>,
    /// Java field `scriptFlatten`, initialised to null.
    script_flatten: RefCell<Option<ComScript>>,
    /// Java field `scriptNewst3dFindA`, initialised to null.
    script_newst3d_find_a: RefCell<Option<ComScript>>,
    /// Java field `scriptNewst3dFindB`, initialised to null.
    script_newst3d_find_b: RefCell<Option<ComScript>>,
    /// Java field `scriptBlend3dFindA`, initialised to null.
    script_blend3d_find_a: RefCell<Option<ComScript>>,
    /// Java field `scriptBlend3dFindB`, initialised to null.
    script_blend3d_find_b: RefCell<Option<ComScript>>,
    /// Java field `scriptTilt3dFindA`, initialised to null.
    script_tilt3d_find_a: RefCell<Option<ComScript>>,
    /// Java field `scriptTilt3dFindB`, initialised to null.
    script_tilt3d_find_b: RefCell<Option<ComScript>>,
    /// Java field `scriptFindBeads3dA`, initialised to null.
    script_find_beads3d_a: RefCell<Option<ComScript>>,
    /// Java field `scriptFindBeads3dB`, initialised to null.
    script_find_beads3d_b: RefCell<Option<ComScript>>,
    /// Java field `scriptTilt3dFindReprojectA`, initialised to null.
    script_tilt3d_find_reproject_a: RefCell<Option<ComScript>>,
    /// Java field `scriptTilt3dFindReprojectB`, initialised to null.
    script_tilt3d_find_reproject_b: RefCell<Option<ComScript>>,
    /// Java field `scriptXcorrPtA`, initialised to null.
    script_xcorr_pt_a: RefCell<Option<ComScript>>,
    /// Java field `scriptXcorrPtB`, initialised to null.
    script_xcorr_pt_b: RefCell<Option<ComScript>>,
    /// Java field `scriptSirtsetupA`, initialised to null.
    script_sirtsetup_a: RefCell<Option<ComScript>>,
    /// Java field `scriptSirtsetupB`, initialised to null.
    script_sirtsetup_b: RefCell<Option<ComScript>>,
    /// Java field `scriptTiltForSirtA`, initialised to null.
    script_tilt_for_sirt_a: RefCell<Option<ComScript>>,
    /// Java field `scriptTiltForSirtB`, initialised to null.
    script_tilt_for_sirt_b: RefCell<Option<ComScript>>,
    /// Java field `scriptAutofidseedA`, initialised to null.
    script_autofidseed_a: RefCell<Option<ComScript>>,
    /// Java field `scriptAutofidseedB`, initialised to null.
    script_autofidseed_b: RefCell<Option<ComScript>>,
    /// Java field `scriptGoldEraserA`, initialised to null.
    script_gold_eraser_a: RefCell<Option<ComScript>>,
    /// Java field `scriptGoldEraserB`, initialised to null.
    script_gold_eraser_b: RefCell<Option<ComScript>>,
    /// Java field `scriptCryoPositionA`, initialised to null.
    script_cryo_position_a: RefCell<Option<ComScript>>,
    /// Java field `scriptCryoPositionB`, initialised to null.
    script_cryo_position_b: RefCell<Option<ComScript>>,
    /// Java field `scriptMultifiltSetupA`, initialised to null.
    script_multifilt_setup_a: RefCell<Option<ComScript>>,
    /// Java field `scriptMultifiltSetupB`, initialised to null.
    script_multifilt_setup_b: RefCell<Option<ComScript>>,
    /// Java field `scriptCtf3dSetupA`, initialised to null.
    script_ctf3d_setup_a: RefCell<Option<ComScript>>,
    /// Java field `scriptCtf3dSetupB`, initialised to null.
    script_ctf3d_setup_b: RefCell<Option<ComScript>>,
    /// Java field `scriptSubtomoSetup`, initialised to null.
    script_subtomo_setup: RefCell<Option<ComScript>>,
    /// Java field `scriptAltTomoSetup`, initialised to null.
    script_alt_tomo_setup: RefCell<Option<ComScript>>,
    /// Java field `scriptRestrictAlign`, initialised to null.
    script_restrict_align: RefCell<Option<ComScript>>,
    /// Java field `scriptReduceFiltVol`, initialised to null.
    script_reduce_filt_vol: RefCell<Option<ComScript>>,
}

impl ComScriptManager {
    /// Java `ComScriptManager(ApplicationManager)`.
    pub fn new(app_manager: &'static ApplicationManager) -> ComScriptManager {
        ComScriptManager {
            app_manager,
            ui_harness: &ui_harness::INSTANCE,
            script_eraser_a: RefCell::new(None),
            script_eraser_b: RefCell::new(None),
            script_xcorr_a: RefCell::new(None),
            script_xcorr_b: RefCell::new(None),
            script_prenewst_a: RefCell::new(None),
            script_prenewst_b: RefCell::new(None),
            script_track_a: RefCell::new(None),
            script_track_b: RefCell::new(None),
            script_align_a: RefCell::new(None),
            script_align_b: RefCell::new(None),
            script_newst_a: RefCell::new(None),
            script_newst_b: RefCell::new(None),
            script_tilt_a: RefCell::new(None),
            script_tilt_b: RefCell::new(None),
            script_mtf_filter_a: RefCell::new(None),
            script_mtf_filter_b: RefCell::new(None),
            script_preblend_a: RefCell::new(None),
            script_preblend_b: RefCell::new(None),
            script_blend_a: RefCell::new(None),
            script_blend_b: RefCell::new(None),
            script_undistort_a: RefCell::new(None),
            script_undistort_b: RefCell::new(None),
            script_matchvol1: RefCell::new(None),
            script_solvematch: RefCell::new(None),
            script_dualvolmatch: RefCell::new(None),
            script_solvematchshift: RefCell::new(None),
            script_solvematchmod: RefCell::new(None),
            script_patchcorr: RefCell::new(None),
            script_matchorwarp: RefCell::new(None),
            script_tomopitch_a: RefCell::new(None),
            script_tomopitch_b: RefCell::new(None),
            script_combine: RefCell::new(None),
            script_volcombine: RefCell::new(None),
            script_ctf_correction_a: RefCell::new(None),
            script_ctf_correction_b: RefCell::new(None),
            script_ctf_plotter_a: RefCell::new(None),
            script_ctf_plotter_b: RefCell::new(None),
            script_flatten: RefCell::new(None),
            script_newst3d_find_a: RefCell::new(None),
            script_newst3d_find_b: RefCell::new(None),
            script_blend3d_find_a: RefCell::new(None),
            script_blend3d_find_b: RefCell::new(None),
            script_tilt3d_find_a: RefCell::new(None),
            script_tilt3d_find_b: RefCell::new(None),
            script_find_beads3d_a: RefCell::new(None),
            script_find_beads3d_b: RefCell::new(None),
            script_tilt3d_find_reproject_a: RefCell::new(None),
            script_tilt3d_find_reproject_b: RefCell::new(None),
            script_xcorr_pt_a: RefCell::new(None),
            script_xcorr_pt_b: RefCell::new(None),
            script_sirtsetup_a: RefCell::new(None),
            script_sirtsetup_b: RefCell::new(None),
            script_tilt_for_sirt_a: RefCell::new(None),
            script_tilt_for_sirt_b: RefCell::new(None),
            script_autofidseed_a: RefCell::new(None),
            script_autofidseed_b: RefCell::new(None),
            script_gold_eraser_a: RefCell::new(None),
            script_gold_eraser_b: RefCell::new(None),
            script_cryo_position_a: RefCell::new(None),
            script_cryo_position_b: RefCell::new(None),
            script_multifilt_setup_a: RefCell::new(None),
            script_multifilt_setup_b: RefCell::new(None),
            script_ctf3d_setup_a: RefCell::new(None),
            script_ctf3d_setup_b: RefCell::new(None),
            script_subtomo_setup: RefCell::new(None),
            script_alt_tomo_setup: RefCell::new(None),
            script_restrict_align: RefCell::new(None),
            script_reduce_filt_vol: RefCell::new(None),
        }
    }

    /// Java `loadEraser`.  Load the specified eraser com script.
    pub fn load_eraser(&self, axis_id: AxisID) {
        // Assign the new ComScript object object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_eraser_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "eraser",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_eraser_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "eraser",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadGoldEraser`.  Returns true if script has loaded.
    pub fn load_gold_eraser(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_gold_eraser_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                file_type::CLASS
                    .gold_eraser_comscript
                    .get_file_name(Some(self.app_manager), Some(axis_id))
                    .as_deref(),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_gold_eraser_b.borrow().is_some();
        }
        *self.script_gold_eraser_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .gold_eraser_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_gold_eraser_a.borrow().is_some()
    }

    /// Java `loadMultifiltSetup`.  Returns true if script has loaded.
    pub fn load_multifilt_setup(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_multifilt_setup_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                file_type::CLASS
                    .multifilt_setup_comscript
                    .get_file_name(Some(self.app_manager), Some(axis_id))
                    .as_deref(),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_multifilt_setup_b.borrow().is_some();
        }
        *self.script_multifilt_setup_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .multifilt_setup_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_multifilt_setup_a.borrow().is_some()
    }

    /// Java `loadCtf3dSetup`.
    pub fn load_ctf3d_setup(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        let com_script = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .ctf_3d_setup_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        let loaded = com_script.is_some();
        if axis_id == AxisID::Second {
            *self.script_ctf3d_setup_b.borrow_mut() = com_script;
        } else {
            *self.script_ctf3d_setup_a.borrow_mut() = com_script;
        }
        loaded
    }

    /// Java `loadSubtomoSetup`.
    pub fn load_subtomo_setup(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        let com_script = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .subtomo_setup_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        let loaded = com_script.is_some();
        *self.script_subtomo_setup.borrow_mut() = com_script;
        loaded
    }

    /// Java `loadAltTomoSetup`.
    pub fn load_alt_tomo_setup(&self, axis_id: AxisID, required: bool) -> bool {
        let com_script = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .alt_tomo_setup_comscript
                .get_file_name_with_axis_type(
                    Some(self.app_manager),
                    None,
                    Some(AxisType::SingleAxis),
                    Some(axis_id),
                )
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        let loaded = com_script.is_some();
        *self.script_alt_tomo_setup.borrow_mut() = com_script;
        loaded
    }

    /// Java `loadRestrictAlign`.
    pub fn load_restrict_align(&self, axis_id: AxisID, required: bool) -> bool {
        let com_script = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .restrict_align_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        let loaded = com_script.is_some();
        *self.script_restrict_align.borrow_mut() = com_script;
        loaded
    }

    /// Java `loadReduceFiltVol`.
    pub fn load_reduce_filt_vol(&self, axis_id: AxisID, required: bool) -> bool {
        let com_script = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .reduce_filt_vol_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        let loaded = com_script.is_some();
        *self.script_reduce_filt_vol.borrow_mut() = com_script;
        loaded
    }

    /// Java `getSetEnvParamFromMultifiltSetup`.
    pub fn get_set_env_param_from_multifilt_setup(
        &self,
        axis_id: AxisID,
        env_var: &str,
    ) -> Option<SetEnvParam> {
        let com_script = if axis_id == AxisID::Second {
            &self.script_multifilt_setup_b
        } else {
            &self.script_multifilt_setup_a
        };
        // Initialize a SetEnvParam object from the com script command
        // object
        let mut param = SetEnvParam::new(Some(env_var));
        // Assuming its the first setenv command in the comfile.
        if !ComScriptUtil::initialize_previous_command_required_option(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            set_env_param::COMMAND_NAME,
            axis_id,
            true,
            None,
            true,
            false,
            false,
            false,
            Some(env_var),
        ) {
            return None;
        }
        Some(param)
    }

    /// Java `getMultifiltSetupParam`.
    pub fn get_multifilt_setup_param(&self, axis_id: AxisID) -> MultifiltSetupParam {
        // Get a reference to the appropriate script object
        let com_script = if axis_id == AxisID::Second {
            &self.script_multifilt_setup_b
        } else {
            &self.script_multifilt_setup_a
        };

        // Initialize a MultifiltSetupParam object from the com script command object
        let mut param = MultifiltSetupParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            &ProcessName::MULTIFILT_SETUP.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getCtf3dSetupParam`.
    pub fn get_ctf3d_setup_param(&self, axis_id: AxisID) -> Ctf3dSetupParam {
        // Get a reference to the appropriate script object
        let com_script = if axis_id == AxisID::Second {
            &self.script_ctf3d_setup_b
        } else {
            &self.script_ctf3d_setup_a
        };

        // Initialize a Ctf3dSetupParam object from the com script command object
        let mut param = Ctf3dSetupParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            &ProcessName::CTF_3D_SETUP.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getSubtomoSetupParam`.
    pub fn get_subtomo_setup_param(&self, axis_id: AxisID) -> SubtomoSetupParam {
        // Get a reference to the appropriate script object
        let com_script = &self.script_subtomo_setup;

        // Initialize a SubtomoSetupParam object from the com script command object
        let mut param = SubtomoSetupParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            &ProcessName::SUBTOMO_SETUP.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getAltTomoSetupParam`.
    pub fn get_alt_tomo_setup_param(&self, axis_id: AxisID) -> AltTomoSetupParam {
        let com_script = &self.script_alt_tomo_setup;
        let mut param = AltTomoSetupParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            &ProcessName::ALT_TOMO_SETUP.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getRestrictAlignParam`.
    pub fn get_restrict_align_param(&self, axis_id: AxisID) -> RestrictalignParam {
        let com_script = &self.script_restrict_align;
        let mut param = RestrictalignParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            &ProcessName::RESTRICTALIGN.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getReduceFiltVolParam`.
    pub fn get_reduce_filt_vol_param(&self, axis_id: AxisID) -> ReduceFiltVolParam {
        let com_script = &self.script_reduce_filt_vol;
        let mut param = ReduceFiltVolParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            &ProcessName::REDUCE_FILT_VOL.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveMultifiltSetup(SetEnvParam, AxisID, String)`.
    pub fn save_multifilt_setup_set_env(
        &self,
        param: &SetEnvParam,
        axis_id: AxisID,
        env_var: &str,
    ) {
        let script = if axis_id == AxisID::Second {
            &self.script_multifilt_setup_b
        } else {
            &self.script_multifilt_setup_a
        };
        ComScriptUtil::add_modify_command_required_option(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            set_env_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            Some(env_var),
        );
    }

    /// Java `saveMultifiltSetup(MultifiltSetupParam, AxisID)`.
    pub fn save_multifilt_setup_multifilt_setup(
        &self,
        param: &MultifiltSetupParam,
        axis_id: AxisID,
    ) {
        let script = if axis_id == AxisID::Second {
            &self.script_multifilt_setup_b
        } else {
            &self.script_multifilt_setup_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::MULTIFILT_SETUP.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveCtf3dSetup`.
    pub fn save_ctf3d_setup(&self, param: &Ctf3dSetupParam, axis_id: AxisID) {
        let script = if axis_id == AxisID::Second {
            &self.script_ctf3d_setup_b
        } else {
            &self.script_ctf3d_setup_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::CTF_3D_SETUP.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveSubtomoSetup`.
    pub fn save_subtomo_setup(&self, param: &SubtomoSetupParam) {
        let script = &self.script_subtomo_setup;

        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::SUBTOMO_SETUP.to_string(),
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `saveAltTomoSetup`.
    pub fn save_alt_tomo_setup(&self, param: &AltTomoSetupParam, axis_id: AxisID) {
        let script = &self.script_alt_tomo_setup;
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::ALT_TOMO_SETUP.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveRestrictAlign`.
    pub fn save_restrict_align(&self, param: &RestrictalignParam, axis_id: AxisID) {
        let script = &self.script_restrict_align;

        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::RESTRICTALIGN.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveReduceFiltVol`.
    pub fn save_reduce_filt_vol(&self, param: &ReduceFiltVolParam, axis_id: AxisID) {
        let script = &self.script_reduce_filt_vol;

        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::REDUCE_FILT_VOL.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadCryoPosition`.
    pub fn load_cryo_position(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_cryo_position_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                file_type::CLASS
                    .cryo_position_comscript
                    .get_file_name(Some(self.app_manager), Some(axis_id))
                    .as_deref(),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_cryo_position_b.borrow().is_some();
        }
        *self.script_cryo_position_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .cryo_position_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_cryo_position_a.borrow().is_some()
    }

    /// Java `getCryoPositionParam`.  (`axisType` is unused, as in the source.)
    pub fn get_cryo_position_param(
        &self,
        axis_id: AxisID,
        axis_type: AxisType,
    ) -> CryoPositionParam {
        let _ = axis_type;
        let script = if axis_id == AxisID::Second {
            &self.script_cryo_position_b
        } else {
            &self.script_cryo_position_a
        };
        // Initialize
        let mut param = CryoPositionParam::new(axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            script.borrow_mut().as_mut(),
            &ProcessName::CRYO_POSITION.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getCCDEraserParam`.  Get the CCD eraser parameters from the specified
    /// eraser script object.
    pub fn get_ccd_eraser_param(
        &self,
        axis_id: AxisID,
        mode: Option<ccd_eraser_param::Mode>,
    ) -> CCDEraserParam {
        // Get a reference to the appropriate script object
        let eraser = if axis_id == AxisID::Second {
            &self.script_eraser_b
        } else {
            &self.script_eraser_a
        };

        // Initialize a CCDEraserParam object from the com script command object
        let mut ccd_eraser_param = CCDEraserParam::new(self.app_manager, axis_id, mode);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut ccd_eraser_param,
            eraser.borrow_mut().as_mut(),
            "ccderaser",
            axis_id,
            false,
            false,
            true,
        );
        ccd_eraser_param
    }

    /// Java `getCCDEraserParamFromGoldEraser`.
    pub fn get_ccd_eraser_param_from_gold_eraser(
        &self,
        axis_id: AxisID,
        mode: Option<ccd_eraser_param::Mode>,
    ) -> CCDEraserParam {
        // Get a reference to the appropriate script object
        let script = if axis_id == AxisID::Second {
            &self.script_gold_eraser_b
        } else {
            &self.script_gold_eraser_a
        };

        // Initialize a param object from the com script command object
        let mut param = CCDEraserParam::new(self.app_manager, axis_id, mode);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            script.borrow_mut().as_mut(),
            &ccd_eraser_param::command_name(),
            axis_id,
            false,
            false,
            true,
        );
        param.is_expand_circle_iterations_set();
        param
    }

    /// Java `getGoldEraserParam`.  Get the CCD eraser parameters from the gold eraser
    /// script object.
    pub fn get_gold_eraser_param(
        &self,
        axis_id: AxisID,
        mode: Option<ccd_eraser_param::Mode>,
    ) -> CCDEraserParam {
        // Get a reference to the appropriate script object
        let eraser = if axis_id == AxisID::Second {
            &self.script_gold_eraser_b
        } else {
            &self.script_gold_eraser_a
        };

        // Initialize a CCDEraserParam object from the com script command object
        let mut ccd_eraser_param = CCDEraserParam::new(self.app_manager, axis_id, mode);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut ccd_eraser_param,
            eraser.borrow_mut().as_mut(),
            "ccderaser",
            axis_id,
            false,
            false,
            true,
        );
        ccd_eraser_param
    }

    /// Java `saveEraser`.  Save the specified eraser com script updating the ccderaser
    /// parmaeters.
    pub fn save_eraser(&self, ccd_eraser_param: &CCDEraserParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_eraser = if axis_id == AxisID::Second {
            &self.script_eraser_b
        } else {
            &self.script_eraser_a
        };

        // update the ccderaser parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_eraser.borrow_mut().as_mut(),
            ccd_eraser_param,
            "ccderaser",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveGoldEraser`.  Save the specified gold eraser com script updating the
    /// ccderaser parmaeters.
    pub fn save_gold_eraser(&self, ccd_eraser_param: &CCDEraserParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_eraser = if axis_id == AxisID::Second {
            &self.script_gold_eraser_b
        } else {
            &self.script_gold_eraser_a
        };

        // update the ccderaser parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_eraser.borrow_mut().as_mut(),
            ccd_eraser_param,
            "ccderaser",
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadXcorr`.  Load the specified xcorr com script.
    pub fn load_xcorr(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_xcorr_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "xcorr",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_xcorr_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "xcorr",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadUndistort`.  Load or create undistort.com.
    pub fn load_undistort(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.script_undistort_b.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Undistort),
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_undistort_a.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Undistort),
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `getTiltxcorrParamFromXcorrPt`.  Get the tiltxcorr parameters from the
    /// specified xcorr_pt script object.
    pub fn get_tiltxcorr_param_from_xcorr_pt(&self, axis_id: AxisID) -> TiltxcorrParam {
        // Get a reference to the appropriate script object
        let xcorr_pt = if axis_id == AxisID::Second {
            &self.script_xcorr_pt_b
        } else {
            &self.script_xcorr_pt_a
        };

        // Initialize a TiltxcorrParam object from the com script command object
        let mut tilt_xcorr_param =
            TiltxcorrParam::new(self.app_manager, axis_id, ProcessName::XCORR_PT);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tilt_xcorr_param,
            xcorr_pt.borrow_mut().as_mut(),
            "tiltxcorr",
            axis_id,
            false,
            false,
            true,
        );
        tilt_xcorr_param
    }

    /// Java `getImodchopcontsParam`.
    pub fn get_imodchopconts_param(&self, axis_id: AxisID) -> ImodchopcontsParam {
        // Get a reference to the appropriate script object
        let xcorr_pt = if axis_id == AxisID::Second {
            &self.script_xcorr_pt_b
        } else {
            &self.script_xcorr_pt_a
        };

        // Initialize a TiltxcorrParam object from the com script command object
        let mut param = ImodchopcontsParam::new();
        ComScriptUtil::initialize_optional_command(
            self.app_manager,
            &mut param,
            xcorr_pt.borrow_mut().as_mut(),
            "imodchopconts",
            axis_id,
            false,
            false,
            true,
            true,
            None,
        );
        param
    }

    /// Java `getGotoParamFromXcorrPt`.  Get the first goto command from xcorr.com.
    pub fn get_goto_param_from_xcorr_pt(
        &self,
        axis_id: AxisID,
        required: bool,
    ) -> Option<GotoParam> {
        let xcorr_pt = if axis_id == AxisID::Second {
            &self.script_xcorr_pt_b
        } else {
            &self.script_xcorr_pt_a
        };
        // Initialize a GotoParam object from the com script command
        // object
        let mut goto_param = GotoParam::new();
        if !ComScriptUtil::initialize(
            self.app_manager,
            &mut goto_param,
            xcorr_pt.borrow_mut().as_mut(),
            goto_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            required,
        ) {
            return None;
        }
        Some(goto_param)
    }

    /// Java `getSetEnvParamFromXcorr`.
    pub fn get_set_env_param_from_xcorr(
        &self,
        axis_id: AxisID,
        name: &str,
        required: bool,
    ) -> Option<SetEnvParam> {
        let xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };
        // Initialize a SetEnvParam object from the com script command
        // object
        let mut param = SetEnvParam::new(Some(name));
        // Assuming its the first setenv command in the comfile.
        if !ComScriptUtil::initialize_previous_command(
            self.app_manager,
            &mut param,
            xcorr.borrow_mut().as_mut(),
            set_env_param::COMMAND_NAME,
            axis_id,
            true,
            None,
            true,
            false,
            false,
            required,
        ) {
            return None;
        }
        Some(param)
    }

    /// Java `getAutofidseedParam`.  Get the autofidseed parameters.
    pub fn get_autofidseed_param(&self, axis_id: AxisID) -> AutofidseedParam {
        // Get a reference to the appropriate script object
        let com_script = if axis_id == AxisID::Second {
            &self.script_autofidseed_b
        } else {
            &self.script_autofidseed_a
        };

        // Initialize a TiltxcorrParam object from the com script command object
        let mut param = AutofidseedParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            com_script.borrow_mut().as_mut(),
            "autofidseed",
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getTiltxcorrParam`.  Get the tiltxcorr parameters from the specified xcorr
    /// script object.
    pub fn get_tiltxcorr_param(&self, axis_id: AxisID) -> TiltxcorrParam {
        // Get a reference to the appropriate script object
        let xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };

        // Initialize a TiltxcorrParam object from the com script command object
        let mut tilt_xcorr_param =
            TiltxcorrParam::new(self.app_manager, axis_id, ProcessName::XCORR);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tilt_xcorr_param,
            xcorr.borrow_mut().as_mut(),
            "tiltxcorr",
            axis_id,
            false,
            false,
            true,
        );
        tilt_xcorr_param
    }

    /// Java `saveXcorrPt(TiltxcorrParam, AxisID)`.  Save the specified xcorr_pt com
    /// script updating the tiltxcorr parameters.
    pub fn save_xcorr_pt_tiltxcorr(&self, tilt_xcorr_param: &TiltxcorrParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_xcorr_pt = if axis_id == AxisID::Second {
            &self.script_xcorr_pt_b
        } else {
            &self.script_xcorr_pt_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_xcorr_pt.borrow_mut().as_mut(),
            tilt_xcorr_param,
            "tiltxcorr",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXcorrPt(ImodchopcontsParam, AxisID)`.
    pub fn save_xcorr_pt_imodchopconts(&self, param: &ImodchopcontsParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_xcorr_pt = if axis_id == AxisID::Second {
            &self.script_xcorr_pt_b
        } else {
            &self.script_xcorr_pt_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_xcorr_pt.borrow_mut().as_mut(),
            param,
            "imodchopconts",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXcorrPt(GotoParam, AxisID)`.
    pub fn save_xcorr_pt_goto(&self, param: &GotoParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_xcorr_pt = if axis_id == AxisID::Second {
            &self.script_xcorr_pt_b
        } else {
            &self.script_xcorr_pt_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_xcorr_pt.borrow_mut().as_mut(),
            param,
            "goto",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveAutofidseed`.  Save the specified autofidseed com script, updating the
    /// autofidseed parameters.
    pub fn save_autofidseed(&self, param: &AutofidseedParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let com_script = if axis_id == AxisID::Second {
            &self.script_autofidseed_b
        } else {
            &self.script_autofidseed_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            com_script.borrow_mut().as_mut(),
            param,
            "autofidseed",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXcorr(TiltxcorrParam, AxisID)`.  Save the specified xcorr com script
    /// updating the tiltxcorr parameters.
    pub fn save_xcorr_tiltxcorr(&self, tilt_xcorr_param: &TiltxcorrParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_xcorr.borrow_mut().as_mut(),
            tilt_xcorr_param,
            "tiltxcorr",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXcorr(BlendmontParam, AxisID)`.  Save the blendmont param object to
    /// xcorr.com.
    pub fn save_xcorr_blendmont(&self, blendmont_param: &BlendmontParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_xcorr.borrow_mut().as_mut(),
            blendmont_param,
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXcorrToUndistort`.
    pub fn save_xcorr_to_undistort(&self, blendmont_param: &BlendmontParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let (script_undistort, script_xcorr) = if axis_id == AxisID::Second {
            (&self.script_undistort_b, &self.script_xcorr_b)
        } else {
            (&self.script_undistort_a, &self.script_xcorr_a)
        };
        ComScriptUtil::add_modify_command_from_to(
            self.app_manager,
            script_xcorr.borrow_mut().as_mut(),
            script_undistort.borrow_mut().as_mut(),
            blendmont_param,
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXcorr(GotoParam, AxisID)`.  Save the goto param object to xcorr.com.
    /// Saves to the first instance of the goto command in xcorr.com.
    pub fn save_xcorr_goto(&self, goto_param: &GotoParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_xcorr.borrow_mut().as_mut(),
            goto_param,
            goto_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadPrenewst`.  Load the specified prenewst com script.
    pub fn load_prenewst(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_prenewst_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "prenewst",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_prenewst_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "prenewst",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadPreblend`.
    pub fn load_preblend(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_preblend_b.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Preblend),
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_preblend_a.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Preblend),
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadBlend`.
    pub fn load_blend(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_blend_b.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Blend),
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_blend_a.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                BlendmontParam::get_process_name_for_mode(blendmont_param::Mode::Blend),
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadBlend3dFind`.
    pub fn load_blend3d_find(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_blend3d_find_b.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                ProcessName::BLEND_3D_FIND,
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_blend3d_find_a.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                ProcessName::BLEND_3D_FIND,
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadAutofidseed`.  Returns true if script has loaded.
    pub fn load_autofidseed(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_autofidseed_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                file_type::CLASS
                    .autofidseed_comscript
                    .get_file_name(Some(self.app_manager), Some(axis_id))
                    .as_deref(),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_autofidseed_b.borrow().is_some();
        }
        *self.script_autofidseed_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .autofidseed_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_autofidseed_a.borrow().is_some()
    }

    /// Java `loadXcorrPt`.  Returns true if script has loaded.
    pub fn load_xcorr_pt(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_xcorr_pt_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                file_type::CLASS
                    .patch_tracking_comscript
                    .get_file_name(Some(self.app_manager), Some(axis_id))
                    .as_deref(),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_xcorr_pt_b.borrow().is_some();
        }
        *self.script_xcorr_pt_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .patch_tracking_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_xcorr_pt_a.borrow().is_some()
    }

    /// Java `loadSirtsetup`.
    pub fn load_sirtsetup(&self, axis_id: AxisID) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_sirtsetup_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                file_type::CLASS
                    .sirtsetup_comscript
                    .get_file_name(Some(self.app_manager), Some(axis_id))
                    .as_deref(),
                axis_id,
                true,
                false,
                false,
                false,
            );
            return self.script_sirtsetup_b.borrow().is_some();
        }
        *self.script_sirtsetup_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            file_type::CLASS
                .sirtsetup_comscript
                .get_file_name(Some(self.app_manager), Some(axis_id))
                .as_deref(),
            axis_id,
            true,
            false,
            false,
            false,
        );
        self.script_sirtsetup_a.borrow().is_some()
    }

    /// Java `getSirtsetupParam`.
    pub fn get_sirtsetup_param(&self, axis_id: AxisID) -> SirtsetupParam {
        // Get a reference to the appropriate script object
        let sirtsetup = if axis_id == AxisID::Second {
            &self.script_sirtsetup_b
        } else {
            &self.script_sirtsetup_a
        };

        // Initialize a SirtsetupParam object from the com script command object
        let mut param = SirtsetupParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            sirtsetup.borrow_mut().as_mut(),
            &ProcessName::SIRTSETUP.to_string(),
            axis_id,
            true,
            true,
            true,
        );
        param
    }

    /// Java `saveSirtsetup`.
    pub fn save_sirtsetup(&self, param: &SirtsetupParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script = if axis_id == AxisID::Second {
            &self.script_sirtsetup_b
        } else {
            &self.script_sirtsetup_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::SIRTSETUP.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadCtfPlotter`.  Returns true if script has loaded.
    pub fn load_ctf_plotter(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_ctf_plotter_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                Some(&ProcessName::CTF_PLOTTER.get_comscript(axis_id)),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_ctf_plotter_b.borrow().is_some();
        }
        *self.script_ctf_plotter_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            Some(&ProcessName::CTF_PLOTTER.get_comscript(axis_id)),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_ctf_plotter_a.borrow().is_some()
    }

    /// Java `loadCtfCorrection`.
    pub fn load_ctf_correction(&self, axis_id: AxisID, required: bool) -> bool {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_ctf_correction_b.borrow_mut() = ComScriptUtil::load_com_script_file_name(
                self.app_manager,
                Some(&ProcessName::CTF_CORRECTION.get_comscript(axis_id)),
                axis_id,
                true,
                required,
                false,
                false,
            );
            return self.script_ctf_correction_b.borrow().is_some();
        }
        *self.script_ctf_correction_a.borrow_mut() = ComScriptUtil::load_com_script_file_name(
            self.app_manager,
            Some(&ProcessName::CTF_CORRECTION.get_comscript(axis_id)),
            axis_id,
            true,
            required,
            false,
            false,
        );
        self.script_ctf_correction_a.borrow().is_some()
    }

    /// Java `getPrenewstParam`.  Get the newstack parameters from the specified
    /// prenewst script object.
    pub fn get_prenewst_param(&self, axis_id: AxisID) -> NewstParam {
        // Get a reference to the appropriate script object
        let script_prenewst = if axis_id == AxisID::Second {
            &self.script_prenewst_b
        } else {
            &self.script_prenewst_a
        };

        // Initialize a NewstParam object from the com script command object
        let mut prenewst_param = NewstParam::get_instance(self.app_manager, axis_id);

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        let cmd_name = self.newst_or_newstack(script_prenewst.borrow().as_ref());
        ComScriptUtil::initialize(
            self.app_manager,
            &mut prenewst_param,
            script_prenewst.borrow_mut().as_mut(),
            &cmd_name,
            axis_id,
            false,
            false,
            true,
        );
        prenewst_param
    }

    /// Java `getPreblendParam`.
    pub fn get_preblend_param(&self, axis_id: AxisID) -> BlendmontParam {
        // Get a reference to the appropriate script object
        let script_preblend = if axis_id == AxisID::Second {
            &self.script_preblend_b
        } else {
            &self.script_preblend_a
        };

        // Initialize a BlendmontParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut preblend_param = BlendmontParam::new_with_mode(
            self.app_manager,
            Some(&dataset_name),
            axis_id,
            blendmont_param::Mode::Preblend,
        );
        ComScriptUtil::initialize(
            self.app_manager,
            &mut preblend_param,
            script_preblend.borrow_mut().as_mut(),
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        preblend_param
    }

    /// Java `getBlendParam`.
    pub fn get_blend_param(&self, axis_id: AxisID) -> BlendmontParam {
        // Get a reference to the appropriate script object
        let script_blend = if axis_id == AxisID::Second {
            &self.script_blend_b
        } else {
            &self.script_blend_a
        };

        // Initialize a BlendmontParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut blend_param = BlendmontParam::new_with_mode(
            self.app_manager,
            Some(&dataset_name),
            axis_id,
            blendmont_param::Mode::Blend,
        );
        ComScriptUtil::initialize(
            self.app_manager,
            &mut blend_param,
            script_blend.borrow_mut().as_mut(),
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        blend_param
    }

    /// Java `getBlendParamFromBlend3dFind`.  Get BlendmontParam from blend_3dfind.com.
    pub fn get_blend_param_from_blend3d_find(&self, axis_id: AxisID) -> BlendmontParam {
        // Get a reference to the appropriate script object
        let script_blend3d_find = if axis_id == AxisID::Second {
            &self.script_blend3d_find_b
        } else {
            &self.script_blend3d_find_a
        };

        // Initialize a BlendmontParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut blend_param = BlendmontParam::new_with_mode(
            self.app_manager,
            Some(&dataset_name),
            axis_id,
            blendmont_param::Mode::Blend3dFind,
        );
        ComScriptUtil::initialize(
            self.app_manager,
            &mut blend_param,
            script_blend3d_find.borrow_mut().as_mut(),
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        blend_param
    }

    /// Java `getMrcTaperParamFromBlend3dFind`.  Get MrcTaperParam from
    /// blend_3dfind.com.
    pub fn get_mrc_taper_param_from_blend3d_find(&self, axis_id: AxisID) -> MrcTaperParam {
        // Get a reference to the appropriate script object
        let script_blend3d_find = if axis_id == AxisID::Second {
            &self.script_blend3d_find_b
        } else {
            &self.script_blend3d_find_a
        };

        // Initialize a MrcTaperParam object from the com script command object
        let mut param = MrcTaperParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            script_blend3d_find.borrow_mut().as_mut(),
            mrc_taper_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `savePrenewst`.  Save the specified prenewst com script updating the newst
    /// parameters.
    pub fn save_prenewst(&self, prenewst_param: &NewstParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_prenewst = if axis_id == AxisID::Second {
            &self.script_prenewst_b
        } else {
            &self.script_prenewst_a
        };

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        let cmd_name = self.newst_or_newstack(script_prenewst.borrow().as_ref());
        ComScriptUtil::modify_command(
            self.app_manager,
            script_prenewst.borrow_mut().as_mut(),
            prenewst_param,
            &cmd_name,
            axis_id,
            false,
            false,
        );
    }

    /// Java `savePreblend`.
    pub fn save_preblend(&self, blendmont_param: &BlendmontParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_preblend = if axis_id == AxisID::Second {
            &self.script_preblend_b
        } else {
            &self.script_preblend_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_preblend.borrow_mut().as_mut(),
            blendmont_param,
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveCryoPosition`.
    pub fn save_cryo_position(&self, param: &CryoPositionParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script = if axis_id == AxisID::Second {
            &self.script_cryo_position_b
        } else {
            &self.script_cryo_position_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::CRYO_POSITION.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveBlend`.
    pub fn save_blend(&self, blendmont_param: &BlendmontParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_blend = if axis_id == AxisID::Second {
            &self.script_blend_b
        } else {
            &self.script_blend_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_blend.borrow_mut().as_mut(),
            blendmont_param,
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveBlend3dFind(BlendmontParam, AxisID)`.
    pub fn save_blend3d_find_blendmont(&self, blendmont_param: &BlendmontParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_blend3d_find = if axis_id == AxisID::Second {
            &self.script_blend3d_find_b
        } else {
            &self.script_blend3d_find_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_blend3d_find.borrow_mut().as_mut(),
            blendmont_param,
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveBlend3dFind(MrcTaperParam, AxisID)`.
    pub fn save_blend3d_find_mrc_taper(&self, param: &MrcTaperParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_blend3d_find = if axis_id == AxisID::Second {
            &self.script_blend3d_find_b
        } else {
            &self.script_blend3d_find_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_blend3d_find.borrow_mut().as_mut(),
            param,
            mrc_taper_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveFindBeads3d`.
    pub fn save_find_beads3d(&self, param: &FindBeads3dParam, axis_id: AxisID) {
        let script = if axis_id == AxisID::Second {
            &self.script_find_beads3d_b
        } else {
            &self.script_find_beads3d_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::FIND_BEADS_3D.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveCtfPlotter`.
    pub fn save_ctf_plotter(&self, param: &CtfPlotterParam, axis_id: AxisID) {
        let script = if axis_id == AxisID::Second {
            &self.script_ctf_plotter_b
        } else {
            &self.script_ctf_plotter_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            &ProcessName::CTF_PLOTTER.to_string(),
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveCtfPhaseFlip`.
    pub fn save_ctf_phase_flip(&self, param: &CtfPhaseFlipParam, axis_id: AxisID) {
        let script = if axis_id == AxisID::Second {
            &self.script_ctf_correction_b
        } else {
            &self.script_ctf_correction_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script.borrow_mut().as_mut(),
            param,
            ctf_phase_flip_param::COMMAND,
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadTrack`.  Load the specified track com script.
    pub fn load_track(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_track_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "track",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_track_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "track",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadFindBeads3d`.  Load the specified findbeads3d com script.
    pub fn load_find_beads3d(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_find_beads3d_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                &ProcessName::FIND_BEADS_3D.to_string(),
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_find_beads3d_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                &ProcessName::FIND_BEADS_3D.to_string(),
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `getBeadtrackParam`.  Get the beadtrack parameters from the specified track
    /// script object.
    pub fn get_beadtrack_param(&self, axis_id: AxisID) -> BeadtrackParam {
        // Get a reference to the appropriate script object
        let track = if axis_id == AxisID::Second {
            &self.script_track_b
        } else {
            &self.script_track_a
        };

        // Initialize a BeadtrckParam object from the com script command object
        let mut beadtrack_param = BeadtrackParam::new(axis_id, self.app_manager);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut beadtrack_param,
            track.borrow_mut().as_mut(),
            "beadtrack",
            axis_id,
            false,
            false,
            true,
        );
        beadtrack_param
    }

    /// Java `saveTrack`.  Save the specified track com script updating the beadtrack
    /// parameters.
    pub fn save_track(&self, beadtrack_param: &BeadtrackParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_track = if axis_id == AxisID::Second {
            &self.script_track_b
        } else {
            &self.script_track_a
        };
        // update the beadtrack parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_track.borrow_mut().as_mut(),
            beadtrack_param,
            "beadtrack",
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadAlign`.  Load the specified align com script object.
    pub fn load_align(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_align_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "align",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_align_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "align",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `resetAlign`.
    pub fn reset_align(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.script_align_b.borrow_mut() = None;
        } else {
            *self.script_align_a.borrow_mut() = None;
        }
    }

    /// Java `getTiltalignParam`.  Get the tiltalign parameters from the specified align
    /// script object.
    pub fn get_tiltalign_param(&self, axis_id: AxisID) -> TiltalignParam {
        // Get a reference to the appropriate script object
        let align = if axis_id == AxisID::Second {
            &self.script_align_b
        } else {
            &self.script_align_a
        };

        // Initialize a BeadtrckParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut tiltalign_param =
            TiltalignParam::new(self.app_manager, Some(&dataset_name), axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tiltalign_param,
            align.borrow_mut().as_mut(),
            "tiltalign",
            axis_id,
            false,
            false,
            true,
        );
        tiltalign_param
    }

    /// Java `getXfproductInAlign`.  Get the xfproduct parameter from the align script.
    pub fn get_xfproduct_in_align(&self, axis_id: AxisID) -> XfproductParam {
        // Get a reference to the appropriate script object
        let align = if axis_id == AxisID::Second {
            &self.script_align_b
        } else {
            &self.script_align_a
        };

        // Initialize a BeadtrckParam object from the com script command object
        let mut xfproduct_param = XfproductParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut xfproduct_param,
            align.borrow_mut().as_mut(),
            "xfproduct",
            axis_id,
            false,
            false,
            true,
        );
        xfproduct_param
    }

    /// Java `saveAlign`.  Save the specified align com script updating the tiltalign
    /// parameters.
    pub fn save_align(&self, tiltalign_param: &TiltalignParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_align = if axis_id == AxisID::Second {
            &self.script_align_b
        } else {
            &self.script_align_a
        };

        // update the tiltalign parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_align.borrow_mut().as_mut(),
            tiltalign_param,
            "tiltalign",
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveXfproductInAlign`.  Save the xfproduct command to the specified align
    /// com script.
    pub fn save_xfproduct_in_align(&self, xfproduct_param: &XfproductParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_align = if axis_id == AxisID::Second {
            &self.script_align_b
        } else {
            &self.script_align_a
        };

        // update the tiltalign parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_align.borrow_mut().as_mut(),
            xfproduct_param,
            "xfproduct",
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadNewst`.  Load the specified newst com script.
    pub fn load_newst(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_newst_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "newst",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_newst_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "newst",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `loadNewst3dFind`.  Load the specified newst_3dfind com script.
    pub fn load_newst3d_find(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_newst3d_find_b.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                ProcessName::NEWST_3D_FIND,
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_newst3d_find_a.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                ProcessName::NEWST_3D_FIND,
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `isScriptNewst3dFindNull`.
    pub fn is_script_newst3d_find_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.script_newst3d_find_b.borrow().is_none();
        }
        self.script_newst3d_find_a.borrow().is_none()
    }

    /// Java `isScriptBlend3dFindNull`.
    pub fn is_script_blend3d_find_null(&self, axis_id: AxisID) -> bool {
        if axis_id == AxisID::Second {
            return self.script_blend3d_find_b.borrow().is_none();
        }
        self.script_blend3d_find_a.borrow().is_none()
    }

    /// Java `getNewstComNewstParam`.  Get the newst parameters from the specified newst
    /// script object.
    pub fn get_newst_com_newst_param(&self, axis_id: AxisID) -> NewstParam {
        // Get a reference to the appropriate script object
        let script_newst = if axis_id == AxisID::Second {
            &self.script_newst_b
        } else {
            &self.script_newst_a
        };

        // Initialize a NewstParam object from the com script command object
        let mut newst_param = NewstParam::get_instance(self.app_manager, axis_id);

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        let cmd_name = self.newst_or_newstack(script_newst.borrow().as_ref());
        ComScriptUtil::initialize(
            self.app_manager,
            &mut newst_param,
            script_newst.borrow_mut().as_mut(),
            &cmd_name,
            axis_id,
            false,
            false,
            true,
        );
        newst_param
    }

    /// Java `getNewstParamFromNewst3dFind`.  Get the newst_3dfind parameters from the
    /// specified newst_3dfind script object.
    pub fn get_newst_param_from_newst3d_find(&self, axis_id: AxisID) -> NewstParam {
        // Get a reference to the appropriate script object
        let script_newst3dfind = if axis_id == AxisID::Second {
            &self.script_newst3d_find_b
        } else {
            &self.script_newst3d_find_a
        };

        // Initialize a NewstParam object from the com script command object
        let mut newst_param = NewstParam::get_instance(self.app_manager, axis_id);

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        let cmd_name = self.newst_or_newstack(script_newst3dfind.borrow().as_ref());
        ComScriptUtil::initialize(
            self.app_manager,
            &mut newst_param,
            script_newst3dfind.borrow_mut().as_mut(),
            &cmd_name,
            axis_id,
            false,
            false,
            true,
        );
        newst_param
    }

    /// Java `getFindBeads3dParam`.
    pub fn get_find_beads3d_param(&self, axis_id: AxisID) -> FindBeads3dParam {
        let script = if axis_id == AxisID::Second {
            &self.script_find_beads3d_b
        } else {
            &self.script_find_beads3d_a
        };
        let mut param = FindBeads3dParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            script.borrow_mut().as_mut(),
            &ProcessName::FIND_BEADS_3D.to_string(),
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `getMrcTaperParamFromNewst3dFind`.  Get the newst_3dfind parameters from
    /// the specified newst_3dfind script object.
    pub fn get_mrc_taper_param_from_newst3d_find(&self, axis_id: AxisID) -> MrcTaperParam {
        // Get a reference to the appropriate script object
        let script_newst3dfind = if axis_id == AxisID::Second {
            &self.script_newst3d_find_b
        } else {
            &self.script_newst3d_find_a
        };

        // Initialize a NewstParam object from the com script command object
        let mut param = MrcTaperParam::new(self.app_manager, axis_id);

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            script_newst3dfind.borrow_mut().as_mut(),
            mrc_taper_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveNewst`.  Save the specified newst com script updating the newst
    /// parameters.
    pub fn save_newst(&self, newst_param: &NewstParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_newst = if axis_id == AxisID::Second {
            &self.script_newst_b
        } else {
            &self.script_newst_a
        };

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        let cmd_name = self.newst_or_newstack(script_newst.borrow().as_ref());

        // update the newst parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_newst.borrow_mut().as_mut(),
            newst_param,
            &cmd_name,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveNewst3dFind(NewstParam, AxisID)`.  Save the specified newst_3dfind com
    /// script updating the newst_3dfind parameters.
    pub fn save_newst3d_find_newst(&self, newst_param: &NewstParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_newst3d_find = if axis_id == AxisID::Second {
            &self.script_newst3d_find_b
        } else {
            &self.script_newst3d_find_a
        };

        // Implementation note: since the name of the command newst was changed to
        // newstack we need to figure out which one it is before calling initialize.
        let cmd_name = self.newst_or_newstack(script_newst3d_find.borrow().as_ref());

        // update the newst parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_newst3d_find.borrow_mut().as_mut(),
            newst_param,
            &cmd_name,
            axis_id,
            false,
            false,
        );
    }

    /// Java `saveNewst3dFind(MrcTaperParam, AxisID)`.  Save the specified newst_3dfind
    /// com script updating the MrcTaperParam command.
    pub fn save_newst3d_find_mrc_taper(&self, param: &MrcTaperParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_newst3d_find = if axis_id == AxisID::Second {
            &self.script_newst3d_find_b
        } else {
            &self.script_newst3d_find_a
        };
        // update the newst parameters
        ComScriptUtil::modify_command(
            self.app_manager,
            script_newst3d_find.borrow_mut().as_mut(),
            param,
            mrc_taper_param::COMMAND_NAME,
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadTilt`.  Load the specified tilt com script.
    pub fn load_tilt(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_tilt_b.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                ProcessName::TILT,
                axis_id,
                false,
                true,
                true,
            );
        } else {
            *self.script_tilt_a.borrow_mut() = ComScriptUtil::load_com_script_process_name(
                self.app_manager,
                ProcessName::TILT,
                axis_id,
                false,
                true,
                true,
            );
        }
    }

    /// Java `loadTiltForSirt`.  Load the specified tilt_for_sirt com script.
    pub fn load_tilt_for_sirt(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_tilt_for_sirt_b.borrow_mut() = ComScriptUtil::load_com_script_file_type(
                self.app_manager,
                &file_type::CLASS.tilt_for_sirt_comscript,
                axis_id,
                false,
                true,
                true,
            );
        } else {
            *self.script_tilt_for_sirt_a.borrow_mut() = ComScriptUtil::load_com_script_file_type(
                self.app_manager,
                &file_type::CLASS.tilt_for_sirt_comscript,
                axis_id,
                false,
                true,
                true,
            );
        }
    }

    /// Java `loadTilt3dFind`.  Load the specified tilt_3dfind com script.
    pub fn load_tilt3d_find(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_tilt3d_find_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "tilt_3dfind",
                axis_id,
                false,
                true,
                true,
            );
        } else {
            *self.script_tilt3d_find_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "tilt_3dfind",
                axis_id,
                false,
                true,
                true,
            );
        }
    }

    /// Java `loadTilt3dFindReproject`.  Load the specified tilt_3dfind_reproject com
    /// script.
    pub fn load_tilt3d_find_reproject(&self, axis_id: AxisID) {
        if axis_id == AxisID::Second {
            *self.script_tilt3d_find_reproject_b.borrow_mut() =
                ComScriptUtil::load_com_script_process_name(
                    self.app_manager,
                    ProcessName::TILT_3D_FIND_REPROJECT,
                    axis_id,
                    false,
                    true,
                    true,
                );
        } else {
            *self.script_tilt3d_find_reproject_a.borrow_mut() =
                ComScriptUtil::load_com_script_process_name(
                    self.app_manager,
                    ProcessName::TILT_3D_FIND_REPROJECT,
                    axis_id,
                    false,
                    true,
                    true,
                );
        }
    }

    /// Java `getTiltParam`.  Get the tilt parameters from the specified tilt script
    /// object.
    pub fn get_tilt_param(&self, axis_id: AxisID) -> TiltParam {
        // Get a reference to the appropriate script object
        let tilt = if axis_id == AxisID::Second {
            &self.script_tilt_b
        } else {
            &self.script_tilt_a
        };

        // Initialize a TiltParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut tilt_param = TiltParam::new(self.app_manager, Some(&dataset_name), axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tilt_param,
            tilt.borrow_mut().as_mut(),
            "tilt",
            axis_id,
            true,
            true,
            true,
        );
        tilt_param
    }

    /// Java `getTiltParamFromTiltForSirt`.  Get the tilt parameters from the tilt for
    /// sirt com script.
    pub fn get_tilt_param_from_tilt_for_sirt(&self, axis_id: AxisID) -> TiltParam {
        // Get a reference to the appropriate script object
        let com_script = if axis_id == AxisID::Second {
            &self.script_tilt_for_sirt_b
        } else {
            &self.script_tilt_for_sirt_a
        };
        // Initialize a TiltParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut tilt_param = TiltParam::new(self.app_manager, Some(&dataset_name), axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tilt_param,
            com_script.borrow_mut().as_mut(),
            "tilt",
            axis_id,
            true,
            true,
            true,
        );
        tilt_param
    }

    /// Java `getTiltParamFromTilt3dFind`.  Get the tilt parameters from the
    /// tilt_3dfind com script.
    pub fn get_tilt_param_from_tilt3d_find(&self, axis_id: AxisID) -> TiltParam {
        // Get a reference to the appropriate script object
        let tilt = if axis_id == AxisID::Second {
            &self.script_tilt3d_find_b
        } else {
            &self.script_tilt3d_find_a
        };

        // Initialize a TiltParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut tilt_param = TiltParam::new(self.app_manager, Some(&dataset_name), axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tilt_param,
            tilt.borrow_mut().as_mut(),
            "tilt",
            axis_id,
            true,
            true,
            true,
        );
        tilt_param
    }

    /// Java `getTiltParamFromTilt3dFindReproject`.  Get the tilt parameters from the
    /// tilt_3dfind_reproject com script.
    pub fn get_tilt_param_from_tilt3d_find_reproject(&self, axis_id: AxisID) -> TiltParam {
        // Get a reference to the appropriate script object
        let tilt = if axis_id == AxisID::Second {
            &self.script_tilt3d_find_reproject_b
        } else {
            &self.script_tilt3d_find_reproject_a
        };

        // Initialize a TiltParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut tilt_param = TiltParam::new(self.app_manager, Some(&dataset_name), axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tilt_param,
            tilt.borrow_mut().as_mut(),
            "tilt",
            axis_id,
            true,
            true,
            true,
        );
        tilt_param
    }

    /// Java `saveTilt`.  Save the specified tilt com script updating the tilt
    /// parameters.
    pub fn save_tilt(&self, tilt_param: &TiltParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_tilt = if axis_id == AxisID::Second {
            &self.script_tilt_b
        } else {
            &self.script_tilt_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_tilt.borrow_mut().as_mut(),
            tilt_param,
            "tilt",
            axis_id,
            true,
            true,
        );
    }

    /// Java `saveTilt3dFind`.  Save the specified tilt_3dfind com script updating the
    /// tilt parameters.
    pub fn save_tilt3d_find(&self, tilt_param: &TiltParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_tilt = if axis_id == AxisID::Second {
            &self.script_tilt3d_find_b
        } else {
            &self.script_tilt3d_find_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_tilt.borrow_mut().as_mut(),
            tilt_param,
            "tilt",
            axis_id,
            true,
            true,
        );
    }

    /// Java `saveTilt3dFindReproject`.  Save the specified tilt_3dfind_reproject com
    /// script updating the tilt parameters.
    pub fn save_tilt3d_find_reproject(&self, tilt_param: &TiltParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_tilt = if axis_id == AxisID::Second {
            &self.script_tilt3d_find_reproject_b
        } else {
            &self.script_tilt3d_find_reproject_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_tilt.borrow_mut().as_mut(),
            tilt_param,
            "tilt",
            axis_id,
            true,
            true,
        );
    }

    /// Java `loadTomopitch`.  Load the specified tomopitch com script.
    pub fn load_tomopitch(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_tomopitch_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "tomopitch",
                axis_id,
                true,
                false,
                false,
            );
        } else {
            *self.script_tomopitch_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                "tomopitch",
                axis_id,
                true,
                false,
                false,
            );
        }
    }

    /// Java `getTomopitchParam`.  Get the tomopitch parameters from the specified
    /// tomopitch script object.
    pub fn get_tomopitch_param(&self, axis_id: AxisID) -> TomopitchParam {
        // Get a reference to the appropriate script object
        let tomopitch = if axis_id == AxisID::Second {
            &self.script_tomopitch_b
        } else {
            &self.script_tomopitch_a
        };

        // Initialize a TomopitchParam object from the com script command object
        let mut tomopitch_param = TomopitchParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut tomopitch_param,
            tomopitch.borrow_mut().as_mut(),
            "tomopitch",
            axis_id,
            false,
            false,
            true,
        );
        tomopitch_param
    }

    /// Java `saveTomopitch`.  Save the specified tomopitch com script updating the
    /// tomopitch parameters.
    pub fn save_tomopitch(&self, tomopitch_param: &TomopitchParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_tomopitch = if axis_id == AxisID::Second {
            &self.script_tomopitch_b
        } else {
            &self.script_tomopitch_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_tomopitch.borrow_mut().as_mut(),
            tomopitch_param,
            "tomopitch",
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadMTFFilter`.
    pub fn load_mtf_filter(&self, axis_id: AxisID) {
        // Assign the new ComScriptObject object to the appropriate reference
        if axis_id == AxisID::Second {
            *self.script_mtf_filter_b.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                MTFFILTER_COMMAND,
                axis_id,
                false,
                false,
                false,
            );
        } else {
            *self.script_mtf_filter_a.borrow_mut() = ComScriptUtil::load_com_script_script_name(
                self.app_manager,
                MTFFILTER_COMMAND,
                axis_id,
                false,
                false,
                false,
            );
        }
    }

    /// Java `getCtfPhaseFlipParam`.
    pub fn get_ctf_phase_flip_param(&self, axis_id: AxisID) -> CtfPhaseFlipParam {
        // Get a reference to the appropriate script object
        let ctf_phaseflip = if axis_id == AxisID::Second {
            &self.script_ctf_correction_b
        } else {
            &self.script_ctf_correction_a
        };

        // Initialize from the com script command object
        let mut ctf_phase_flip_param = CtfPhaseFlipParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut ctf_phase_flip_param,
            ctf_phaseflip.borrow_mut().as_mut(),
            ctf_phase_flip_param::COMMAND,
            axis_id,
            false,
            false,
            true,
        );
        ctf_phase_flip_param
    }

    /// Java `getCtfPlotterParam`.
    pub fn get_ctf_plotter_param(&self, axis_id: AxisID) -> CtfPlotterParam {
        // Get a reference to the appropriate script object
        let ctf_plotter = if axis_id == AxisID::Second {
            &self.script_ctf_plotter_b
        } else {
            &self.script_ctf_plotter_a
        };

        // Initialize from the com script command object
        let mut ctf_plotter_param = CtfPlotterParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut ctf_plotter_param,
            ctf_plotter.borrow_mut().as_mut(),
            ctf_plotter_param::COMMAND,
            axis_id,
            false,
            false,
            true,
        );
        ctf_plotter_param
    }

    /// Java `getMTFFilterParam`.
    pub fn get_mtf_filter_param(&self, axis_id: AxisID) -> MTFFilterParam {
        // Get a reference to the appropriate script object
        let mtf_filter = if axis_id == AxisID::Second {
            &self.script_mtf_filter_b
        } else {
            &self.script_mtf_filter_a
        };

        // Initialize a TiltParam object from the com script command object
        let mut mtf_filter_param = MTFFilterParam::new(self.app_manager, axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut mtf_filter_param,
            mtf_filter.borrow_mut().as_mut(),
            MTFFILTER_COMMAND,
            axis_id,
            false,
            false,
            true,
        );
        mtf_filter_param
    }

    /// Java `saveMTFFilter`.
    pub fn save_mtf_filter(&self, mtf_filter_param: &MTFFilterParam, axis_id: AxisID) {
        // Get a reference to the appropriate script object
        let script_mtf_filter = if axis_id == AxisID::Second {
            &self.script_mtf_filter_b
        } else {
            &self.script_mtf_filter_a
        };
        ComScriptUtil::modify_command(
            self.app_manager,
            script_mtf_filter.borrow_mut().as_mut(),
            mtf_filter_param,
            MTFFILTER_COMMAND,
            axis_id,
            false,
            false,
        );
    }

    /// Java `loadSolvematch`.  Load in the solvematch com script.
    pub fn load_solvematch(&self) {
        *self.script_solvematch.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            "solvematch",
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `loadDualvolmatch`.
    pub fn load_dualvolmatch(&self) {
        *self.script_dualvolmatch.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            &ProcessName::DUALVOLMATCH.to_string(),
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `getSolvematch`.  Return the solvematch parameter object from the currently
    /// loaded solvematch com script.
    pub fn get_solvematch(&self) -> SolvematchParam {
        // Initialize a SolvematchParam object from the com script command
        // object
        let mut solve_match_param = SolvematchParam::new(self.app_manager);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut solve_match_param,
            self.script_solvematch.borrow_mut().as_mut(),
            "solvematch",
            AxisID::Only,
            false,
            false,
            true,
        );
        solve_match_param
    }

    /// Java `getDualvolmatch`.
    pub fn get_dualvolmatch(&self) -> DualvolmatchParam {
        let mut param = DualvolmatchParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            self.script_dualvolmatch.borrow_mut().as_mut(),
            &ProcessName::DUALVOLMATCH.to_string(),
            AxisID::Only,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveSolvematch(SolvematchParam)`.  Replace the solvematch command in the
    /// solvematch com script with the info in the specified SolvematchParam object.
    pub fn save_solvematch_solvematch(&self, solve_match_param: &SolvematchParam) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_solvematch.borrow_mut().as_mut(),
            solve_match_param,
            "solvematch",
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `saveDualvolmatch`.
    pub fn save_dualvolmatch(&self, param: &DualvolmatchParam) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_dualvolmatch.borrow_mut().as_mut(),
            param,
            &ProcessName::DUALVOLMATCH.to_string(),
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `saveSolvematch(MatchshiftsParam)`.  Save the matchshifts command to the
    /// solvematch com script.
    pub fn save_solvematch_matchshifts(&self, matchshifts_param: &MatchshiftsParam) {
        ComScriptUtil::modify_command_add_new(
            self.app_manager,
            self.script_solvematch.borrow_mut().as_mut(),
            matchshifts_param,
            &matchshifts_param.get_command(),
            AxisID::Only,
            true,
            false,
            false,
            false,
            None,
        );
    }

    /// Java `loadSolvematchshift`.  Load the solvematchshift com script.
    pub fn load_solvematchshift(&self) {
        *self.script_solvematchshift.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            "solvematchshift",
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `getSolvematchshift`.  Parse the solvematch command from the
    /// solvematchshift script.
    pub fn get_solvematchshift(&self) -> SolvematchshiftParam {
        // Initialize a SolvematchshiftParam object from the com script command
        // object
        let mut solve_matchshift_param = SolvematchshiftParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut solve_matchshift_param,
            self.script_solvematchshift.borrow_mut().as_mut(),
            "solvematch",
            AxisID::Only,
            false,
            false,
            true,
        );
        solve_matchshift_param
    }

    /// Java `getMatchshiftsFromSolvematchshifts`.
    pub fn get_matchshifts_from_solvematchshifts(&self) -> MatchshiftsParam {
        // Initialize a MatchshiftsParam object from the com script command
        // object
        let mut matchshifts_param = MatchshiftsParam::new();
        ComScriptUtil::initialize_optional_command(
            self.app_manager,
            &mut matchshifts_param,
            self.script_solvematchshift.borrow_mut().as_mut(),
            "matchshifts",
            AxisID::Only,
            false,
            false,
            true,
            true,
            None,
        );
        matchshifts_param
    }

    /// Java `getBlendmontParamFromTiltxcorr`.  Get the blendmont command from
    /// xcorr.com.
    pub fn get_blendmont_param_from_tiltxcorr(&self, axis_id: AxisID) -> BlendmontParam {
        // Get a reference to the appropriate script object
        let xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };

        // Initialize a TiltxcorrParam object from the com script command object
        let dataset_name = self.app_manager.get_meta_data().get_dataset_name();
        let mut blendmont_param =
            BlendmontParam::new(self.app_manager, Some(&dataset_name), axis_id);
        ComScriptUtil::initialize(
            self.app_manager,
            &mut blendmont_param,
            xcorr.borrow_mut().as_mut(),
            blendmont_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        );
        blendmont_param
    }

    /// Java `getGotoParamFromTiltxcorr`.  Get the first goto command from xcorr.com.
    pub fn get_goto_param_from_tiltxcorr(&self, axis_id: AxisID) -> Option<GotoParam> {
        let xcorr = if axis_id == AxisID::Second {
            &self.script_xcorr_b
        } else {
            &self.script_xcorr_a
        };
        // Initialize a GotoParam object from the com script command
        // object
        let mut goto_param = GotoParam::new();
        if !ComScriptUtil::initialize(
            self.app_manager,
            &mut goto_param,
            xcorr.borrow_mut().as_mut(),
            goto_param::COMMAND_NAME,
            axis_id,
            false,
            false,
            true,
        ) {
            return None;
        }
        Some(goto_param)
    }

    /// Java `saveSolvematchshift`.  Save the solvematchshift com script updating the
    /// solveMatchshiftParam parameters.
    pub fn save_solvematchshift(&self, solve_matchshift_param: &SolvematchshiftParam) {
        // `new Exception().printStackTrace()`: the JVM's stack trace is not
        // reproduced; the exception line is.
        eprintln!("java.lang.Exception");
        eprintln!("WARNING: call to saveSolvematchshift");
        ComScriptUtil::modify_command_add_new(
            self.app_manager,
            self.script_solvematchshift.borrow_mut().as_mut(),
            solve_matchshift_param,
            "solvematch",
            AxisID::Only,
            false,
            false,
            true,
            true,
            None,
        );
    }

    /// Java `loadSolvematchmod`.  Load the solvematchmod com script.
    pub fn load_solvematchmod(&self) {
        *self.script_solvematchmod.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            "solvematchmod",
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `getSolvematchmod`.  Parse the solvematch command from the solvematchmod
    /// script.
    pub fn get_solvematchmod(&self) -> SolvematchmodParam {
        // Initialize a SolvematchmodParam object from the com script command
        // object
        let mut solve_matchmod_param = SolvematchmodParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut solve_matchmod_param,
            self.script_solvematchmod.borrow_mut().as_mut(),
            "solvematch",
            AxisID::Only,
            false,
            false,
            true,
        );
        solve_matchmod_param
    }

    /// Java `saveSolvematchmod`.  Save the solvematchmod com script updating the
    /// solveMatchmodParam parameters.
    pub fn save_solvematchmod(&self, solve_matchmod_param: &SolvematchmodParam) {
        // `new Exception().printStackTrace()`: the JVM's stack trace is not
        // reproduced; the exception line is.
        eprintln!("java.lang.Exception");
        eprintln!("WARNING: call to saveSolvematchshift");
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_solvematchmod.borrow_mut().as_mut(),
            solve_matchmod_param,
            "solvematch",
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `loadPatchcorr`.  Load the patchcorr com script.
    pub fn load_patchcorr(&self) {
        *self.script_patchcorr.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            "patchcorr",
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `getPatchcrawl3D`.  Parse the patchrawl3D command from the patchcorr
    /// script.
    pub fn get_patchcrawl3_d(&self) -> Patchcrawl3DParam {
        // Initialize a Patchcrawl3DParam object from the com script command object
        let mut patchcrawl3_d_param = Patchcrawl3DParam::new(self.app_manager);
        if !ComScriptUtil::initialize_optional_command(
            self.app_manager,
            &mut patchcrawl3_d_param,
            self.script_patchcorr.borrow_mut().as_mut(),
            patchcrawl3d_param::COMMAND,
            AxisID::Only,
            true,
            false,
            false,
            true,
            None,
        ) {
            ComScriptUtil::initialize(
                self.app_manager,
                &mut patchcrawl3_d_param,
                self.script_patchcorr.borrow_mut().as_mut(),
                patchcrawl3d_pre_pip_param::COMMAND,
                AxisID::Only,
                false,
                false,
                true,
            );
        }
        patchcrawl3_d_param
    }

    /// Java `savePatchcorr`.  Save the patchcorr com script updating the patchcrawl3d
    /// parameters.
    pub fn save_patchcorr(&self, patchcrawl3_d_param: &Patchcrawl3DParam) {
        if !ComScriptUtil::modify_optional_command(
            self.app_manager,
            self.script_patchcorr.borrow_mut().as_mut(),
            patchcrawl3_d_param,
            patchcrawl3d_param::COMMAND,
            AxisID::Only,
            false,
            false,
        ) {
            ComScriptUtil::delete_command(
                self.app_manager,
                self.script_patchcorr.borrow_mut().as_mut(),
                patchcrawl3d_pre_pip_param::COMMAND,
                AxisID::Only,
                false,
                false,
            );
            ComScriptUtil::add_modify_command_previous_index(
                self.app_manager,
                self.script_patchcorr.borrow_mut().as_mut(),
                patchcrawl3_d_param,
                patchcrawl3d_param::COMMAND,
                AxisID::Only,
                -1,
                false,
                false,
                false,
            );
        }
    }

    /// Java `loadMatchvol1`.  Load the matchvol1 com script.
    pub fn load_matchvol1(&self) {
        *self.script_matchvol1.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            &(matchvol_param::COMMAND.to_string() + "1"),
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `getMatchvolParam`.  Parse the matchvol command from the matchvol1 script.
    pub fn get_matchvol_param(&self) -> MatchvolParam {
        // Initialize a MatchvolParam object from the com script command object
        let mut param = MatchvolParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            self.script_matchvol1.borrow_mut().as_mut(),
            matchvol_param::COMMAND,
            AxisID::Only,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveMatchvol`.  Save the matchvol1 com script updating the matchvol
    /// parameters.
    pub fn save_matchvol(&self, matchvol_param: &MatchvolParam) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_matchvol1.borrow_mut().as_mut(),
            matchvol_param,
            matchvol_param::COMMAND,
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `loadMatchorwarp`.  Load the matchorwarp com script.
    pub fn load_matchorwarp(&self) {
        *self.script_matchorwarp.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            "matchorwarp",
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `getMatchorwarParam`.  Parse the matchorwarp command from the matchorwarp
    /// script.
    pub fn get_matchorwar_param(&self) -> MatchorwarpParam {
        // Initialize a MatchorwarpParam object from the com script command object
        let mut matchorwarp_param = MatchorwarpParam::new();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut matchorwarp_param,
            self.script_matchorwarp.borrow_mut().as_mut(),
            "matchorwarp",
            AxisID::Only,
            false,
            false,
            true,
        );
        matchorwarp_param
    }

    /// Java `saveMatchorwarp`.  Save the matchorwarp com script updating the
    /// matchorwarp parameters.
    pub fn save_matchorwarp(&self, matchorwarp_param: &MatchorwarpParam) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_matchorwarp.borrow_mut().as_mut(),
            matchorwarp_param,
            "matchorwarp",
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `getCombineComscript`.
    pub fn get_combine_comscript(
        &self,
        initial_volume_matching: bool,
    ) -> Option<CombineComscriptState> {
        // Initialize a CombineComscript object from the com script command object
        let mut combine_comscript_state = CombineComscriptState::new(initial_volume_matching);
        if !combine_comscript_state.initialize(self) {
            return None;
        }
        Some(combine_comscript_state)
    }

    /// Java `loadCombine`.  Load the combine com script.
    pub fn load_combine(&self) {
        *self.script_combine.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            combine_comscript_state::COMSCRIPT_NAME,
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `loadVolcombine`.
    pub fn load_volcombine(&self) {
        *self.script_volcombine.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            "volcombine",
            AxisID::Only,
            true,
            false,
            false,
        );
    }

    /// Java `saveCombine(GotoParam)`.
    pub fn save_combine_goto(&self, goto_param: &GotoParam) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_combine.borrow_mut().as_mut(),
            goto_param,
            goto_param::COMMAND_NAME,
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `saveCombine(EchoParam, String)`.  Returns index of saved command.
    pub fn save_combine_echo(&self, echo_param: &EchoParam, previous_command: &str) -> i32 {
        ComScriptUtil::add_modify_command_previous_command(
            self.app_manager,
            self.script_combine.borrow_mut().as_mut(),
            echo_param,
            echo_param::COMMAND_NAME,
            AxisID::Only,
            previous_command,
            false,
            false,
        )
    }

    /// Java `saveCombine(ExitParam, int)`.
    pub fn save_combine_exit(&self, exit_param: &ExitParam, previous_command_index: i32) {
        ComScriptUtil::add_modify_command_previous_index(
            self.app_manager,
            self.script_combine.borrow_mut().as_mut(),
            exit_param,
            exit_param::COMMAND_NAME,
            AxisID::Only,
            previous_command_index,
            false,
            false,
            false,
        );
    }

    /// Java `saveCombine(GotoParam, int)`.
    pub fn save_combine_goto_index(&self, goto_param: &GotoParam, previous_command_index: i32) {
        ComScriptUtil::add_modify_command_previous_index(
            self.app_manager,
            self.script_combine.borrow_mut().as_mut(),
            goto_param,
            goto_param::COMMAND_NAME,
            AxisID::Only,
            previous_command_index,
            false,
            false,
            false,
        );
    }

    /// Java `saveVolcombine(SetParam)`.
    pub fn save_volcombine(&self, set_param: &SetParam) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_volcombine.borrow_mut().as_mut(),
            set_param,
            set_param::COMMAND_NAME,
            AxisID::Only,
            false,
            false,
        );
    }

    /// Java `saveVolcombine(SetParam, String)`.
    pub fn save_volcombine_previous_command(&self, set_param: &SetParam, previous_command: &str) {
        ComScriptUtil::modify_command_previous_command(
            self.app_manager,
            self.script_volcombine.borrow_mut().as_mut(),
            set_param,
            set_param::COMMAND_NAME,
            AxisID::Only,
            false,
            false,
            previous_command,
            false,
            false,
        );
    }

    /// Java `deleteFromCombine`.
    pub fn delete_from_combine(&self, command: &str, previous_command: &str) {
        ComScriptUtil::delete_command_previous_command(
            self.app_manager,
            self.script_combine.borrow_mut().as_mut(),
            command,
            AxisID::Only,
            previous_command,
            false,
            false,
        );
    }

    /// Java `getGotoParamFromCombine`.
    pub fn get_goto_param_from_combine(&self) -> Option<GotoParam> {
        // Initialize a GotoParam object from the com script command
        // object
        let mut goto_param = GotoParam::new();
        if !ComScriptUtil::initialize_optional_command(
            self.app_manager,
            &mut goto_param,
            self.script_combine.borrow_mut().as_mut(),
            goto_param::COMMAND_NAME,
            AxisID::Only,
            true,
            false,
            false,
            true,
            None,
        ) {
            return None;
        }
        Some(goto_param)
    }

    /// Java `getSetParamFromVolcombine(String, EtomoNumber.Type)`.
    pub fn get_set_param_from_volcombine(&self, name: &str, r#type: Type) -> Option<SetParam> {
        let mut set_param = SetParam::new(name, r#type);
        if !ComScriptUtil::initialize_optional_command(
            self.app_manager,
            &mut set_param,
            self.script_volcombine.borrow_mut().as_mut(),
            set_param::COMMAND_NAME,
            AxisID::Only,
            true,
            false,
            false,
            true,
            None,
        ) {
            return None;
        }
        Some(set_param)
    }

    /// Java `getSetParamFromVolcombine(String, EtomoNumber.Type, String)`.
    pub fn get_set_param_from_volcombine_previous_command(
        &self,
        name: &str,
        r#type: Type,
        previous_command: Option<&str>,
    ) -> Option<SetParam> {
        let mut set_param = SetParam::new(name, r#type);
        if !ComScriptUtil::initialize_previous_command(
            self.app_manager,
            &mut set_param,
            self.script_volcombine.borrow_mut().as_mut(),
            set_param::COMMAND_NAME,
            AxisID::Only,
            true,
            previous_command,
            true,
            false,
            false,
            true,
        ) {
            return None;
        }
        Some(set_param)
    }

    /// Java `getEchoParamFromCombine`.
    pub fn get_echo_param_from_combine(&self, previous_command: &str) -> Option<EchoParam> {
        // Initialize an EchoParam object from a location after previousCommand
        // in the com script command
        // object
        let mut echo_param = EchoParam::new();
        if !ComScriptUtil::initialize_previous_command(
            self.app_manager,
            &mut echo_param,
            self.script_combine.borrow_mut().as_mut(),
            echo_param::COMMAND_NAME,
            AxisID::Only,
            false,
            Some(previous_command),
            false,
            false,
            false,
            true,
        ) {
            return None;
        }
        Some(echo_param)
    }

    /// Java `isDualvolmatchLabelInCombine`.
    pub fn is_dualvolmatch_label_in_combine(&self) -> bool {
        if self.script_combine.borrow().is_none() {
            self.load_combine();
        }
        if self.script_combine.borrow().is_none() {
            return false;
        }
        let mut param = LabelParam::new(ProcessName::DUALVOLMATCH);
        let label = param.get_label();
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            self.script_combine.borrow_mut().as_mut(),
            &label,
            AxisID::Only,
            false,
            false,
            false,
        )
    }

    /// Java `loadFlatten`.
    pub fn load_flatten(&self, axis_id: AxisID) {
        *self.script_flatten.borrow_mut() = ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            &ProcessName::FLATTEN.to_string(),
            axis_id,
            true,
            false,
            false,
        );
    }

    /// Java `isWarpVolParamInFlatten`.
    ///
    /// Fixed in translation: `loadComScript` returns null when flatten.com cannot be
    /// parsed (after its error dialog), and `.isCommandLoaded()` then throws
    /// `NullPointerException`; a script that did not load answers false (`BUGS.md`).
    pub fn is_warp_vol_param_in_flatten(&self, axis_id: AxisID) -> bool {
        match ComScriptUtil::load_com_script_script_name(
            self.app_manager,
            &ProcessName::FLATTEN.to_string(),
            axis_id,
            true,
            false,
            false,
        ) {
            None => false,
            Some(com_script) => com_script.is_command_loaded(),
        }
    }

    /// Java `getWarpVolParamFromFlatten`.
    pub fn get_warp_vol_param_from_flatten(&self, axis_id: AxisID) -> WarpVolParam {
        // Initialize a WarpVolParam object from the com script command
        // object
        let mut param = WarpVolParam::new(
            self.app_manager,
            axis_id,
            Some(warp_vol_param::Mode::PostProcessing),
        );
        ComScriptUtil::initialize(
            self.app_manager,
            &mut param,
            self.script_flatten.borrow_mut().as_mut(),
            warp_vol_param::COMMAND,
            axis_id,
            false,
            false,
            true,
        );
        param
    }

    /// Java `saveFlatten`.  Save the WarpVolParam command to the flatten com script.
    pub fn save_flatten(&self, param: &WarpVolParam, axis_id: AxisID) {
        ComScriptUtil::modify_command(
            self.app_manager,
            self.script_flatten.borrow_mut().as_mut(),
            param,
            warp_vol_param::COMMAND,
            axis_id,
            true,
            false,
        );
    }

    /// Java private `newstOrNewstack`.  Examine the com script to see whether it
    /// contains newst or newstack commands.
    fn newst_or_newstack(&self, com_script: Option<&ComScript>) -> String {
        let com_script = match com_script {
            None => return String::new(),
            Some(com_script) => com_script,
        };
        let commands = com_script.get_command_array();
        for i in 0..commands.len() {
            if commands[i].as_deref() == Some("newst") {
                return "newst".to_string();
            }
            if commands[i].as_deref() == Some("newstack") {
                return "newstack".to_string();
            }
        }
        String::new()
    }
}
