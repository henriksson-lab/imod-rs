//! Translation of `IMOD/flib/tiltalign/alivar.h`.
//!
//! Type-only header: the `AlignVariables` struct that the tiltalign units reach
//! through a pointer usually named `av` (`alivar.h:6`).
//!
//! # How later units receive it
//!
//! In the C++ each unit keeps its own file-scope copy of the pointer, set by a
//! `*SetPointers` call (`static AlignVariables *av;` in `funct.cpp:57`,
//! `leaveout.cpp:16`, `utilfuncs.cpp:17`, `beamtilt.cpp:18`, `map_vars.cpp:28`,
//! `input_model.cpp:20`, `proc_vars.cpp:21`, `tiltali.cpp:21`).  The object
//! itself is a file-scope `static AlignVariables alignVars;` in the program
//! main (`tiltalign.cpp:29`, `beadtrack.cpp:36`).  The translation owns one
//! `AlignVariables` in the program's state and passes it down as
//! `&mut AlignVariables` (or `&AlignVariables` for a reader) to every function
//! whose C body dereferences `av`.  The `*SetPointers` functions then have
//! nothing to store; translate each as the assignment it is and let the
//! parameter carry the reference, as `FortModel` does for `use fortmodel`
//! (`flib/subrs/model/fortmodel.rs`).  Do not reintroduce a `static mut`.
//!
//! # Representation
//!
//! - Every `float *`/`int *` member is a `Vec`, sized where the source
//!   `B3DMALLOC`s it (`allocateAlivar`, `utilfuncs.cpp:50-117`, and the patch
//!   track arrays at `utilfuncs.cpp:30-47`; the leave-out arrays in
//!   `leaveout.cpp`; `imodObjNum`/`ifill*` in `input_model.cpp`).  No member is
//!   ever pointed into another array — the only non-`B3DMALLOC` pointer store is
//!   `av->realInTestSet = NULL` (`tiltalign.cpp:196`) — so owning `Vec`s lose
//!   nothing.
//! - A `NULL` pointer is an **empty `Vec`**.  Where the source passes a
//!   possibly-`NULL` member to a pointer parameter it tests
//!   (`countNumInView(..., realInTestSet, ...)`, `utilfuncs.cpp:143-151`), the
//!   caller passes `(!v.is_empty()).then_some(&v[..])`.
//! - `B3DMALLOC` leaves memory uninitialised; a `Vec` is zeroed.  Output that
//!   depended on reading unwritten elements would differ (`NATIVE.md` §4).
//! - Arrays are 0-based, but many *contain* 1-based indexes (the "CIF 1"
//!   comments, `alivar.h:7-8`); copy the source's `- 1`s exactly.
//! - `int` flags stay `i32` (`av->xyzFixed = false` stores 0).
//! - `std::map<int,float>` is `BTreeMap<i32, f32>`: both are ordered maps, and
//!   `count`/`find` become `contains_key`/`get`.
//! - The static instance is zero-initialised, which `Default` reproduces.

use std::collections::BTreeMap;

/// Original: `struct AlignVariables` (`alivar.h:9-126`) — shared variables in
/// structure usually pointed to by "av".  Arrays are indexed from 0 but many
/// contain 1-based indexes; "CIF" marks those that contain indexes from.
#[derive(Clone, Debug, Default)]
pub struct AlignVariables {
    /// Original: `float *dxy` (`alivar.h:11`) — X/Y shifts of each view (X/Y sequential).
    pub dxy: Vec<f32>,
    /// Original: `float *xyz` (`alivar.h:12`) — X/Y/Z coords of each real point (X/Y/Z sequential).
    pub xyz: Vec<f32>,
    /// Original: `float *xx` (`alivar.h:13`) — X and Y coordinates of 2D point.
    pub xx: Vec<f32>,
    /// Original: `float *yy` (`alivar.h:14`).
    pub yy: Vec<f32>,
    /// Original: `float *xresid` (`alivar.h:15`).
    pub xresid: Vec<f32>,
    /// Original: `float *yresid` (`alivar.h:16`).
    pub yresid: Vec<f32>,
    /// Original: `float *weight` (`alivar.h:17`).
    pub weight: Vec<f32>,
    /// Original: `float *trackResid` (`alivar.h:18`).
    pub track_resid: Vec<f32>,
    /// Original: `float *viewMedianRes` (`alivar.h:19`).
    pub view_median_res: Vec<f32>,
    /// Original: `int *irealStr` (`alivar.h:20`) — Index of start of real point in coord/view arrays  CIF 1.
    pub ireal_str: Vec<i32>,
    /// Original: `int *isecView` (`alivar.h:21`) — View number of a 2D point  IF 0, CIF 1 (view #'s).
    pub isec_view: Vec<i32>,
    /// Original: `int *imodObjNum` (`alivar.h:22`) — IMOD object number of each real point in search.
    pub imod_obj_num: Vec<i32>,
    /// Original: `int *ifillRealStart` (`alivar.h:23`).
    pub ifill_real_start: Vec<i32>,
    /// Original: `int *ifillView` (`alivar.h:24`).
    pub ifill_view: Vec<i32>,
    /// Original: `float *yfillProj` (`alivar.h:25`).
    pub yfill_proj: Vec<f32>,
    /// Original: `float *xfillProj` (`alivar.h:26`).
    pub xfill_proj: Vec<f32>,
    /// Original: `int *mapComp` (`alivar.h:27`).
    pub map_comp: Vec<i32>,
    /// Original: `int *mapGmag` (`alivar.h:28`).
    pub map_gmag: Vec<i32>,
    /// Original: `int *mapTilt` (`alivar.h:29`).
    pub map_tilt: Vec<i32>,
    /// Original: `int *mapRot` (`alivar.h:30`).
    pub map_rot: Vec<i32>,
    /// Original: `int *mapSkew` (`alivar.h:31`).
    pub map_skew: Vec<i32>,
    /// Original: `int *mapDmag` (`alivar.h:32`).
    pub map_dmag: Vec<i32>,
    /// Original: `int *linComp` (`alivar.h:33`).
    pub lin_comp: Vec<i32>,
    /// Original: `int *linGmag` (`alivar.h:34`).
    pub lin_gmag: Vec<i32>,
    /// Original: `int *linTilt` (`alivar.h:35`).
    pub lin_tilt: Vec<i32>,
    /// Original: `int *linRot` (`alivar.h:36`).
    pub lin_rot: Vec<i32>,
    /// Original: `int *linSkew` (`alivar.h:37`).
    pub lin_skew: Vec<i32>,
    /// Original: `int *linDmag` (`alivar.h:38`).
    pub lin_dmag: Vec<i32>,
    /// Original: `int *linAlf` (`alivar.h:39`).
    pub lin_alf: Vec<i32>,
    /// Original: `int *mapAlf` (`alivar.h:40`).
    pub map_alf: Vec<i32>,
    /// Original: `int nrealPt` (`alivar.h:41`).
    pub nreal_pt: i32,
    /// Original: `int mapDmagStart` (`alivar.h:42`).
    pub map_dmag_start: i32,
    /// Original: `int mapDumDmag` (`alivar.h:43`).
    pub map_dum_dmag: i32,
    /// Original: `int ifAnyAlf` (`alivar.h:44`).
    pub if_any_alf: i32,
    /// Original: `int nview` (`alivar.h:45`).
    pub nview: i32,
    /// Original: `int ifRotFix` (`alivar.h:46`).
    pub if_rot_fix: i32,
    /// Original: `float dumDmagFac` (`alivar.h:47`).
    pub dum_dmag_fac: f32,
    /// Original: `float *frcComp` (`alivar.h:48`).
    pub frc_comp: Vec<f32>,
    /// Original: `float *frcGmag` (`alivar.h:49`).
    pub frc_gmag: Vec<f32>,
    /// Original: `float *frcTilt` (`alivar.h:50`).
    pub frc_tilt: Vec<f32>,
    /// Original: `float *frcRot` (`alivar.h:51`).
    pub frc_rot: Vec<f32>,
    /// Original: `float *frcSkew` (`alivar.h:52`).
    pub frc_skew: Vec<f32>,
    /// Original: `float *frcDmag` (`alivar.h:53`).
    pub frc_dmag: Vec<f32>,
    /// Original: `float *comp` (`alivar.h:54`).
    pub comp: Vec<f32>,
    /// Original: `float *gmag` (`alivar.h:55`).
    pub gmag: Vec<f32>,
    /// Original: `float *tilt` (`alivar.h:56`).
    pub tilt: Vec<f32>,
    /// Original: `float *rot` (`alivar.h:57`).
    pub rot: Vec<f32>,
    /// Original: `float *skew` (`alivar.h:58`).
    pub skew: Vec<f32>,
    /// Original: `float *dmag` (`alivar.h:59`).
    pub dmag: Vec<f32>,
    /// Original: `float *tiltInc` (`alivar.h:60`).
    pub tilt_inc: Vec<f32>,
    /// Original: `float *alf` (`alivar.h:61`).
    pub alf: Vec<f32>,
    /// Original: `float *frcAlf` (`alivar.h:62`).
    pub frc_alf: Vec<f32>,
    /// Original: `float fixedTilt` (`alivar.h:63`).
    pub fixed_tilt: f32,
    /// Original: `float fixedGmag` (`alivar.h:64`).
    pub fixed_gmag: f32,
    /// Original: `float fixedComp` (`alivar.h:65`).
    pub fixed_comp: f32,
    /// Original: `float fixedDmag` (`alivar.h:66`).
    pub fixed_dmag: f32,
    /// Original: `float fixedSkew` (`alivar.h:67`).
    pub fixed_skew: f32,
    /// Original: `float fixedTilt2` (`alivar.h:68`).
    pub fixed_tilt2: f32,
    /// Original: `float fixedRot` (`alivar.h:69`).
    pub fixed_rot: f32,
    /// Original: `float fixedAlf` (`alivar.h:70`).
    pub fixed_alf: f32,
    /// Original: `float projStrRot` (`alivar.h:71`).
    pub proj_str_rot: f32,
    /// Original: `float projSkew` (`alivar.h:72`).
    pub proj_skew: f32,
    /// Original: `float beamTilt` (`alivar.h:73`).
    pub beam_tilt: f32,
    /// Original: `float kfacRobust` (`alivar.h:74`).
    pub kfac_robust: f32,
    /// Original: `float smallWgtMaxFrac` (`alivar.h:75`).
    pub small_wgt_max_frac: f32,
    /// Original: `float smallWgtThreshold` (`alivar.h:76`).
    pub small_wgt_threshold: f32,
    /// Original: `int *mapFileToView` (`alivar.h:77`).
    pub map_file_to_view: Vec<i32>,
    /// Original: `int *mapViewToFile` (`alivar.h:78`).
    pub map_view_to_file: Vec<i32>,
    /// Original: `int *mapTrackToReal` (`alivar.h:79`) — Index from full track components to "real" tracks  CIF 1.
    pub map_track_to_real: Vec<i32>,
    /// Original: `int *mapRealToTrack` (`alivar.h:80`) — Index from a real track to the full track it is in  CIF 1.
    pub map_real_to_track: Vec<i32>,
    /// Original: `int *indFullTrack` (`alivar.h:81`) — Starting indexes of full track in mapTrackToReal  CIF 1.
    pub ind_full_track: Vec<i32>,
    /// Original: `int nfileViews` (`alivar.h:82`).
    pub nfile_views: i32,
    /// Original: `int mapProjStretch` (`alivar.h:83`).
    pub map_proj_stretch: i32,
    /// Original: `int mapBeamTilt` (`alivar.h:84`).
    pub map_beam_tilt: i32,
    /// Original: `float *glbAlf` (`alivar.h:85`).
    pub glb_alf: Vec<f32>,
    /// Original: `float *glbTilt` (`alivar.h:86`).
    pub glb_tilt: Vec<f32>,
    /// Original: `float *glbRot` (`alivar.h:87`).
    pub glb_rot: Vec<f32>,
    /// Original: `float *glbSkew` (`alivar.h:88`).
    pub glb_skew: Vec<f32>,
    /// Original: `float *glbDmag` (`alivar.h:89`).
    pub glb_dmag: Vec<f32>,
    /// Original: `float *glbGmag` (`alivar.h:90`).
    pub glb_gmag: Vec<f32>,
    /// Original: `int incrGmag` (`alivar.h:91`).
    pub incr_gmag: i32,
    /// Original: `int incrDmag` (`alivar.h:92`).
    pub incr_dmag: i32,
    /// Original: `int incrSkew` (`alivar.h:93`).
    pub incr_skew: i32,
    /// Original: `int incrTilt` (`alivar.h:94`).
    pub incr_tilt: i32,
    /// Original: `int incrAlf` (`alivar.h:95`).
    pub incr_alf: i32,
    /// Original: `int incrRot` (`alivar.h:96`).
    pub incr_rot: i32,
    /// Original: `int firstFunct` (`alivar.h:97`).
    pub first_funct: i32,
    /// Original: `int xyzFixed` (`alivar.h:98`).
    pub xyz_fixed: i32,
    /// Original: `int applyExtraWeights` (`alivar.h:99`).
    pub apply_extra_weights: i32,
    /// Original: `std::map<int,float> objectWeightMap` (`alivar.h:100`).
    pub object_weight_map: BTreeMap<i32, f32>,
    /// Original: `int robustWeights` (`alivar.h:101`).
    pub robust_weights: i32,
    /// Original: `int patchTrackModel` (`alivar.h:102`).
    pub patch_track_model: i32,
    /// Original: `int robustByTrack` (`alivar.h:103`).
    pub robust_by_track: i32,
    /// Original: `int projectFillPoints` (`alivar.h:104`).
    pub project_fill_points: i32,
    /// Original: `int numWgtGroups` (`alivar.h:105`).
    pub num_wgt_groups: i32,
    /// Original: `int numFullPatchTracks` (`alivar.h:106`).
    pub num_full_patch_tracks: i32,
    /// Original: `int numFullTracksUsed` (`alivar.h:107`).
    pub num_full_tracks_used: i32,
    /// Original: `int numTrackGroups` (`alivar.h:108`).
    pub num_track_groups: i32,
    /// Original: `int *ivStartWgtGroup` (`alivar.h:109`).
    pub iv_start_wgt_group: Vec<i32>,
    /// Original: `int *indProjWgtList` (`alivar.h:110`).
    pub ind_proj_wgt_list: Vec<i32>,
    /// Original: `int *ipStartWgtView` (`alivar.h:111`).
    pub ip_start_wgt_view: Vec<i32>,
    /// Original: `int *itrackGroup` (`alivar.h:112`).
    pub itrack_group: Vec<i32>,
    /// Original: `int leavingOut` (`alivar.h:113`).
    pub leaving_out: i32,
    /// Original: `int *realLeftOut` (`alivar.h:114`).
    pub real_left_out: Vec<i32>,
    /// Original: `int *projLeftOut` (`alivar.h:115`).
    pub proj_left_out: Vec<i32>,
    /// Original: `int *projToPredict` (`alivar.h:116`).
    pub proj_to_predict: Vec<i32>,
    /// Original: `int *timesLeftOut` (`alivar.h:117`).
    pub times_left_out: Vec<i32>,
    /// Original: `int *realInTestSet` (`alivar.h:118`).
    pub real_in_test_set: Vec<i32>,
    /// Original: `float testSetFracStep` (`alivar.h:119`).
    pub test_set_frac_step: f32,
    /// Original: `int numLvOutErr[6]` (`alivar.h:120`).
    pub num_lv_out_err: [i32; 6],
    /// Original: `double lvOutErrSum[6]` (`alivar.h:121`).
    pub lv_out_err_sum: [f64; 6],
    /// Original: `double lvOutErrSqSum[6]` (`alivar.h:122`).
    pub lv_out_err_sq_sum: [f64; 6],
    /// Original: `int numLvOutWgtErr[6]` (`alivar.h:123`).
    pub num_lv_out_wgt_err: [i32; 6],
    /// Original: `double lvOutWgtSum[6]` (`alivar.h:124`).
    pub lv_out_wgt_sum: [f64; 6],
    /// Original: `double lvOutWgtSqSum[6]` (`alivar.h:125`).
    pub lv_out_wgt_sq_sum: [f64; 6],
}
