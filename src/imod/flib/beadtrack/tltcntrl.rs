//! Translation of `IMOD/flib/beadtrack/tltcntrl.h`.
//!
//! Type-only header: the tilt-alignment control variables beadtrack shares
//! between `beadtrack.cpp`, `proc_vars.cpp` and `tiltali.cpp` through a
//! pointer named `tc` (`static TiltControl *tc;`, `proc_vars.cpp:19`,
//! `tiltali.cpp:20`; the object is `static TiltControl tltCntrl;`,
//! `beadtrack.cpp:34`).  Later units take `&mut TiltControl` wherever the C
//! body dereferences `tc`; conventions as in `tiltalign/alivar.rs` (`Vec` per
//! `B3DMALLOC`ed pointer, empty for `NULL`, zeroed where the C is
//! uninitialised, `Default` for the zero-initialised static).
//!
//! Sizes, for reference: `tiltOrig`/`gmagOrig`/`rotOrig`/`tiltAll` are
//! `maxView`, `dxySave` `2 * maxView`, `ivsepIn` `maxView * MAXGRP`
//! (`beadtrack.cpp:337-342`); `iobjSeq`/`xyzSave` `mMaxAllReal`/`3 *` that
//! (`beadtrack.cpp:419-420`); `iobjAli` at `beadtrack.cpp:1330`; `izExclude`
//! is the `parselist` result (`beadtrack.cpp:470`); `H` is freed and
//! re-allocated at `maxH / 2` doubles when `maxH` grows (`tiltali.cpp:128-137`)
//! — its `!tc->H` test there is an allocation-failure check.

use crate::imod::flib::tiltalign::arraymaxes::MAXGRP;

/// Original: `struct TiltControl` (`tltcntrl.h:5-55`).
#[derive(Clone, Debug, Default)]
pub struct TiltControl {
    /// Original: `float *rotOrig` (`tltcntrl.h:7`).
    pub rot_orig: Vec<f32>,
    /// Original: `float *gmagOrig` (`tltcntrl.h:8`).
    pub gmag_orig: Vec<f32>,
    /// Original: `float *tiltOrig` (`tltcntrl.h:9`).
    pub tilt_orig: Vec<f32>,
    /// Original: `float *tiltAll` (`tltcntrl.h:10`).
    pub tilt_all: Vec<f32>,
    /// Original: `float *dxySave` (`tltcntrl.h:11`).
    pub dxy_save: Vec<f32>,
    /// Original: `int *izExclude` (`tltcntrl.h:12`).
    pub iz_exclude: Vec<i32>,
    /// Original: `int *ivsepIn` (`tltcntrl.h:13`).
    pub ivsep_in: Vec<i32>,
    /// Original: `int *iobjAli` (`tltcntrl.h:14`).
    pub iobj_ali: Vec<i32>,
    /// Original: `int nmapSpecMag[MAXGRP]` (`tltcntrl.h:15`).
    pub nmap_spec_mag: [i32; MAXGRP as usize],
    /// Original: `int ivSpecStrMag[MAXGRP]` (`tltcntrl.h:16`).
    pub iv_spec_str_mag: [i32; MAXGRP as usize],
    /// Original: `int ivSpecEndMag[MAXGRP]` (`tltcntrl.h:17`).
    pub iv_spec_end_mag: [i32; MAXGRP as usize],
    /// Original: `int nmapSpecRot[MAXGRP]` (`tltcntrl.h:18`).
    pub nmap_spec_rot: [i32; MAXGRP as usize],
    /// Original: `int ivSpecStrRot[MAXGRP]` (`tltcntrl.h:19`).
    pub iv_spec_str_rot: [i32; MAXGRP as usize],
    /// Original: `int ivSpecEndRot[MAXGRP]` (`tltcntrl.h:20`).
    pub iv_spec_end_rot: [i32; MAXGRP as usize],
    /// Original: `int nmapSpecTilt[MAXGRP]` (`tltcntrl.h:21`).
    pub nmap_spec_tilt: [i32; MAXGRP as usize],
    /// Original: `int ivSpecStrTilt[MAXGRP]` (`tltcntrl.h:22`).
    pub iv_spec_str_tilt: [i32; MAXGRP as usize],
    /// Original: `int ivSpecEndTilt[MAXGRP]` (`tltcntrl.h:23`).
    pub iv_spec_end_tilt: [i32; MAXGRP as usize],
    /// Original: `int nsepInGrpIn[MAXGRP]` (`tltcntrl.h:24`).
    pub nsep_in_grp_in: [i32; MAXGRP as usize],
    /// Original: `int nviewAll` (`tltcntrl.h:25`).
    pub nview_all: i32,
    /// Original: `int minTiltInd` (`tltcntrl.h:26`).
    pub min_tilt_ind: i32,
    /// Original: `int minInView` (`tltcntrl.h:27`).
    pub min_in_view: i32,
    /// Original: `int minViewsTiltAli` (`tltcntrl.h:28`).
    pub min_views_tilt_ali: i32,
    /// Original: `int initXyzDone` (`tltcntrl.h:29`).
    pub init_xyz_done: i32,
    /// Original: `float rangeDoAxis` (`tltcntrl.h:30`).
    pub range_do_axis: f32,
    /// Original: `float rangeDoTilt` (`tltcntrl.h:31`).
    pub range_do_tilt: f32,
    /// Original: `float scaleXY` (`tltcntrl.h:32`).
    pub scale_xy: f32,
    /// Original: `float xcen` (`tltcntrl.h:33`).
    pub xcen: f32,
    /// Original: `float ycen` (`tltcntrl.h:34`).
    pub ycen: f32,
    /// Original: `float xorig` (`tltcntrl.h:35`).
    pub xorig: f32,
    /// Original: `float yorig` (`tltcntrl.h:36`).
    pub yorig: f32,
    /// Original: `float xdelt` (`tltcntrl.h:37`).
    pub xdelt: f32,
    /// Original: `float ydelt` (`tltcntrl.h:38`).
    pub ydelt: f32,
    /// Original: `float facMetro` (`tltcntrl.h:39`).
    pub fac_metro: f32,
    /// Original: `float eps` (`tltcntrl.h:40`).
    pub eps: f32,
    /// Original: `int nCycle` (`tltcntrl.h:41`).
    pub n_cycle: i32,
    /// Original: `int nviewLocal` (`tltcntrl.h:42`).
    pub nview_local: i32,
    /// Original: `int numObjDo` (`tltcntrl.h:43`).
    pub num_obj_do: i32,
    /// Original: `int maxH` (`tltcntrl.h:44`).
    pub max_h: i32,
    /// Original: `int numExclude` (`tltcntrl.h:45`).
    pub num_exclude: i32,
    /// Original: `int nmapMag` (`tltcntrl.h:46`).
    pub nmap_mag: i32,
    /// Original: `int nmapRot` (`tltcntrl.h:47`).
    pub nmap_rot: i32,
    /// Original: `int nmapTilt` (`tltcntrl.h:48`).
    pub nmap_tilt: i32,
    /// Original: `int nRanSpecMag` (`tltcntrl.h:49`).
    pub n_ran_spec_mag: i32,
    /// Original: `int nRanSpecRot` (`tltcntrl.h:50`).
    pub n_ran_spec_rot: i32,
    /// Original: `int nRanSpecTilt` (`tltcntrl.h:51`).
    pub n_ran_spec_tilt: i32,
    /// Original: `int *iobjSeq` (`tltcntrl.h:52`).
    pub iobj_seq: Vec<i32>,
    /// Original: `float *xyzSave` (`tltcntrl.h:53`).
    pub xyz_save: Vec<f32>,
    /// Original: `double *H` (`tltcntrl.h:54`).
    pub h: Vec<f64>,
}
