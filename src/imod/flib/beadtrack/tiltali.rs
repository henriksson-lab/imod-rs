//! Translation of `IMOD/flib/beadtrack/tiltali.cpp` — runs tilt alignment for
//! beadtrack.
//!
//! Compiled into `beadtrack` only, whose `solve_xyzd`/`map_vars`/`utilfuncs`
//! are the `-DBEADTRACK` build, so `solveXyzd` is called with
//! `BEADTRACK = true`.
//!
//! # File-scope statics
//!
//! `tc`, `av`, `mx`, `sEvalFunct` (`tiltali.cpp:20-23`), set by
//! `tiltAliSetPointers`, are parameters (`alivar.rs`, `tltcntrl.rs`).  The
//! callees reach two more objects through *their* own statics, which are
//! therefore parameters here too: `proc_model` reads the `fortmodel` arrays
//! (`&FortModel`) and `proc_vars` the `MapSepGroups` set by
//! `procVarsSetPointers` (`&mut MapSepGroups`).
//!
//! # Representation
//!
//! - `var`, `grad`, `varSave` are `B3DMALLOC`ed and uninitialised in the
//!   source; they are zeroed `Vec`s here.  `proc_vars` and the xyz packing
//!   write every element `funct`/`metroSearch` read, and `copyArray` copies
//!   only the written range.  The allocation-failure `exitError` cannot occur.
//! - `tc->H` is a `double *` allocated to `maxH / 2` doubles, handed to
//!   `solveXyzd` as `double *sprod` and to `metroSearch` as `(float *)tc->H`
//!   (`tiltali.cpp:132-135`: "H is a double so it can be passed to solveXyzd
//!   as sprod but is allocated to give maxH real elements because metro uses
//!   reals").  `tc.h` stays a `Vec<f64>`, and the metro call reinterprets the
//!   same storage as `2 * len` `f32`s, which is what the cast does.  (Both
//!   callees write their part of the array before reading it, so nothing
//!   observable crosses between them; the reinterpretation keeps the storage
//!   and its size exactly the source's.)
//! - `float resTmp[25000]` is written and never read; it is kept.
//!
//! # Arithmetic
//!
//! `float DTOR = 0.0174532;` is the double literal rounded to `float`.
//! `pow(float, 2.f)` is the C++ `float` overload, which g++ folds to a
//! single-precision `x * x` (the reference `tiltali.o` imports `sqrtf` and no
//! `pow`), so the residual sums are `f32` throughout.  `B3DMAX(int, pow(...,
//! 2.))` compares in `double` and truncates back to `int`; `nsum > 0.2 *
//! nrealPt` compares in `double`; `tiltRange < 0.4` widens the `float`.

use std::io::Write;

use super::proc_model::proc_model;
use super::proc_vars::proc_vars;
use super::tltcntrl::TiltControl;
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::tiltalign::alivar::AlignVariables;
use crate::imod::flib::tiltalign::arraymaxes::{ArrayMaxes, MAX_REAL_FOR_DIRECT_INIT};
use crate::imod::flib::tiltalign::evalfunct::EvalFunct;
use crate::imod::flib::tiltalign::funct::funct;
use crate::imod::flib::tiltalign::mapsepgroups::MapSepGroups;
use crate::imod::flib::tiltalign::solve_xyzd::solve_xyzd;
use crate::imod::flib::tiltalign::utilfuncs::copy_array;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};
use crate::imod::libcfshr::metro::metro_search;
use crate::imod::libcfshr::parse_params::exit_error;

/// Original: `tiltAliSetPointers` (`tiltali.cpp:25`).
///
/// The source stores the four pointers in file-scope statics; [`tilt_ali`]
/// takes them as parameters instead, so there is nothing to store.
pub fn tilt_ali_set_pointers(
    tlcnt: &mut TiltControl,
    av_in: &mut AlignVariables,
    mx_in: &mut ArrayMaxes,
    evfn: &mut EvalFunct,
) {
    let _ = (tlcnt, av_in, mx_in, evfn);
}

/// Original: `tiltAli` (`tiltali.cpp:37`).
///
/// Routine to run the tiltalign operation.
#[allow(clippy::too_many_arguments)]
pub fn tilt_ali(
    tc: &mut TiltControl,
    av: &mut AlignVariables,
    mx: &ArrayMaxes,
    s_eval_funct: &mut EvalFunct,
    sg: &mut MapSepGroups,
    fm: &FortModel,
    if_did_align: &mut i32,
    if_align_done: &mut i32,
    res_mean: &mut [f32],
    iview: i32,
    ali_mean_res: &mut f32,
) {
    const MAX_METRO_TRIALS: usize = 5;

    //
    let mut var: Vec<f32>;
    let mut var_save: Vec<f32>;
    let mut grad: Vec<f32>;
    let mut error: f64 = 0.;
    let dtor: f32 = 0.0174532_f64 as f32;
    let mut imin_tilt_solv: i32 = 0;
    let mut i: i32;
    let mut nprojpt: i32 = 0;
    let mut iv: i32;
    let mut max_var: i32;
    let mut tilt_solve_min: f32;
    let mut tilt_solve_max: f32;
    let mut angle: f32;
    let tilt_range: f32;
    let mut f_init: f32 = 0.;
    let rms_scale: f32;
    let mut nvar_search: i32 = 0;
    let mut if_map_tilt: i32;
    let mut jpt: i32;
    let mut kpt: i32;
    let _nvar_geometric: i32;
    let mut ier: i32 = 0;
    let mut kount: i32 = 0;
    let mut nsum: i32;
    let mut f: f32 = 0.;
    let mut f_final: f32 = 0.;
    let mut rsum: f32;
    let mut ior: i32;
    let mut ipt: i32;
    let mut iv_orig: usize;
    let mut metro_loop: i32;
    let trial_scale: [f32; MAX_METRO_TRIALS] = [1.0, 0.9, 1.1, 0.75, 0.5];
    let mut res_tmp = vec![0f32; 25000];

    let nalloc = (5 * mx.max_view + 3 * mx.max_real).max(0) as usize;
    var = vec![0.; nalloc];
    grad = vec![0.; nalloc];
    var_save = vec![0.; nalloc];

    av.xyz_fixed = 0;
    av.robust_weights = 0;
    av.if_any_alf = 0;
    {
        let iz_exclude = (!tc.iz_exclude.is_empty()).then_some(&tc.iz_exclude[..]);
        proc_model(
            fm,
            tc.xcen,
            tc.ycen,
            tc.xdelt,
            tc.ydelt,
            tc.xorig,
            tc.yorig,
            tc.scale_xy,
            tc.nview_all,
            tc.min_in_view,
            iview,
            tc.nview_local,
            &tc.iobj_seq,
            tc.num_obj_do,
            &mut av.map_file_to_view,
            &mut av.map_view_to_file,
            &mut av.xx,
            &mut av.yy,
            &mut av.isec_view,
            mx.max_proj_pt,
            mx.max_real,
            &mut av.ireal_str,
            &mut tc.iobj_ali,
            &mut av.nview,
            &mut nprojpt,
            &mut av.nreal_pt,
            iz_exclude,
            tc.num_exclude,
        );
    }
    *if_did_align = 0;
    if av.nview >= tc.min_views_tilt_ali {
        //
        // if enough views, set up for solving tilt axis and tilt
        // angles depending on range of tilt angles
        // reload tilt from nominal angles to get the increments right
        // Find minimum tilt for this angle range
        //
        tilt_solve_min = 1.0e10;
        tilt_solve_max = -1.0e10;
        rsum = 1.0e10;
        iv = 1;
        while iv <= av.nview {
            angle = tc.tilt_all[(av.map_view_to_file[(iv - 1) as usize] - 1) as usize];
            tilt_solve_min = if tilt_solve_min < angle {
                tilt_solve_min
            } else {
                angle
            };
            tilt_solve_max = if tilt_solve_max > angle {
                tilt_solve_max
            } else {
                angle
            };
            av.tilt[(iv - 1) as usize] = dtor * angle;
            if (if angle >= 0. { angle } else { -angle }) < rsum {
                rsum = if angle >= 0. { angle } else { -angle };
                imin_tilt_solv = iv;
            }
            iv += 1;
        }
        tilt_range = tilt_solve_max - tilt_solve_min;
        if (tilt_range as f64) < 0.4 {
            exit_error(
                c_format(
                    "%s %5.1f %s %d %s\n",
                    &[
                        CArg::Str("Insufficient tilt range to do tilt alignment ("),
                        CArg::Dbl(tilt_range as f64),
                        CArg::Str(" deg over"),
                        CArg::Int(av.nview as i64),
                        CArg::Str(" views) - increase minimum # of views for tilt alignment"),
                    ],
                )
                .as_bytes(),
            );
        }
        av.if_rot_fix = 0;
        if tilt_range < tc.range_do_axis {
            av.if_rot_fix = imin_tilt_solv;
        }
        if_map_tilt = 1;
        if tilt_range < tc.range_do_tilt {
            if_map_tilt = 0;
        }
        // print *,'ifrotfix', ifRotFix, '  ifmaptilt', ifMapTilt, &
        // '  imintilt&solv', minTiltInd, iminTiltSolv
        proc_vars(
            tc,
            sg,
            av,
            mx,
            if_map_tilt,
            imin_tilt_solv,
            &mut var,
            &mut nvar_search,
        );
        //
        // find out how many points have not been done yet and decide
        // whether to reinitialize dxy and xyz
        //
        nsum = 0;
        jpt = 0;
        while jpt < av.nreal_pt {
            i = 0;
            while i < 3 {
                if tc.xyz_save[((tc.iobj_ali[jpt as usize] - 1) * 3 + i) as usize] == 0. {
                    nsum += 1;
                }
                i += 1;
            }
            jpt += 1;
        }
        if nsum as f64 > 0.2 * av.nreal_pt as f64 {
            tc.init_xyz_done = 0;
        }

        //
        // check h allocation; if it is not enough, try  to make it enough for the
        // full set of views
        max_var = nvar_search + 3 * av.nreal_pt;
        {
            let a = (max_var + 3) * max_var;
            let m = if av.nreal_pt < MAX_REAL_FOR_DIRECT_INIT {
                av.nreal_pt
            } else {
                MAX_REAL_FOR_DIRECT_INIT
            };
            let b = ((3 * m) as f64).powf(2.);
            iv = (if a as f64 > b { a as f64 } else { b }) as i32;
        }
        if iv > tc.max_h {
            max_var = (tc.nview_all + av.nview - 1) * nvar_search / av.nview;
            iv = if (max_var + 3) * max_var > iv {
                (max_var + 3) * max_var
            } else {
                iv
            };
            tc.max_h = iv;
            // print *,'Allocated h to', maxH, ' based on', maxVar, maxVar + 3
            // Here H is a double so it can be passed to solveXyzd as sprod but is allocated to
            // give maxH real elements because metro uses reals
            tc.h = vec![0.; (tc.max_h / 2).max(0) as usize];
        }
        //
        if tc.init_xyz_done == 0 {
            //
            // first time, initialize xyz and dxy
            //
            s_eval_funct.remap_params(av, &mut var);
            //
            solve_xyzd::<true>(
                &av.xx,
                &av.yy,
                &av.isec_view,
                &av.ireal_str,
                av.nview,
                av.nreal_pt,
                &av.tilt,
                &av.rot,
                &av.gmag,
                &av.comp,
                &mut av.xyz,
                &mut av.dxy,
                0.,
                &mut tc.h,
                &mut error,
                &mut ier,
            );
            if ier != 0 {
                let _ = ImodFile::Stdout.write_all(
                    b"\nWARNING: failed to initialize X/Y/Z coordinates for tiltalign solution\n",
                );
            }
            tc.init_xyz_done = 1;
        } else {
            jpt = 1;
            while jpt <= av.nreal_pt {
                kpt = tc.iobj_ali[(jpt - 1) as usize];
                av.xyz[(jpt * 3 - 3) as usize] = tc.xyz_save[(kpt * 3 - 3) as usize] / tc.scale_xy;
                av.xyz[(jpt * 3 - 2) as usize] = tc.xyz_save[(kpt * 3 - 2) as usize] / tc.scale_xy;
                av.xyz[(jpt * 3 - 1) as usize] = tc.xyz_save[(kpt * 3 - 1) as usize] / tc.scale_xy;
                jpt += 1;
            }
            iv = 1;
            while iv <= av.nview {
                av.dxy[(iv * 2 - 2) as usize] = tc.dxy_save
                    [(av.map_view_to_file[(iv - 1) as usize] * 2 - 2) as usize]
                    / tc.scale_xy;
                av.dxy[(iv * 2 - 1) as usize] = tc.dxy_save
                    [(av.map_view_to_file[(iv - 1) as usize] * 2 - 1) as usize]
                    / tc.scale_xy;
                iv += 1;
            }
        }
        //
        // pack the xyz into the var list
        //
        _nvar_geometric = nvar_search;
        jpt = 0;
        while jpt < av.nreal_pt - 1 {
            i = 0;
            while i < 3 {
                var[nvar_search as usize] = av.xyz[(jpt * 3 + i) as usize];
                nvar_search += 1;
                i += 1;
            }
            jpt += 1;
        }
        //
        // save the variable list for multiple trials and
        // ability to restart on errors.
        // 1 / 25 / 06: changed to restart on all errros, including too many cycles,
        // since varying metro factor  works for this too
        //
        copy_array(&mut var_save, 1, nvar_search, &var, 1);
        //
        metro_loop = 1;
        ier = 1;
        rms_scale = tc.scale_xy * tc.scale_xy / nprojpt as f32;
        while metro_loop <= MAX_METRO_TRIALS as i32 && ier != 0 {
            av.first_funct = 1;
            funct(
                s_eval_funct,
                av,
                mx,
                nvar_search,
                &mut var,
                &mut f_init,
                &mut grad,
            );
            // WRITE(6, 70) fInit
            // 70      FORMAT(/' Variable Metric minimization',T50, &
            // 'Initial F:',T67,E14.7)
            //
            let step = tc.fac_metro * trial_scale[(metro_loop - 1) as usize];
            // SAFETY: `(float *)tc->H` (`tiltali.cpp:171`).  `tc.h` is a live,
            // uniquely borrowed `Vec<f64>`; its buffer is 8-aligned (so
            // 4-aligned), spans exactly `2 * len` `f32`s, and every bit
            // pattern is a valid `f32`.  No other reference to it exists while
            // the slice is alive.
            let h_real: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(tc.h.as_mut_ptr() as *mut f32, tc.h.len() * 2)
            };
            metro_search(
                nvar_search,
                &mut var,
                &mut |n: i32, x: &mut [f32], fv: &mut f32, g: &mut [f32]| {
                    funct(s_eval_funct, av, mx, n, x, fv, g)
                },
                &mut f,
                &mut grad,
                step,
                tc.eps,
                -tc.n_cycle,
                &mut ier,
                h_real,
                &mut kount,
                rms_scale,
            );
            metro_loop += 1;
            if ier != 0 && metro_loop <= MAX_METRO_TRIALS as i32 {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "Metro error #%d, Restarting with step factor of %.2f\n",
                    &[
                        CArg::Int(ier as i64),
                        CArg::Dbl((tc.fac_metro * trial_scale[(metro_loop - 1) as usize]) as f64),
                    ],
                ));
                let _ = ImodFile::Stdout.flush();
                copy_array(&mut var, 1, nvar_search, &var_save, 1);
            }
        }
        //
        // Final call to FUNCT
        funct(
            s_eval_funct,
            av,
            mx,
            nvar_search,
            &mut var,
            &mut f_final,
            &mut grad,
        );
        //
        // unscale all the points, dx, dy, and restore angles to
        // degrees
        //
        *ali_mean_res = 0.;
        if_map_tilt = 0;
        i = 1;
        while i <= av.nreal_pt {
            ior = tc.iobj_ali[(i - 1) as usize];
            av.xyz[(3 * i - 3) as usize] *= tc.scale_xy;
            av.xyz[(3 * i - 2) as usize] *= tc.scale_xy;
            av.xyz[(3 * i - 1) as usize] *= tc.scale_xy;
            tc.xyz_save[(3 * ior - 3) as usize] = av.xyz[(3 * i - 3) as usize];
            tc.xyz_save[(3 * ior - 2) as usize] = av.xyz[(3 * i - 2) as usize];
            tc.xyz_save[(3 * ior - 1) as usize] = av.xyz[(3 * i - 1) as usize];
            rsum = 0.;
            ipt = av.ireal_str[(i - 1) as usize];
            while ipt <= av.ireal_str[i as usize] - 1 {
                let xr = av.xresid[(ipt - 1) as usize];
                let yr = av.yresid[(ipt - 1) as usize];
                rsum += (xr * xr + yr * yr).sqrt();
                res_tmp[(ipt + 1 - av.ireal_str[(i - 1) as usize] - 1) as usize] =
                    tc.scale_xy * (xr * xr + yr * yr).sqrt();
                ipt += 1;
            }
            res_mean[(ior - 1) as usize] = rsum * tc.scale_xy
                / (av.ireal_str[i as usize] - av.ireal_str[(i - 1) as usize]) as f32;
            // write(*,'(i4,(10f7.3))') ior, resMean(ior), (resTmp(ipt), &
            // ipt = 1, min(9, irealStr(i + 1) - irealStr(i)))
            *ali_mean_res += rsum * tc.scale_xy;
            if_map_tilt += av.ireal_str[i as usize] - av.ireal_str[(i - 1) as usize];
            i += 1;
        }
        *ali_mean_res /= if_map_tilt as f32;
        //
        let _ = ImodFile::Stdout.write_all(&c_format_bytes(
            "%4d views,%5d cycles, F =%13.6f, mean residual =%8.2f\n",
            &[
                CArg::Int(av.nview as i64),
                CArg::Int(kount as i64),
                CArg::Dbl((f_final * rms_scale).sqrt() as f64),
                CArg::Dbl(*ali_mean_res as f64),
            ],
        ));
        let _ = ImodFile::Stdout.flush();
        //
        // Error reports:
        if ier != 0 {
            let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                "tiltalign error going to view %d even after varying step factor\n",
                &[CArg::Int(iview as i64)],
            ));
            let _ = ImodFile::Stdout.flush();
        }
        //
        iv = 0;
        while iv < av.nview {
            iv_orig = (av.map_view_to_file[iv as usize] - 1) as usize;
            tc.dxy_save[iv_orig * 2] = av.dxy[(iv * 2) as usize] * tc.scale_xy;
            tc.dxy_save[iv_orig * 2 + 1] = av.dxy[(iv * 2 + 1) as usize] * tc.scale_xy;
            tc.rot_orig[iv_orig] = av.rot[iv as usize] / dtor;
            tc.tilt_orig[iv_orig] = av.tilt[iv as usize] / dtor;
            tc.gmag_orig[iv_orig] = av.gmag[iv as usize];

            ////$        ifMapTilt = 0
            ////$        rsum = 0
            ////$        do i = 1, nprojpt
            ////$        if (iv == isecView(i)) then
            ////$        rsum = rsum + scaleXY * sqrt(xresid(i)**2 + yresid(i)**2)
            ////$        ifMapTilt = ifMapTilt + 1
            ////$        endif
            ////$        enddo
            ////$        write(*,'(i4,3f8.2,f8.4,2f10.2,f8.2)') ivOrig, rotOrig(ivOrig), &
            ////$                    tiltOrig(ivOrig), tiltOrig(ivOrig) - tiltAll(ivOrig), gmag(iv), &
            ////$                    dxySave(1, ivOrig), dxySave(2, ivOrig), rsum / ifMapTilt
            iv += 1;
        }
        ////$      do ior = 1, max_mod_obj
        ////$      if (numberInList(ior, iobjAli, nrealPt, 0)) &
        ////$                    write(*,'(i4,3f10.2)') ior, (xyzSave(j, ior), j = 1, 3)
        ////$      enddo

        *if_did_align = 1;
        *if_align_done = 1;
    }
    // Args assigned to: ifDidAlign aliMeanRes ifAlignDone
}
