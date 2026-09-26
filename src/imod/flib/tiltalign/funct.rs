//! Translation of `IMOD/flib/tiltalign/funct.cpp` — computes the error and
//! gradient values for the `metro` variable-metric minimiser, plus the
//! member functions of `class EvalFunct` (`evalfunct.h`; the struct and its
//! data members are in `evalfunct.rs`).
//!
//! Compiled into both `tiltalign` and `beadtrack` (`flib/beadtrack/Makefile`
//! lists `funct.o`), with no `BEADTRACK` conditional anywhere in the unit, so
//! there is one build and no `const BEADTRACK` parameter.
//!
//! # File-scope statics
//!
//! - `sEvalFunct`, `av`, `mx` (`funct.cpp:56-58`), set by
//!   `allocateFunctVars`: per `alivar.rs`/`evalfunct.rs` these are
//!   parameters.  The free function [`funct`] that `metroSearch` calls back
//!   takes the `EvalFunct`, `&mut AlignVariables` and `&ArrayMaxes`
//!   explicitly; a caller wraps it in a closure capturing all three.
//! - The per-thread gradient scratch `sGradSums`, `sCoefX`, `sCoefY`,
//!   `sThreadAlloc`, `sNumThreads` (`:61-64`) and `funct`'s function-local
//!   `static double errorMin` (`:152`) are process-wide state that persists
//!   across calls and across `EvalFunct` instances, so they live in one
//!   process-global [`FunctStatics`] behind a `Mutex`, locked once per
//!   [`EvalFunct::funct`] call.  `freeThreadAllocations` takes the locked
//!   state as a parameter so it is not re-locked.
//! - `functWallCum`/`functNumCalls` (`:142-143`) are extern globals that
//!   `beamtilt.cpp:211-212` resets; they are [`FUNCT_WALL_CUM`] and
//!   [`FUNCT_NUM_CALLS`].
//!
//! # OpenMP (`funct.cpp:713`)
//!
//! The coordinate-gradient loop is `#pragma omp parallel for` over real
//! points with no `schedule` clause, which GCC compiles to the static
//! schedule: thread `t` of `n` gets `q + (t < r)` iterations starting at
//! `t*q + min(t, r)` (`q = nrealPt / n`, `r = nrealPt % n`).  Each thread
//! accumulates into its own `sGradSums[t]`, and the sums are then added in
//! thread order starting from `0.`.  That is a floating-point reduction whose
//! grouping depends on the thread count, so native output can in principle
//! depend on `sNumThreads`.  Measured (2026-09-25, `/big/henriksson/realbench/
//! wave3-funct`): native at `OMP_NUM_THREADS` 1-6 gave bit-identical `funct`
//! output over 444 evaluations in 12 configurations, and identical results
//! through 10 full native `metroSearch` minimisations — the double partials
//! are sums of `2. * w * float` terms, whose regrouping differences are far
//! below the `float` rounding of `grad`.  The translation reproduces the
//! partition exactly: it computes `sNumThreads` through `numOMPthreads` as
//! the source does, runs each static-schedule chunk as its own task into its
//! own `grad_sums[t]` (concurrently on `metro::OMP_TEAM_POOL`, the caller running chunk 0,
//! when `sNumThreads > 1`),
//! and reduces them serially in thread order.  Which worker runs a chunk
//! cannot change a value, so the result is bit-identical to native at every
//! thread count and to the serial run at the same `sNumThreads`, including
//! `OMP_NUM_THREADS=1`, where it is the plain serial order.
//!
//! # Arithmetic
//!
//! Declared types are kept: `error`, `gradSum`, `wsum` and `matrix_to_coef`'s
//! `tmp` are `double`, everything else `float`.  `2. * …` and `1. - frc`
//! are double literals, so those subexpressions evaluate in double while
//! the `float * float` products inside them round to single first.
//! `pow(float, 2)` is C++11's promoted overload (the reference compiles as
//! the default `gnu++17`): `pow(double(x), 2.)`, which is the exact square,
//! written here as `x as f64 * x as f64`.  `grad[i] += double` is
//! `grad = (float)((double)grad + expr)`.
//!
//! # Allocation
//!
//! `B3DMALLOC` becomes a zero-filled `Vec` (`NATIVE.md` §4); the `ierr`
//! test in `allocateFunctVars` and `memoryError` see every array allocated.

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, Ordering};

use super::alivar::AlignVariables;
use super::arraymaxes::ArrayMaxes;
use super::evalfunct::EvalFunct;
use super::fill_matrices::{
    fill_beam_matrices, fill_dist_matrix, fill_proj_matrix, fill_rot_matrix, fill_xtilt_matrix,
    fill_ytilt_matrix, mat_product, zero_matrix,
};
use super::utilfuncs::memory_error;
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes, num_omp_threads, wall_time};
use crate::imod::libcfshr::metro::OMP_TEAM_POOL;
use crate::imod::libcfshr::regression::mult_regress;

/// Original: `#define MAX_THREADS 8` (`funct.cpp:18`).
const MAX_THREADS: usize = 8;

/// The file-scope thread statics of `funct.cpp:61-64` plus `funct`'s
/// function-local `static double errorMin` (`:152`).
///
/// `numOMPthreads` can return more than `MAX_THREADS` only under
/// `IMOD_FORCE_OMP_THREADS`; the source then writes past the static arrays.
/// Fixed in translation (2026-09-26, `BUGS.md`): the thread count is limited
/// to `MAX_THREADS`.
pub struct FunctStatics {
    /// Original: `static double *sGradSums[MAX_THREADS]`.
    grad_sums: [Vec<f64>; MAX_THREADS],
    /// Original: `static float *sCoefX[MAX_THREADS]`.
    coef_x: [Vec<f32>; MAX_THREADS],
    /// Original: `static float *sCoefY[MAX_THREADS]`.
    coef_y: [Vec<f32>; MAX_THREADS],
    /// Original: `static int sThreadAlloc = 0`.
    thread_alloc: i32,
    /// Original: `static int sNumThreads = 1`.
    num_threads: i32,
    /// Original: `static double errorMin` in `EvalFunct::funct`.
    error_min: f64,
}

static FUNCT_STATICS: Mutex<FunctStatics> = Mutex::new(FunctStatics {
    grad_sums: [const { Vec::new() }; MAX_THREADS],
    coef_x: [const { Vec::new() }; MAX_THREADS],
    coef_y: [const { Vec::new() }; MAX_THREADS],
    thread_alloc: 0,
    num_threads: 1,
    error_min: 0.,
});

/// One partition of the `funct.cpp:713` region: thread index and that
/// thread's `sCoefX`, `sCoefY` and `sGradSums`.
type FunctPart<'a> = (i32, &'a mut Vec<f32>, &'a mut Vec<f32>, &'a mut Vec<f64>);

/// Original: `double functWallCum = 0.;` (`funct.cpp:142`).
pub static FUNCT_WALL_CUM: Mutex<f64> = Mutex::new(0.);
/// Original: `int functNumCalls = 0;` (`funct.cpp:143`).
pub static FUNCT_NUM_CALLS: AtomicI32 = AtomicI32::new(0);

/// Original: `funct` (`funct.cpp:66`) — the free function `metroSearch` calls
/// back; forwards to the member through the `sEvalFunct` static, which is the
/// `eval_funct` parameter here.
pub fn funct(
    eval_funct: &mut EvalFunct,
    av: &mut AlignVariables,
    mx: &ArrayMaxes,
    nvar_search_in: i32,
    var: &mut [f32],
    ferror: &mut f32,
    grad: &mut [f32],
) {
    eval_funct.funct(av, mx, nvar_search_in, var, ferror, grad);
}

impl EvalFunct {
    /// Original: `EvalFunct::allocateFunctVars` (`funct.cpp:72`).
    ///
    /// The source stores `this`, `avIn` and `mxIn` in the file statics; here
    /// they are passed to each member that reads them, so `av_in` is unused.
    pub fn allocate_funct_vars(
        &mut self,
        av_in: &AlignVariables,
        mx_in: &ArrayMaxes,
        ierr: &mut i32,
    ) {
        let _ = av_in;
        let mx = mx_in;
        let ms = mx_in.max_view as usize;
        *ierr = 0;
        let max_real = mx.max_real as usize;
        let max_proj_pt = mx.max_proj_pt as usize;
        self.m_real_in_view = vec![0; max_real * ms];
        self.m_xbar = vec![0.; ms];
        self.m_ybar = vec![0.; ms];
        self.m_xproj = vec![0.; max_proj_pt];
        self.m_yproj = vec![0.; max_proj_pt];
        self.m_xcen = vec![0.; ms];
        self.m_ycen = vec![0.; ms];
        self.m_zcen = vec![0.; ms];
        self.m_a = vec![0.; ms];
        self.m_b = vec![0.; ms];
        self.m_c = vec![0.; ms];
        self.m_d = vec![0.; ms];
        self.m_e = vec![0.; ms];
        self.m_f = vec![0.; ms];
        self.m_a_over_n = vec![0.; ms];
        self.m_b_over_n = vec![0.; ms];
        self.m_c_over_n = vec![0.; ms];
        self.m_d_over_n = vec![0.; ms];
        self.m_e_over_n = vec![0.; ms];
        self.m_f_over_n = vec![0.; ms];
        self.m_cos_bet = vec![0.; ms];
        self.m_sin_bet = vec![0.; ms];
        self.m_cos_alf = vec![0.; ms];
        self.m_sin_alf = vec![0.; ms];
        self.m_cos_gam = vec![0.; ms];
        self.m_sin_gam = vec![0.; ms];
        self.m_cos_del = vec![0.; ms];
        self.m_sin_del = vec![0.; ms];
        self.m_mag = vec![0.; ms];
        self.m_indv_real = vec![0; max_real * ms];
        self.m_npt_in_view = vec![0; ms];
        self.m_indv_proj = vec![0; max_real * ms];
        //mCoefX = B3DMALLOC(float, 3 * mx->maxReal);
        //mCoefY = B3DMALLOC(float, 3 * mx->maxReal);
        self.m_res_prod = vec![0.; 6 * max_proj_pt];
        self.m_dmat = vec![0.; 9 * ms];
        self.m_xtmat = vec![0.; 9 * ms];
        self.m_ytmat = vec![0.; 9 * ms];
        self.m_rmat = vec![0.; 4 * ms];
        self.m_beam_inv = vec![0.; 9 * ms];
        self.m_beam_mat = vec![0.; 6 * ms];
        self.m_xyzmat = vec![0.; 8 * ms];
        // The source's NULL test cannot fail: a Vec allocation failure aborts.
    }

    /// Original: `EvalFunct::freeThreadAllocations` (`funct.cpp:130`).
    ///
    /// Touches only the file statics, which the caller has locked.
    fn free_thread_allocations(&mut self, statics: &mut FunctStatics) {
        if statics.thread_alloc <= 0 {
            return;
        }
        for ii in 0..statics.num_threads as usize {
            statics.coef_x[ii] = Vec::new();
            statics.coef_y[ii] = Vec::new();
            statics.grad_sums[ii] = Vec::new();
        }
        statics.thread_alloc = 0;
    }

    /// Original: `EvalFunct::funct` (`funct.cpp:145`).
    pub fn funct(
        &mut self,
        av: &mut AlignVariables,
        mx: &ArrayMaxes,
        nvar_search_in: i32,
        var: &mut [f32],
        ferror: &mut f32,
        grad: &mut [f32],
    ) {
        let mut statics_guard = FUNCT_STATICS.lock().unwrap_or_else(|e| e.into_inner());
        let statics = &mut *statics_guard;
        //
        //
        let nvar_search = nvar_search_in;
        //
        let mut error: f64;
        let mut grad_sum: f64;
        //
        let mut dermat = [0f32; 9];
        let mut proj_mat = [0f32; 4];
        let mut umat = [0f32; 9];
        //
        let num_proj_pt: i32;
        let mut iv: i32;
        let mut jpt: i32;
        let mut ii: i32;
        let (mut afac, mut bfac, mut cfac, mut dfac, mut efac, mut ffac) =
            (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
        let mut val_add: f32;
        let (mut cos_p_skew, mut sin_p_skew, mut cos_beam, mut sin_beam) = (0f32, 0f32, 0f32, 0f32);
        let mut wgt: f32;
        let mut wsum: f64 = 0.;
        let (mut cos2rot, mut sin2rot) = (0f32, 0f32);
        let nreal_pt = av.nreal_pt;
        let max_real = mx.max_real;
        let mut star: bool;
        let wall_start = wall_time();
        //
        // Stretch type: 1 for dmag = stretch on X axis, skew = X axis rotation
        // 2 for dmag = stretch on X axis, skew = +Y - axis and - X - axis rotation
        // 3 for dmag = +Y - axis and - X - axis stretch, skew = +Y - axis and - X - axis
        // rotation
        // Since the X - stretch frequently occurs because of thinning, the third
        // formulation would often affect mag inappropriately.
        // Neither the second nor the third seemed to reduce solved rotation
        //
        let istr_type: i32 = 1;
        //
        // first time in, precompute the mean projection coords in each view
        // and build indexes to the points in each view.
        //
        num_proj_pt = av.ireal_str[(av.nreal_pt + 1 - 1) as usize] - 1;
        if av.first_funct != 0 {
            for ii in 0..av.nview as usize {
                self.m_xbar[ii] = 0.;
                self.m_ybar[ii] = 0.;
                self.m_npt_in_view[ii] = 0;
            }

            // Initialize the weights unless one of the special setups is doing so
            if !(av.robust_weights != 0 || av.leaving_out != 0 || av.test_set_frac_step > 0.) {
                for ii in 0..num_proj_pt.max(0) as usize {
                    av.weight[ii] = 1.;
                }
            }

            // Apply extra weights: first get sum so they can be normalized
            if av.apply_extra_weights != 0 {
                for jpt in 0..av.nreal_pt.max(0) as usize {
                    wgt = 1.;
                    if let Some(&value) = av.object_weight_map.get(&av.imod_obj_num[jpt]) {
                        wgt = value;
                    }
                    wsum += wgt as f64;
                }

                // Then apply to each point weight
                for jpt in 0..av.nreal_pt.max(0) as usize {
                    wgt = 1.;
                    if let Some(&value) = av.object_weight_map.get(&av.imod_obj_num[jpt]) {
                        wgt = value;
                    }
                    wgt = (wgt as f64 * (av.nreal_pt as f64 / wsum)) as f32;
                    ii = av.ireal_str[jpt] - 1;
                    while ii < av.ireal_str[jpt + 1] - 1 {
                        av.weight[ii as usize] *= wgt;
                        ii += 1;
                    }
                }
            }

            self.m_real_in_view[..(mx.max_real * av.nview).max(0) as usize].fill(0);
            //
            jpt = 1;
            while jpt <= av.nreal_pt {
                ii = av.ireal_str[(jpt - 1) as usize];
                while ii <= av.ireal_str[(jpt + 1 - 1) as usize] - 1 {
                    iv = av.isec_view[(ii - 1) as usize];
                    let ivm = (iv - 1) as usize;
                    self.m_npt_in_view[ivm] = self.m_npt_in_view[ivm] + 1;
                    let slot = ((iv - 1) * mx.max_real + self.m_npt_in_view[ivm] - 1) as usize;
                    self.m_indv_real[slot] = jpt as i16;
                    self.m_indv_proj[slot] = ii;
                    self.m_real_in_view[((iv - 1) * mx.max_real + jpt - 1) as usize] = 1;
                    self.m_xbar[ivm] = self.m_xbar[ivm] + av.xx[(ii - 1) as usize];
                    self.m_ybar[ivm] = self.m_ybar[ivm] + av.yy[(ii - 1) as usize];
                    ii += 1;
                }
                jpt += 1;
            }
            //
            for iv in 0..av.nview.max(0) as usize {
                self.m_xbar[iv] = self.m_xbar[iv] / self.m_npt_in_view[iv] as f32;
                self.m_ybar[iv] = self.m_ybar[iv] / self.m_npt_in_view[iv] as f32;
            }
            av.first_funct = 0;
            self.remap_params(av, var);

            statics.error_min = 1.0e37;
            ii = (av.nreal_pt as f64 / 16.0 + 0.5).floor() as i32;
            ii = 1.max(6.min(ii));
            ii = num_omp_threads(ii).min(MAX_THREADS as i32);
            if 3 * av.nreal_pt > statics.thread_alloc || ii != statics.num_threads {
                self.free_thread_allocations(statics);
                statics.thread_alloc = 3 * av.nreal_pt;
                for iv in 0..ii as usize {
                    statics.grad_sums[iv] = vec![0.; statics.thread_alloc as usize];
                    statics.coef_x[iv] = vec![0.; statics.thread_alloc as usize];
                    statics.coef_y[iv] = vec![0.; statics.thread_alloc as usize];
                    memory_error(true, "arrays for computing gradients in threads");
                }
            }
            statics.num_threads = ii;
        }
        //
        // precompute the a - f and items related to them.  Store the component
        // matrices to use for computing gradient coefficients
        //
        self.remap_params(av, var);
        //

        fill_proj_matrix(
            av.proj_str_rot,
            av.proj_skew,
            &mut proj_mat,
            &mut cos_p_skew,
            &mut sin_p_skew,
            &mut cos2rot,
            &mut sin2rot,
        );
        fill_beam_matrices(
            av.beam_tilt,
            &mut self.m_beam_inv,
            &mut self.m_beam_mat,
            &mut cos_beam,
            &mut sin_beam,
        );
        for ii in 1..=av.nview.max(0) as usize {
            self.m_mag[ii - 1] = av.gmag[ii - 1] + av.dmag[ii - 1];
            fill_dist_matrix(
                av.gmag[ii - 1],
                av.dmag[ii - 1],
                av.skew[ii - 1],
                av.comp[ii - 1],
                istr_type,
                &mut self.m_dmat[ii * 9 - 9..],
                &mut self.m_cos_del[ii - 1],
                &mut self.m_sin_del[ii - 1],
            );
            fill_xtilt_matrix(
                av.alf[ii - 1],
                av.if_any_alf,
                &mut self.m_xtmat[ii * 9 - 9..],
                &mut self.m_cos_alf[ii - 1],
                &mut self.m_sin_alf[ii - 1],
            );
            fill_ytilt_matrix(
                av.tilt[ii - 1],
                &mut self.m_ytmat[ii * 9 - 9..],
                &mut self.m_cos_bet[ii - 1],
                &mut self.m_sin_bet[ii - 1],
            );
            fill_rot_matrix(
                av.rot[ii - 1],
                &mut self.m_rmat[ii * 4 - 4..],
                &mut self.m_cos_gam[ii - 1],
                &mut self.m_sin_gam[ii - 1],
            );
            EvalFunct::matrix_to_coef(
                &self.m_dmat[ii * 9 - 9..],
                &self.m_xtmat[ii * 9 - 9..],
                &self.m_beam_inv,
                &self.m_ytmat[ii * 9 - 9..],
                &self.m_beam_mat,
                &proj_mat,
                &self.m_rmat[ii * 4 - 4..],
                &mut self.m_a[ii - 1],
                &mut self.m_b[ii - 1],
                &mut self.m_c[ii - 1],
                &mut self.m_d[ii - 1],
                &mut self.m_e[ii - 1],
                &mut self.m_f[ii - 1],
            );
        }
        //
        for ii in 1..=av.nview.max(0) as usize {
            let n = self.m_npt_in_view[ii - 1] as f32;
            self.m_a_over_n[ii - 1] = -self.m_a[ii - 1] / n;
            self.m_b_over_n[ii - 1] = -self.m_b[ii - 1] / n;
            self.m_c_over_n[ii - 1] = -self.m_c[ii - 1] / n;
            self.m_d_over_n[ii - 1] = -self.m_d[ii - 1] / n;
            self.m_e_over_n[ii - 1] = -self.m_e[ii - 1] / n;
            self.m_f_over_n[ii - 1] = -self.m_f[ii - 1] / n;
        }
        //
        let nvmat = 3 * (av.nreal_pt - 1); //# of x, y, z variables
        let icoord_bas = nvar_search - nvmat; //offset to x, y, z"s
        let kzlas = (icoord_bas + av.nreal_pt * 3) as usize; //indexes of x, y, z of last point
        let kylas = kzlas - 1;
        let kxlas = kylas - 1;
        //
        // get xproj and yproj: for now, these will be projected points minus
        // the dx, dy values
        // compute the coordinates of the last point: minus the sum of the rest
        //
        for iv in 0..av.nview.max(0) as usize {
            self.m_xcen[iv] = 0.;
            self.m_ycen[iv] = 0.;
            self.m_zcen[iv] = 0.;
        }
        var[kxlas - 1] = 0.;
        var[kylas - 1] = 0.;
        var[kzlas - 1] = 0.;
        //
        jpt = 1;
        while jpt <= av.nreal_pt {
            let kz = (icoord_bas + jpt * 3) as usize;
            let ky = kz - 1;
            let kx = ky - 1;
            //
            if jpt < av.nreal_pt {
                //accumulate last point coords
                var[kxlas - 1] = var[kxlas - 1] - var[kx - 1];
                var[kylas - 1] = var[kylas - 1] - var[ky - 1];
                var[kzlas - 1] = var[kzlas - 1] - var[kz - 1];
            }
            //
            let j3 = (jpt * 3) as usize;
            av.xyz[j3 - 3] = var[kx - 1]; //unpack the coordinates
            av.xyz[j3 - 2] = var[ky - 1];
            av.xyz[j3 - 1] = var[kz - 1];
            //
            ii = av.ireal_str[(jpt - 1) as usize];
            while ii <= av.ireal_str[(jpt + 1 - 1) as usize] - 1 {
                let ivm = (av.isec_view[(ii - 1) as usize] - 1) as usize;
                let im = (ii - 1) as usize;
                self.m_xproj[im] = self.m_a[ivm] * var[kx - 1]
                    + self.m_b[ivm] * var[ky - 1]
                    + self.m_c[ivm] * var[kz - 1];
                self.m_yproj[im] = self.m_d[ivm] * var[kx - 1]
                    + self.m_e[ivm] * var[ky - 1]
                    + self.m_f[ivm] * var[kz - 1];
                self.m_xcen[ivm] = self.m_xcen[ivm] + var[kx - 1];
                self.m_ycen[ivm] = self.m_ycen[ivm] + var[ky - 1];
                self.m_zcen[ivm] = self.m_zcen[ivm] + var[kz - 1];
                ii += 1;
            }
            jpt += 1;
        }
        //
        // get xcen, ycen, zcen scaled, and get the dx and dy
        //
        for iv in 1..=av.nview.max(0) as usize {
            let n = self.m_npt_in_view[iv - 1] as f32;
            self.m_xcen[iv - 1] = self.m_xcen[iv - 1] / n;
            self.m_ycen[iv - 1] = self.m_ycen[iv - 1] / n;
            self.m_zcen[iv - 1] = self.m_zcen[iv - 1] / n;
            av.dxy[iv * 2 - 2] = self.m_xbar[iv - 1]
                - self.m_a[iv - 1] * self.m_xcen[iv - 1]
                - self.m_b[iv - 1] * self.m_ycen[iv - 1]
                - self.m_c[iv - 1] * self.m_zcen[iv - 1];
            av.dxy[iv * 2 - 1] = self.m_ybar[iv - 1]
                - self.m_d[iv - 1] * self.m_xcen[iv - 1]
                - self.m_e[iv - 1] * self.m_ycen[iv - 1]
                - self.m_f[iv - 1] * self.m_zcen[iv - 1];
        }
        //
        // adjust xproj&yproj by dxy, get residuals and errors
        //
        error = 0.;
        for ii in 1..=num_proj_pt.max(0) as usize {
            let iv = av.isec_view[ii - 1] as usize;
            self.m_xproj[ii - 1] = self.m_xproj[ii - 1] + av.dxy[iv * 2 - 2];
            self.m_yproj[ii - 1] = self.m_yproj[ii - 1] + av.dxy[iv * 2 - 1];
            av.xresid[ii - 1] = self.m_xproj[ii - 1] - av.xx[ii - 1];
            av.yresid[ii - 1] = self.m_yproj[ii - 1] - av.yy[ii - 1];
            let xr = av.xresid[ii - 1] as f64;
            let yr = av.yresid[ii - 1] as f64;
            error = error + (xr * xr + yr * yr) * av.weight[ii - 1] as f64;
        }
        //
        // Compute projections to fill in model
        if av.project_fill_points != 0 {
            for jpt in 1..=av.nreal_pt.max(0) as usize {
                ii = av.ifill_real_start[jpt - 1];
                while ii <= av.ifill_real_start[jpt + 1 - 1] - 1 {
                    let iv = av.ifill_view[(ii - 1) as usize] as usize;
                    av.xfill_proj[(ii - 1) as usize] = self.m_a[iv - 1] * av.xyz[jpt * 3 - 3]
                        + self.m_b[iv - 1] * av.xyz[jpt * 3 - 2]
                        + self.m_c[iv - 1] * av.xyz[jpt * 3 - 1]
                        + av.dxy[iv * 2 - 2];
                    av.yfill_proj[(ii - 1) as usize] = self.m_d[iv - 1] * av.xyz[jpt * 3 - 3]
                        + self.m_e[iv - 1] * av.xyz[jpt * 3 - 2]
                        + self.m_f[iv - 1] * av.xyz[jpt * 3 - 1]
                        + av.dxy[iv * 2 - 1];
                    ii += 1;
                }
            }
        }

        //
        // precompute products needed for gradients
        //
        for iv in 1..=av.nview.max(0) as usize {
            for ipt_in_v in 1..=self.m_npt_in_view[iv - 1].max(0) as usize {
                let slot = (iv - 1) * max_real as usize + ipt_in_v - 1;
                let ipt = self.m_indv_proj[slot] as usize;
                let jpt = self.m_indv_real[slot] as usize;
                let xr = av.xresid[ipt - 1] as f64;
                let yr = av.yresid[ipt - 1] as f64;
                let w = av.weight[ipt - 1] as f64;
                let dx = (av.xyz[jpt * 3 - 3] - self.m_xcen[iv - 1]) as f64;
                let dy = (av.xyz[jpt * 3 - 2] - self.m_ycen[iv - 1]) as f64;
                let dz = (av.xyz[jpt * 3 - 1] - self.m_zcen[iv - 1]) as f64;
                self.m_res_prod[ipt * 6 - 6] = (2. * dx * xr * w) as f32;
                self.m_res_prod[ipt * 6 - 5] = (2. * dy * xr * w) as f32;
                self.m_res_prod[ipt * 6 - 4] = (2. * dz * xr * w) as f32;
                self.m_res_prod[ipt * 6 - 3] = (2. * dx * yr * w) as f32;
                self.m_res_prod[ipt * 6 - 2] = (2. * dy * yr * w) as f32;
                self.m_res_prod[ipt * 6 - 1] = (2. * dz * yr * w) as f32;
            }
        }

        // Keep track of minimum error for trace output
        *ferror = error as f32;
        star = false;
        if error < statics.error_min {
            statics.error_min = error;
            star = true;
        }
        let _ = star;
        // write(*,'(f55.15,a)') error, star
        //
        // compute derivatives of error w / r to search parameters
        // first clear out all the gradients
        //
        for ivar in 1..=nvar_search.max(0) as usize {
            grad[ivar - 1] = 0.;
        }
        //
        // loop on views: consider each of the parameters
        //
        for iv in 1..=av.nview.max(0) as usize {
            let mr = (iv - 1) * max_real as usize;
            let npt = self.m_npt_in_view[iv - 1];
            //
            // rotation: add gradient for this view to any variables that it is
            // mapped to
            // These equations are valid as long as rotation is the last operation
            //
            grad_sum = 0.;
            if av.map_rot[iv - 1] > 0 {
                for ipt_in_v in 1..=npt.max(0) as usize {
                    let ipt = self.m_indv_proj[mr + ipt_in_v - 1] as usize;
                    grad_sum = grad_sum
                        + 2. * av.weight[ipt - 1] as f64
                            * ((self.m_ybar[iv - 1] - self.m_yproj[ipt - 1]) * av.xresid[ipt - 1]
                                + (self.m_xproj[ipt - 1] - self.m_xbar[iv - 1])
                                    * av.yresid[ipt - 1]) as f64;
                }
                let m = (av.map_rot[iv - 1] - 1) as usize;
                grad[m] = (grad[m] as f64 + av.frc_rot[iv - 1] as f64 * grad_sum) as f32;
                if av.lin_rot[iv - 1] > 0 {
                    let l = (av.lin_rot[iv - 1] - 1) as usize;
                    grad[l] = (grad[l] as f64 + (1. - av.frc_rot[iv - 1] as f64) * grad_sum) as f32;
                }
            }
            //
            // tilt: add gradient for this tilt angle to the variable it is mapped
            // from, if any
            //
            if av.map_tilt[iv - 1] != 0 {
                zero_matrix(&mut dermat, 9);
                dermat[0] = -self.m_sin_bet[iv - 1];
                dermat[2] = self.m_cos_bet[iv - 1];
                dermat[6] = -self.m_cos_bet[iv - 1];
                dermat[8] = -self.m_sin_bet[iv - 1];

                EvalFunct::matrix_to_coef(
                    &self.m_dmat[iv * 9 - 9..],
                    &self.m_xtmat[iv * 9 - 9..],
                    &self.m_beam_inv,
                    &dermat,
                    &self.m_beam_mat,
                    &proj_mat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[mr..],
                    npt,
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                let m = (av.map_tilt[iv - 1] - 1) as usize;
                grad[m] = (grad[m] as f64 + av.frc_tilt[iv - 1] as f64 * grad_sum) as f32;
                if av.lin_tilt[iv - 1] > 0 {
                    let l = (av.lin_tilt[iv - 1] - 1) as usize;
                    grad[l] =
                        (grad[l] as f64 + (1. - av.frc_tilt[iv - 1] as f64) * grad_sum) as f32;
                }
                //
            }
            //
            // mag: add gradient for this view to the variable it is mapped from
            //
            if av.map_gmag[iv - 1] > 0 {
                zero_matrix(&mut dermat, 9);
                if istr_type == 1 {
                    dermat[0] = self.m_cos_del[iv - 1];
                    dermat[3] = self.m_sin_del[iv - 1];
                    dermat[4] = 1.;
                } else {
                    dermat[0] = self.m_cos_del[iv - 1];
                    dermat[1] = -self.m_sin_del[iv - 1];
                    dermat[3] = -self.m_sin_del[iv - 1];
                    dermat[4] = self.m_cos_del[iv - 1];
                }
                dermat[8] = av.comp[iv - 1];
                EvalFunct::matrix_to_coef(
                    &dermat,
                    &self.m_xtmat[iv * 9 - 9..],
                    &self.m_beam_inv,
                    &self.m_ytmat[iv * 9 - 9..],
                    &self.m_beam_mat,
                    &proj_mat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[mr..],
                    npt,
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                // write(*,'(i4,3f9.5,f16.10)') iv, gmag(iv), dmag(iv), skew(iv), gradSum
                let m = (av.map_gmag[iv - 1] - 1) as usize;
                grad[m] = (grad[m] as f64 + av.frc_gmag[iv - 1] as f64 * grad_sum) as f32;
                if av.lin_gmag[iv - 1] > 0 {
                    let l = (av.lin_gmag[iv - 1] - 1) as usize;
                    grad[l] =
                        (grad[l] as f64 + (1. - av.frc_gmag[iv - 1] as f64) * grad_sum) as f32;
                }
            }
            //
            // comp: add gradient for this view to the variable it is mapped from
            // These equation are valid as long as the there is nothing else in
            // the final column of the distortion matrix
            //
            if av.map_comp[iv - 1] > 0 {
                grad_sum = 0.;
                cfac = self.m_c[iv - 1] / av.comp[iv - 1];
                ffac = self.m_f[iv - 1] / av.comp[iv - 1];
                for ipt_in_v in 1..=npt.max(0) as usize {
                    let ipt = self.m_indv_proj[mr + ipt_in_v - 1] as usize;
                    grad_sum = grad_sum
                        + (cfac * self.m_res_prod[ipt * 6 - 4]) as f64
                        + (ffac * self.m_res_prod[ipt * 6 - 1]) as f64;
                }
                let m = (av.map_comp[iv - 1] - 1) as usize;
                grad[m] = (grad[m] as f64 + av.frc_comp[iv - 1] as f64 * grad_sum) as f32;
                if av.lin_comp[iv - 1] > 0 {
                    let l = (av.lin_comp[iv - 1] - 1) as usize;
                    grad[l] =
                        (grad[l] as f64 + (1. - av.frc_comp[iv - 1] as f64) * grad_sum) as f32;
                }
            }
            //
            // dmag: add gradient for this view to the variable it is mapped from
            //
            if av.map_dmag[iv - 1] > 0 {
                zero_matrix(&mut dermat, 9);
                if istr_type == 1 {
                    dermat[0] = self.m_cos_del[iv - 1];
                    dermat[3] = self.m_sin_del[iv - 1];
                } else if istr_type == 2 {
                    dermat[0] = self.m_cos_del[iv - 1];
                    dermat[3] = -self.m_sin_del[iv - 1];
                } else {
                    dermat[0] = -self.m_cos_del[iv - 1];
                    dermat[1] = -self.m_sin_del[iv - 1];
                    dermat[3] = self.m_sin_del[iv - 1];
                    dermat[4] = self.m_cos_del[iv - 1];
                }
                EvalFunct::matrix_to_coef(
                    &dermat,
                    &self.m_xtmat[iv * 9 - 9..],
                    &self.m_beam_inv,
                    &self.m_ytmat[iv * 9 - 9..],
                    &self.m_beam_mat,
                    &proj_mat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[mr..],
                    npt,
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                //
                // if this parameter maps to the dummy dmag, then need to subtract
                // the fraction times the gradient sum from gradient of every real
                // variable
                //
                if av.map_dmag[iv - 1] == av.map_dum_dmag || av.lin_dmag[iv - 1] == av.map_dum_dmag
                {
                    if av.map_dmag[iv - 1] == av.map_dum_dmag {
                        val_add =
                            ((av.dum_dmag_fac * av.frc_dmag[iv - 1]) as f64 * grad_sum) as f32;
                        if av.lin_dmag[iv - 1] > 0 {
                            let l = (av.lin_dmag[iv - 1] - 1) as usize;
                            grad[l] = (grad[l] as f64
                                + (1. - av.frc_dmag[iv - 1] as f64) * grad_sum)
                                as f32;
                        }
                    } else {
                        val_add = (av.dum_dmag_fac as f64
                            * (1. - av.frc_dmag[iv - 1] as f64)
                            * grad_sum) as f32;
                        // Fixed in translation (2026-09-26, `BUGS.md`): the source
                        // assigns here rather than accumulating (`funct.cpp:583`), so a
                        // view whose `linDmag` is the dummy overwrites what earlier
                        // views sharing `mapDmag` added; it accumulates here, as every
                        // other gradient site does.
                        let m = (av.map_dmag[iv - 1] - 1) as usize;
                        grad[m] = (grad[m] as f64 + av.frc_dmag[iv - 1] as f64 * grad_sum) as f32;
                    }
                    for jj in av.map_dmag_start..=av.map_dum_dmag - 1 {
                        grad[(jj - 1) as usize] = grad[(jj - 1) as usize] + val_add;
                    }
                } else {
                    let m = (av.map_dmag[iv - 1] - 1) as usize;
                    grad[m] = (grad[m] as f64 + av.frc_dmag[iv - 1] as f64 * grad_sum) as f32;
                    if av.lin_dmag[iv - 1] > 0 {
                        let l = (av.lin_dmag[iv - 1] - 1) as usize;
                        grad[l] =
                            (grad[l] as f64 + (1. - av.frc_dmag[iv - 1] as f64) * grad_sum) as f32;
                    }
                }
            }
            //
            // skew: add gradient for this view to the variable it is mapped from
            //
            if av.map_skew[iv - 1] > 0 {
                zero_matrix(&mut dermat, 9);
                if istr_type == 1 {
                    dermat[0] = -self.m_mag[iv - 1] * self.m_sin_del[iv - 1];
                    dermat[3] = self.m_mag[iv - 1] * self.m_cos_del[iv - 1];
                } else if istr_type == 2 {
                    dermat[0] = -self.m_mag[iv - 1] * self.m_sin_del[iv - 1];
                    dermat[1] = -av.gmag[iv - 1] * self.m_cos_del[iv - 1];
                    dermat[3] = -self.m_mag[iv - 1] * self.m_cos_del[iv - 1];
                    dermat[4] = -av.gmag[iv - 1] * self.m_sin_del[iv - 1];
                } else {
                    dermat[0] = -(av.gmag[iv - 1] - av.dmag[iv - 1]) * self.m_sin_del[iv - 1];
                    dermat[1] = -self.m_mag[iv - 1] * self.m_cos_del[iv - 1];
                    dermat[3] = -(av.gmag[iv - 1] - av.dmag[iv - 1]) * self.m_cos_del[iv - 1];
                    dermat[4] = -self.m_mag[iv - 1] * self.m_sin_del[iv - 1];
                }
                EvalFunct::matrix_to_coef(
                    &dermat,
                    &self.m_xtmat[iv * 9 - 9..],
                    &self.m_beam_inv,
                    &self.m_ytmat[iv * 9 - 9..],
                    &self.m_beam_mat,
                    &proj_mat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[mr..],
                    npt,
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                let m = (av.map_skew[iv - 1] - 1) as usize;
                grad[m] = (grad[m] as f64 + av.frc_skew[iv - 1] as f64 * grad_sum) as f32;
                if av.lin_skew[iv - 1] > 0 {
                    let l = (av.lin_skew[iv - 1] - 1) as usize;
                    grad[l] =
                        (grad[l] as f64 + (1. - av.frc_skew[iv - 1] as f64) * grad_sum) as f32;
                }

                //
            }
            //
            // alpha: add gradient for this view to the variable it is mapped from
            //
            if av.map_alf[iv - 1] > 0 {
                zero_matrix(&mut dermat, 9);
                dermat[4] = -self.m_sin_alf[iv - 1];
                dermat[5] = -self.m_cos_alf[iv - 1];
                dermat[7] = self.m_cos_alf[iv - 1];
                dermat[8] = -self.m_sin_alf[iv - 1];
                EvalFunct::matrix_to_coef(
                    &self.m_dmat[iv * 9 - 9..],
                    &dermat,
                    &self.m_beam_inv,
                    &self.m_ytmat[iv * 9 - 9..],
                    &self.m_beam_mat,
                    &proj_mat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[mr..],
                    npt,
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                // write(*,'(3i4,f7.4,f16.10)') iv, mapAlf(iv), linAlf(iv) &
                // , frcAlf(iv), gradSum
                let m = (av.map_alf[iv - 1] - 1) as usize;
                grad[m] = (grad[m] as f64 + av.frc_alf[iv - 1] as f64 * grad_sum) as f32;
                if av.lin_alf[iv - 1] > 0 {
                    let l = (av.lin_alf[iv - 1] - 1) as usize;
                    grad[l] = (grad[l] as f64 + (1. - av.frc_alf[iv - 1] as f64) * grad_sum) as f32;
                }
            }
        }
        let _ = ImodFile::Stdout.flush();
        //
        // projection skew: do gradient
        //
        if av.map_proj_stretch > 0 {
            //
            dermat[0] = -sin_p_skew + cos_p_skew * sin2rot;
            dermat[1] = -cos_p_skew * cos2rot;
            dermat[3 - 1] = dermat[1];
            dermat[3] = -sin_p_skew - cos_p_skew * sin2rot;
            for iv in 1..=av.nview.max(0) as usize {
                EvalFunct::matrix_to_coef(
                    &self.m_dmat[iv * 9 - 9..],
                    &self.m_xtmat[iv * 9 - 9..],
                    &self.m_beam_inv,
                    &self.m_ytmat[iv * 9 - 9..],
                    &self.m_beam_mat,
                    &dermat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[(iv - 1) * max_real as usize..],
                    self.m_npt_in_view[iv - 1],
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                let m = (av.map_proj_stretch - 1) as usize;
                grad[m] = (grad[m] as f64 + grad_sum) as f32;
            }
        }
        //
        // beam tilt: the derivative is of the product of beamInv, ytmat and
        // beamMat, so pass two unit matrices and derivative
        //
        if av.map_beam_tilt > 0 {
            dermat[0] = 0.;
            zero_matrix(&mut umat, 9);
            umat[0] = 1.;
            umat[4] = 1.;
            umat[9 - 1] = 1.;
            for iv in 1..=av.nview.max(0) as usize {
                dermat[1] = -cos_beam * self.m_sin_bet[iv - 1];
                dermat[3 - 1] = -sin_beam * self.m_sin_bet[iv - 1];
                dermat[3] = cos_beam * self.m_sin_bet[iv - 1];
                dermat[4] =
                    (2. * cos_beam as f64 * sin_beam as f64 * (self.m_cos_bet[iv - 1] as f64 - 1.))
                        as f32;
                dermat[5] = ((cos_beam as f64 * cos_beam as f64
                    - sin_beam as f64 * sin_beam as f64)
                    * (1. - self.m_cos_bet[iv - 1]) as f64) as f32;
                EvalFunct::matrix_to_coef(
                    &self.m_dmat[iv * 9 - 9..],
                    &self.m_xtmat[iv * 9 - 9..],
                    &umat,
                    &umat,
                    &dermat,
                    &proj_mat,
                    &self.m_rmat[iv * 4 - 4..],
                    &mut afac,
                    &mut bfac,
                    &mut cfac,
                    &mut dfac,
                    &mut efac,
                    &mut ffac,
                );
                grad_sum = self.gradient_sum(
                    &self.m_indv_proj[(iv - 1) * max_real as usize..],
                    self.m_npt_in_view[iv - 1],
                    &self.m_res_prod,
                    afac,
                    bfac,
                    cfac,
                    dfac,
                    efac,
                    ffac,
                );
                let m = (av.map_beam_tilt - 1) as usize;
                grad[m] = (grad[m] as f64 + grad_sum) as f32;
            }
        }
        //
        // loop on points, get derivatives w / r to x, y, or z
        //
        //functWallCum += wallTime() - wallStart;
        if av.xyz_fixed != 0 {
            *FUNCT_WALL_CUM.lock().unwrap_or_else(|e| e.into_inner()) += wall_time() - wall_start;
            FUNCT_NUM_CALLS.fetch_add(1, Ordering::SeqCst);
            return;
        }
        //double wallStart = wallTime();

        let nvmat_u = nvmat.max(0) as usize;
        for ii in 0..statics.num_threads as usize {
            statics.grad_sums[ii][..nvmat_u].fill(0.);
        }

        // `#pragma omp parallel for num_threads(sNumThreads)` (`funct.cpp:713`).
        // GCC's static schedule gives thread `indThread` the iterations
        // `start + 1 ..= start + q` below, and each thread accumulates into its
        // own `sGradSums[indThread]` using its own `sCoefX/Y[indThread]`, so
        // each partition is an independent task: running the partitions one
        // after another or concurrently produces the same per-thread sums,
        // which are then added in thread order below exactly as the source
        // does.  The partition, not the worker that runs it, fixes every
        // floating-point grouping, so the result depends only on
        // `sNumThreads` — which `numOMPthreads` computes as native does.
        //
        // Per projection `ii`, the source's `kpt` loop stores one of three
        // sextuples into the coefficient arrays: `m? + ?Rlast` for `kpt ==
        // jpt`, `?Rlast` when point `kpt` is in the view, `m?OverN + ?Rlast`
        // otherwise.  Each sum's operands do not depend on `kpt`, so the three
        // sextuples are formed once per `ii` (the same `float` additions of the
        // same operands, hence the same values) and the loop only selects and
        // stores them.
        let num_threads = statics.num_threads;
        let nm1 = (nreal_pt - 1).max(0) as usize;
        let max_real_u = max_real as usize;
        let real_in_view = &self.m_real_in_view[..];
        let (m_a, m_b, m_c) = (&self.m_a[..], &self.m_b[..], &self.m_c[..]);
        let (m_d, m_e, m_f) = (&self.m_d[..], &self.m_e[..], &self.m_f[..]);
        let (m_aon, m_bon, m_con) = (
            &self.m_a_over_n[..],
            &self.m_b_over_n[..],
            &self.m_c_over_n[..],
        );
        let (m_don, m_eon, m_fon) = (
            &self.m_d_over_n[..],
            &self.m_e_over_n[..],
            &self.m_f_over_n[..],
        );
        let ireal_str = &av.ireal_str[..];
        let isec_view = &av.isec_view[..];
        let weight = &av.weight[..];
        let xresid = &av.xresid[..];
        let yresid = &av.yresid[..];
        let run_thread = |(ind_thread, coef_x, coef_y, grad_sums): FunctPart| {
            let mut q = nreal_pt.max(0) / num_threads;
            let mut r = nreal_pt.max(0) % num_threads;
            if ind_thread < r {
                q += 1;
                r = 0;
            }
            let chunk_start = q * ind_thread + r;
            let chunk_end = chunk_start + q;
            let coef_x = &mut coef_x[..nvmat_u];
            let coef_y = &mut coef_y[..nvmat_u];
            let grad_sums = &mut grad_sums[..nvmat_u];
            for jpt in chunk_start + 1..=chunk_end {
                //
                // for each projection of the real point, find how that point
                // contributes to the derivative w / r to each of the x, y, z
                //
                for ii in ireal_str[(jpt - 1) as usize]..=ireal_str[jpt as usize] - 1 {
                    let ivm = (isec_view[(ii - 1) as usize] - 1) as usize;
                    //
                    // the relation between the projection (x, y) and the set of (x, y, z)
                    // contains the term dxy, which is actually a sum of the (x, y, z) .
                    // There is a 3 by 3 matrix of possibilities: the first set of 3
                    // possibilities is whether this real point is the last one, or
                    // whether the last point is projected in this view or not.
                    //
                    let (xpx_rlast, xpy_rlast, xpz_rlast, ypx_rlast, ypy_rlast, ypz_rlast): (
                        f32,
                        f32,
                        f32,
                        f32,
                        f32,
                        f32,
                    ) = if jpt == nreal_pt {
                        (
                            -m_a[ivm], -m_b[ivm], -m_c[ivm], -m_d[ivm], -m_e[ivm], -m_f[ivm],
                        )
                    } else if real_in_view[ivm * max_real_u + (nreal_pt - 1) as usize] != 0 {
                        (0., 0., 0., 0., 0., 0.)
                    } else {
                        (
                            -m_aon[ivm],
                            -m_bon[ivm],
                            -m_con[ivm],
                            -m_don[ivm],
                            -m_eon[ivm],
                            -m_fon[ivm],
                        )
                    };
                    //
                    // The second set of three possibilities is whether the point whose
                    // coordinate that we are taking the derivative w / r to is the same
                    // as the real point whose projections are being considered
                    // (kpt == jpt), and whether the former point is or is not projected
                    // in the view being considered.
                    //
                    let same = [
                        m_a[ivm] + xpx_rlast,
                        m_b[ivm] + xpy_rlast,
                        m_c[ivm] + xpz_rlast,
                        m_d[ivm] + ypx_rlast,
                        m_e[ivm] + ypy_rlast,
                        m_f[ivm] + ypz_rlast,
                    ];
                    let in_view = [
                        xpx_rlast, xpy_rlast, xpz_rlast, ypx_rlast, ypy_rlast, ypz_rlast,
                    ];
                    let not_in_view = [
                        m_aon[ivm] + xpx_rlast,
                        m_bon[ivm] + xpy_rlast,
                        m_con[ivm] + xpz_rlast,
                        m_don[ivm] + ypx_rlast,
                        m_eon[ivm] + ypy_rlast,
                        m_fon[ivm] + ypz_rlast,
                    ];
                    let riv = &real_in_view[ivm * max_real_u..][..nm1];
                    // `kpt == jpt` happens at most once, at index `jpt - 1`
                    // (never when `jpt == nrealPt`); the stretches before and
                    // after it only choose between the other two sextuples.
                    let fill = |cx: &mut [f32], cy: &mut [f32], riv: &[i8]| {
                        for ((cx, cy), &inv) in
                            cx.chunks_exact_mut(3).zip(cy.chunks_exact_mut(3)).zip(riv)
                        {
                            let v = if inv != 0 { in_view } else { not_in_view };
                            cx[0] = v[0];
                            cx[1] = v[1];
                            cx[2] = v[2];
                            cy[0] = v[3];
                            cy[1] = v[4];
                            cy[2] = v[5];
                        }
                    };
                    let jsame = (jpt - 1) as usize;
                    if jsame < nm1 {
                        let (cx_lo, cx_hi) = coef_x.split_at_mut(3 * jsame);
                        let (cy_lo, cy_hi) = coef_y.split_at_mut(3 * jsame);
                        fill(cx_lo, cy_lo, &riv[..jsame]);
                        cx_hi[..3].copy_from_slice(&same[..3]);
                        cy_hi[..3].copy_from_slice(&same[3..]);
                        fill(&mut cx_hi[3..], &mut cy_hi[3..], &riv[jsame + 1..]);
                    } else {
                        fill(coef_x, coef_y, riv);
                    }
                    //
                    // The coefficients directly yield derivatives
                    //
                    let w2 = 2. * weight[(ii - 1) as usize] as f64;
                    let xr = xresid[(ii - 1) as usize];
                    let yr = yresid[(ii - 1) as usize];
                    for ((sum, &cx), &cy) in grad_sums.iter_mut().zip(&*coef_x).zip(&*coef_y) {
                        *sum += w2 * (xr * cx + yr * cy) as f64;
                    }
                    //
                }
            }
        };
        let parts: Vec<FunctPart> = statics
            .coef_x
            .iter_mut()
            .zip(statics.coef_y.iter_mut())
            .zip(statics.grad_sums.iter_mut())
            .take(num_threads as usize)
            .enumerate()
            .map(|(it, ((cx, cy), gs))| (it as i32, cx, cy, gs))
            .collect();
        if num_threads > 1 {
            // The calling thread runs thread 0's partition itself, as the
            // OpenMP master does, and the shared region pool
            // (`metro::OMP_TEAM_POOL`) runs the others; `in_place_scope`
            // returns when all of them are done.
            let team = OMP_TEAM_POOL.get_or_init(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads((num_omp_threads(6) - 1).max(1) as usize)
                    .build()
                    .expect("funct: cannot start the thread pool")
            });
            let mut jobs = parts.into_iter();
            let first = jobs.next();
            team.in_place_scope(|s| {
                for part in jobs {
                    s.spawn(move |_| run_thread(part));
                }
                if let Some(part) = first {
                    run_thread(part);
                }
            });
        } else {
            parts.into_iter().for_each(run_thread);
        }

        // Add up the gradients from the threads, keeping the sum double until assignment
        for ivar in 0..nvmat_u {
            grad_sum = 0.;
            for ii in 0..statics.num_threads as usize {
                grad_sum += statics.grad_sums[ii][ivar];
            }
            grad[ivar + icoord_bas as usize] = grad_sum as f32;
        }

        //
        // write(*,'(i4,2f16.10)') (i, var(i), grad(i), i = 1, nvarSearch)
        *FUNCT_WALL_CUM.lock().unwrap_or_else(|e| e.into_inner()) += wall_time() - wall_start;
        FUNCT_NUM_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    /// Original: `EvalFunct::gradientSum` (`funct.cpp:841`) — forms the
    /// standard gradient sum over the points in a view for the given factors
    /// `afac` - `ffac`.  `indv_proj` has indices from point in view to residual
    /// products in `res_prod`; `npt_in_view` is the number of points in the
    /// view.  Each `float * float` product rounds to single before the double
    /// add.
    #[allow(clippy::too_many_arguments)]
    fn gradient_sum(
        &self,
        indv_proj: &[i32],
        npt_in_view: i32,
        res_prod: &[f32],
        afac: f32,
        bfac: f32,
        cfac: f32,
        dfac: f32,
        efac: f32,
        ffac: f32,
    ) -> f64 {
        let mut grad_sum: f64;
        grad_sum = 0.;
        // Each point's six products are sliced once (one bounds check instead
        // of six); the same elements in the same order.
        for &ipt in &indv_proj[..npt_in_view.max(0) as usize] {
            let ipt = ipt as usize;
            let rp = &res_prod[ipt * 6 - 6..ipt * 6];
            grad_sum = grad_sum
                + (afac * rp[0]) as f64
                + (bfac * rp[1]) as f64
                + (cfac * rp[2]) as f64
                + (dfac * rp[3]) as f64
                + (efac * rp[4]) as f64
                + (ffac * rp[5]) as f64;
        }
        grad_sum
    }

    /// Original: `EvalFunct::remap_params` (`funct.cpp:862`) — returns the
    /// complete set of geometric variables based on the current values of the
    /// search parameters.  `var_list` is written and restored when there is a
    /// dummy dmag variable.
    pub fn remap_params(&self, av: &mut AlignVariables, var_list: &mut [f32]) {
        let mut sum: f32;
        let mut var_save: f32 = 0.;
        //
        // 4 / 10 / 05: eliminated global rotation, it is just like the others now
        // so extra arguments to map_one_var were removed
        //
        self.map_one_var(
            var_list,
            &mut av.rot,
            &av.map_rot,
            &av.frc_rot,
            &av.lin_rot,
            av.fixed_rot,
            av.nview,
            &av.glb_rot,
            av.incr_rot,
        );
        //
        for ii in 1..=av.nview.max(0) as usize {
            if av.map_tilt[ii - 1] > 0 {
                let frc = av.frc_tilt[ii - 1];
                let v = var_list[(av.map_tilt[ii - 1] - 1) as usize];
                if av.lin_tilt[ii - 1] > 0 {
                    av.tilt[ii - 1] = ((frc * v) as f64
                        + (1. - frc as f64) * var_list[(av.lin_tilt[ii - 1] - 1) as usize] as f64
                        + av.tilt_inc[ii - 1] as f64) as f32;
                } else if av.lin_tilt[ii - 1] == -1 {
                    av.tilt[ii - 1] = ((frc * v) as f64
                        + (1. - frc as f64) * av.fixed_tilt as f64
                        + av.tilt_inc[ii - 1] as f64) as f32;
                } else if av.lin_tilt[ii - 1] == -2 {
                    av.tilt[ii - 1] = ((frc * v) as f64
                        + (1. - frc as f64) * av.fixed_tilt2 as f64
                        + av.tilt_inc[ii - 1] as f64) as f32;
                } else {
                    av.tilt[ii - 1] = v + av.tilt_inc[ii - 1];
                }
            }
        }
        //
        self.map_one_var(
            var_list,
            &mut av.gmag,
            &av.map_gmag,
            &av.frc_gmag,
            &av.lin_gmag,
            av.fixed_gmag,
            av.nview,
            &av.glb_gmag,
            av.incr_gmag,
        );
        //
        self.map_one_var(
            var_list,
            &mut av.comp,
            &av.map_comp,
            &av.frc_comp,
            &av.lin_comp,
            av.fixed_comp,
            av.nview,
            &av.glb_gmag,
            0,
        );
        //
        self.map_one_var(
            var_list,
            &mut av.skew,
            &av.map_skew,
            &av.frc_skew,
            &av.lin_skew,
            av.fixed_skew,
            av.nview,
            &av.glb_skew,
            av.incr_skew,
        );
        //
        if av.map_dum_dmag > av.map_dmag_start {
            //
            // if there are any dmag variables, the dummy variable is some factor
            // times the sum of the real variables.  Save that position on
            // varList, put the value there, and compose all of the view
            // parameters as usual
            //
            sum = 0.;
            for ii in av.map_dmag_start..=av.map_dum_dmag - 1 {
                sum = sum + var_list[(ii - 1) as usize];
            }
            var_save = var_list[(av.map_dum_dmag - 1) as usize];
            var_list[(av.map_dum_dmag - 1) as usize] = av.dum_dmag_fac * sum;
        }
        //
        self.map_one_var(
            var_list,
            &mut av.dmag,
            &av.map_dmag,
            &av.frc_dmag,
            &av.lin_dmag,
            av.fixed_dmag,
            av.nview,
            &av.glb_dmag,
            av.incr_dmag,
        );
        if av.map_dum_dmag > av.map_dmag_start {
            var_list[(av.map_dum_dmag - 1) as usize] = var_save;
        }
        //
        if av.if_any_alf != 0 {
            self.map_one_var(
                var_list,
                &mut av.alf,
                &av.map_alf,
                &av.frc_alf,
                &av.lin_alf,
                av.fixed_alf,
                av.nview,
                &av.glb_alf,
                av.incr_alf,
            );
        }
        //
        if av.map_proj_stretch > 0 {
            // projStretch = varList(mapProjStretch)
            av.proj_skew = var_list[(av.map_proj_stretch - 1) as usize];
        }
        if av.map_beam_tilt > 0 {
            av.beam_tilt = var_list[(av.map_beam_tilt - 1) as usize];
        }
    }

    /// Original: `EvalFunct::map_one_var` (`funct.cpp:934`).
    ///
    /// `glb` is only read when `incr != 0`, and `remap_params` passes
    /// `glbGmag` with `incr` 0 for `comp`, so an empty `glb` is fine then.
    #[allow(clippy::too_many_arguments)]
    fn map_one_var(
        &self,
        var_list: &[f32],
        val: &mut [f32],
        map: &[i32],
        frc: &[f32],
        lin: &[i32],
        fixed: f32,
        nview: i32,
        glb: &[f32],
        incr: i32,
    ) {
        for ii in 0..nview.max(0) as usize {
            let map_val = map[ii] - 1;
            if map_val >= 0 {
                if lin[ii] > 0 {
                    val[ii] = ((frc[ii] * var_list[map_val as usize]) as f64
                        + (1. - frc[ii] as f64) * var_list[(lin[ii] - 1) as usize] as f64)
                        as f32;
                } else if lin[ii] < 0 {
                    val[ii] = ((frc[ii] * var_list[map_val as usize]) as f64
                        + (1. - frc[ii] as f64) * fixed as f64)
                        as f32;
                } else {
                    val[ii] = var_list[map_val as usize];
                }
                if incr != 0 {
                    val[ii] = val[ii] + glb[ii];
                }
            }
        }
    }

    /// Original: `EvalFunct::matrix_to_coef` (`funct.cpp:962`) — takes the
    /// distortion matrix `dist`, X-axis tilt matrix `xtilt`, Y-axis tilt
    /// matrix `y_tilt`, projection stretch matrix `proj_str`, and rotation
    /// matrix `rot`, and computes the 6 components of the 2x3 product in `a`
    /// - `f`.
    ///
    /// The member uses no data member, so it is an associated function: the
    /// source's own calls pass member arrays in and member elements out, which
    /// a `&self` receiver could not borrow alongside.  `tiltalign.cpp:1490`
    /// calls it as `EvalFunct::matrix_to_coef(...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn matrix_to_coef(
        dist: &[f32],
        xtilt: &[f32],
        beam_inv: &[f32],
        y_tilt: &[f32],
        beam_mat: &[f32],
        proj_str: &[f32],
        rot: &[f32],
        a: &mut f32,
        b: &mut f32,
        c: &mut f32,
        d: &mut f32,
        e: &mut f32,
        f: &mut f32,
    ) {
        let mut tmp = [0f64; 9];

        for ii in 0..9 {
            tmp[ii] = dist[ii] as f64;
        }
        mat_product(&mut tmp, 3, 3, xtilt, 3, 3);
        mat_product(&mut tmp, 3, 3, beam_inv, 3, 3);
        mat_product(&mut tmp, 3, 3, y_tilt, 3, 3);
        mat_product(&mut tmp, 3, 3, beam_mat, 2, 3);
        mat_product(&mut tmp, 2, 3, rot, 2, 2);
        mat_product(&mut tmp, 2, 3, proj_str, 2, 2);
        *a = tmp[0] as f32;
        *b = tmp[1] as f32;
        *c = tmp[2] as f32;
        *d = tmp[3] as f32;
        *e = tmp[4] as f32;
        *f = tmp[5] as f32;
    }

    /// Original: `EvalFunct::solveLeftOutXYZs` (`funct.cpp:988`) — using the
    /// current solution, uses a simple linear relationship to the tabulated
    /// coefficients to solve for the best X/Y/Z for a contour left out.
    pub fn solve_left_out_xyzs(&mut self, av: &mut AlignVariables, test_set: bool) {
        let mut x_sd = [0f32; 4];
        let mut x_mean = [0f32; 4];
        let mut sol = [0f32; 3];
        let mut work = [0f32; 32];
        let mut num_data: i32;
        for ind_real in 0..av.nreal_pt.max(0) as usize {
            if (!test_set
                && (av.real_left_out[ind_real] == 0 || av.real_in_test_set[ind_real] != 0))
                || (test_set && av.real_in_test_set[ind_real] == 0)
            {
                continue;
            }
            num_data = 0;
            for ii in
                (av.ireal_str[ind_real] - 1) as usize..(av.ireal_str[ind_real + 1] - 1) as usize
            {
                let iv = (av.isec_view[ii] - 1) as usize;
                let nd = (4 * num_data) as usize;
                self.m_xyzmat[nd] = self.m_a[iv];
                self.m_xyzmat[nd + 1] = self.m_b[iv];
                self.m_xyzmat[nd + 2] = self.m_c[iv];
                self.m_xyzmat[nd + 3] = av.xx[ii] - av.dxy[iv * 2];
                num_data += 1;
                let nd = (4 * num_data) as usize;
                self.m_xyzmat[nd] = self.m_d[iv];
                self.m_xyzmat[nd + 1] = self.m_e[iv];
                self.m_xyzmat[nd + 2] = self.m_f[iv];
                self.m_xyzmat[nd + 3] = av.yy[ii] - av.dxy[iv * 2 + 1];
                num_data += 1;
            }

            if mult_regress(
                &self.m_xyzmat,
                4,
                1,
                3,
                num_data,
                1,
                0,
                &mut sol,
                3,
                None,
                &mut x_mean,
                &mut x_sd,
                &mut work,
            ) != 0
            {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "Error solving for X/Y/Z of point %d; using original estimate\n",
                    &[CArg::Int(ind_real as i64)],
                ));
            } else {
                // Replace XYZ and compute the residuals
                av.xyz[ind_real * 3] = sol[0];
                av.xyz[ind_real * 3 + 1] = sol[1];
                av.xyz[ind_real * 3 + 2] = sol[2];
                for ii in
                    (av.ireal_str[ind_real] - 1) as usize..(av.ireal_str[ind_real + 1] - 1) as usize
                {
                    let iv = (av.isec_view[ii] - 1) as usize;
                    av.xresid[ii] = av.xx[ii]
                        - (sol[0] * self.m_a[iv]
                            + sol[1] * self.m_b[iv]
                            + sol[2] * self.m_c[iv]
                            + av.dxy[iv * 2]);
                    av.yresid[ii] = av.yy[ii]
                        - (sol[0] * self.m_d[iv]
                            + sol[1] * self.m_e[iv]
                            + sol[2] * self.m_f[iv]
                            + av.dxy[iv * 2 + 1]);
                }
            }
        }
    }
}
