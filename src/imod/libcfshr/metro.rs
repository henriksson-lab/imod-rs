//! Translation of `IMOD/libcfshr/metro.c`.
//!
//! The Fortran wrapper `metro` (`metro.c:365`) is not translated: a translated
//! caller calls [`metro_search`] directly (see `DEAD_CODE.md`).
//!
//! # The three OpenMP regions
//!
//! `metroSearch` runs three `omp parallel for` loops over the rows of the
//! Hessian, with `numThreads = numOMPthreads(B3DNINT(n / 150.))` clamped to
//! 1..4 — so more than one thread only from `n >= 225`.
//!
//! - **`metro.c:300` (H0 update) and `metro.c:315` (H1 update)** write row
//!   `j` of the matrix from row-independent values; no sum crosses rows.  The
//!   output cannot depend on the thread count, and they are split into
//!   disjoint row groups (`chunks_mut`, one pool task per group), as `TO_OPT.md`'s
//!   parallelisation round prescribes.
//! - **`metro.c:270`** reduces `gammaHGamma` and `posdef` — and here the
//!   thread count **is** observable.  Each iteration adds into
//!   `thr*[BREAK_SUMS * threadNum + j % BREAK_SUMS]`, and those 8 x
//!   `numThreads` partial sums are then added in index order, so the grouping
//!   of the double-precision sum depends on which `j` each thread ran.  The
//!   source's comment says as much ("1 thread gave different sums from more
//!   than 1").  With no `schedule` clause libgomp uses its static schedule:
//!   thread `t` of `T` gets `q + (t < r)` consecutive iterations starting at
//!   `t * q + min(t, r)`, with `q = n / T`, `r = n % T`
//!   (`gomp_iter_static_next`).  That partition is reproduced exactly here, so
//!   the result equals native at the *same* `numThreads` — which
//!   [`num_omp_threads`] computes from `OMP_NUM_THREADS` exactly as native
//!   does — at every thread count, and serial at 1.  Each partition runs as
//!   one rayon task with its own 8 bins, and the bins are combined serially in
//!   the source's index order; the parallelism therefore changes nothing but
//!   the wall time.
//!
//! All three regions run on [`OMP_TEAM_POOL`], with the calling thread taking
//! the first share as the OpenMP master does.  The inner loops run over
//! length-`n` slices (`zip`), so they carry one bounds check per row instead
//! of several per element; the operations and their order are the source's.

use std::io::Write;
use std::sync::OnceLock;

use super::b3dutil::{CArg, ImodFile, c_format, num_omp_threads};

/// `MAX_THREADS` (`metro.c:31`).
const MAX_THREADS: usize = 8;
/// `BREAK_SUMS` (`metro.c:32`).
const BREAK_SUMS: usize = 8;

/// Rust-side worker pool for the OpenMP regions of `metroSearch` and of
/// tiltalign's `EvalFunct::funct` (`funct.cpp:713`), which native runs on
/// libgomp's thread team.  Not in the source.  It holds
/// `numOMPthreads(6) - 1` workers (6 is `funct`'s cap, 4 `metroSearch`'s);
/// the calling thread is the team's thread 0 and runs a share itself.  A
/// small pool shared by the two alternating regions keeps its workers warm:
/// the 20-worker global pool (or one private team per region) let the
/// workers fall asleep between regions, and the futex wake-ups then cost as
/// much as the parallel work saved.
pub static OMP_TEAM_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Original `copyArray` (`metro.c:9`): copies `fromArr(from1..)` into
/// `toArr(to1..to2)`, 1-based inclusive, as one `memcpy`.
fn copy_array(to_arr: &mut [f32], to1: i32, to2: i32, from_arr: &[f32], from1: i32) {
    let size = (to2 + 1 - to1) as usize;
    let out = (to1 - 1) as usize;
    let inp = (from1 - 1) as usize;
    to_arr[out..out + size].copy_from_slice(&from_arr[inp..inp + size]);
}

/// Original `metroSearch` (`metro.c:57`).
///
/// `funct` is the C's `MetroFunct` (`cfsemshare.h:338`), called as
/// `funct(n, X, f, G)`; the C passes `n` by pointer but no caller writes it
/// (`funct.cpp:149` copies it), so it is passed by value.  `h` is the scratch
/// array of at least `n * n + 3 * n` floats.
#[allow(clippy::too_many_arguments)]
pub fn metro_search(
    n: i32,
    x: &mut [f32],
    funct: &mut dyn FnMut(i32, &mut [f32], &mut f32, &mut [f32]),
    f: &mut f32,
    g: &mut [f32],
    step_initial: f32,
    epsilon: f32,
    limit_in: i32,
    ier: &mut i32,
    h: &mut [f32],
    num_iter: &mut i32,
    rms_scale: f32,
) {
    let mut cut_by_10: i32;
    let mut reinitialize: i32;
    let mut del_x_gamma: f64;
    let mut step_dbl: f64;
    let mut backoff_step: f64 = 0.;
    let mut f_old: f64;
    let mut f_new: f64;
    let mut dir_vec_dot_g_old: f64;
    let mut dir_vec_dot_g_new: f64;
    let mut z: f64;
    let mut w: f64;
    let mut delta_x: f64;
    let mut delta_g: f64;
    let mut del_x_length: f64;
    let mut gamma_h_gamma: f64;
    let mut del_x_dot_g: f64;
    let mu: f64;
    let mut posdef: f64;
    let mut step: f32;
    let mut g_dot_g: f32;
    let mut num_threads: i32;
    let mut thr_ghg = [0f64; MAX_THREADS * BREAK_SUMS];
    let mut thr_posdef = [0f64; MAX_THREADS * BREAK_SUMS];
    let mut sqrt_ghg: f64;
    let nu = n as usize;

    // Determine number of threads based on number of variables
    num_threads = ((n as f64 / 150.) + 0.5).floor() as i32;
    num_threads = num_omp_threads(num_threads);
    // `B3DCLAMP(numThreads, 1, 4)` is `numThreads = B3DMAX(1, B3DMIN(4, numThreads))`.
    let clamped_high = if 4 < num_threads { 4 } else { num_threads };
    num_threads = if 1 > clamped_high { 1 } else { clamped_high };

    mu = 1.0e-04;
    //
    // SCRATCH VECTOR STORAGE IN H:
    // H(1 -- n**2)            - COVARIANCE MATRIX
    // H(n**2 + 1 -- n**2 + n)     - OLD ARGUMENT VECTOR
    // H(n**2 + n + 1 -- n**2 + 2n)  - OLD GRADIENT VECTOR
    // H(n**2 + 2n + 1 -- n**2 + 3n) - DIRECTION VECTOR,
    // OVERWRITTEN w HG WHEN UPDATING MA
    //
    // 2/8/07: removed est, it was assigned to step just before factor is

    // Eval initial arg, generate identity matrix scaled by norm grad

    step = step_initial;
    let ib_arg = n * n;
    let ib_grad = n * n + n;
    let ib_dir_vec = n * n + 2 * n;
    let ib_h_gamma = ib_dir_vec;
    let (ib_arg_u, ib_grad_u, ib_dir_u) = (ib_arg as usize, ib_grad as usize, ib_dir_vec as usize);
    *ier = 0;
    *num_iter = 0;
    let iter_limit = if limit_in >= 0 { limit_in } else { -limit_in };
    funct(n, x, f, g);
    reinitialize = 1;
    //
    // Iteration loop
    while *num_iter <= iter_limit {
        if reinitialize != 0 {
            // print *,'Initializing'
            g_dot_g = dot_product(g, g, n) as f32;
            h[..nu * nu].fill(0.);
            for j in 1..=nu {
                h[(j - 1) * nu + j - 1] = (1. / ((g_dot_g / n as f32) as f64).sqrt()) as f32;
            }
        }
        *num_iter += 1;
        //
        if limit_in > 0 && (*num_iter % 10) == 0 {
            let mut out = ImodFile::Stdout;
            if rms_scale == 0. {
                let _ = out.write_all(
                    c_format(
                        "                                               Cycle %4d       %14.7f\n",
                        &[CArg::Int(*num_iter as i64), CArg::Dbl(*f as f64)],
                    )
                    .as_bytes(),
                );
            } else {
                let _ = out.write_all(
                    c_format(
                        "                                               Cycle %4d   %14.6f\n",
                        &[
                            CArg::Int(*num_iter as i64),
                            CArg::Dbl(((*f * rms_scale) as f64).sqrt()),
                        ],
                    )
                    .as_bytes(),
                );
            }
            let _ = out.flush();
        }
        //
        // Save old f, X, G
        copy_array(h, ib_arg + 1, ib_arg + n, x, 1);
        copy_array(h, ib_grad + 1, ib_grad + n, g, 1);
        //
        // Compute direction vector
        for j in 1..=nu {
            let value = -dot_product(&h[(j - 1) * nu..], g, n);
            h[ib_dir_u + j - 1] = value as f32;
        }
        //
        // Compute component of new gradient along search vector
        f_new = *f as f64;
        g_dot_g = dot_product(g, g, n) as f32;
        dir_vec_dot_g_new = dot_product(&h[ib_dir_u..], g, n);
        //printf("GdotG %19.10f\n", GdotG);
        //
        // CONVERGED: normal return (of limited value, you will never get exactly 0)
        if g_dot_g == 0. {
            return;
        }
        //
        // Can't reduce f along this line -- retry steepest
        reinitialize = if dir_vec_dot_g_new >= 0.0 { 1 } else { 0 };
        if reinitialize != 0 {
            continue;
        }
        //
        // ------------------------------   EXTRAPOLATE -------------------------------
        f_old = f_new;
        dir_vec_dot_g_old = dir_vec_dot_g_new;
        for j in 0..nu {
            x[j] += step * h[ib_dir_u + j];
        }
        funct(n, x, f, g);
        f_new = *f as f64;
        dir_vec_dot_g_new = dot_product(&h[ib_dir_u..], g, n);
        del_x_dot_g = 0.0;
        for j in 1..=nu {
            delta_x = (x[j - 1] - h[ib_arg_u + j - 1]) as f64;
            del_x_dot_g += delta_x * h[ib_grad_u + j - 1] as f64;
        }
        //printf("delXdotG %19.15f\n", delXdotG);
        //
        // Something's flaky: dg >= 0.  But it is OK if vector length is very small
        if del_x_dot_g >= 0.0 {
            del_x_length = delta_xlength(h, x, ib_arg, n);
            //printf("DG > 0, xlength %19.15f %19.15f\n", delXlength,epsilon);
            if del_x_length < epsilon as f64 / 5. {
                return;
            }
            *ier = 1;
            return;
        }

        if (f_new - f_old) / del_x_dot_g >= mu {
            //
            // Good improvement w/out linear search, use full shift next time
            step = step_initial;
            //printf("Restoring initial step %11.9f\n", step);
        } else {
            //
            // not enough improvement
            cut_by_10 = if dir_vec_dot_g_new < 0. && f_new < f_old {
                1
            } else {
                0
            };
            if cut_by_10 == 0 {
                //
                // try cubic interpolation if step was too large and pass initial test
                step_dbl = step as f64;
                z = 3.0 * (f_old - f_new) / step_dbl + dir_vec_dot_g_old + dir_vec_dot_g_new;
                w = (z * z - dir_vec_dot_g_old * dir_vec_dot_g_new).sqrt();
                backoff_step = step_dbl * (dir_vec_dot_g_new + w - z)
                    / (dir_vec_dot_g_new - dir_vec_dot_g_old + 2.0 * w);
                let abs_backoff = if backoff_step >= 0. {
                    backoff_step
                } else {
                    -backoff_step
                };
                if abs_backoff > step_dbl {
                    backoff_step *= 0.90 * step_dbl / abs_backoff;
                }
                //printf("backing off step by %11.9f\n", backoffStep);
                for j in 0..nu {
                    x[j] = (x[j] as f64 - backoff_step * h[ib_dir_u + j] as f64) as f32;
                }
                funct(n, x, f, g);
                cut_by_10 = if *f as f64 > f_old || *f as f64 > f_new {
                    1
                } else {
                    0
                }; //interpolation failed
            }
            //
            if cut_by_10 != 0 {
                //
                // try stepsize OF 0.1**n if either test fails, restore last position
                step = (0.1 * step as f64) as f32;
                if ((step / step_initial) as f64) < 1.0E-06 {
                    //
                    // DMN 11/27/13: if step is too small, just terminate if delta X is small enough
                    del_x_length = delta_xlength(h, x, ib_arg, n);
                    //printf("step too small, xlength %14.12f %14.12f\n", delXlength,epsilon);
                    if del_x_length < epsilon as f64 {
                        return;
                    }
                    *ier = 2; // Linear search lost
                    return;
                }
                //
                // DNM 11/27/13:
                // Restore the old f too so that tests and interpolation are correct in next round
                copy_array(x, 1, n, h, ib_arg + 1);
                copy_array(g, 1, n, h, ib_grad + 1);
                *f = f_old as f32;
                //printf("restored previous arg/grad; cut step by 10 to %12.9f\n", step);
                continue; //try it
            } else {
                //
                // Otherwise decrease step size next time.  This would make sense if the backoff
                // was a factor and not the step itself...
                let factor = if 1. - backoff_step > 1.0e-1 {
                    1. - backoff_step
                } else {
                    1.0e-1
                };
                step = (step as f64 * factor) as f32;
                //printf("Cut step by 1 - backoff to %11.9f\n", step);
            }
        }
        // -------------------------- End extrapolation -----------------------------
        //
        // Check convergence
        del_x_length = 0.;
        del_x_gamma = 0.0;
        del_x_dot_g = 0.0;
        for j in 1..=nu {
            delta_x = (x[j - 1] - h[ib_arg_u + j - 1]) as f64;
            delta_g = (g[j - 1] - h[ib_grad_u + j - 1]) as f64;
            del_x_dot_g += delta_x * h[ib_grad_u + j - 1] as f64;
            del_x_gamma += delta_x * delta_g;
            del_x_length += delta_x * delta_x;
        }
        //
        // Normal return
        if del_x_length.sqrt() <= epsilon as f64 {
            return;
        }
        //
        // ERROR: dg >= 0 even if length is big
        if del_x_dot_g >= 0.0 {
            //printf("DG > 0, xlength %14.8f %14.8f\n", sqrt(delXlength),epsilon);
            *ier = 1;
            return;
        }
        // numIter has reached limit
        if *num_iter >= iter_limit {
            *ier = 3;
            return;
        }

        // With parallelization, 1 thread gave different sums from more than 1, but
        // breaking up the sums took care of this
        for j in 0..BREAK_SUMS * num_threads as usize {
            thr_ghg[j] = 0.;
            thr_posdef[j] = 0.;
        }
        //
        // Update covariance matrix according to switching algorithm of Fletcher
        gamma_h_gamma = 0.0;
        posdef = 0.0;
        {
            // `metro.c:270`.  The matrix, the saved gradient and `HGamma` are
            // disjoint parts of `H`; the loop reads the first two and writes
            // the third, one element per `j`.
            let (mat, tail) = h.split_at_mut(ib_arg_u);
            let (arg_grad, h_gamma) = tail.split_at_mut(ib_dir_u - ib_arg_u);
            let grad = &arg_grad[nu..2 * nu];
            let g = &g[..nu];
            let mat = &*mat;
            // One partition of the libgomp static schedule: thread `thr`'s
            // `j` range (0-based here) and its own run of `HGamma`, returning
            // its `BREAK_SUMS` bins of each sum.
            let run_thread = |(thr, j0, h_gamma_part): (usize, usize, &mut [f32])| {
                let mut ghg = [0f64; BREAK_SUMS];
                let mut pos = [0f64; BREAK_SUMS];
                for (jj, h_gamma_j) in h_gamma_part.iter_mut().enumerate() {
                    let j = j0 + jj + 1;
                    let ind_thr = j % BREAK_SUMS;
                    let mut sig = 0.0f64;
                    let mut h_dot_gamma = 0.0f64;
                    let row = &mat[(j - 1) * nu..j * nu];
                    // `k = 1..=n` in order over three length-`n` slices.
                    for ((&hk, &gk), &grad_k) in row.iter().zip(g).zip(grad) {
                        let hdbl = hk as f64;
                        sig += hdbl * gk as f64;
                        h_dot_gamma += hdbl * (gk - grad_k) as f64;
                    }
                    *h_gamma_j = h_dot_gamma as f32;
                    ghg[ind_thr] += (g[j - 1] - grad[j - 1]) as f64 * h_dot_gamma;
                    pos[ind_thr] += g[j - 1] as f64 * sig;
                }
                (thr, ghg, pos)
            };
            let nthr = num_threads as usize;
            let q = nu / nthr;
            let r = nu % nthr;
            let mut parts: Vec<(usize, usize, &mut [f32])> = Vec::with_capacity(nthr);
            let mut remaining = &mut h_gamma[..nu];
            for thr in 0..nthr {
                let count = q + usize::from(thr < r);
                let start = thr * q + thr.min(r);
                let (part, rest) = remaining.split_at_mut(count);
                remaining = rest;
                parts.push((thr, start, part));
            }
            let mut results = [(0usize, [0f64; BREAK_SUMS], [0f64; BREAK_SUMS]); MAX_THREADS];
            if nthr > 1 {
                // The calling thread runs partition 0 itself, as the OpenMP
                // master does; the pool runs the others.
                let team = OMP_TEAM_POOL.get_or_init(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads((num_omp_threads(6) - 1).max(1) as usize)
                        .build()
                        .expect("metroSearch: cannot start the thread pool")
                });
                let mut jobs = parts.into_iter().zip(results.iter_mut());
                let first = jobs.next();
                team.in_place_scope(|s| {
                    for (part, slot) in jobs {
                        s.spawn(move |_| *slot = run_thread(part));
                    }
                    if let Some((part, slot)) = first {
                        *slot = run_thread(part);
                    }
                });
            } else {
                for (part, slot) in parts.into_iter().zip(results.iter_mut()) {
                    *slot = run_thread(part);
                }
            }
            for &(thr, ghg, pos) in &results[..nthr] {
                for ind in 0..BREAK_SUMS {
                    thr_ghg[BREAK_SUMS * thr + ind] += ghg[ind];
                    thr_posdef[BREAK_SUMS * thr + ind] += pos[ind];
                }
            }
        }

        for ind_thr in 0..BREAK_SUMS * num_threads as usize {
            gamma_h_gamma += thr_ghg[ind_thr];
            posdef += thr_posdef[ind_thr];
        }

        //
        // ERROR: matrix non-positive definite
        if posdef < 0. || gamma_h_gamma < 0. {
            *ier = 4;
            return;
        }
        //
        // H0 algorithm (`metro.c:300`): each row of the matrix is written from
        // `X`, the saved argument and `HGamma` alone.
        {
            let (mat, tail) = h.split_at_mut(ib_arg_u);
            let arg = &tail[..nu];
            let h_gamma = &tail[(ib_h_gamma - ib_arg) as usize..][..nu];
            let x = &x[..nu];
            let run_rows = |(group, rows): (usize, &mut [f32])| {
                for (jj, row) in rows.chunks_mut(nu).enumerate() {
                    let j = group + jj + 1;
                    // `X[j-1] - H[ibArg+j-1]` and `H[ibHGamma+j-1]` do not
                    // depend on `k`; `k = 1..=n` runs in order over slices of
                    // length `n`, so there is no bounds check per element.
                    let dxj = x[j - 1] - arg[j - 1];
                    let hgj = h_gamma[j - 1];
                    for (((hk, &xk), &ak), &hgk) in row.iter_mut().zip(x).zip(arg).zip(h_gamma) {
                        let del_x_del_x = (dxj * (xk - ak)) as f64;
                        let hggh = (hgj * hgk) as f64;
                        *hk = (*hk as f64 + (del_x_del_x / del_x_gamma - hggh / gamma_h_gamma))
                            as f32;
                    }
                }
            };
            // Consecutive row groups, one per thread; the rows are
            // independent, so the grouping cannot change a value.
            let rows_per_group = nu.div_ceil(num_threads as usize).max(1);
            if num_threads > 1 && nu > 0 {
                // The calling thread takes the first row group itself.
                let team = OMP_TEAM_POOL.get_or_init(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads((num_omp_threads(6) - 1).max(1) as usize)
                        .build()
                        .expect("metroSearch: cannot start the thread pool")
                });
                let mut groups = mat[..nu * nu].chunks_mut(rows_per_group * nu).enumerate();
                let first = groups.next();
                team.in_place_scope(|s| {
                    for (grp, rows) in groups {
                        s.spawn(move |_| run_rows((grp * rows_per_group, rows)));
                    }
                    if let Some((grp, rows)) = first {
                        run_rows((grp * rows_per_group, rows));
                    }
                });
            } else if nu > 0 {
                run_rows((0, &mut mat[..nu * nu]));
            }
        }
        //
        // H1 algorithm
        if del_x_gamma < gamma_h_gamma {
            continue;
        }
        sqrt_ghg = gamma_h_gamma.sqrt();
        {
            // `metro.c:315`: the same row independence as H0.
            let (mat, tail) = h.split_at_mut(ib_arg_u);
            let arg = &tail[..nu];
            let h_gamma = &tail[(ib_h_gamma - ib_arg) as usize..][..nu];
            let x = &x[..nu];
            let run_rows = |(group, rows): (usize, &mut [f32])| {
                for (jj, row) in rows.chunks_mut(nu).enumerate() {
                    let j = group + jj + 1;
                    let nuj = sqrt_ghg
                        * ((x[j - 1] - arg[j - 1]) as f64 / del_x_gamma
                            - h_gamma[j - 1] as f64 / gamma_h_gamma);
                    for (((hk, &xk), &ak), &hgk) in row.iter_mut().zip(x).zip(arg).zip(h_gamma) {
                        let nuk = sqrt_ghg
                            * ((xk - ak) as f64 / del_x_gamma - hgk as f64 / gamma_h_gamma);
                        *hk = (*hk as f64 + nuj * nuk) as f32;
                    }
                }
            };
            // Consecutive row groups, one per thread; the rows are
            // independent, so the grouping cannot change a value.
            let rows_per_group = nu.div_ceil(num_threads as usize).max(1);
            if num_threads > 1 && nu > 0 {
                // The calling thread takes the first row group itself.
                let team = OMP_TEAM_POOL.get_or_init(|| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads((num_omp_threads(6) - 1).max(1) as usize)
                        .build()
                        .expect("metroSearch: cannot start the thread pool")
                });
                let mut groups = mat[..nu * nu].chunks_mut(rows_per_group * nu).enumerate();
                let first = groups.next();
                team.in_place_scope(|s| {
                    for (grp, rows) in groups {
                        s.spawn(move |_| run_rows((grp * rows_per_group, rows)));
                    }
                    if let Some((grp, rows)) = first {
                        run_rows((grp * rows_per_group, rows));
                    }
                });
            } else if nu > 0 {
                run_rows((0, &mut mat[..nu * nu]));
            }
        }
    } // Continue with next direction
    *ier = 3; // numIter has reached limit after a cycle statement
}

// DNM 11/27/13: Compute the argument vector change in full precision
/// Original `deltaXlength` (`metro.c:331`).
fn delta_xlength(h: &[f32], x: &[f32], ib_arg: i32, n: i32) -> f64 {
    let mut delta_x: f64;
    let mut length: f64 = 0.;
    // `j = 1..=n` in order over two length-`n` slices.
    let nu = n.max(0) as usize;
    let ib = ib_arg as usize;
    for (&xj, &hj) in x[..nu].iter().zip(&h[ib..ib + nu]) {
        delta_x = (xj - hj) as f64;
        length += delta_x * delta_x;
    }
    length.sqrt()
}

// **     DOT PRODUCT
/// Original `dotProduct` (`metro.c:346`).
fn dot_product(a: &[f32], b: &[f32], n: i32) -> f64 {
    let mut ad: f64;
    let mut bd: f64;
    let mut product: f64 = 0.0;
    //

    // `j = 1..=n` in order over two length-`n` slices: one bounds check per
    // call instead of two per element.
    let nu = n.max(0) as usize;
    for (&aj, &bj) in a[..nu].iter().zip(&b[..nu]) {
        ad = aj as f64;
        bd = bj as f64;
        product += ad * bd;
    }

    // DNM 11/27/13: get rid of assignment to float then back
    product
}
