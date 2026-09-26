//! Translation of `IMOD/libfft/mdftkd.c` -- the multi-dimensional complex
//! Fourier transform kernel driver and its radix kernels.
//!
//! The C source hands each kernel several `float *` that all point into the
//! **same** caller array at different biases (`&x[stepAlong]`,
//! `&x[2 * stepAlong]`, ...), and `cmplft` itself is always called with
//! `y = &x[1]`, so the real and imaginary walkers alias too.  There is no safe
//! Rust spelling for several `&mut [f32]` over one buffer, so the whole array
//! is one `&mut [f32]` and every walker is a `usize` base index into it: the
//! bias the C applies with `&x[i * stepAlong]` is applied at the same place,
//! in `mdftkd`, as `x + i * step_along`.  A kernel's `x0[k]` is therefore
//! `data[x0 + k as usize]`, with the source's index arithmetic unchanged.
//!
//! **Unchecked access in the radix kernels** (`r2cftk`..`r8cftk`, and
//! `rpcftk` with the offsets `m * stepAlong` added to the bound).  With a
//! bounds check on each of the 8-32 loads and stores, the innermost loop ran
//! ~1.5x the C's time.  The kernels therefore index through `dp`, the raw
//! pointer of `data`, and prove the bound once per innermost loop instead of
//! per access.  Soundness: every access in that loop is `base + k` with
//! `base` one of the kernel's walkers (all `<= top`), and `k` runs from
//! `l - 1` upward in steps of `sep_between` while `k < k1`, so it is at
//! least `l - 1` and at most the last point of that progression below
//! `k1`, `kLast = (l - 1) + (k1 - l) / sep_between * sep_between`.  The
//! `assert!` ahead of a non-empty loop checks `l - 1 >= 0`,
//! `sep_between > 0` and `top + kLast < len` (with a checked add), so every
//! `base + k` lies in `0..len`.  An input
//! the checked indexing would have rejected mid-loop now panics before the
//! loop instead; any input that does not panic computes exactly the same
//! operations in the same order, so no output byte can change.  Raw-pointer
//! access leaves LLVM assuming the walkers alias, which is also what the C
//! compiler must assume of the source's `float *` walkers.

use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use std::io::Write as _;
/// C `mdftkd`.
pub fn mdftkd(n: i32, factor: &[i32], dim: &[i32], data: &mut [f32], x: usize, y: usize) {
    let mut ind_fac: i32;
    let mut n_reduced: i32;
    let mut p_fac: i32;
    let mut step_along: i32;
    let sep_along: i32;

    sep_along = dim[2];
    ind_fac = 0;
    n_reduced = n;
    while factor[(ind_fac + 1) as usize] != 0 {
        ind_fac = ind_fac + 1;
        p_fac = factor[ind_fac as usize];
        n_reduced = n_reduced / p_fac;
        step_along = n_reduced * sep_along;
        match p_fac {
            1 => {}

            2 => {
                r2cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    dim,
                );
            }

            3 => {
                r3cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    dim,
                );
            }

            4 => {
                r4cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    x + (3 * step_along) as usize,
                    y + (3 * step_along) as usize,
                    dim,
                );
            }

            5 => {
                r5cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    x + (3 * step_along) as usize,
                    y + (3 * step_along) as usize,
                    x + (4 * step_along) as usize,
                    y + (4 * step_along) as usize,
                    dim,
                );
            }

            8 => {
                r8cftk(
                    n,
                    n_reduced,
                    data,
                    x,
                    y,
                    x + step_along as usize,
                    y + step_along as usize,
                    x + (2 * step_along) as usize,
                    y + (2 * step_along) as usize,
                    x + (3 * step_along) as usize,
                    y + (3 * step_along) as usize,
                    x + (4 * step_along) as usize,
                    y + (4 * step_along) as usize,
                    x + (5 * step_along) as usize,
                    y + (5 * step_along) as usize,
                    x + (6 * step_along) as usize,
                    y + (6 * step_along) as usize,
                    x + (7 * step_along) as usize,
                    y + (7 * step_along) as usize,
                    dim,
                );
            }

            6 => {
                // `mdftkd.c:57`, `printf("\ntransfer error detected in
                // mdftkd\n\n")`.  `srfp` never emits a factor of 6 -- it
                // splits every composite into primes and regroups only powers
                // of two -- so this arm is unreachable, and the stream it
                // would print on is not exercised by any command.
                // `mdftkd.c:57`.  On `ImodFile::Stdout`, the **C** stream,
                // not Rust's: `cmplft` and `srfp` write to the C stream, and C
                // stdio is block-buffered under redirection while Rust's is
                // not, so mixing the two reorders a captured file.  The arm
                // is unreachable in practice — `srfp` splits composites into
                // primes and regroups only powers of two, so it never emits a
                // factor of 6 — but the stream is part of the behaviour.
                {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "\ntransfer error detected in mdftkd\n\n",
                        &[],
                    ));
                }
                return;
            }

            // `mdftkd.c:60-61`, `case 7:` falls through to `default:`.
            _ => {
                rpcftk(n, n_reduced, p_fac, step_along, data, x, y, dim);
            }
        }
    }
}

/// C `r2cftk`.
///
/// radix 2 multi-dimensional complex fourier transform kernel
pub fn r2cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    dim: &[i32],
) {
    // Unchecked-access bound: see the module comment.
    let len = data.len();
    let top = x0.max(y0).max(x1).max(y1);
    let dp = data.as_mut_ptr();

    let n_red2: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red2: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let mut c: f32 = 0.0;
    let mut sep_along: f32 = 0.0;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red2: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red2 = n_reduced * 2;
    fn_red2 = n_red2 as f32;
    n_red_ov2p1 = n_reduced / 2 + 1;
    sep_n_red2 = sep * n_red2;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red2) as f64;
        let zero = angle == 0.0;
        if !zero {
            c = angle.cos() as f32;
            sep_along = angle.sin() as f32;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                // Whether every per-`l` assertion below would pass, so it may
                // be skipped (TO_OPT.md, "combinefft single-thread"): a
                // `false` keeps it.  Exact: `(k1 - 1 - k)` is `extent_between`
                // for every `l`, so the loop runs iff `extent_between >= 0`
                // and its last index `l - 1 + extent_between / sep_between *
                // sep_between` increases with `l`; with `lim_along > 0` the
                // `l` run is `kk ..= kk + (tot_floats - kk) / lim_along *
                // lim_along`, so the assertions all hold iff `kk - 1 >= 0`
                // and the bound holds at the last `l`.  In `i64`/`u128`, so
                // nothing here overflows.
                let kk_ok = if kk > tot_floats {
                    true
                } else if lim_along <= 0 || sep_between <= 0 || extent_between < 0 {
                    false
                } else {
                    let (kkl, totl, laln, ebl, sbl) = (
                        kk as i64,
                        tot_floats as i64,
                        lim_along as i64,
                        extent_between as i64,
                        sep_between as i64,
                    );
                    let last_l = kkl + (totl - kkl) / laln * laln;
                    let last = last_l - 1 + ebl / sbl * sbl;
                    kkl - 1 >= 0
                        && last <= i32::MAX as i64
                        && (top as u128 + last as u128) < len as u128
                };
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    if !kk_ok && k < k1 {
                        assert!(
                            k >= 0
                                && sep_between > 0
                                && top
                                    .checked_add(
                                        (k + (k1 - 1 - k) / sep_between * sep_between) as usize
                                    )
                                    .is_some_and(|last| last < len)
                        );
                    }
                    while k < k1 {
                        unsafe {
                            let rs = *dp.add(x0 + k as usize) + *dp.add(x1 + k as usize);
                            let is = *dp.add(y0 + k as usize) + *dp.add(y1 + k as usize);
                            let ru = *dp.add(x0 + k as usize) - *dp.add(x1 + k as usize);
                            let iu = *dp.add(y0 + k as usize) - *dp.add(y1 + k as usize);
                            *dp.add(x0 + k as usize) = rs;
                            *dp.add(y0 + k as usize) = is;
                            if !zero {
                                *dp.add(x1 + k as usize) = ru * c + iu * sep_along;
                                *dp.add(y1 + k as usize) = iu * c - ru * sep_along;
                            } else {
                                *dp.add(x1 + k as usize) = ru;
                                *dp.add(y1 + k as usize) = iu;
                            }
                            k += sep_between;
                        }
                    }
                    l += lim_along;
                }
                kk += sep_n_red2;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            c = -c;
        }
    }
}

/// C `r3cftk`.
///
/// radix 3 multi-dimensional complex fourier transform kernel
pub fn r3cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    dim: &[i32],
) {
    // Unchecked-access bound: see the module comment.
    let len = data.len();
    let top = x0.max(y0).max(x1).max(y1).max(x2).max(y2);
    let dp = data.as_mut_ptr();

    let n_red3: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red3: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let a: f32 = -0.5;
    let b: f32 = 0.86602540;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red3: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red3 = n_reduced * 3;
    fn_red3 = n_red3 as f32;
    sep_n_red3 = sep * n_red3;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red3) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                // Whether every per-`l` assertion below would pass, so it may
                // be skipped (TO_OPT.md, "combinefft single-thread"): a
                // `false` keeps it.  Exact: `(k1 - 1 - k)` is `extent_between`
                // for every `l`, so the loop runs iff `extent_between >= 0`
                // and its last index `l - 1 + extent_between / sep_between *
                // sep_between` increases with `l`; with `lim_along > 0` the
                // `l` run is `kk ..= kk + (tot_floats - kk) / lim_along *
                // lim_along`, so the assertions all hold iff `kk - 1 >= 0`
                // and the bound holds at the last `l`.  In `i64`/`u128`, so
                // nothing here overflows.
                let kk_ok = if kk > tot_floats {
                    true
                } else if lim_along <= 0 || sep_between <= 0 || extent_between < 0 {
                    false
                } else {
                    let (kkl, totl, laln, ebl, sbl) = (
                        kk as i64,
                        tot_floats as i64,
                        lim_along as i64,
                        extent_between as i64,
                        sep_between as i64,
                    );
                    let last_l = kkl + (totl - kkl) / laln * laln;
                    let last = last_l - 1 + ebl / sbl * sbl;
                    kkl - 1 >= 0
                        && last <= i32::MAX as i64
                        && (top as u128 + last as u128) < len as u128
                };
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    if !kk_ok && k < k1 {
                        assert!(
                            k >= 0
                                && sep_between > 0
                                && top
                                    .checked_add(
                                        (k + (k1 - 1 - k) / sep_between * sep_between) as usize
                                    )
                                    .is_some_and(|last| last < len)
                        );
                    }
                    while k < k1 {
                        unsafe {
                            let r0 = *dp.add(x0 + k as usize);
                            let i0 = *dp.add(y0 + k as usize);
                            let rs = *dp.add(x1 + k as usize) + *dp.add(x2 + k as usize);
                            let is = *dp.add(y1 + k as usize) + *dp.add(y2 + k as usize);
                            *dp.add(x0 + k as usize) = r0 + rs;
                            *dp.add(y0 + k as usize) = i0 + is;
                            let ra = r0 + rs * a;
                            let ia = i0 + is * a;
                            let rb = (*dp.add(x1 + k as usize) - *dp.add(x2 + k as usize)) * b;
                            let ib = (*dp.add(y1 + k as usize) - *dp.add(y2 + k as usize)) * b;
                            if !zero {
                                let r1 = ra + ib;
                                let i1 = ia - rb;
                                let r2 = ra - ib;
                                let i2 = ia + rb;
                                *dp.add(x1 + k as usize) = r1 * c1 + i1 * s1;
                                *dp.add(y1 + k as usize) = i1 * c1 - r1 * s1;
                                *dp.add(x2 + k as usize) = r2 * c2 + i2 * s2;
                                *dp.add(y2 + k as usize) = i2 * c2 - r2 * s2;
                            } else {
                                *dp.add(x1 + k as usize) = ra + ib;
                                *dp.add(y1 + k as usize) = ia - rb;
                                *dp.add(x2 + k as usize) = ra - ib;
                                *dp.add(y2 + k as usize) = ia + rb;
                            }
                            k += sep_between;
                        }
                    }
                    l += lim_along;
                }
                kk += sep_n_red3;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            t = c1 * a + s1 * b;
            s1 = c1 * b - s1 * a;
            c1 = t;
            t = c2 * a - s2 * b;
            s2 = -c2 * b - s2 * a;
            c2 = t;
        }
    }
}

/// C `r4cftk`.
///
/// radix 4 multi-dimensional complex fourier transform kernel
pub fn r4cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    x3: usize,
    y3: usize,
    dim: &[i32],
) {
    // Unchecked-access bound: see the module comment.
    let len = data.len();
    let top = x0.max(y0).max(x1).max(y1).max(x2).max(y2).max(x3).max(y3);
    let dp = data.as_mut_ptr();

    let n_red4: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red4: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red4: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red4 = n_reduced * 4;
    fn_red4 = n_red4 as f32;
    sep_n_red4 = sep * n_red4;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red4) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                // Whether every per-`l` assertion below would pass, so it may
                // be skipped (TO_OPT.md, "combinefft single-thread"): a
                // `false` keeps it.  Exact: `(k1 - 1 - k)` is `extent_between`
                // for every `l`, so the loop runs iff `extent_between >= 0`
                // and its last index `l - 1 + extent_between / sep_between *
                // sep_between` increases with `l`; with `lim_along > 0` the
                // `l` run is `kk ..= kk + (tot_floats - kk) / lim_along *
                // lim_along`, so the assertions all hold iff `kk - 1 >= 0`
                // and the bound holds at the last `l`.  In `i64`/`u128`, so
                // nothing here overflows.
                let kk_ok = if kk > tot_floats {
                    true
                } else if lim_along <= 0 || sep_between <= 0 || extent_between < 0 {
                    false
                } else {
                    let (kkl, totl, laln, ebl, sbl) = (
                        kk as i64,
                        tot_floats as i64,
                        lim_along as i64,
                        extent_between as i64,
                        sep_between as i64,
                    );
                    let last_l = kkl + (totl - kkl) / laln * laln;
                    let last = last_l - 1 + ebl / sbl * sbl;
                    kkl - 1 >= 0
                        && last <= i32::MAX as i64
                        && (top as u128 + last as u128) < len as u128
                };
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    if !kk_ok && k < k1 {
                        assert!(
                            k >= 0
                                && sep_between > 0
                                && top
                                    .checked_add(
                                        (k + (k1 - 1 - k) / sep_between * sep_between) as usize
                                    )
                                    .is_some_and(|last| last < len)
                        );
                    }
                    while k < k1 {
                        unsafe {
                            let rs0 = *dp.add(x0 + k as usize) + *dp.add(x2 + k as usize);
                            let is0 = *dp.add(y0 + k as usize) + *dp.add(y2 + k as usize);
                            let ru0 = *dp.add(x0 + k as usize) - *dp.add(x2 + k as usize);
                            let iu0 = *dp.add(y0 + k as usize) - *dp.add(y2 + k as usize);
                            let rs1 = *dp.add(x1 + k as usize) + *dp.add(x3 + k as usize);
                            let is1 = *dp.add(y1 + k as usize) + *dp.add(y3 + k as usize);
                            let ru1 = *dp.add(x1 + k as usize) - *dp.add(x3 + k as usize);
                            let iu1 = *dp.add(y1 + k as usize) - *dp.add(y3 + k as usize);
                            *dp.add(x0 + k as usize) = rs0 + rs1;
                            *dp.add(y0 + k as usize) = is0 + is1;
                            if !zero {
                                let r1 = ru0 + iu1;
                                let i1 = iu0 - ru1;
                                let r2 = rs0 - rs1;
                                let i2 = is0 - is1;
                                let r3 = ru0 - iu1;
                                let i3 = iu0 + ru1;
                                *dp.add(x2 + k as usize) = r1 * c1 + i1 * s1;
                                *dp.add(y2 + k as usize) = i1 * c1 - r1 * s1;
                                *dp.add(x1 + k as usize) = r2 * c2 + i2 * s2;
                                *dp.add(y1 + k as usize) = i2 * c2 - r2 * s2;
                                *dp.add(x3 + k as usize) = r3 * c3 + i3 * s3;
                                *dp.add(y3 + k as usize) = i3 * c3 - r3 * s3;
                            } else {
                                *dp.add(x2 + k as usize) = ru0 + iu1;
                                *dp.add(y2 + k as usize) = iu0 - ru1;
                                *dp.add(x1 + k as usize) = rs0 - rs1;
                                *dp.add(y1 + k as usize) = is0 - is1;
                                *dp.add(x3 + k as usize) = ru0 - iu1;
                                *dp.add(y3 + k as usize) = iu0 + ru1;
                            }
                            k += sep_between;
                        }
                    }
                    l += lim_along;
                }
                kk += sep_n_red4;
            }

            k0 = (n_reduced + 1 - j) * sep + 1;
            t = c1;
            c1 = s1;
            s1 = t;
            c2 = -c2;
            t = c3;
            c3 = -s3;
            s3 = -t;
        }
    }
}

/// C `r5cftk`.
///
/// radix 5 multi-dimensional complex fourier transform kernel
pub fn r5cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    x3: usize,
    y3: usize,
    x4: usize,
    y4: usize,
    dim: &[i32],
) {
    // Unchecked-access bound: see the module comment.
    let len = data.len();
    let top = x0
        .max(y0)
        .max(x1)
        .max(y1)
        .max(x2)
        .max(y2)
        .max(x3)
        .max(y3)
        .max(x4)
        .max(y4);
    let dp = data.as_mut_ptr();

    let n_red5: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red5: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let a1: f32 = 0.30901699;
    let a2: f32 = -0.80901699;
    let b1: f32 = 0.95105652;
    let b2: f32 = 0.58778525;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut c4: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut s4: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red5: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red5 = n_reduced * 5;
    fn_red5 = n_red5 as f32;
    sep_n_red5 = sep * n_red5;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red5) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
            c4 = c2 * c2 - s2 * s2;
            s4 = s2 * c2 + c2 * s2;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                // Whether every per-`l` assertion below would pass, so it may
                // be skipped (TO_OPT.md, "combinefft single-thread"): a
                // `false` keeps it.  Exact: `(k1 - 1 - k)` is `extent_between`
                // for every `l`, so the loop runs iff `extent_between >= 0`
                // and its last index `l - 1 + extent_between / sep_between *
                // sep_between` increases with `l`; with `lim_along > 0` the
                // `l` run is `kk ..= kk + (tot_floats - kk) / lim_along *
                // lim_along`, so the assertions all hold iff `kk - 1 >= 0`
                // and the bound holds at the last `l`.  In `i64`/`u128`, so
                // nothing here overflows.
                let kk_ok = if kk > tot_floats {
                    true
                } else if lim_along <= 0 || sep_between <= 0 || extent_between < 0 {
                    false
                } else {
                    let (kkl, totl, laln, ebl, sbl) = (
                        kk as i64,
                        tot_floats as i64,
                        lim_along as i64,
                        extent_between as i64,
                        sep_between as i64,
                    );
                    let last_l = kkl + (totl - kkl) / laln * laln;
                    let last = last_l - 1 + ebl / sbl * sbl;
                    kkl - 1 >= 0
                        && last <= i32::MAX as i64
                        && (top as u128 + last as u128) < len as u128
                };
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    if !kk_ok && k < k1 {
                        assert!(
                            k >= 0
                                && sep_between > 0
                                && top
                                    .checked_add(
                                        (k + (k1 - 1 - k) / sep_between * sep_between) as usize
                                    )
                                    .is_some_and(|last| last < len)
                        );
                    }
                    while k < k1 {
                        unsafe {
                            let r0 = *dp.add(x0 + k as usize);
                            let i0 = *dp.add(y0 + k as usize);
                            let rs1 = *dp.add(x1 + k as usize) + *dp.add(x4 + k as usize);
                            let is1 = *dp.add(y1 + k as usize) + *dp.add(y4 + k as usize);
                            let ru1 = *dp.add(x1 + k as usize) - *dp.add(x4 + k as usize);
                            let iu1 = *dp.add(y1 + k as usize) - *dp.add(y4 + k as usize);
                            let rs2 = *dp.add(x2 + k as usize) + *dp.add(x3 + k as usize);
                            let is2 = *dp.add(y2 + k as usize) + *dp.add(y3 + k as usize);
                            let ru2 = *dp.add(x2 + k as usize) - *dp.add(x3 + k as usize);
                            let iu2 = *dp.add(y2 + k as usize) - *dp.add(y3 + k as usize);
                            *dp.add(x0 + k as usize) = r0 + rs1 + rs2;
                            *dp.add(y0 + k as usize) = i0 + is1 + is2;
                            let ra1 = r0 + rs1 * a1 + rs2 * a2;
                            let ia1 = i0 + is1 * a1 + is2 * a2;
                            let ra2 = r0 + rs1 * a2 + rs2 * a1;
                            let ia2 = i0 + is1 * a2 + is2 * a1;
                            let rb1 = ru1 * b1 + ru2 * b2;
                            let ib1 = iu1 * b1 + iu2 * b2;
                            let rb2 = ru1 * b2 - ru2 * b1;
                            let ib2 = iu1 * b2 - iu2 * b1;
                            if !zero {
                                let r1 = ra1 + ib1;
                                let i1 = ia1 - rb1;
                                let r2 = ra2 + ib2;
                                let i2 = ia2 - rb2;
                                let r3 = ra2 - ib2;
                                let i3 = ia2 + rb2;
                                let r4 = ra1 - ib1;
                                let i4 = ia1 + rb1;
                                *dp.add(x1 + k as usize) = r1 * c1 + i1 * s1;
                                *dp.add(y1 + k as usize) = i1 * c1 - r1 * s1;
                                *dp.add(x2 + k as usize) = r2 * c2 + i2 * s2;
                                *dp.add(y2 + k as usize) = i2 * c2 - r2 * s2;
                                *dp.add(x3 + k as usize) = r3 * c3 + i3 * s3;
                                *dp.add(y3 + k as usize) = i3 * c3 - r3 * s3;
                                *dp.add(x4 + k as usize) = r4 * c4 + i4 * s4;
                                *dp.add(y4 + k as usize) = i4 * c4 - r4 * s4;
                            } else {
                                *dp.add(x1 + k as usize) = ra1 + ib1;
                                *dp.add(y1 + k as usize) = ia1 - rb1;
                                *dp.add(x2 + k as usize) = ra2 + ib2;
                                *dp.add(y2 + k as usize) = ia2 - rb2;
                                *dp.add(x3 + k as usize) = ra2 - ib2;
                                *dp.add(y3 + k as usize) = ia2 + rb2;
                                *dp.add(x4 + k as usize) = ra1 - ib1;
                                *dp.add(y4 + k as usize) = ia1 + rb1;
                            }
                            k += sep_between;
                        }
                    }
                    l += lim_along;
                }
                kk += sep_n_red5;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            t = c1 * a1 + s1 * b1;
            s1 = c1 * b1 - s1 * a1;
            c1 = t;
            t = c2 * a2 + s2 * b2;
            s2 = c2 * b2 - s2 * a2;
            c2 = t;
            t = c3 * a2 - s3 * b2;
            s3 = -c3 * b2 - s3 * a2;
            c3 = t;
            t = c4 * a1 - s4 * b1;
            s4 = -c4 * b1 - s4 * a1;
            c4 = t;
        }
    }
}

/// C `r8cftk`.
///
/// radix 8 multi-dimensional complex fourier transform kernel
pub fn r8cftk(
    n: i32,
    n_reduced: i32,
    data: &mut [f32],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    x3: usize,
    y3: usize,
    x4: usize,
    y4: usize,
    x5: usize,
    y5: usize,
    x6: usize,
    y6: usize,
    x7: usize,
    y7: usize,
    dim: &[i32],
) {
    // Unchecked-access bound: see the module comment.
    let len = data.len();
    let top = x0
        .max(y0)
        .max(x1)
        .max(y1)
        .max(x2)
        .max(y2)
        .max(x3)
        .max(y3)
        .max(x4)
        .max(y4)
        .max(x5)
        .max(y5)
        .max(x6)
        .max(y6)
        .max(x7)
        .max(y7);
    let dp = data.as_mut_ptr();

    let n_red8: i32;
    let n_red_ov2p1: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_red8: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;
    let mut c1: f32 = 0.0;
    let mut c2: f32 = 0.0;
    let mut c3: f32 = 0.0;
    let mut c4: f32 = 0.0;
    let mut c5: f32 = 0.0;
    let mut c6: f32 = 0.0;
    let mut c7: f32 = 0.0;
    let e: f32 = 0.70710678;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut s4: f32 = 0.0;
    let mut s5: f32 = 0.0;
    let mut s6: f32 = 0.0;
    let mut s7: f32 = 0.0;
    let mut t: f32;
    let twopi: f32 = 6.2831853;
    let mut fjm1: f32;
    let fn_red8: f32;

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red8 = n_reduced * 8;
    fn_red8 = n_red8 as f32;
    sep_n_red8 = sep * n_red8;
    n_red_ov2p1 = n_reduced / 2 + 1;

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_red8) as f64;
        let zero = angle == 0.0;
        if !zero {
            c1 = angle.cos() as f32;
            s1 = angle.sin() as f32;
            c2 = c1 * c1 - s1 * s1;
            s2 = s1 * c1 + c1 * s1;
            c3 = c2 * c1 - s2 * s1;
            s3 = s2 * c1 + c2 * s1;
            c4 = c2 * c2 - s2 * s2;
            s4 = s2 * c2 + c2 * s2;
            c5 = c4 * c1 - s4 * s1;
            s5 = s4 * c1 + c4 * s1;
            c6 = c4 * c2 - s4 * s2;
            s6 = s4 * c2 + c4 * s2;
            c7 = c4 * c3 - s4 * s3;
            s7 = s4 * c3 + c4 * s3;
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                // Whether every per-`l` assertion below would pass, so it may
                // be skipped (TO_OPT.md, "combinefft single-thread"): a
                // `false` keeps it.  Exact: `(k1 - 1 - k)` is `extent_between`
                // for every `l`, so the loop runs iff `extent_between >= 0`
                // and its last index `l - 1 + extent_between / sep_between *
                // sep_between` increases with `l`; with `lim_along > 0` the
                // `l` run is `kk ..= kk + (tot_floats - kk) / lim_along *
                // lim_along`, so the assertions all hold iff `kk - 1 >= 0`
                // and the bound holds at the last `l`.  In `i64`/`u128`, so
                // nothing here overflows.
                let kk_ok = if kk > tot_floats {
                    true
                } else if lim_along <= 0 || sep_between <= 0 || extent_between < 0 {
                    false
                } else {
                    let (kkl, totl, laln, ebl, sbl) = (
                        kk as i64,
                        tot_floats as i64,
                        lim_along as i64,
                        extent_between as i64,
                        sep_between as i64,
                    );
                    let last_l = kkl + (totl - kkl) / laln * laln;
                    let last = last_l - 1 + ebl / sbl * sbl;
                    kkl - 1 >= 0
                        && last <= i32::MAX as i64
                        && (top as u128 + last as u128) < len as u128
                };
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    if !kk_ok && k < k1 {
                        assert!(
                            k >= 0
                                && sep_between > 0
                                && top
                                    .checked_add(
                                        (k + (k1 - 1 - k) / sep_between * sep_between) as usize
                                    )
                                    .is_some_and(|last| last < len)
                        );
                    }
                    while k < k1 {
                        unsafe {
                            let rs0 = *dp.add(x0 + k as usize) + *dp.add(x4 + k as usize);
                            let is0 = *dp.add(y0 + k as usize) + *dp.add(y4 + k as usize);
                            let ru0 = *dp.add(x0 + k as usize) - *dp.add(x4 + k as usize);
                            let iu0 = *dp.add(y0 + k as usize) - *dp.add(y4 + k as usize);
                            let rs1 = *dp.add(x1 + k as usize) + *dp.add(x5 + k as usize);
                            let is1 = *dp.add(y1 + k as usize) + *dp.add(y5 + k as usize);
                            let ru1 = *dp.add(x1 + k as usize) - *dp.add(x5 + k as usize);
                            let iu1 = *dp.add(y1 + k as usize) - *dp.add(y5 + k as usize);
                            let rs2 = *dp.add(x2 + k as usize) + *dp.add(x6 + k as usize);
                            let is2 = *dp.add(y2 + k as usize) + *dp.add(y6 + k as usize);
                            let ru2 = *dp.add(x2 + k as usize) - *dp.add(x6 + k as usize);
                            let iu2 = *dp.add(y2 + k as usize) - *dp.add(y6 + k as usize);
                            let rs3 = *dp.add(x3 + k as usize) + *dp.add(x7 + k as usize);
                            let is3 = *dp.add(y3 + k as usize) + *dp.add(y7 + k as usize);
                            let ru3 = *dp.add(x3 + k as usize) - *dp.add(x7 + k as usize);
                            let iu3 = *dp.add(y3 + k as usize) - *dp.add(y7 + k as usize);
                            let rss0 = rs0 + rs2;
                            let iss0 = is0 + is2;
                            let rsu0 = rs0 - rs2;
                            let isu0 = is0 - is2;
                            let rss1 = rs1 + rs3;
                            let iss1 = is1 + is3;
                            let rsu1 = rs1 - rs3;
                            let isu1 = is1 - is3;
                            let rus0 = ru0 - iu2;
                            let ius0 = iu0 + ru2;
                            let ruu0 = ru0 + iu2;
                            let iuu0 = iu0 - ru2;
                            let mut rus1 = ru1 - iu3;
                            let mut ius1 = iu1 + ru3;
                            let mut ruu1 = ru1 + iu3;
                            let mut iuu1 = iu1 - ru3;
                            t = (rus1 + ius1) * e;
                            ius1 = (ius1 - rus1) * e;
                            rus1 = t;
                            t = (ruu1 + iuu1) * e;
                            iuu1 = (iuu1 - ruu1) * e;
                            ruu1 = t;
                            *dp.add(x0 + k as usize) = rss0 + rss1;
                            *dp.add(y0 + k as usize) = iss0 + iss1;
                            if !zero {
                                let r1 = ruu0 + ruu1;
                                let i1 = iuu0 + iuu1;
                                let r2 = rsu0 + isu1;
                                let i2 = isu0 - rsu1;
                                let r3 = rus0 + ius1;
                                let i3 = ius0 - rus1;
                                let r4 = rss0 - rss1;
                                let i4 = iss0 - iss1;
                                let r5 = ruu0 - ruu1;
                                let i5 = iuu0 - iuu1;
                                let r6 = rsu0 - isu1;
                                let i6 = isu0 + rsu1;
                                let r7 = rus0 - ius1;
                                let i7 = ius0 + rus1;
                                *dp.add(x4 + k as usize) = r1 * c1 + i1 * s1;
                                *dp.add(y4 + k as usize) = i1 * c1 - r1 * s1;
                                *dp.add(x2 + k as usize) = r2 * c2 + i2 * s2;
                                *dp.add(y2 + k as usize) = i2 * c2 - r2 * s2;
                                *dp.add(x6 + k as usize) = r3 * c3 + i3 * s3;
                                *dp.add(y6 + k as usize) = i3 * c3 - r3 * s3;
                                *dp.add(x1 + k as usize) = r4 * c4 + i4 * s4;
                                *dp.add(y1 + k as usize) = i4 * c4 - r4 * s4;
                                *dp.add(x5 + k as usize) = r5 * c5 + i5 * s5;
                                *dp.add(y5 + k as usize) = i5 * c5 - r5 * s5;
                                *dp.add(x3 + k as usize) = r6 * c6 + i6 * s6;
                                *dp.add(y3 + k as usize) = i6 * c6 - r6 * s6;
                                *dp.add(x7 + k as usize) = r7 * c7 + i7 * s7;
                                *dp.add(y7 + k as usize) = i7 * c7 - r7 * s7;
                            } else {
                                *dp.add(x4 + k as usize) = ruu0 + ruu1;
                                *dp.add(y4 + k as usize) = iuu0 + iuu1;
                                *dp.add(x2 + k as usize) = rsu0 + isu1;
                                *dp.add(y2 + k as usize) = isu0 - rsu1;
                                *dp.add(x6 + k as usize) = rus0 + ius1;
                                *dp.add(y6 + k as usize) = ius0 - rus1;
                                *dp.add(x1 + k as usize) = rss0 - rss1;
                                *dp.add(y1 + k as usize) = iss0 - iss1;
                                *dp.add(x5 + k as usize) = ruu0 - ruu1;
                                *dp.add(y5 + k as usize) = iuu0 - iuu1;
                                *dp.add(x3 + k as usize) = rsu0 - isu1;
                                *dp.add(y3 + k as usize) = isu0 + rsu1;
                                *dp.add(x7 + k as usize) = rus0 - ius1;
                                *dp.add(y7 + k as usize) = ius0 + rus1;
                            }
                            k += sep_between;
                        }
                    }
                    l += lim_along;
                }
                kk += sep_n_red8;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            t = (c1 + s1) * e;
            s1 = (c1 - s1) * e;
            c1 = t;
            t = s2;
            s2 = c2;
            c2 = t;
            t = (-c3 + s3) * e;
            s3 = (c3 + s3) * e;
            c3 = t;
            c4 = -c4;
            t = -(c5 + s5) * e;
            s5 = (-c5 + s5) * e;
            c5 = t;
            t = -s6;
            s6 = -c6;
            c6 = t;
            t = (c7 - s7) * e;
            s7 = -(c7 + s7) * e;
            c7 = t;
        }
    }
}

/// C `rpcftk`.
///
/// radix prime multi-dimensional complex fourier transform kernel
pub fn rpcftk(
    n: i32,
    n_reduced: i32,
    p_fac: i32,
    step_along: i32,
    data: &mut [f32],
    x: usize,
    y: usize,
    dim: &[i32],
) {
    // Unchecked-access bound: see the module comment; here every access
    // is `x` or `y` plus `k + m * step_along` for `0 <= m <= pm`.
    let len = data.len();
    let top = x.max(y);
    let dp = data.as_mut_ptr();

    let twopi: f32 = 6.2831853;
    let mut t: f32;
    let mut xt: f32;
    let mut yt: f32;
    let mut fu: f32;
    let fp: f32;
    let mut fjm1: f32;
    let fn_redp: f32;
    let mut jj: i32;
    let n_red_ov2p1: i32;
    let n_redp: i32;
    let pm: i32;
    let pp: i32;
    let sep_between: i32;
    let lim_along: i32;
    let sep_n_redp: i32;
    let tot_floats: i32;
    let extent_between: i32;
    let sep: i32;
    let n_sep: i32;

    // `rpcftk` declares these on the stack without initialising them; every
    // element read below is written first (`pp <= 9` and `pm <= 18` for the
    // largest factor `srfp` allows).
    let mut aa = [[0.0_f32; 10]; 10];
    let mut bb = [[0.0_f32; 10]; 10];
    let mut a = [0.0_f32; 19];
    let mut b = [0.0_f32; 19];
    let mut c = [0.0_f32; 19];
    let mut sep_along = [0.0_f32; 19];
    // `ra`/`ia`/`rb`/`ib` are kept as one four-lane group per `v`, and the
    // coefficients each lane multiplies by as the matching group: lane `l`
    // of `acc[v]` is `ra[v]`, `ia[v]`, `rb[v]`, `ib[v]` for `l = 0..3`, and
    // `cf[u][v]` is `aa[v][u], aa[v][u], bb[v][u], bb[v][u]`.  Every lane
    // performs exactly the scalar `float` operation the source writes on
    // that array (`ra[v] + rs * aa[v][u]`, ...), in the same order; the
    // grouping only lets the four independent lanes share one SSE
    // multiply and add, as gcc's SLP vectoriser does with the source.
    // Elementwise SSE `mulps`/`addps` are the IEEE single operations of
    // `mulss`/`addss` lane by lane (no FMA: the target has none enabled).
    let mut acc = [[0.0_f32; 4]; 10];
    let mut cf = [[[0.0_f32; 4]; 10]; 10];

    tot_floats = dim[1];
    sep = dim[2];
    lim_along = dim[3];
    extent_between = dim[4] - 1;
    sep_between = dim[5];
    n_sep = n * sep;
    n_red_ov2p1 = n_reduced / 2 + 1;
    n_redp = n_reduced * p_fac;
    fn_redp = n_redp as f32;
    sep_n_redp = sep * n_redp;
    pp = p_fac / 2;
    pm = p_fac - 1;
    // `aa`/`bb` are `[10][10]` and `a`/`b`/`c`/`sep_along` hold 19, so the
    // setup loops below index out of range (and panic, before any data is
    // touched) unless `pp <= 9`, i.e. `pm <= 18`.  Stating that bound up
    // front lets the compiler drop the per-access checks on those small
    // arrays in the innermost `v` loop; the loops also run over half-open
    // ranges (`1..pp + 1`, the same values as `1..=pp`), which compile to a
    // plain counted loop.
    assert!(pp <= 9 && pm <= 18);
    fp = p_fac as f32;
    fu = 0.0;
    for u in 1..pp + 1 {
        fu = (fu as f64 + 1.0) as f32;
        let angle = (twopi * fu / fp) as f64;
        jj = p_fac - u;
        a[u as usize] = angle.cos() as f32;
        b[u as usize] = angle.sin() as f32;
        a[jj as usize] = a[u as usize];
        b[jj as usize] = -b[u as usize];
    }
    for u in 1..pp + 1 {
        for v in 1..pp + 1 {
            jj = u * v - u * v / p_fac * p_fac;
            aa[v as usize][u as usize] = a[jj as usize];
            bb[v as usize][u as usize] = b[jj as usize];
        }
    }
    for u in 1..pp + 1 {
        for v in 1..pp + 1 {
            let (av, bv) = (aa[v as usize][u as usize], bb[v as usize][u as usize]);
            cf[u as usize][v as usize] = [av, av, bv, bv];
        }
    }

    fjm1 = -1.0;
    for j in 1..=n_red_ov2p1 {
        let fold = j > 1 && 2 * j < n_reduced + 2;
        let mut k0 = (j - 1) * sep + 1;
        fjm1 = (fjm1 as f64 + 1.0) as f32;
        let angle = (twopi * fjm1 / fn_redp) as f64;
        let zero = angle == 0.0;
        if !zero {
            c[1] = angle.cos() as f32;
            sep_along[1] = angle.sin() as f32;
            for u in 2..pm + 1 {
                c[u as usize] =
                    c[(u - 1) as usize] * c[1] - sep_along[(u - 1) as usize] * sep_along[1];
                sep_along[u as usize] =
                    sep_along[(u - 1) as usize] * c[1] + c[(u - 1) as usize] * sep_along[1];
            }
        }
        let ntrip = if fold { 2 } else { 1 };
        for _itrip in 0..ntrip {
            let mut kk = k0;
            while kk <= n_sep {
                let mut l = kk;
                while l <= tot_floats {
                    let k1 = l + extent_between;
                    let mut k = l - 1;
                    if k < k1 {
                        let last = (k + (k1 - 1 - k) / sep_between.max(1) * sep_between.max(1))
                            as i64
                            + pm as i64 * step_along as i64;
                        assert!(
                            k >= 0
                                && sep_between > 0
                                && step_along >= 0
                                && pm >= 0
                                && last <= i32::MAX as i64
                                && top.checked_add(last as usize).is_some_and(|end| end < len)
                        );
                    }
                    while k < k1 {
                        unsafe {
                            xt = *dp.add(x + k as usize);
                            yt = *dp.add(y + k as usize);
                            let mut rs = *dp.add(x + (k + step_along) as usize)
                                + *dp.add(x + (k + step_along * pm) as usize);
                            let mut is = *dp.add(y + (k + step_along) as usize)
                                + *dp.add(y + (k + step_along * pm) as usize);
                            let mut ru = *dp.add(x + (k + step_along) as usize)
                                - *dp.add(x + (k + step_along * pm) as usize);
                            let mut iu = *dp.add(y + (k + step_along) as usize)
                                - *dp.add(y + (k + step_along * pm) as usize);
                            for u in 1..pp + 1 {
                                let c1 = cf[1][u as usize];
                                acc[u as usize] =
                                    [xt + rs * c1[0], yt + is * c1[1], ru * c1[2], iu * c1[3]];
                            }
                            xt = xt + rs;
                            yt = yt + is;

                            for u in 2..pp + 1 {
                                // u numbers from 1 not 0
                                jj = p_fac - u;
                                rs = *dp.add(x + (k + u * step_along) as usize)
                                    + *dp.add(x + (k + jj * step_along) as usize);
                                is = *dp.add(y + (k + u * step_along) as usize)
                                    + *dp.add(y + (k + jj * step_along) as usize);
                                ru = *dp.add(x + (k + u * step_along) as usize)
                                    - *dp.add(x + (k + jj * step_along) as usize);
                                iu = *dp.add(y + (k + u * step_along) as usize)
                                    - *dp.add(y + (k + jj * step_along) as usize);
                                xt = xt + rs;
                                yt = yt + is;
                                let m = [rs, is, ru, iu];
                                let cu = &cf[u as usize];
                                for v in 1..pp + 1 {
                                    let c = cu[v as usize];
                                    let g = &mut acc[v as usize];
                                    g[0] = g[0] + m[0] * c[0];
                                    g[1] = g[1] + m[1] * c[1];
                                    g[2] = g[2] + m[2] * c[2];
                                    g[3] = g[3] + m[3] * c[3];
                                }
                            }
                            *dp.add(x + k as usize) = xt;
                            *dp.add(y + k as usize) = yt;
                            for u in 1..pp + 1 {
                                jj = p_fac - u;
                                if !zero {
                                    xt = acc[u as usize][0] + acc[u as usize][3];
                                    yt = acc[u as usize][1] - acc[u as usize][2];
                                    *dp.add(x + (k + u * step_along) as usize) =
                                        xt * c[u as usize] + yt * sep_along[u as usize];
                                    *dp.add(y + (k + u * step_along) as usize) =
                                        yt * c[u as usize] - xt * sep_along[u as usize];
                                    xt = acc[u as usize][0] - acc[u as usize][3];
                                    yt = acc[u as usize][1] + acc[u as usize][2];
                                    *dp.add(x + (k + jj * step_along) as usize) =
                                        xt * c[jj as usize] + yt * sep_along[jj as usize];
                                    *dp.add(y + (k + jj * step_along) as usize) =
                                        yt * c[jj as usize] - xt * sep_along[jj as usize];
                                } else {
                                    *dp.add(x + (k + u * step_along) as usize) =
                                        acc[u as usize][0] + acc[u as usize][3];
                                    *dp.add(y + (k + u * step_along) as usize) =
                                        acc[u as usize][1] - acc[u as usize][2];
                                    *dp.add(x + (k + jj * step_along) as usize) =
                                        acc[u as usize][0] - acc[u as usize][3];
                                    *dp.add(y + (k + jj * step_along) as usize) =
                                        acc[u as usize][1] + acc[u as usize][2];
                                }
                            }
                            k += sep_between;
                        }
                    }
                    l += lim_along;
                }
                kk += sep_n_redp;
            }
            if !fold {
                break;
            }
            k0 = (n_reduced + 1 - j) * sep + 1;
            for u in 1..pm + 1 {
                t = c[u as usize] * a[u as usize] + sep_along[u as usize] * b[u as usize];
                sep_along[u as usize] =
                    -sep_along[u as usize] * a[u as usize] + c[u as usize] * b[u as usize];
                c[u as usize] = t;
            }
        }
    }
}
