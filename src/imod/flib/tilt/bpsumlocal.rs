//! Translation of `IMOD/flib/tilt/bpsumlocal.f90`.
//!
//! Does back-projection with local alignment information using5 steps at a time; see
//! note in bpsumnox.f90
//!
//! Array and index conventions are described in `super` (`tilt/mod.rs`).

/// Original `bpSumLocal` (`bpsumlocal.f90:4`).
///
/// Everything is `real*4`.  `nprojSuper * (...)` is an integer times a real, so
/// the factor converts to single; the tail loop applies that formula even when
/// `nprojSuper == 1`, as the source does (`1 * (x - 0.5) + 0.5` rounds back to
/// `x` for every `x >= 0.5`, so this is only observable below that).  The five-step body writes
/// `outSlice(index)` and `outSlice(index + 1)` before advancing `index` by 2, then
/// by 1 three times, as the source does; `index` is returned advanced by the
/// number of points.
///
/// The `DO j = jStart, jEndFast, numDiv` trip count is Fortran's
/// `max(0, (jEndFast - jStart + numDiv) / numDiv)`.
#[allow(clippy::too_many_arguments)]
pub fn bp_sum_local(
    out_slice: &mut [f32],
    index: &mut i32,
    input_arr: &[f32],
    zz: f32,
    x_proj_f: &[f32],
    x_proj_z: &[f32],
    y_proj_f: &[f32],
    y_proj_z: &[f32],
    ipoint: i32,
    ip_del: i32,
    lslice: i32,
    j_start: i32,
    j_end: i32,
    nproj_super: i32,
) {
    let num_div: i32 = 5;
    let num_fast = (j_end + 1 - j_start) / num_div;
    let j_end_fast = j_start + (num_fast - 1) * num_div;
    let j_start_slow = j_start + num_fast * num_div;
    let fast_trips = ((j_end_fast - j_start + num_div) / num_div).max(0);
    let slow_trips = (j_end - j_start_slow + 1).max(0);
    let super_f = nproj_super as f32;
    if fast_trips + slow_trips == 0 {
        return;
    }
    // Hoisted once: the outSlice(index : ...) elements the loops below advance
    // through, one per j; the five-step blocks first, then the tail loop.  The
    // index itself is carried in a local and stored once at the end.
    let n_out = (num_div * fast_trips + slow_trips) as usize;
    let out = &mut out_slice[(*index - 1) as usize..(*index - 1) as usize + n_out];
    let (out5, out_slow) = out.split_at_mut((num_div * fast_trips) as usize);
    let in_len = input_arr.len() as i64;
    let del_lo = (ip_del as i64).min(0);
    let del_hi = (ip_del as i64).max(0);

    // One five-step block of either branch of the source: the ten projections,
    // then the five bilinear interpolations into blk[0..5] (the source's
    // outSlice(index), outSlice(index + 1), then index + 2, + 3, + 4).
    macro_rules! five_steps {
        ($blk:expr, $j:expr, $x_of:expr) => {{
            let k0 = ($j - 1) as usize;
            let xf: &[f32; 5] = x_proj_f[k0..k0 + 5].try_into().unwrap();
            let xz: &[f32; 5] = x_proj_z[k0..k0 + 5].try_into().unwrap();
            let yf: &[f32; 5] = y_proj_f[k0..k0 + 5].try_into().unwrap();
            let yz: &[f32; 5] = y_proj_z[k0..k0 + 5].try_into().unwrap();
            let mut x_proj = [0f32; 5];
            let mut y_proj = [0f32; 5];
            for m in 0..5 {
                x_proj[m] = $x_of(xf[m], xz[m]);
                y_proj[m] = yf[m] + zz * yz[m];
            }
            let mut i_proj = [0i32; 5];
            let mut j_proj = [0i32; 5];
            for m in 0..5 {
                i_proj[m] = fortran_int!(f32: x_proj[m]);
                j_proj[m] = fortran_int!(f32: y_proj[m]);
            }
            let mut ip1 = [0i32; 5];
            for m in 0..5 {
                ip1[m] = ipoint + (j_proj[m] - lslice) * ip_del + i_proj[m];
            }
            // Each step reads inputArr(ip1), inputArr(ip1 + 1), inputArr(ip2) and
            // inputArr(ip2 + 1), ip2 = ip1 + ipDel, so the block is in bounds
            // exactly when those hold at the smallest and largest ip1; one test
            // here stands for the twenty reads below.
            let lo = *ip1.iter().min().unwrap() as i64 - 1 + del_lo;
            let hi = *ip1.iter().max().unwrap() as i64 + del_hi;
            if lo < 0 || hi >= in_len {
                panic!("bpSumLocal: inputArr index {lo}..={hi} outside the array (Fortran would read out of bounds)");
            }
            for m in 0..5 {
                let x_frac = x_proj[m] - i_proj[m] as f32;
                let y_frac = y_proj[m] - j_proj[m] as f32;
                let ip2 = ip1[m] + ip_del;
                // SAFETY: ip1 - 1, ip1, ip2 - 1 and ip2 all lie in [lo, hi],
                // checked against 0 and input_arr.len() just above.
                let (a1, b1, a2, b2) = unsafe {
                    (
                        *input_arr.get_unchecked((ip1[m] - 1) as usize),
                        *input_arr.get_unchecked(ip1[m] as usize),
                        *input_arr.get_unchecked((ip2 - 1) as usize),
                        *input_arr.get_unchecked(ip2 as usize),
                    )
                };
                $blk[m] = $blk[m]
                    + (1. - y_frac) * ((1. - x_frac) * a1 + x_frac * b1)
                    + y_frac * ((1. - x_frac) * a2 + x_frac * b2);
            }
        }};
    }

    // It took 20% longer to do the loop with the proj factor of 1, so duplicate it
    if nproj_super == 1 {
        for (t, blk) in out5.chunks_exact_mut(5).enumerate() {
            let j = j_start + t as i32 * num_div;
            five_steps!(blk, j, |f: f32, z: f32| f + zz * z);
        }
    } else {
        for (t, blk) in out5.chunks_exact_mut(5).enumerate() {
            let j = j_start + t as i32 * num_div;
            five_steps!(blk, j, |f: f32, z: f32| super_f * (f + zz * z - 0.5) + 0.5);
        }
    }

    for (t, o) in out_slow.iter_mut().enumerate() {
        let j = j_start_slow + t as i32;
        let k = (j - 1) as usize;
        let x_proj = super_f * (x_proj_f[k] + zz * x_proj_z[k] - 0.5) + 0.5;
        let y_proj = y_proj_f[k] + zz * y_proj_z[k];
        let i_proj = fortran_int!(f32: x_proj);
        let x_frac = x_proj - i_proj as f32;
        let j_proj = fortran_int!(f32: y_proj);
        let y_frac = y_proj - j_proj as f32;
        //
        let ip1 = ipoint + (j_proj - lslice) * ip_del + i_proj;
        let ip2 = ip1 + ip_del;
        *o = *o
            + (1. - y_frac)
                * ((1. - x_frac) * input_arr[(ip1 - 1) as usize]
                    + x_frac * input_arr[ip1 as usize])
            + y_frac
                * ((1. - x_frac) * input_arr[(ip2 - 1) as usize]
                    + x_frac * input_arr[ip2 as usize]);
    }
    *index += n_out as i32;
}
