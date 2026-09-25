//! Translation of `IMOD/flib/tilt/bpsumxtilt.f90`.
//!
//! Does back-projection with X-axis tilt using 8 steps at a time; see note in
//! bpsumnox.f90
//!
//! Array and index conventions are described in `super` (`tilt/mod.rs`).

/// Original `bpSumXtilt` (`bpsumxtilt.f90:3`).
///
/// The source has no `implicit none`: only `outSlice`, `inputArr` and the eight
/// `xProj*` (`real*8`) are declared, so by the implicit rules `ind1`, `ipBase`,
/// `ipdel`, `n`, `n8`, `numLeft`, `j`, `iProj*`, `ip1`, `ip2` are `integer` and
/// `cosBeta`, `yfrac`, `omyfrac`, `xFrac*` are default `real` (`real*4`) — the
/// reference build has no `-fdefault-real-8`.  So the accumulation is single
/// precision throughout, as in `bpSumNoX`.
#[allow(clippy::too_many_arguments)]
pub fn bp_sum_xtilt(
    out_slice: &mut [f32],
    ind1: &mut i32,
    input_arr: &[f32],
    ip_base: i32,
    ipdel: i32,
    n: i32,
    x_proj1: &mut f64,
    cos_beta: f32,
    yfrac: f32,
    omyfrac: f32,
) {
    let n8 = n / 8;
    let num_left = n - 8 * n8;
    let cos_beta8 = cos_beta as f64;
    if n <= 0 {
        return;
    }
    // Hoisted once: outSlice(ind1 : ind1 + n - 1), the only elements written,
    // split into the 8-step blocks and the `numLeft` tail.  `ind1` and `xProj1`
    // are carried in locals and stored once at the end.
    let out = &mut out_slice[(*ind1 - 1) as usize..(*ind1 - 1 + n) as usize];
    let (out8, out_left) = out.split_at_mut(8 * n8 as usize);
    let in_len = input_arr.len() as i64;
    let mut x_proj1_l = *x_proj1;
    let mut ind1_l = *ind1;

    for blk in out8.chunks_exact_mut(8) {
        let mut x_proj = [0f64; 8];
        x_proj[0] = x_proj1_l;
        for k in 1..8 {
            x_proj[k] = x_proj[k - 1] + cos_beta8;
        }
        let mut i_proj = [0i32; 8];
        for k in 0..8 {
            i_proj[k] = fortran_int!(f64: x_proj[k]);
        }
        let mut x_frac = [0f32; 8];
        for k in 0..8 {
            x_frac[k] = (x_proj[k] - i_proj[k] as f64) as f32;
        }
        // Each step reads inputArr(ip1), inputArr(ip1 + 1), inputArr(ip2) and
        // inputArr(ip2 + 1) with ip1 = ipBase + iProj and ip2 = ip1 + ipdel, so the
        // block is in bounds exactly when those four hold at the smallest and
        // largest iProj; one test here stands for the thirty-two below.
        let i_min = *i_proj.iter().min().unwrap() as i64;
        let i_max = *i_proj.iter().max().unwrap() as i64;
        let lo = ip_base as i64 + i_min - 1 + (ipdel as i64).min(0);
        let hi = ip_base as i64 + i_max + (ipdel as i64).max(0);
        if lo < 0 || hi >= in_len {
            panic!(
                "bpSumXtilt: inputArr index {lo}..={hi} outside the array (Fortran would read out of bounds)"
            );
        }
        for k in 0..8 {
            let ip1 = ip_base + i_proj[k];
            let ip2 = ip1 + ipdel;
            // SAFETY: ip1 - 1, ip1, ip2 - 1 and ip2 all lie in [lo, hi], checked
            // against 0 and input_arr.len() just above.
            let (a1, b1, a2, b2) = unsafe {
                (
                    *input_arr.get_unchecked((ip1 - 1) as usize),
                    *input_arr.get_unchecked(ip1 as usize),
                    *input_arr.get_unchecked((ip2 - 1) as usize),
                    *input_arr.get_unchecked(ip2 as usize),
                )
            };
            blk[k] = blk[k]
                + (1. - x_frac[k]) * (omyfrac * a1 + yfrac * a2)
                + x_frac[k] * (omyfrac * b1 + yfrac * b2);
            ind1_l += 1;
        }
        x_proj1_l = x_proj[7] + cos_beta8;
    }

    for o in out_left.iter_mut().take(num_left as usize) {
        let i_proj1 = fortran_int!(f64: x_proj1_l);
        let x_frac1 = (x_proj1_l - i_proj1 as f64) as f32;
        let ip1 = ip_base + i_proj1;
        let ip2 = ip1 + ipdel;
        *o = *o
            + (1. - x_frac1)
                * (omyfrac * input_arr[(ip1 - 1) as usize] + yfrac * input_arr[(ip2 - 1) as usize])
            + x_frac1 * (omyfrac * input_arr[ip1 as usize] + yfrac * input_arr[ip2 as usize]);
        ind1_l += 1;
        x_proj1_l += cos_beta8;
    }
    *ind1 = ind1_l;
    *x_proj1 = x_proj1_l;
}
