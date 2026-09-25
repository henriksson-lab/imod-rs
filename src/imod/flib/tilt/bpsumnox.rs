//! Translation of `IMOD/flib/tilt/bpsumnox.f90`.
//!
//! Does back-projection with no X-axis tilt using 8 steps at a time.
//! This and the other two bpsum routines were originally made so that assembly code
//! could be modified to minimize slow switching between rounding modes.
//! They are retained for two reasons: 1) the multiple steps run faster than a single
//! loop. 2) C translations ran ~50% slower than Fortran with Intel compilers, despite
//! various efforts to optimize the code
//!
//! Array and index conventions are described in `super` (`tilt/mod.rs`).

/// Original `bpSumNoX` (`bpsumnox.f90:8`).
///
/// `outSlice(ind1 ..)` accumulates `n` interpolated values from the projection
/// line starting at `inputArr(ipoint + 1)`.  `xProj*` are `real*8`, `cosBeta` and
/// `xFrac*` `real*4`: each `xFrac` is the double difference rounded to single, and
/// the accumulation is entirely single precision, left to right.  `ind1` and
/// `xProj1` are updated as in the source (the C wrapper `Tilt::bpSumNoX` passes a
/// copy of `xproj1`, so only `ind1` is seen by the caller).
#[allow(clippy::too_many_arguments)]
pub fn bp_sum_no_x(
    out_slice: &mut [f32],
    ind1: &mut i32,
    input_arr: &[f32],
    ipoint: i32,
    n: i32,
    x_proj1: &mut f64,
    cos_beta: f32,
) {
    let n8 = n / 8;
    let num_left = n - 8 * n8;
    let ip_p1 = ipoint + 1;
    let cos_beta8 = cos_beta as f64;
    if n <= 0 {
        return;
    }
    // Hoisted once: outSlice(ind1 : ind1 + n - 1), the only elements the source
    // touches, split into the 8-step blocks and the `numLeft` tail.
    let out = &mut out_slice[(*ind1 - 1) as usize..(*ind1 - 1 + n) as usize];
    let (out8, out_left) = out.split_at_mut(8 * n8 as usize);
    let in_len = input_arr.len() as i64;
    // `ind1` and `xProj1` are carried in locals and stored once at the end.
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
        // The block reads inputArr(ipoint + iProj) and inputArr(ipP1 + iProj) for
        // each of its eight iProj, so it is in bounds exactly when the smallest
        // and largest are; one test here stands for the sixteen below.
        let lo = ipoint as i64 - 1 + *i_proj.iter().min().unwrap() as i64;
        let hi = ip_p1 as i64 - 1 + *i_proj.iter().max().unwrap() as i64;
        if lo < 0 || hi >= in_len {
            panic!(
                "bpSumNoX: inputArr index {lo}..={hi} outside the array (Fortran would read out of bounds)"
            );
        }
        for k in 0..8 {
            // SAFETY: both indices lie in [lo, hi], checked against 0 and
            // input_arr.len() just above.
            let (a, b) = unsafe {
                (
                    *input_arr.get_unchecked((ipoint + i_proj[k] - 1) as usize),
                    *input_arr.get_unchecked((ip_p1 + i_proj[k] - 1) as usize),
                )
            };
            blk[k] = blk[k] + (1. - x_frac[k]) * a + x_frac[k] * b;
        }
        ind1_l += 8;
        x_proj1_l = x_proj[7] + cos_beta8;
    }

    for o in out_left.iter_mut().take(num_left as usize) {
        let i_proj1 = fortran_int!(f64: x_proj1_l);
        let x_frac1 = (x_proj1_l - i_proj1 as f64) as f32;
        *o = *o
            + (1. - x_frac1) * input_arr[(ipoint + i_proj1 - 1) as usize]
            + x_frac1 * input_arr[(ip_p1 + i_proj1 - 1) as usize];
        ind1_l += 1;
        x_proj1_l += cos_beta8;
    }
    *ind1 = ind1_l;
    *x_proj1 = x_proj1_l;
}

/// Original `bpSumAreaNoX` (`bpsumnox.f90:87`).
///
/// Does backprojection into a line of with areas of intersection between
/// projection rays and pixels in the reconstruction.
///
/// `ind1` is only read by the source (the caller advances its index itself), so
/// it is taken by value.  `rayDist = 0.5 + iproj - xProj1` adds in single
/// precision first (`0.5 + iproj` is `real*4`), then subtracts the `real*8`
/// `xProj1` in double and rounds to single; `lutInd = rayDist * maxDist + 1` is a
/// single-precision product and sum truncated to integer.
#[allow(clippy::too_many_arguments)]
pub fn bp_sum_area_no_x(
    out_slice: &mut [f32],
    ind1: i32,
    input_arr: &[f32],
    ipoint: i32,
    n: i32,
    x_proj1: &mut f64,
    cos_beta: f32,
    nx_out: i32,
    ind_del_ray: &[i32],
    num_rays_hit: &[i32],
    ray_areas: &[f32],
    max_dist: i32,
) {
    // Loop-invariant bounds, tested once.  `lutInd` is clamped to
    // [1, max(1, maxDist)], so the two lookup tables are in bounds for every
    // pixel when they hold that many entries (tilt.cpp sizes them maxDist + 1).
    // A ray is used only when 1 <= iray <= nxOut, so inputArr(ipoint + iray) is
    // in bounds when inputArr(ipoint + 1 : ipoint + nxOut) is, and outSlice(j)
    // when outSlice(ind1 : ind1 + n - 1) is.  When all three hold the loop runs
    // with those accesses unchecked; otherwise the same loop runs checked, so a
    // panic happens exactly where the source would read or write out of bounds.
    let n_lut = max_dist.max(1) as usize;
    let luts_ok = num_rays_hit.len() >= n_lut && ind_del_ray.len() >= n_lut;
    let line_ok =
        ipoint >= 0 && nx_out >= 0 && (ipoint as i64 + nx_out as i64) <= input_arr.len() as i64;
    let out_ok = n <= 0 || (ind1 >= 1 && (ind1 as i64 - 1 + n as i64) <= out_slice.len() as i64);
    let mut x_proj1_l = *x_proj1;

    // The source's pixel loop; `$fast` is a literal, so each instantiation is
    // one straight loop.
    macro_rules! pixel_loop {
        ($fast:literal) => {
            //
            // Loop on the pixels, for each pixel add in intersecting rays
            for j in ind1..ind1 + n {
                let iproj = fortran_int!(f64: x_proj1_l + 0.5);
                let ray_dist = ((0.5f32 + iproj as f32) as f64 - x_proj1_l) as f32;
                let mut lut_ind = fortran_int!(f32: ray_dist * max_dist as f32 + 1f32);
                lut_ind = 1.max(max_dist.min(lut_ind));
                let lut = (lut_ind - 1) as usize;
                let (num_hit, del_ray) = if $fast {
                    // SAFETY: lut < n_lut <= both lengths (luts_ok).
                    unsafe { (*num_rays_hit.get_unchecked(lut), *ind_del_ray.get_unchecked(lut)) }
                } else {
                    (num_rays_hit[lut], ind_del_ray[lut])
                };
                // rayAreas(3 * (lutInd - 1) + indRay) for indRay <= numRaysHit(lutInd):
                // one test per pixel for all its rays.
                let areas_ok = 3 * lut as i64 + num_hit as i64 <= ray_areas.len() as i64;
                for ind_ray in 1..num_hit + 1 {
                    let iray = iproj + del_ray + ind_ray - 1;
                    if iray > 0 && iray <= nx_out {
                        let ia = (3 * (lut_ind - 1) + ind_ray - 1) as usize;
                        let area = if areas_ok {
                            // SAFETY: 0 <= ia <= 3 * lut + num_hit - 1 < ray_areas.len()
                            // (areas_ok, 1 <= ind_ray <= num_hit).
                            unsafe { *ray_areas.get_unchecked(ia) }
                        } else {
                            ray_areas[ia]
                        };
                        let ii = (ipoint + iray - 1) as usize;
                        let jj = (j - 1) as usize;
                        if $fast {
                            // SAFETY: ipoint <= ii < ipoint + nxOut <= input_arr.len()
                            // (line_ok, 1 <= iray <= nxOut) and
                            // ind1 - 1 <= jj < ind1 - 1 + n <= out_slice.len() (out_ok).
                            unsafe {
                                let o = out_slice.get_unchecked_mut(jj);
                                *o = *o + area * *input_arr.get_unchecked(ii);
                            }
                        } else {
                            out_slice[jj] = out_slice[jj] + area * input_arr[ii];
                        }
                    }
                }
                x_proj1_l += cos_beta as f64;
            }
        };
    }
    if luts_ok && line_ok && out_ok {
        pixel_loop!(true);
    } else {
        pixel_loop!(false);
    }
    *x_proj1 = x_proj1_l;
}
