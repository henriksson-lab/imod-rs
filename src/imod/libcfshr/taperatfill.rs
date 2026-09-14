//! Translation of `IMOD/libcfshr/taperatfill.c`.
#![allow(dead_code, static_mut_refs)]

use crate::imod::libcfshr::islice::{Islice, slice_get_val, slice_init, slice_put_val};

const MAX_TAPER: i32 = 256;
const MAX_AVG_OUT: usize = 16;
static mut S_FOUND_FILL: i32 = 0;
static mut S_LAST_FILL_VAL: f32 = 0.0;

/// C `sliceTaperAtFill`.
pub unsafe fn slice_taper_at_fill(sl: *mut Islice, mut ntaper: i32, inside: i32) -> i32 {
    unsafe {
        ntaper = ntaper.min(MAX_TAPER);
        let tapersq = (ntaper + 1) * (ntaper + 1);
        let inside = if inside != 0 { 1 } else { 0 };
        let xsize = (*sl).xsize;
        let ysize = (*sl).ysize;
        S_FOUND_FILL = 0;
        let (mut fillval, mut longest, mut longix, mut longiy, mut dir) = (0., 0, 0, 0, 0);
        slice_find_fill_value(
            sl,
            &mut fillval,
            &mut longest,
            &mut longix,
            &mut longiy,
            &mut dir,
        );
        if longest < 10 {
            return 0;
        }
        (*sl).mean = fillval;
        let dxout = [0, 1, 0, -1];
        let dyout = [-1, 0, 1, 0];
        let dxnext = [1, 0, -1, 0];
        let dynext = [0, 1, 0, -1];
        let mut ix = longix + dxnext[(dir % 2) as usize] * longest / 2;
        let mut iy = longiy + dynext[(dir % 2) as usize] * longest / 2;
        let mut found = false;
        while !found {
            ix -= dxout[dir as usize];
            iy -= dyout[dir as usize];
            if ix < 0 || ix >= xsize || iy < 0 || iy >= ysize {
                break;
            }
            let mut val = [0.; 4];
            slice_get_val(sl, ix, iy, &mut val);
            if val[0] != fillval {
                found = true;
            }
        }
        if !found {
            return 0;
        }
        S_FOUND_FILL = 1;
        S_LAST_FILL_VAL = fillval;
        let mut pixmap = vec![0u8; (xsize * ysize) as usize];
        let dirstart = dir;
        let xstart = ix;
        let ystart = iy;
        let taper_dir = 1 - 2 * inside;
        let mut elist = vec![(ix, iy)];
        let mut plist: Vec<(i16, i16, usize)> = Vec::new();
        let mut lastout = 1;
        loop {
            let mut ncol = 1;
            let mut xnext = ix + dxout[dir as usize] + dxnext[dir as usize];
            let mut ynext = iy + dyout[dir as usize] + dynext[dir as usize];
            let mut val = [0.; 4];
            slice_get_val(sl, xnext, ynext, &mut val);
            let mut ind = 1;
            if val[0] != fillval {
                ix = xnext;
                iy = ynext;
                dir = (dir + 3) % 4;
                if inside != 0 {
                    ncol = ntaper + 1;
                }
            } else {
                xnext = ix + dxnext[dir as usize];
                ynext = iy + dynext[dir as usize];
                slice_get_val(sl, xnext, ynext, &mut val);
                if val[0] != fillval {
                    ix = xnext;
                    iy = ynext;
                } else {
                    dir = (dir + 1) % 4;
                    if inside == 0 {
                        ncol = ntaper + 1;
                    }
                    ind = lastout;
                }
            }
            let xpout = ix + dxout[dir as usize];
            let ypout = iy + dyout[dir as usize];
            lastout = if xpout < 0 || xpout >= xsize || ypout < 0 || ypout >= ysize {
                1
            } else {
                0
            };
            if lastout == 0 {
                if ind != 0 {
                    elist.push((ix, iy));
                }
                let (mut lim1, mut lim2);
                if dxnext[dir as usize] != 0 {
                    lim1 = ix / dxnext[dir as usize];
                    lim2 = (ix + 1 - xsize) / dxnext[dir as usize];
                } else {
                    lim1 = iy / dynext[dir as usize];
                    lim2 = (iy + 1 - ysize) / dynext[dir as usize];
                }
                let mut collim = ncol - 1;
                if lim1 >= 0 && lim1 < collim {
                    collim = lim1;
                }
                if lim2 >= 0 && lim2 < collim {
                    collim = lim2;
                }
                if dxout[dir as usize] != 0 {
                    lim1 = -ix / (taper_dir * dxout[dir as usize]);
                    lim2 = (xsize - 1 - ix) / (taper_dir * dxout[dir as usize]);
                } else {
                    lim1 = -iy / (taper_dir * dyout[dir as usize]);
                    lim2 = (ysize - 1 - iy) / (taper_dir * dyout[dir as usize]);
                }
                let mut rowlim = ntaper - inside;
                if lim1 >= 1 - inside && lim1 < rowlim {
                    rowlim = lim1;
                }
                if lim2 >= 1 - inside && lim2 < rowlim {
                    rowlim = lim2;
                }
                for col in 0..=collim {
                    for row in 1 - inside..=rowlim {
                        let xp =
                            ix + taper_dir * row * dxout[dir as usize] - col * dxnext[dir as usize];
                        let yp =
                            iy + taper_dir * row * dyout[dir as usize] - col * dynext[dir as usize];
                        let pind = (xsize * yp + xp) as usize;
                        if pixmap[pind] != 0 {
                            continue;
                        }
                        slice_get_val(sl, xp, yp, &mut val);
                        if (inside != 0 && val[0] == fillval) || (inside == 0 && val[0] != fillval)
                        {
                            continue;
                        }
                        if inside == 0 {
                            let (mut v1, mut v2, mut v3, mut v4) =
                                ([0.; 4], [0.; 4], [0.; 4], [0.; 4]);
                            slice_get_val(sl, xp + 1, yp, &mut v1);
                            slice_get_val(sl, xp, yp + 1, &mut v2);
                            slice_get_val(sl, xp - 1, yp, &mut v3);
                            slice_get_val(sl, xp, yp - 1, &mut v4);
                            if v1[0] != fillval
                                && v2[0] != fillval
                                && v3[0] != fillval
                                && v4[0] != fillval
                            {
                                continue;
                            }
                        }
                        pixmap[pind] = 1;
                        plist.push(((xp - ix) as i16, (yp - iy) as i16, elist.len() - 1));
                    }
                }
            }
            if ix == xstart && iy == ystart && dir == dirstart {
                break;
            }
        }
        let mut fracs = vec![0_f32; (ntaper * 10 + 1) as usize];
        let mut fillpart = fracs.clone();
        for i in 1..=ntaper * 10 {
            let dist = if inside != 0 { i } else { ntaper * 10 + 1 - i };
            fracs[i as usize] = dist as f32 / (ntaper * 10 + 1) as f32;
            // `taperatfill.c:236` is `(1. - fracs[i]) * fillval` with `1.` a
            // double, so both the subtraction and the product are double and
            // only the store to `fillpart` rounds.
            fillpart[i as usize] = ((1. - fracs[i as usize] as f64) * fillval as f64) as f32;
        }
        for (dx, dy, edgeind) in plist {
            let mut distmin = [0_i32; MAX_AVG_OUT];
            let mut minedge = [0_usize; MAX_AVG_OUT];
            distmin[0] = dx as i32 * dx as i32 + dy as i32 * dy as i32;
            minedge[0] = edgeind;
            let mut num_mins = 1usize;
            let distmax = (1.5 * (distmin[0].max(tapersq)) as f32) as i32;
            let px = elist[edgeind].0 + dx as i32;
            let py = elist[edgeind].1 + dy as i32;
            for walkdir in [-1_i32, 1] {
                let mut k = edgeind;
                for _ in 0..elist.len() {
                    k = ((k as i32 + walkdir + elist.len() as i32) % elist.len() as i32) as usize;
                    let xx = px - elist[k].0;
                    let yy = py - elist[k].1;
                    let dist = xx * xx + yy * yy;
                    if dist < distmin[num_mins - 1] || (inside == 0 && num_mins < 15) {
                        if inside != 0 {
                            distmin[0] = dist;
                            minedge[0] = k;
                        } else {
                            let mut newind = num_mins.min(14);
                            while newind > 0 && dist < distmin[newind - 1] {
                                newind -= 1;
                            }
                            let mut mm = 13.min(num_mins - 1);
                            while mm >= newind {
                                distmin[mm + 1] = distmin[mm];
                                minedge[mm + 1] = minedge[mm];
                                if mm == 0 {
                                    break;
                                }
                                mm -= 1;
                            }
                            num_mins = (num_mins + 1).min(15);
                            distmin[newind] = dist;
                            minedge[newind] = k;
                        }
                    }
                    if dist > distmax {
                        break;
                    }
                }
            }
            let find = (10.0 * ((distmin[0] as f64).sqrt() + inside as f64)) as i32;
            if find > ntaper * 10 {
                continue;
            }
            let mut val = [0.; 4];
            if inside != 0 {
                slice_get_val(sl, px, py, &mut val);
            } else {
                let (mut wsum, mut valsum) = (0., 0.);
                for mm in 0..num_mins {
                    // `taperatfill.c:319` is `weight = 1. / (4. + sqrt(distmin[mm]))`
                    // with `weight` a float: the whole reciprocal is computed in
                    // double and rounded once, not `(4 + sqrt)` rounded first.
                    let weight = (1. / (4. + (distmin[mm] as f64).sqrt())) as f32;
                    wsum += weight;
                    slice_get_val(sl, elist[minedge[mm]].0, elist[minedge[mm]].1, &mut val);
                    valsum += weight * val[0];
                }
                val[0] = valsum / wsum;
            }
            val[0] = fracs[find as usize] * val[0] + fillpart[find as usize];
            slice_put_val(sl, px, py, val);
        }
        0
    }
}

/// C `taperAtFill`.
pub unsafe fn taper_at_fill(
    array: *mut core::ffi::c_void,
    data_type: i32,
    nx: i32,
    ny: i32,
    ntaper: i32,
    inside: i32,
) -> i32 {
    unsafe {
        let mut slice = core::mem::zeroed::<Islice>();
        slice_init(&mut slice, nx, ny, data_type, array);
        slice_taper_at_fill(&mut slice, ntaper, inside)
    }
}
/// C Fortran wrapper `taperatfill`.
pub unsafe fn taper_at_fill_fortran(
    array: *mut f32,
    nx: *const i32,
    ny: *const i32,
    ntaper: *const i32,
    inside: *const i32,
) -> i32 {
    unsafe { taper_at_fill(array.cast(), 2, *nx, *ny, *ntaper, *inside) }
}
/// C `getLastTaperFillValue`.
pub unsafe fn get_last_taper_fill_value(value: *mut f32) -> i32 {
    unsafe {
        *value = S_LAST_FILL_VAL;
        S_FOUND_FILL
    }
}
/// C Fortran wrapper `getlasttaperfillvalue`.
pub unsafe fn get_last_taper_fill_value_fortran(value: *mut f32) -> i32 {
    unsafe { get_last_taper_fill_value(value) }
}

/// C `sliceFindFillValue`.
pub unsafe fn slice_find_fill_value(
    sl: *mut Islice,
    fillval_p: *mut f32,
    longest_p: *mut i32,
    longix_p: *mut i32,
    longiy_p: *mut i32,
    dir_p: *mut i32,
) {
    unsafe {
        let (xsize, ysize) = ((*sl).xsize, (*sl).ysize);
        let (mut longest, mut fillval, mut longix, mut longiy, mut direction) = (0, 0., 0, 0, 0);
        for iy in [0, ysize - 1] {
            let mut len = 0;
            let mut val = [0.; 4];
            slice_get_val(sl, 0, iy, &mut val);
            let mut lastval = val[0];
            let mut lastix = 0;
            for ix in 1..xsize {
                slice_get_val(sl, ix, iy, &mut val);
                if val[0] == lastval {
                    len += 1;
                    if len > longest {
                        longest = len;
                        fillval = lastval;
                        longix = lastix;
                        longiy = iy;
                        direction = if iy != 0 { 2 } else { 0 };
                    }
                } else {
                    len = 0;
                    lastval = val[0];
                    lastix = ix;
                }
            }
        }
        for ix in [0, xsize - 1] {
            let mut len = 0;
            let mut val = [0.; 4];
            slice_get_val(sl, ix, 0, &mut val);
            let mut lastval = val[0];
            let mut lastiy = 0;
            for iy in 1..ysize {
                slice_get_val(sl, ix, iy, &mut val);
                if val[0] == lastval {
                    len += 1;
                    if len > longest {
                        longest = len;
                        fillval = lastval;
                        longix = ix;
                        longiy = lastiy;
                        direction = if ix != 0 { 1 } else { 3 };
                    }
                } else {
                    len = 0;
                    lastval = val[0];
                    lastiy = iy;
                }
            }
        }
        *longest_p = longest;
        *fillval_p = fillval;
        *longix_p = longix;
        *longiy_p = longiy;
        *dir_p = direction;
    }
}

/// C `sliceReplaceFill`.
pub unsafe fn slice_replace_fill(
    sl: *mut Islice,
    median: i32,
    have_fill: i32,
    old_fill: *mut f32,
    new_fill: *mut f32,
) -> i32 {
    unsafe {
        let (xsize, ysize) = ((*sl).xsize, (*sl).ysize);
        let mut fillval;
        if have_fill != 0 {
            fillval = *old_fill;
        } else {
            let (mut longest, mut longix, mut longiy, mut direction) = (0, 0, 0, 0);
            fillval = 0.;
            slice_find_fill_value(
                sl,
                &mut fillval,
                &mut longest,
                &mut longix,
                &mut longiy,
                &mut direction,
            );
            if longest < 10 {
                return 1;
            }
            *old_fill = fillval;
        }
        let mut top = vec![ysize - 1; xsize as usize];
        let mut bot = vec![0; xsize as usize];
        let mut edge_vals = Vec::new();
        let mut sum = 0_f64;
        let mut nsum = 0usize;
        for (edge_x, scan_dir) in [(0, 1), (0, -1), (xsize - 1, 1), (xsize - 1, -1)] {
            let (mut y, stop) = if scan_dir > 0 {
                (0, ysize / 2)
            } else {
                (ysize - 1, ysize / 2 + 1)
            };
            while (stop - y) * scan_dir >= 0 {
                let mut step = 0;
                while step < xsize {
                    let xx = edge_x + if edge_x == 0 { step } else { -step };
                    let ind = (xx + xsize * y) as usize;
                    let mut val = [0.; 4];
                    slice_get_val(sl, xx, y, &mut val);
                    if val[0] != fillval {
                        if median != 0 {
                            edge_vals.push(val[0]);
                        } else {
                            sum += val[0] as f64;
                        }
                        nsum += 1;
                        break;
                    } else {
                        let starts = if scan_dir > 0 { &mut bot } else { &mut top };
                        if starts[xx as usize] == y - scan_dir {
                            starts[xx as usize] = y;
                        }
                    }
                    step += 1;
                }
                y += scan_dir;
            }
        }
        for (edge_y, scan_dir) in [(0, 1), (ysize - 1, -1)] {
            for x in 0..xsize {
                let start = if scan_dir > 0 {
                    bot[x as usize]
                } else {
                    ysize - 1 - top[x as usize]
                };
                for step in start..ysize {
                    let yy = edge_y + scan_dir * step;
                    if yy < 0 || yy >= ysize {
                        break;
                    }
                    let mut val = [0.; 4];
                    slice_get_val(sl, x, yy, &mut val);
                    if val[0] != fillval {
                        if median != 0 {
                            edge_vals.push(val[0]);
                        } else {
                            sum += val[0] as f64;
                        }
                        nsum += 1;
                        break;
                    }
                }
            }
        }
        if nsum < 10 {
            return 2;
        }
        if median != 0 {
            if edge_vals.len() > 15000 {
                let skip = edge_vals.len() / 15000;
                let remain = edge_vals.len() % 15000;
                let mut reduced = Vec::with_capacity(15000);
                let mut at = 0;
                for _ in 0..remain {
                    reduced.push(edge_vals[at]);
                    at += skip + 1;
                }
                while reduced.len() < 15000 {
                    reduced.push(edge_vals[at]);
                    at += skip;
                }
                edge_vals = reduced;
            }
            let n = edge_vals.len() as i32;
            crate::imod::libcfshr::robuststat::rs_fast_median_in_place(
                &mut edge_vals,
                n,
                &mut *new_fill,
            );
        } else {
            *new_fill = (sum / nsum as f64) as f32;
        }
        for pass in 0..4 {
            if pass < 2 {
                let edge_y = if pass == 0 { 0 } else { ysize - 1 };
                let dir = if pass == 0 { 1 } else { -1 };
                for x in 0..xsize {
                    let start = if dir > 0 {
                        bot[x as usize]
                    } else {
                        ysize - 1 - top[x as usize]
                    };
                    for step in start..ysize {
                        let y = edge_y + dir * step;
                        if y < 0 || y >= ysize {
                            break;
                        }
                        let mut v = [0.; 4];
                        slice_get_val(sl, x, y, &mut v);
                        if v[0] != fillval && v[0] != *new_fill {
                            break;
                        }
                        v[0] = *new_fill;
                        slice_put_val(sl, x, y, v);
                    }
                }
            } else {
                let edge_x = if pass == 2 { 0 } else { xsize - 1 };
                let dir = if pass == 2 { 1 } else { -1 };
                for y in 0..ysize {
                    for step in 0..xsize {
                        let x = edge_x + dir * step;
                        if x < 0 || x >= xsize {
                            break;
                        }
                        let mut v = [0.; 4];
                        slice_get_val(sl, x, y, &mut v);
                        if v[0] != fillval && v[0] != *new_fill {
                            break;
                        }
                        v[0] = *new_fill;
                        slice_put_val(sl, x, y, v);
                    }
                }
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finds_and_reports_edge_fill() {
        unsafe {
            let mut image = vec![0_f32; 12 * 12];
            for y in 1..11 {
                for x in 1..11 {
                    image[y * 12 + x] = 10.;
                }
            }
            let mut sl = core::mem::zeroed::<Islice>();
            slice_init(&mut sl, 12, 12, 2, image.as_mut_ptr().cast());
            let (mut f, mut len, mut x, mut y, mut d) = (0., 0, 0, 0, 0);
            slice_find_fill_value(&mut sl, &mut f, &mut len, &mut x, &mut y, &mut d);
            assert_eq!(f, 0.);
            assert!(len >= 10);
            assert_eq!(slice_taper_at_fill(&mut sl, 2, 0), 0);
            let mut last = 0.;
            assert_eq!(get_last_taper_fill_value(&mut last), 1);
            assert_eq!(last, 0.);
        }
    }

    #[test]
    fn replaces_fill_with_adjacent_mean() {
        unsafe {
            let mut image = vec![0_f32; 12 * 12];
            for y in 1..11 {
                for x in 1..11 {
                    image[y * 12 + x] = 10.;
                }
            }
            let mut sl = core::mem::zeroed::<Islice>();
            slice_init(&mut sl, 12, 12, 2, image.as_mut_ptr().cast());
            let mut old = 0.;
            let mut new = 0.;
            assert_eq!(slice_replace_fill(&mut sl, 0, 0, &mut old, &mut new), 0);
            assert_eq!((old, new), (0., 10.));
            assert!(image.iter().all(|value| *value == 10.));
        }
    }
}
