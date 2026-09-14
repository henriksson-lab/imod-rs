//! Translation of `IMOD/clip/threshminsize.cpp`.
#![allow(dead_code)]

use crate::imod::clip::clip::{ClipOptions, PlaneConnectedPoints, ZConnectedSets};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format};
use crate::imod::libcfshr::islice::{Islice, slice_create, slice_free, slice_put_val};
use crate::imod::libiimod::mrcfiles::MrcHeader;
use std::io::Write as _;

/// C++ `thresholdWithMinSize`.
pub unsafe fn threshold_with_min_size(
    hin: &mut MrcHeader,
    hout: &mut MrcHeader,
    opt: &mut ClipOptions,
    thresh_lo: f32,
    thresh_hi: f32,
    mut z_write: i32,
) -> i32 {
    unsafe {
        crate::imod::libiimod::mrcfiles::mrc_head_label(
            &mut *hout,
            b"clip: thresholded with minimum size constraint",
        );
        if opt.min_size == 0 {
            crate::imod::clip::clip::show_error("CLIP - The minimum size entry must be non-zero");
            return -1;
        }
        let mut min_size = opt.min_size;
        let mut direction = 1.;
        let mut thresh = opt.thresh;
        let (fill, set) = if min_size < 0 {
            min_size = -min_size;
            direction = -1.;
            (thresh_hi, thresh_lo)
        } else {
            thresh *= 1.0000006;
            (thresh_lo, thresh_hi)
        };
        let (nx, ny) = (opt.ix, opt.iy);
        let mut shift = 0_u32;
        let mut mask = 0_u32;
        let mut nyleft = ny - 1;
        let mut nxcheck = nx - 1;
        while nyleft > 0 {
            shift += 1;
            nyleft >>= 1;
            nxcheck <<= 1;
            mask = (mask << 1) | 1;
            if nyleft > 0 && (nxcheck as u32 & 0x80000000) != 0 {
                crate::imod::clip::clip::show_error(
                    "CLIP - Images are too large in X and Y for the thresholding procedure",
                );
                return -1;
            }
        }
        let out = slice_create(nx, ny, opt.mode);
        if out.is_null() {
            // `threshminsize.cpp:126-129`.  Note the doubled space in the
            // source's text.
            let _ = ImodFile::Stdout.write_all(b"ERROR: CLIP - Allocating  memory\n");
            return -1;
        }
        // The source's two `catch` blocks (`threshminsize.cpp:358-365`,
        // `std::bad_alloc` and `exception&`) have no counterpart: Rust has no
        // exceptions, and the STL containers they guard are `Vec`s here.  Their
        // texts are therefore unreachable rather than omitted.
        let mut grouped = vec![0_u8; (nx * ny) as usize];
        let mut planes: Vec<Vec<PlaneConnectedPoints>> =
            (0..opt.nofsecs).map(|_| Vec::new()).collect();
        let mut sets: Vec<ZConnectedSets> = Vec::new();
        let offset = if opt.dim == 3 { 1 } else { 0 };
        let mut kout = 0;
        let mut next: *mut Islice = core::ptr::null_mut();
        let mut last: *mut Islice = core::ptr::null_mut();
        let mut input: *mut Islice = core::ptr::null_mut();
        for kin in -offset..opt.nofsecs {
            if kin + offset < opt.nofsecs {
                let z = opt.secs[(kin + offset) as usize];
                next = crate::imod::libiimod::mrcslice::slice_read_subm(
                    hin,
                    z,
                    b'z' as i8,
                    nx,
                    ny,
                    opt.cx as i32,
                    opt.cy as i32,
                );
                if next.is_null() || crate::imod::libiimod::mrcslice::slice_float(next) < 0 {
                    // `threshminsize.cpp:143`: the first `%s` is chosen by
                    // `nextSlice` and the third by `inSlice`, which are not the
                    // same test -- a conversion failure on the very first slice
                    // prints "Converting slice N from file".
                    let _ = ImodFile::Stdout.write_all(
                        c_format(
                            "ERROR: CLIP - %s slice %d %s\n",
                            &[
                                CArg::Str(if !next.is_null() {
                                    "Converting"
                                } else {
                                    "Reading"
                                }),
                                CArg::Int(z as i64),
                                CArg::Str(if !input.is_null() {
                                    "to floating point"
                                } else {
                                    "from file"
                                }),
                            ],
                        )
                        .as_bytes(),
                    );
                    if !next.is_null() {
                        slice_free(next)
                    };
                    if !last.is_null() {
                        slice_free(last)
                    };
                    slice_free(out);
                    return -1;
                }
                if offset == 0 || kin == -1 {
                    input = next
                }
                if kin == -1 {
                    continue;
                }
            }
            let data = (*input).data.f;
            grouped.fill(0);
            for ly in 0..ny {
                for lx in 0..nx {
                    let li = lx + nx * ly;
                    if grouped[li as usize] != 0
                        || direction * (*data.add(li as usize) - thresh) < 0.
                    {
                        continue;
                    }
                    let mut checks = vec![((lx as u32) << shift) | ly as u32];
                    let mut plane = PlaneConnectedPoints {
                        points: std::collections::BTreeSet::new(),
                        xmin: lx,
                        xmax: lx,
                        ymin: ly,
                        ymax: ly,
                    };
                    plane.points.insert(checks[0]);
                    grouped[li as usize] = 1;
                    let mut ci = 0;
                    while ci < checks.len() {
                        let combo = checks[ci];
                        let (x, y) = ((combo >> shift) as i32, (combo & mask) as i32);
                        for (cx, cy) in [
                            (0.max(x - 1), y),
                            ((nx - 1).min(x + 1), y),
                            (x, 0.max(y - 1)),
                            (x, (ny - 1).min(y + 1)),
                        ] {
                            let ind = cx + nx * cy;
                            if grouped[ind as usize] == 0
                                && direction * (*data.add(ind as usize) - thresh) >= 0.
                            {
                                let xy = ((cx as u32) << shift) | cy as u32;
                                checks.push(xy);
                                plane.points.insert(xy);
                                plane.xmin = plane.xmin.min(cx);
                                plane.xmax = plane.xmax.max(cx);
                                plane.ymin = plane.ymin.min(cy);
                                plane.ymax = plane.ymax.max(cy);
                                grouped[ind as usize] = 1;
                            }
                        }
                        ci += 1;
                    }
                    let lone = if ci == 1 && !last.is_null() && !next.is_null() {
                        let last_value = *((*last).data.f).add(li as usize);
                        let next_value = *((*next).data.f).add(li as usize);
                        direction * (last_value - thresh) < 0.
                            && direction * (next_value - thresh) < 0.
                    } else {
                        false
                    };
                    if !lone {
                        planes[kin as usize].push(plane)
                    }
                }
            }
            if offset != 0 {
                if !last.is_null() {
                    slice_free(last)
                };
                last = input;
                input = next;
                next = core::ptr::null_mut()
            } else {
                slice_free(input)
            }
            for pi in 0..planes[kin as usize].len() {
                let plane = &planes[kin as usize][pi];
                let mut first: Option<usize> = None;
                let mut si = 0;
                while si < sets.len() {
                    let overlaps = !(sets[si].xmax < plane.xmin
                        || sets[si].xmin > plane.xmax
                        || sets[si].ymax < plane.ymin
                        || sets[si].ymin > plane.ymax);
                    let joins = opt.dim == 3
                        && overlaps
                        && (0..sets[si].index.len()).any(|n| {
                            sets[si].plane_z[n] == kin - 1
                                && plane.points.iter().any(|p| {
                                    planes[(kin - 1) as usize][sets[si].index[n] as usize]
                                        .points
                                        .contains(p)
                                })
                        });
                    if joins {
                        if let Some(fi) = first {
                            let other = sets.remove(si);
                            let main = &mut sets[fi];
                            main.plane_z.extend(other.plane_z);
                            main.index.extend(other.index);
                            main.xmin = main.xmin.min(other.xmin);
                            main.xmax = main.xmax.max(other.xmax);
                            main.ymin = main.ymin.min(other.ymin);
                            main.ymax = main.ymax.max(other.ymax);
                            main.zmin = main.zmin.min(other.zmin);
                            main.num_points += other.num_points;
                            continue;
                        } else {
                            let z = &mut sets[si];
                            z.plane_z.push(kin);
                            z.index.push(pi as i32);
                            z.xmin = z.xmin.min(plane.xmin);
                            z.xmax = z.xmax.max(plane.xmax);
                            z.ymin = z.ymin.min(plane.ymin);
                            z.ymax = z.ymax.max(plane.ymax);
                            z.zmax = kin;
                            z.num_points += plane.points.len() as i32;
                            first = Some(si);
                        }
                    }
                    si += 1;
                }
                if first.is_none() {
                    sets.push(ZConnectedSets {
                        plane_z: vec![kin],
                        index: vec![pi as i32],
                        xmin: plane.xmin,
                        xmax: plane.xmax,
                        ymin: plane.ymin,
                        ymax: plane.ymax,
                        zmin: kin,
                        zmax: kin,
                        num_points: plane.points.len() as i32,
                    })
                }
            }
            let active = if kin < opt.nofsecs - 1 {
                kin
            } else {
                opt.nofsecs
            };
            while kout < active {
                if sets.iter().any(|z| z.zmin <= kout && z.zmax == active) {
                    break;
                }
                for y in 0..ny {
                    for x in 0..nx {
                        slice_put_val(out, x, y, [fill; 4]);
                    }
                }
                for z in &sets {
                    if z.num_points >= min_size && z.zmin <= kout && z.zmax >= kout {
                        for n in 0..z.index.len() {
                            if z.plane_z[n] == kout {
                                for p in &planes[kout as usize][z.index[n] as usize].points {
                                    slice_put_val(
                                        out,
                                        (p >> shift) as i32,
                                        (p & mask) as i32,
                                        [set; 4],
                                    );
                                }
                            }
                        }
                    }
                }
                if crate::imod::clip::file_io::clip_write_slice(
                    out,
                    hout,
                    opt,
                    kout,
                    &mut z_write,
                    0,
                ) != 0
                {
                    if !last.is_null() {
                        slice_free(last)
                    }
                    slice_free(out);
                    return -1;
                }
                planes[kout as usize].clear();
                kout += 1
            }
            sets.retain(|z| !((z.num_points < min_size && z.zmax < active) || z.zmax < kout));
        }
        if !last.is_null() {
            slice_free(last)
        }
        slice_free(out);
        crate::imod::clip::file_io::set_mrc_coords(hin, hout, opt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::clip::clip::CameraDefects;
    use crate::imod::libiimod::mrcfiles::mrc_head_new;

    #[test]
    fn threshold_rejects_zero_minimum_size_before_slice_io() {
        let mut hin = unsafe { MrcHeader::default() };
        let mut hout = unsafe { MrcHeader::default() };
        mrc_head_new(&mut hin, 2, 2, 1, 2);
        mrc_head_new(&mut hout, 2, 2, 1, 2);
        let defects = CameraDefects {
            was_scaled: 0,
            rotation_flip: 0,
            k2_type: 0,
            falcon_type: 0,
            usable_top: 0,
            usable_left: 0,
            usable_bottom: 0,
            usable_right: 0,
            num_avg_super_res: 0,
            bad_column_start: vec![],
            bad_column_width: vec![],
            partial_bad_col: vec![],
            partial_bad_width: vec![],
            partial_bad_start_y: vec![],
            partial_bad_end_y: vec![],
            bad_row_start: vec![],
            bad_row_height: vec![],
            partial_bad_row: vec![],
            partial_bad_height: vec![],
            partial_bad_start_x: vec![],
            partial_bad_end_x: vec![],
            bad_pixel_x: vec![],
            bad_pixel_y: vec![],
            pix_use_mean: vec![],
        };
        let mut opt = ClipOptions {
            pname: String::new(),
            command: String::new(),
            process: 44,
            x: 0,
            y: 0,
            z: 0,
            x2: 0,
            y2: 0,
            z2: 0,
            ix: 2,
            iy: 2,
            iz: 1,
            iz2: 0,
            ox: 0,
            oy: 0,
            oz: 1,
            chunk_x: 0,
            chunk_y: 0,
            chunk_z: 0,
            cx: 1.,
            cy: 1.,
            cz: 0.,
            high: 0.,
            low: 0.,
            red: 0.,
            green: 0.,
            blue: 0.,
            thresh: 1.,
            weight: 0.,
            pctl_frac: 0.,
            falloff_frac: 0.,
            min_size: 0,
            pad: 0.,
            mode: 2,
            dim: 3,
            infiles: 1,
            fnames: Vec::new(),
            sano: 0,
            add2file: 0,
            isec: 0,
            val: 0.,
            nofsecs: 1,
            secs: Vec::new(),
            out_before: 0,
            out_after: 0,
            ocanresize: 0,
            ocanchmode: 0,
            from_one: 0,
            ofname: None,
            plname: None,
            super_gain_name: None,
            point_out_name: None,
            new_xoverlap: 0,
            new_yoverlap: 0,
            rotation_flip: 0,
            read_defects: 0,
            defects,
            cam_size_x: 0,
            cam_size_y: 0,
            binning: 0.,
            scale_defects: 0,
        };
        assert_eq!(
            unsafe { threshold_with_min_size(&mut hin, &mut hout, &mut opt, 0., 1., 0) },
            -1
        );
    }
}
