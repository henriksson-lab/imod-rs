//! Translation of `IMOD/flib/blend/shuffler.f90`: `shuffler`,
//! `clearShuffle`, `scaleCachedPieces`, `scaleOneCachedPiece` — the cache of
//! input pieces held in the module array `array`.
//!
//! Every unit here `use`s `blendvars`, so each takes `bv: &mut BlendVars`
//! first (see the design note in [`super::blendvars`]).  The local
//! `real*4 array(*)` declaration is commented out in the source
//! (`shuffler.f90:100`), so `array` is the module variable `bv.array`.

use super::blendvars::BlendVars;
use crate::imod::flib::subrs::hvem::parse_input_params::exit_error;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::irdsec;
use crate::imod::libiimod::unit_fileio::iiu_set_position;
use crate::imod::libwarp::maggradfield::{add_mag_grad_field, make_mag_grad_field};
use crate::imod::libwarp::warpinterp::warp_interp;

/// Original: `subroutine shuffler(izWant, index)` (`shuffler.f90:95`).
///
/// `index` is the 1-based start of the piece in `bv.array`, as in the source.
pub fn shuffler(bv: &mut BlendVars, iz_want: i32, index: &mut i32) {
    //
    // if the section's entry in memIndex exists, return index
    //
    let mut i = bv.mem_index[(iz_want - 1) as usize];
    if i > 0 {
        // `(i - 1) * npixIn + 1` is integer(kind=8), narrowed on assignment.
        *index = ((i - 1) as i64 * bv.npix_in + 1) as i32;
        bv.juse_count += 1;
        bv.last_used[(i - 1) as usize] = bv.juse_count;
        return;
    }
    //
    // Otherwise look for oldest used or unused slot
    //
    let mut min_used = bv.juse_count + 1;
    // `ioldest` is undefined in the source if no slot qualifies, which needs
    // `maxLoad < 1`; 0 here.
    let mut ioldest = 0_i32;
    i = 1;
    while i <= bv.max_load {
        if min_used > bv.last_used[(i - 1) as usize] {
            min_used = bv.last_used[(i - 1) as usize];
            ioldest = i;
        }
        i += 1;
    }
    let iold = (ioldest - 1) as usize;
    //
    // Assign to that slot, clear out previous allocation there
    //
    *index = ((ioldest - 1) as i64 * bv.npix_in + 1) as i32;
    bv.juse_count += 1;
    if bv.iz_mem_list[iold] > 0 {
        bv.mem_index[(bv.iz_mem_list[iold] - 1) as usize] = -1;
    }
    bv.last_used[iold] = bv.juse_count;
    bv.iz_mem_list[iold] = iz_want;
    bv.mem_index[(iz_want - 1) as usize] = ioldest;
    // `imposn` is the C wrapper `imposn` (`unit_fileio.c:492`), which passes
    // its arguments straight to `iiuSetPosition`.
    unsafe { iiu_set_position(1, iz_want - 1, 0) }; //yes izwant starts at 1
    //
    let mut ind_read = *index;
    if bv.do_fields && bv.doing_edge_func {
        ind_read = (bv.max_load as i64 * bv.npix_in + 1) as i32;
    }
    if unsafe { irdsec(1, &mut bv.array[(ind_read - 1) as usize..]) }.is_err() {
        // `99 call exitError('READING FILE')`
        exit_error("READING FILE");
    }
    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    if bv.mult_by_flatfield > 0 {
        for i in 1..=nxin * nyin {
            let k = (ind_read + i - 1 - 1) as usize;
            bv.array[k] = (bv.array[k] - bv.den_zero_base) * bv.flatfield[(i - 1) as usize]
                + bv.den_zero_base;
        }
    }
    //
    if bv.do_fields {
        // Flat offset of `fieldDx(1, 1, ioldest)` in the (lmField, lmField,
        // maxFields) allocation.
        let foff = bv.field_dx_ext[0] * bv.field_dx_ext[1] * iold;
        let foff_y = bv.field_dy_ext[0] * bv.field_dy_ext[1] * iold;
        if bv.do_mag_grad {
            let mag_use = bv.ilistz.min(bv.num_mag_grad);
            let iang_use = bv.ilistz.min(bv.num_angles);
            let xcen: f32;
            let ycen: f32;
            //
            // Get center of tilt as center of frame or montage
            //
            if bv.focus_adjusted {
                xcen = nxin as f32 / 2.;
                ycen = nyin as f32 / 2.;
            } else {
                let n_xoverlap = bv.n_overlap[0];
                let ny_overlap = bv.n_overlap[1];
                xcen = (bv.min_xpiece + bv.nx_pieces * (nxin - n_xoverlap) + n_xoverlap) as f32
                    / 2.
                    - bv.ix_pc_list[(iz_want - 1) as usize] as f32;
                ycen = (bv.min_ypiece + bv.ny_pieces * (nyin - ny_overlap) + ny_overlap) as f32
                    / 2.
                    - bv.iy_pc_list[(iz_want - 1) as usize] as f32;
            }
            let tilt = bv.tilt_angles[(iang_use - 1) as usize];
            let dmag = bv.dmag_per_um[(mag_use - 1) as usize];
            let rot = bv.rot_per_um[(mag_use - 1) as usize];
            //
            // add mag gradient to distortion or make field up
            //
            if bv.undistort {
                add_mag_grad_field(
                    &bv.dist_dx,
                    &bv.dist_dy,
                    &mut bv.field_dx[foff..],
                    &mut bv.field_dy[foff_y..],
                    bv.lm_field,
                    nxin,
                    nyin,
                    bv.nx_field,
                    bv.ny_field,
                    bv.x_field_strt,
                    bv.y_field_strt,
                    bv.x_field_intrv,
                    bv.y_field_intrv,
                    xcen,
                    ycen,
                    bv.pixel_mag_grad,
                    bv.axis_rot,
                    tilt,
                    dmag,
                    rot,
                );
            } else {
                // `makeMagGradField` writes back the grid size, start and
                // interval module variables (`warpwrapfort.c:354-363`).
                make_mag_grad_field(
                    &mut bv.dist_dx,
                    &mut bv.dist_dy,
                    &mut bv.field_dx[foff..],
                    &mut bv.field_dy[foff_y..],
                    bv.lm_field,
                    nxin,
                    nyin,
                    &mut bv.nx_field,
                    &mut bv.ny_field,
                    &mut bv.x_field_strt,
                    &mut bv.y_field_strt,
                    &mut bv.x_field_intrv,
                    &mut bv.y_field_intrv,
                    xcen,
                    ycen,
                    bv.pixel_mag_grad,
                    bv.axis_rot,
                    tilt,
                    dmag,
                    rot,
                );
            }
        } else {
            //
            // Just copy field if distortion only
            //
            for i in 1..=bv.nx_field {
                for j in 1..=bv.ny_field {
                    let (i1, j1) = ((i - 1) as usize, (j - 1) as usize);
                    bv.field_dx[i1 + bv.field_dx_ext[0] * (j1 + bv.field_dx_ext[1] * iold)] =
                        bv.dist_dx[i1 + bv.dist_dx_ext[0] * j1];
                    bv.field_dy[i1 + bv.field_dy_ext[0] * (j1 + bv.field_dy_ext[1] * iold)] =
                        bv.dist_dy[i1 + bv.dist_dy_ext[0] * j1];
                }
            }
        }
        //
        // Undistort the piece for computing edge functions
        //
        if bv.doing_edge_func {
            // `real*4 amat(2,2)` as a dimension-reversed nested array:
            // `amat(i, j)` is `amat[j - 1][i - 1]`.  Its memory order is then
            // exactly what the `warpinterp_` wrapper hands to C as
            // `float amat[2][2]` (`warpwrapfort.c:266-275`), so it is passed
            // as is.
            let mut amat = [[0.0_f32; 2]; 2];
            amat[0][0] = 1.;
            amat[1][1] = 1.;
            amat[1][0] = 0.;
            amat[0][1] = 0.;
            // `array(indRead)` is the input and `array(index)` the output;
            // indRead = maxLoad * npixIn + 1 lies past every cache slot, so the
            // two are disjoint and split at indRead.
            let (lo, hi) = bv.array.split_at_mut((ind_read - 1) as usize);
            warp_interp(
                hi,
                &mut lo[(*index - 1) as usize..],
                nxin,
                nyin,
                nxin,
                nyin,
                &amat,
                nxin as f32 / 2.,
                nyin as f32 / 2.,
                0.,
                0.,
                1.,
                bv.dfill,
                1,
                0,
                &bv.field_dx[foff..],
                &bv.field_dy[foff_y..],
                bv.lm_field,
                bv.nx_field,
                bv.ny_field,
                bv.x_field_strt,
                bv.y_field_strt,
                bv.x_field_intrv,
                bv.y_field_intrv,
            );
        }
    }

    if bv.i_apply_dens_scaling > 0 || bv.x_base_grad_scale != 0. || bv.y_base_grad_scale != 0 as f32
    {
        let (xb, yb) = (bv.x_base_grad_scale, bv.y_base_grad_scale);
        scale_one_cached_piece(bv, ioldest, xb, yb);
    }
}

/// Original: `subroutine clearShuffle()` (`shuffler.f90:211`) — initializes
/// or clears the memory allocation.
pub fn clear_shuffle(bv: &mut BlendVars) {
    bv.juse_count = 0;
    for i in 1..=bv.mem_lim {
        bv.iz_mem_list[(i - 1) as usize] = -1;
        bv.last_used[(i - 1) as usize] = 0;
    }
    for i in 1..=bv.lim_npc {
        bv.mem_index[(i - 1) as usize] = -1;
    }
}

/// Original: `subroutine scaleCachedPieces(izMont)` (`shuffler.f90:231`) —
/// applies the piece and gradient scaling to every loaded piece at montage Z
/// value `izMont`.
pub fn scale_cached_pieces(bv: &mut BlendVars, iz_mont: i32) {
    for i in 1..=bv.mem_lim {
        if bv.iz_mem_list[(i - 1) as usize] > 0 {
            if bv.iz_pc_list[(bv.iz_mem_list[(i - 1) as usize] - 1) as usize] == iz_mont {
                scale_one_cached_piece(bv, i, 0., 0.);
            }
        }
    }
}

/// Original: `subroutine scaleOneCachedPiece(indList, xBaseIn, yBaseIn)`
/// (`shuffler.f90:248`) — applies the piece and density scaling to the piece
/// at cache index `indList`.
pub fn scale_one_cached_piece(bv: &mut BlendVars, ind_list: i32, x_base_in: f32, y_base_in: f32) {
    let do_piece = bv.i_apply_dens_scaling > 0;
    let mut x_grad = x_base_in;
    let mut y_grad = y_base_in;
    let mut do_grad = bv.x_base_grad_scale != 0. || bv.y_base_grad_scale != 0.;
    if bv.i_apply_dens_scaling > 1 {
        do_grad = true;
        x_grad += bv.x_grad_scaling;
        y_grad += bv.y_grad_scaling;
    }

    let nxin = bv.nxyz_in[0];
    let nyin = bv.nxyz_in[1];
    let xcen = nxin as f32 / 2.;
    let ycen = nyin as f32 / 2.;
    // integer(kind=8) product narrowed to `integer*4 index`.
    let index = ((ind_list - 1) as i64 * bv.npix_in) as i32;
    let mut pc_scale = 1.0_f32;
    let den_zero_base = bv.den_zero_base;
    for ix_frame in 1..=bv.nx_pieces {
        for iy_frame in 1..=bv.ny_pieces {
            // Find the piece in the map that has this file Z value
            let (ixf, iyf) = ((ix_frame - 1) as usize, (iy_frame - 1) as usize);
            if bv.map_piece[ixf + bv.map_piece_ext[0] * iyf]
                == bv.iz_mem_list[(ind_list - 1) as usize]
            {
                if do_piece {
                    pc_scale = bv.piece_scaling[ixf + bv.piece_scaling_ext[0] * iyf];
                }
                for iy in 1..=nyin {
                    let ix_base = index + (iy - 1) * nxin;
                    if do_grad {
                        for ix in 1..=nxin {
                            let k = (ix_base + ix - 1) as usize;
                            bv.array[k] = (bv.array[k] - den_zero_base) * pc_scale
                                / (1. + x_grad * (ix as f32 - xcen) + y_grad * (iy as f32 - ycen))
                                + den_zero_base;
                        }
                    } else {
                        for ix in 1..=nxin {
                            let k = (ix_base + ix - 1) as usize;
                            bv.array[k] = (bv.array[k] - den_zero_base) * pc_scale + den_zero_base;
                        }
                    }
                }
            }
        }
    }
}
