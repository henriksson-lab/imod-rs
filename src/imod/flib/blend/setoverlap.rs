//! Translation of `IMOD/flib/blend/setoverlap.f90`.
//!
//! No `use blendvars`, so no `BlendVars` parameter.  `niceFrame` resolves to
//! the `niceframe_` wrapper (`filtxcorr.c:111`), which passes its arguments
//! through unchanged, so the C entry [`nice_frame`] is called directly.

use crate::imod::libcfshr::filtxcorr::nice_frame;

/// Original: `subroutine setOverlap` (`setoverlap.f90:16`).
///
/// `nFrame` is in/out; `nOverlap` is `&mut` because the final search loop
/// may not execute, leaving the caller's value (`setoverlap.f90:74-84`).
#[allow(clippy::too_many_arguments)]
pub fn set_overlap(
    min_tot_pix: i32,
    min_overlap: i32,
    no_fft_sizes: bool,
    n_frame: &mut i32,
    idel_frame: i32,
    num_pieces: &mut i32,
    n_overlap: &mut i32,
    num_tot_pix: &mut i32,
) {
    //
    // make sure overlap is even
    let min_over = 2 * ((min_overlap + 1) / 2);
    // round frame size down if necessary to have no larger prime factors
    if no_fft_sizes {
        *n_frame -= *n_frame % 2;
    } else {
        *n_frame = nice_frame(*n_frame, -idel_frame, 19);
    }
    let mut non_overlap = *n_frame - min_over;
    //
    // here is the minimum number of pieces needed
    *num_pieces = ((min_tot_pix - min_over) + (non_overlap - 1)) / non_overlap;
    //
    // if only one piece, round up frame size to fit # of pixels
    //
    if *num_pieces == 1 {
        *n_frame = idel_frame * ((min_tot_pix + idel_frame - 1) / idel_frame);
        if no_fft_sizes {
            *n_frame += *n_frame % 2;
        } else {
            *n_frame = nice_frame(*n_frame, idel_frame, 19); //round UP if necessary
        }
        *num_tot_pix = *n_frame;
        *n_overlap = min_over;
        return;
    }
    //
    // otherwise find smallest frame size that keeps this number of pieces
    // and has no big prime factors
    //
    loop {
        let mut new_frame: i32;
        if no_fft_sizes {
            new_frame = *n_frame - idel_frame;
            new_frame -= new_frame % 2;
        } else {
            new_frame = nice_frame(*n_frame - idel_frame, -idel_frame, 19);
        }
        non_overlap = new_frame - min_over;
        if ((min_tot_pix - min_over) + (non_overlap - 1)) / non_overlap == *num_pieces {
            *n_frame = new_frame;
        } else {
            break;
        }
    }
    //
    // find overlap that gives this number of pieces and requires fewest
    // extra pixels
    //
    *num_tot_pix = 2 * min_tot_pix;
    let mut lap_over = min_over;
    while ((min_tot_pix - lap_over) + (*n_frame - lap_over - 1)) / (*n_frame - lap_over)
        == *num_pieces
    {
        //
        // total # of pixels required with this overlap
        let ntot_tmp = *num_pieces * (*n_frame - lap_over) + lap_over;
        if ntot_tmp < *num_tot_pix {
            *num_tot_pix = ntot_tmp;
            *n_overlap = lap_over;
        }
        lap_over += 2; //keep overlap even
    }
}
