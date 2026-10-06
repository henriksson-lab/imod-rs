//! Translation of `IMOD/flib/subrs/graphics/screenpak.f90`: the interface
//! between the old-style screen graphics calls and the `qtplax` module
//! ([`super::qtplax`]), with module `scrnvars`.
//!
//! Module variables are thread-local cells (the Fortran program runs on one
//! thread).  `fortgetarg`/`fortiargc`, which `qtplax.cpp` calls back for the
//! argument vector, are [`fortgetarg`] and [`fortiargc`] over
//! `b3dutil::program_args`.

use std::cell::Cell;

use super::qtplax::{
    plax_box, plax_boxo, plax_circ, plax_circo, plax_close, plax_erase, plax_flush, plax_mapcolor,
    plax_next_text_align, plax_open, plax_poly, plax_polyo, plax_sctext, plax_vect,
};
use crate::imod::flib::subrs::compat::gfortran_rt::{adjustl, cvttss2si, format_i, nint_r4};
use crate::imod::libcfshr::b3dutil::program_args;

thread_local! {
    /// Module `scrnvars` (`screenpak.f90:10-12`):
    /// `ifPlaxOn/0/, ixCur, iyCur, icolor, ifReverse/0/`.
    static IF_PLAX_ON: Cell<i32> = const { Cell::new(0) };
    static IX_CUR: Cell<i32> = const { Cell::new(0) };
    static IY_CUR: Cell<i32> = const { Cell::new(0) };
    static ICOLOR: Cell<i32> = const { Cell::new(0) };
    static IF_REVERSE: Cell<i32> = const { Cell::new(0) };
}

/// Original `scrnErase` (`screenpak.f90:15`): initializes graphics if
/// needed, clears the screen and restores the default colors.
pub fn scrn_erase(_ix: i32) {
    if IF_PLAX_ON.get() < 0 {
        return;
    }
    if IF_PLAX_ON.get() == 0 {
        if plax_open() == -1 {
            return;
        }
        IF_PLAX_ON.set(1);
        ICOLOR.set(241);
    }
    plax_erase();
    if IF_REVERSE.get() != 0 {
        plax_mapcolor(0, 255, 255, 255);
        plax_mapcolor(241, 0, 0, 0);
    } else {
        plax_mapcolor(0, 0, 0, 0);
        plax_mapcolor(241, 255, 255, 255);
    }
    plax_mapcolor(250, 255, 0, 0);
    plax_mapcolor(251, 0, 255, 0);
    plax_mapcolor(252, 0, 0, 255);
    plax_mapcolor(253, 255, 255, 0);
    plax_mapcolor(254, 255, 0, 255);
    plax_mapcolor(255, 0, 255, 255);
    plax_box(0, 0, 0, 1279, 1023);
    IX_CUR.set(0);
    IY_CUR.set(0);
    plax_flush();
}

/// Original `scrnClose` (`screenpak.f90:43`): closes the graphics window.
pub fn scrn_close() {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    plax_box(0, 0, 0, 1279, 1023);
    plax_close();
    IF_PLAX_ON.set(0);
}

/// Original `scrnChangeColor` (`screenpak.f90:53`).
pub fn scrn_change_color(ix: i32) {
    ICOLOR.set(ix);
}

/// Original `scrnUpdate` (`screenpak.f90:60`): flush, i.e. force a draw.
pub fn scrn_update(_ix: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    plax_flush();
}

/// Original `scrnOpen` (`screenpak.f90:68`): opens the graphics window or
/// marks it as not to be opened.
pub fn scrn_open(if_off: i32) {
    // if ifOff is 0, want it on, so bump from -1 to 0 or leave as is
    if if_off == 0 {
        IF_PLAX_ON.set(IF_PLAX_ON.get().max(0));
        if IF_PLAX_ON.get() == 0 {
            scrn_erase(1);
        }
    } else {
        // otherwise, want it off; turn it off, mark flag as -1
        scrn_close();
        IF_PLAX_ON.set(-1);
    }
}

/// Original `reverseGraphContrast` (`screenpak.f90:82`).
pub fn reverse_graph_contrast(ival: i32) {
    IF_REVERSE.set(ival);
}

/// Original `scrnMoveAbs` (`screenpak.f90:89`).
pub fn scrn_move_abs(ix: i32, iy: i32) {
    IX_CUR.set(ix);
    IY_CUR.set(iy);
}

/// Original `scrnMoveInc` (`screenpak.f90:97`).
pub fn scrn_move_inc(ix: i32, iy: i32) {
    IX_CUR.set(IX_CUR.get() + ix);
    IY_CUR.set(IY_CUR.get() + iy);
}

/// Original `scrnVectAbs` (`screenpak.f90:105`).
pub fn scrn_vect_abs(ix: i32, iy: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    plax_vect(ICOLOR.get(), IX_CUR.get(), IY_CUR.get(), ix, iy);
    IX_CUR.set(ix);
    IY_CUR.set(iy);
}

/// Original `scrnVectInc` (`screenpak.f90:115`).
pub fn scrn_vect_inc(ix: i32, iy: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    let (x, y) = (IX_CUR.get(), IY_CUR.get());
    plax_vect(ICOLOR.get(), x, y, x + ix, y + iy);
    IX_CUR.set(x + ix);
    IY_CUR.set(y + iy);
}

/// Original `scrnPointAbs` (`screenpak.f90:125`).
pub fn scrn_point_abs(ix: i32, iy: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    IX_CUR.set(ix);
    IY_CUR.set(iy);
    plax_circ(ICOLOR.get(), 1, ix, iy);
}

/// Original `scrnPointInc` (`screenpak.f90:135`).
pub fn scrn_point_inc(ix: i32, iy: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    IX_CUR.set(IX_CUR.get() + ix);
    IY_CUR.set(IY_CUR.get() + iy);
    plax_circ(ICOLOR.get(), 1, IX_CUR.get(), IY_CUR.get());
}

/// Original `scrnPoints` (`screenpak.f90:146`, unused in the source too).
pub fn scrn_points(jx: &[i32], jy: &[i32], np: i32) {
    for i in 0..np.max(0) as usize {
        scrn_point_abs(jx[i], jy[i]);
    }
    scrn_update(1);
}

/// Original `scrnLines` (`screenpak.f90:156`, unused in the source too).
pub fn scrn_lines(jx: &[i32], jy: &[i32], np: i32) {
    for i in 0..np.max(0) as usize {
        scrn_vect_abs(jx[i], jy[i]);
    }
    scrn_update(1);
}

/// Original `scrnSymbol` (`screenpak.f90:166`): draws a symbol of the given
/// type at the absolute position.  `itype` is the caller's variable: a
/// positive type is folded into 1..19 in place, as the source does.
pub fn scrn_symbol(ix: i32, iy: i32, itype: &mut i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    let icolor = ICOLOR.get();
    // size was 5 for Parallax - set to 8 for X windows
    let isize: i32 = 8;
    let isizem1 = isize - 1;
    if *itype < 0 {
        let iscale = nint_r4((2 * (isize + 3)) as f32 / 2.5);
        // `write(dummy, '(i6)') - itype` into a `character*6`
        let dummy = format_i(-*itype, 6).into_bytes();
        plax_next_text_align(5);
        plax_sctext(1, iscale, iscale, icolor, ix, iy, &adjustl(&dummy));
        return;
    }
    if *itype > 0 {
        *itype = (*itype - 1) % 19 + 1;
    }
    let mut ivec = [0i16; 12];
    match *itype {
        1 => {
            plax_boxo(icolor, ix - isize, iy - isize, ix + isize, iy + isize);
            plax_boxo(
                icolor,
                ix - isizem1,
                iy - isizem1,
                ix + isizem1,
                iy + isizem1,
            );
        }
        2 => plax_box(icolor, ix - isize, iy - isize, ix + isize, iy + isize),
        3 | 4 => {
            let len = nint_r4(1.414 * isize as f32);
            ivec[0] = (ix + len) as i16;
            ivec[2] = ix as i16;
            ivec[4] = (ix - len) as i16;
            ivec[6] = ix as i16;
            ivec[1] = iy as i16;
            ivec[3] = (iy + len) as i16;
            ivec[5] = iy as i16;
            ivec[7] = (iy - len) as i16;
            if *itype == 4 {
                plax_poly(icolor, 4, &ivec);
            } else {
                plax_polyo(icolor, 4, &ivec);
            }
        }
        5 | 6 | 11 | 12 => {
            ivec[0] = (ix - isize) as i16;
            ivec[2] = (ix + isize) as i16;
            ivec[4] = ix as i16;
            let len_y = nint_r4(1.732 * isize as f32);
            if *itype < 7 {
                let iy_bot = iy - len_y / 3;
                ivec[1] = iy_bot as i16;
                ivec[3] = iy_bot as i16;
                ivec[5] = (iy_bot + len_y) as i16;
            } else {
                let iy_top = iy + len_y / 3;
                ivec[1] = iy_top as i16;
                ivec[3] = iy_top as i16;
                ivec[5] = (iy_top - len_y) as i16;
            }
            if *itype % 2 == 0 {
                plax_poly(icolor, 3, &ivec);
            } else {
                plax_polyo(icolor, 3, &ivec);
            }
        }
        7 => {
            plax_vect(
                icolor,
                ix - isizem1,
                iy - isizem1,
                ix + isizem1,
                iy + isizem1,
            );
            plax_vect(
                icolor,
                ix + isizem1,
                iy - isizem1,
                ix - isizem1,
                iy + isizem1,
            );
        }
        8 => {
            plax_vect(icolor, ix, iy - isize, ix, iy + isize);
            plax_vect(icolor, ix - isize, iy, ix + isize, iy);
        }
        9 | 16 => {
            plax_circo(icolor, isize, ix, iy);
            plax_circo(icolor, isizem1, ix, iy);
        }
        10 | 17 => plax_circ(icolor, isize, ix, iy),
        13 => {
            let len38 = nint_r4(0.76 * isize as f32);
            let len20 = nint_r4(0.32 * isize as f32);
            let len32 = nint_r4(0.56 * isize as f32);
            plax_vect(icolor, ix - len38, iy + isize, ix - len38, iy - len32);
            plax_vect(icolor, ix - len38, iy - len32, ix - len20, iy - isize);
            plax_vect(icolor, ix - len20, iy - isize, ix + len20, iy - isize);
            plax_vect(icolor, ix + len20, iy - isize, ix + len38, iy - len32);
            plax_vect(icolor, ix + len38, iy - len32, ix + len38, iy + isize);
        }
        14 => {
            let len38 = nint_r4(0.76 * isize as f32);
            plax_vect(icolor, ix - len38, iy - isize, ix + len38, iy - isize);
            plax_vect(icolor, ix + len38, iy - isize, ix + len38, iy);
            plax_vect(icolor, ix + len38, iy, ix - len38, iy);
            plax_vect(icolor, ix - len38, iy, ix - len38, iy + isize);
            plax_vect(icolor, ix - len38, iy + isize, ix + len38, iy + isize);
        }
        15 => {
            plax_circo(icolor, isize, ix, iy);
            plax_vect(icolor, ix, iy - isize, ix, iy + isize);
        }
        18 => plax_circ(icolor, isize / 3, ix, iy),
        19 => plax_vect(icolor, ix - isize, iy, ix + isize, iy),
        // The computed GO TO falls through for 0: `return`
        _ => {}
    }
}

/// Original `scrnGridLine` (`screenpak.f90:255`): draws a grid line with
/// ticks at the given starting position and interval.
pub fn scrn_grid_line(ix0: i32, iy0: i32, idx: i32, idy: i32, num_intervals: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    let icolor = ICOLOR.get();
    let tick_size: f32 = 5.;
    let axis_angle = (idy as f32).atan2(idx as f32);
    // Implicit integers: the products are truncated
    let ix_tick = cvttss2si(-tick_size * axis_angle.sin());
    let iy_tick = cvttss2si(tick_size * axis_angle.cos());
    let mut ix_cen = 0;
    let mut iy_cen = 0;
    for i in 0..=num_intervals {
        ix_cen = ix0 + i * idx;
        iy_cen = iy0 + i * idy;
        plax_vect(
            icolor,
            ix_cen - ix_tick,
            iy_cen - iy_tick,
            ix_cen + ix_tick,
            iy_cen + iy_tick,
        );
    }
    plax_vect(icolor, ix0, iy0, ix_cen, iy_cen);
    scrn_update(1);
}

/// Original `scrnlabel` (`screenpak.f90:275`): prints one integer at the
/// current position.
pub fn scrnlabel(number: i32, num_chars: i32) {
    if IF_PLAX_ON.get() <= 0 {
        return;
    }
    let mut dum2 = [b' '; 8];
    // `write(dummy, '(i8)') number`
    let dummy = format_i(number, 8).into_bytes();
    let n = num_chars.clamp(0, 8) as usize;
    dum2[..n].copy_from_slice(&dummy[8 - n..8]);
    plax_sctext(1, 8, 8, ICOLOR.get(), IX_CUR.get(), IY_CUR.get(), &dum2);
}

/// Original `fortgetarg` (`screenpak.f90:288`): argument `i` (0 is the
/// program).
pub fn fortgetarg(i: i32) -> String {
    program_args()
        .get(i.max(0) as usize)
        .cloned()
        .unwrap_or_default()
}

/// Original `fortiargc` (`screenpak.f90:294`).
pub fn fortiargc() -> i32 {
    program_args().len() as i32 - 1
}
