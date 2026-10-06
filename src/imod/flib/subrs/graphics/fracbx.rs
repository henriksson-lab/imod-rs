//! Translation of `IMOD/flib/subrs/graphics/fracbx.f90`: little boxes
//! filled up to a fraction, with common `/bxparm/`.

use super::psplotpak::{EXTRA_LEN, N_THICK, UNIT_LEN, UNITS_PER_INCH, ps_move_abs, ps_vect_abs};
use super::trnc::trnc;
use crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si;
use std::cell::Cell;

thread_local! {
    /// `common /bxparm/ boxWidth, boxHeight, boxTick` with
    /// `data boxWidth, boxHeight, boxTick/.1, .4, .05/`.
    static BOX_WIDTH: Cell<f32> = const { Cell::new(0.1) };
    static BOX_HEIGHT: Cell<f32> = const { Cell::new(0.4) };
    static BOX_TICK: Cell<f32> = const { Cell::new(0.05) };
}

/// Original `psFracBox` (`fracbx.f90:4`): a box at `x, y` (inches) filled
/// up to `frac`.
pub fn ps_frac_box(x: f32, y: f32, frac: f32) {
    let (box_width, box_height, box_tick) = (BOX_WIDTH.get(), BOX_HEIGHT.get(), BOX_TICK.get());
    let xlf = trnc(x - box_width / 2.);
    let xrt = trnc(x + box_width / 2.);
    let ylo = trnc(y - box_height / 2.);
    let yhi = trnc(y + box_height / 2.);
    ps_move_abs(xlf, ylo);
    ps_vect_abs(xrt, ylo);
    ps_vect_abs(xrt, yhi);
    ps_move_abs(xlf, ylo);
    ps_vect_abs(xlf, yhi);
    ps_vect_abs(xrt, yhi);
    ps_move_abs(xlf, y);
    ps_vect_abs(xlf - box_tick, y);
    ps_move_abs(xrt, y);
    ps_vect_abs(xrt + box_tick, y);
    let num_lines = cvttss2si((xrt - xlf) * UNITS_PER_INCH.get());
    let unit_len = UNIT_LEN.get();
    let y_line_lo = ylo + EXTRA_LEN.get();
    let y_line_hi = y_line_lo + frac * ((yhi - unit_len) - y_line_lo);
    let nthk_save = N_THICK.get();
    N_THICK.set(1);
    for i in 1..=num_lines {
        let xx = xlf + i as f32 * unit_len;
        ps_move_abs(xx, y_line_lo);
        ps_vect_abs(xx, y_line_hi);
    }
    N_THICK.set(nthk_save);
}

/// Original `psFracBoxParams` (`fracbx.f90:36`).
pub fn ps_frac_box_params(wdth: f32, height: f32, tick: f32) {
    if wdth != 0. {
        BOX_WIDTH.set(wdth);
    }
    if height != 0. {
        BOX_HEIGHT.set(height);
    }
    if tick != 0. {
        BOX_TICK.set(tick);
    }
}
