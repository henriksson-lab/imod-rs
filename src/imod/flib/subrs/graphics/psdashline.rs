//! Translation of `IMOD/flib/subrs/graphics/psdashline.f90`: one solid or
//! dashed line on the PostScript plot, in user units transformed by
//! `x (inches) = xscale * (x - xlo) + xAdd`.

use super::psplotpak::{ps_move_abs, ps_setup_thick, ps_vect_abs};
use crate::imod::flib::subrs::compat::gfortran_rt::read_list_stdin;
use crate::imod::flib::subrs::hvem::frefor::ListItem;
use std::io::Write as _;

/// Original `psDashInteractive` (`psdashline.f90:8`): asks for the line's
/// parameters and draws it.
pub fn ps_dash_interactive(xscale: f32, xlo: f32, x_add: f32, y_scale: f32, ylo: f32, y_add: f32) {
    let mut out = std::io::stdout();
    let _ = out.write_all(
        b" slope, intercept, start and end x (user units), thickness (- to switch x&y), \n",
    );
    let _ = out.write_all(b" dash on and off in inches (0, 0 no dash) : ");
    let (mut b, mut a, mut x_start, mut x_end) = (0f32, 0f32, 0f32, 0f32);
    let mut line_thick = 0i32;
    let (mut dash_on, mut dash_off) = (0f32, 0f32);
    read_list_stdin(&mut [
        ListItem::Real(&mut b),
        ListItem::Real(&mut a),
        ListItem::Real(&mut x_start),
        ListItem::Real(&mut x_end),
        ListItem::Integer(&mut line_thick),
        ListItem::Real(&mut dash_on),
        ListItem::Real(&mut dash_off),
    ]);
    ps_dashed_line(
        xscale, xlo, x_add, y_scale, ylo, y_add, b, a, x_start, x_end, line_thick, dash_on,
        dash_off,
    );
}

/// Original `psDashedLine` (`psdashline.f90:20`).
///
/// Fixed in translation (BUGS.md, `psDashedLine`): with a dash and a
/// negative gap that cancel (or a NaN), the source's loop never advances and
/// hangs; here the line ends when a step does not advance.
#[allow(clippy::too_many_arguments)]
pub fn ps_dashed_line(
    xscale: f32,
    xlo: f32,
    x_add: f32,
    y_scale: f32,
    ylo: f32,
    y_add: f32,
    b: f32,
    a: f32,
    x_start: f32,
    x_end: f32,
    line_thick: i32,
    dash_on: f32,
    dash_off: f32,
) {
    let mut dconv = 1. / (xscale * (1. + (b * y_scale / xscale).powi(2)).sqrt());
    if line_thick < 0 {
        dconv = 1. / (y_scale * (1. + (b * xscale / y_scale).powi(2)).sqrt());
    }
    let dash_on_conv = dconv * dash_on;
    let dash_off_conv = dconv * dash_off;
    ps_setup_thick(line_thick.abs());
    let mut x_from = x_start;
    loop {
        let mut x_to = x_end;
        if dash_on_conv > 0. {
            x_to = x_end.min(x_from + dash_on_conv);
        }
        let mut y_from = a + b * x_from;
        let mut y_to = a + b * x_to;
        let mut x_plot_from = x_from;
        let mut x_plot_to = x_to;
        if line_thick < 0 {
            x_plot_from = y_from;
            x_plot_to = y_to;
            y_from = x_from;
            y_to = x_to;
        }
        ps_move_abs(
            xscale * (x_plot_from - xlo) + x_add,
            y_scale * (y_from - ylo) + y_add,
        );
        ps_vect_abs(
            xscale * (x_plot_to - xlo) + x_add,
            y_scale * (y_to - ylo) + y_add,
        );
        let next = x_to + dash_off_conv;
        if !(next > x_from) {
            break;
        }
        x_from = next;
        if !(x_from < x_end) {
            break;
        }
    }
}
