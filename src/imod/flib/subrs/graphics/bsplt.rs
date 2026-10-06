//! Translation of `IMOD/flib/subrs/graphics/bsplt.f90`: `bsplt`, which plots
//! one variable against another on the screen, the terminal and the
//! PostScript plot, its entry points `gnplt`, `errplt` and `boxplt`, and
//! `setpos` and `pctile`.
//!
//! Fixed in translation (BUGS.md, `bsplt`): `gnplt` passes a one-element
//! `colx` that it never sets and the fraction-box symbol (-2) reads one
//! entry per point from it; here `gnplt`'s `colx` is zeros, one per point.
//! A symbol number past the 19-entry terminal symbol table prints a blank
//! instead of reading past it.  The tick arrays (`xtick(310)`,
//! `ytick(310)`) hold as many ticks as entered.  Values typed out to a file
//! are appended (the source reads the file to its end and gfortran then
//! refuses the write, as in `bshst`).

use super::bshst::{open_unit8_append, write_unit};
use super::chrout::chrout;
use super::dsaxes::scrn_axes;
use super::flnam::flnam;
use super::fracbx::{ps_frac_box, ps_frac_box_params};
use super::label_axis::label_axis;
use super::minmax::minmax;
use super::plotvars::{PLOTVARS, lookup_color_index};
use super::psdashline::ps_dash_interactive;
use super::psf::{psframe, pssetcolor};
use super::psgrid::{ps_grid_line, ps_log_grid};
use super::psmisc::ps_misc_items;
use super::psplotpak::{ps_move_abs, ps_setup, ps_setup_thick, ps_vect_abs};
use super::pssymbol::{ps_sym_size, ps_symbol};
use super::qtplax::{plax_drawing_scale, plax_mapcolor, plax_next_text_align, plax_sctext};
use super::screenpak::{scrn_change_color, scrn_move_abs, scrn_symbol, scrn_update, scrn_vect_abs};
use crate::imod::flib::subrs::compat::gfortran_rt::{
    cvttss2si, format_f, format_i, nint_r4, read_list_stdin,
};
use crate::imod::flib::subrs::hvem::frefor::ListItem;
use std::cell::Cell;
use std::io::Write as _;

thread_local! {
    /// `data ifTerm/0/, ifType/0/, ifPlot/0/` with `save`.
    static IF_TERM: Cell<i32> = const { Cell::new(0) };
    static IF_TYPE: Cell<i32> = const { Cell::new(0) };
    static IF_PLOT: Cell<i32> = const { Cell::new(0) };
}

/// `data hiSymb/'-', '|', '<', '>', '/', '\\', 'X', '+', 'O', '@', 'v', '^',
/// '3', '4', '5', '6', '7', '8', '9'/` (`-fbackslash`: one backslash).
const HI_SYMB: [u8; 19] = *b"-|<>/\\X+O@v^3456789";

/// `hiSymb(max(1, isym))`, blank past the table (see the module notes).
fn hi_symb(isym: i32) -> u8 {
    let k = 1.max(isym) as usize;
    if k <= HI_SYMB.len() {
        HI_SYMB[k - 1]
    } else {
        b' '
    }
}

/// `psSetColor` to the color of entry `icl` of `icolors`, or black.
fn ps_color_for(icl: i32) {
    if icl > 0 {
        let c = PLOTVARS.with_borrow(|p| p.icolors[(icl - 1) as usize]);
        pssetcolor(c[1], c[2], c[3]);
    } else {
        pssetcolor(0, 0, 0);
    }
}

/// Original `bsplt` (`bsplt.f90:10`).  `isymbol` and `colx` are the
/// caller's arrays: the screen key loop folds `isymbol` entries in place
/// (`scrnSymbol`), and a box plot sorts values into `colx`.
#[allow(clippy::too_many_arguments)]
pub fn bsplt(
    namex: &[i32],
    xx: &[f32],
    yy: &[f32],
    igroup_num: &[i32],
    num_points: i32,
    isymbol: &mut [i32],
    num_groups: i32,
    irecx: &[i32],
    irecy: &[i32],
    colx: &mut [f32],
    coly: &[f32],
    if_log_x: i32,
    if_log_y: i32,
) {
    let xll_stock: [f32; 4] = [0., 0., 5.2, 5.2];
    let yll_stock: [f32; 4] = [5.2, 0., 5.2, 0.];
    let np = num_points.max(0) as usize;
    let mut out = std::io::stdout();
    let (igen_plt_type, if_no_term, zero_line_dash_len, num_colors, screen) =
        PLOTVARS.with_borrow(|p| {
            (
                p.igen_plt_type,
                p.if_no_term,
                p.zero_line_dash_len,
                p.num_colors,
                (p.screen_ymin, p.screen_ymax, p.screen_xmin, p.screen_xmax),
            )
        });
    //
    // Get the range including error bars
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (0f32, 0f32, 0f32, 0f32);
    minmax(xx, num_points, &mut xmin, &mut xmax);
    minmax(yy, num_points, &mut ymin, &mut ymax);
    let mut if_non_zero = 0;
    if igen_plt_type < 0 {
        if_non_zero = 0;
        for i in 0..np {
            ymin = ymin.min(yy[i] - colx[i]);
            ymax = ymax.max(yy[i] + colx[i]);
            if colx[i] != 0. {
                if_non_zero = 1;
            }
        }
    }
    let (screen_ymin, screen_ymax, screen_xmin, screen_xmax) = screen;
    if screen_ymin != 0. || screen_ymax != 0. {
        let aa = 0.01 * (screen_ymax - screen_ymin);
        ymin = screen_ymin + aa;
        ymax = screen_ymax - aa;
    }
    if screen_xmin != 0. || screen_xmax != 0. {
        let aa = 0.01 * (screen_xmax - screen_xmin);
        xmin = screen_xmin + aa;
        xmax = screen_xmax - aa;
    }
    IF_TERM.set(0);
    if if_no_term == 0 {
        let _ = out.write_all(b" 0 for graphics plot, 1 for plot on terminal: ");
        let mut if_term = IF_TERM.get();
        read_list_stdin(&mut [ListItem::Integer(&mut if_term)]);
        IF_TERM.set(if_term);
    }
    let if_term = IF_TERM.get();
    if if_term > 0 {
        chrout(27);
        chrout(b'[' as i32);
        chrout(b'2' as i32);
        chrout(b'J' as i32);
        setpos(21, 0);
    }
    //
    // Draw the axes then the data, not by group
    let (mut xscale, mut xadd, mut yscale, mut yadd) = (0f32, 0f32, 0f32, 0f32);
    let (mut dx, mut xlow, mut dy, mut ylow) = (0f32, 0f32, 0f32, 0f32);
    scrn_axes(
        xmin,
        &mut xmax,
        ymin,
        &mut ymax,
        &mut xscale,
        &mut xadd,
        &mut yscale,
        &mut yadd,
        &mut dx,
        &mut xlow,
        &mut dy,
        &mut ylow,
    );
    let mut isymb_temp: i32 = 0;
    if if_term == 0 {
        plax_drawing_scale(xscale, xadd, yscale, yadd);
        //
        // Draw a zero line if the dash length is set
        if zero_line_dash_len > 0. && ylow < 0. && ylow + 10. * dy > 0. {
            let iy = cvttss2si(yadd);
            let num_dash = cvttss2si(1f32.max(0.5 / zero_line_dash_len));
            for i in 1..=num_dash {
                let xleft = xlow + 10. * dx * (i - 1) as f32 * 2. * zero_line_dash_len;
                let xright = (xlow + 10. * dx).min(xleft + 10. * dx * zero_line_dash_len);
                let mut ix = cvttss2si(xscale * xleft + xadd);
                scrn_move_abs(ix, iy);
                ix = cvttss2si(xscale * xright + xadd);
                scrn_vect_abs(ix, iy);
            }
        }
        //
        // Set colors
        let mut igrp_last = -1;
        for icl in 1..=num_colors {
            let c = PLOTVARS.with_borrow(|p| p.icolors[(icl - 1) as usize]);
            plax_mapcolor(icl + 100, c[1], c[2], c[3]);
        }
        let if_connect = PLOTVARS.with_borrow(|p| p.if_connect);
        //
        // Draw points
        for i in 1..=np {
            let igroup = igroup_num[i - 1];
            if igroup != igrp_last && num_colors > 0 {
                let mut icolor_use = 241;
                let icl = lookup_color_index(1, igroup);
                if icl > 0 {
                    icolor_use = 100 + icl;
                }
                scrn_change_color(icolor_use);
            }
            igrp_last = igroup;
            let ix = cvttss2si(xscale * xx[i - 1] + xadd);
            let iy = cvttss2si(yscale * yy[i - 1] + yadd);
            isymb_temp = isymbol[(igroup - 1) as usize];
            //
            // if the symbol type is 0, draw connected lines instead
            if igen_plt_type < 0 || isymb_temp == 0 || if_connect != 0 {
                if i > 1 && igroup_num[i - 1] == igroup_num[1.max(i - 1) - 1] {
                    scrn_vect_abs(ix, iy);
                } else {
                    scrn_move_abs(ix, iy);
                }
            }
            //
            // Draw symbols and error bars
            if isymb_temp < 0 {
                isymb_temp = -(i as i32);
            }
            scrn_symbol(ix, iy, &mut isymb_temp);
            if igen_plt_type < 0 {
                let idy = cvttss2si(yscale * colx[i - 1]);
                scrn_move_abs(ix, iy + idy);
                scrn_vect_abs(ix, iy - idy);
                scrn_move_abs(ix, iy);
            }
        }
        if num_colors > 0 {
            scrn_change_color(241);
        }
        //
        // Output symbols and keys on right
        let num_keys = PLOTVARS.with_borrow(|p| p.num_keys);
        for i in 1..=num_keys.min(num_groups) {
            let mut icolor_use = 241;
            let icl = lookup_color_index(1, i);
            if icl > 0 {
                icolor_use = 100 + icl;
            }
            scrn_change_color(icolor_use);
            let ix = cvttss2si((xlow + 10. * dx) * xscale + xadd + 30.);
            let iy = cvttss2si((ylow + (10.5 - i as f32) * dy) * yscale + yadd);
            if isymbol[(i - 1) as usize] > 0 {
                scrn_symbol(ix, iy, &mut isymbol[(i - 1) as usize]);
            }
            plax_next_text_align(14);
            let key = PLOTVARS.with_borrow(|p| p.keys[(i - 1) as usize]);
            plax_sctext(1, 8, 8, icolor_use, ix + 20, iy, &key);
        }
        if num_colors > 0 && num_keys > 0 {
            scrn_change_color(241);
        }
        let xaxis_label = PLOTVARS.with_borrow(|p| p.xaxis_label);
        if xaxis_label.iter().any(|&c| c != b' ') {
            plax_next_text_align(3);
            plax_sctext(
                1,
                8,
                8,
                241,
                nint_r4((xlow + 5. * dx) * xscale + xadd),
                2,
                &xaxis_label,
            );
        }
        PLOTVARS.with_borrow_mut(|p| {
            p.num_keys = 0;
            p.xaxis_label = [b' '; 80];
        });
        scrn_update(1);
    } else if if_term > 0 {
        for i in 1..=np {
            let jcol = cvttss2si(7.9999 * (xx[i - 1] - xlow) / dx);
            if igen_plt_type < 0 {
                let mut irow = cvttss2si(2.09995 * (10. - (yy[i - 1] + colx[i - 1] - ylow) / dy));
                setpos(irow, jcol);
                chrout(b'-' as i32);
                irow = cvttss2si(2.09995 * (10. - (yy[i - 1] - colx[i - 1] - ylow) / dy));
                setpos(irow, jcol);
                chrout(b'-' as i32);
            }
            let irow = cvttss2si(2.09995 * (10. - (yy[i - 1] - ylow) / dy));
            setpos(irow, jcol);
            chrout(hi_symb(isymbol[(igroup_num[i - 1] - 1) as usize]) as i32);
        }
        setpos(22, 65);
        for i in 1..=12 {
            chrout(HI_SYMB[i - 1] as i32);
        }
    }

    let _ =
        out.write_all(b" draw n lines (n+1), type (1), to file (-1), connect (-thick-1), no (0): ");
    let mut if_type = IF_TYPE.get();
    read_list_stdin(&mut [ListItem::Integer(&mut if_type)]);
    let mut num_lines = 0;
    let mut if_plot_connect = 0;
    if if_type > 1 {
        num_lines = if_type - 1;
    }
    if if_type < -1 {
        if_plot_connect = -1 - if_type;
    }
    // if not doing errors, default is to connect all, and addition of
    // 100 indicates connect by group; but if doing errors, default is
    // to do by group only, and addition of 100 is needed to specify
    // connect all
    let by_group = (if_plot_connect > 100 && igen_plt_type >= 0)
        || (if_plot_connect > 0 && if_plot_connect < 100 && igen_plt_type < 0);
    if_plot_connect %= 100;
    if if_type.abs() > 1 {
        if_type = 0;
    }
    IF_TYPE.set(if_type);
    let mut iun_out = 6;
    if if_type < 0 {
        iun_out = 8;
        let _ = out.write_all(b" enter output file name\n");
        let filename = flnam(80, 0, "0");
        // `open(8, ..., err = 26)`: on failure, carry on at 26
        let _ = open_unit8_append(&filename);
    }
    // 26
    let _ = out.write_all(b" plot # (6-10), - # to set new x/y range, or no plot (0): ");
    let mut if_plot = IF_PLOT.get();
    read_list_stdin(&mut [ListItem::Integer(&mut if_plot)]);
    IF_PLOT.set(if_plot);
    let mut iabs_plot = if_plot.abs();
    let do_plot = iabs_plot > 5;
    let mut xtick: Vec<f32> = vec![0.; 310];
    let mut ytick: Vec<f32> = vec![0.; 310];
    let mut num_xticks: i32 = 10;
    let mut num_yticks: i32 = 10;
    let mut x_grid_ofs: f32 = 0.;
    let mut y_grid_ofs: f32 = 0.;
    let mut xhigh = xlow + 10. * dx;
    let mut yhigh = ylow + 10. * dy;
    let mut xrange: f32 = 0.;
    let mut yrange: f32 = 0.;
    let mut sym_width: f32 = 0.1;
    let mut err_len: f32 = 0.8 * sym_width;
    let mut tick_size: f32 = 0.05;
    let mut isym_thick: i32 = 1;
    let mut ithick_grid: i32 = 1;
    let mut if_box: i32 = 0;
    let mut units_per_inch: f32 = 0.;
    let mut ithick_tukey: i32 = 1;
    let mut tukey_width: f32 = 0.4;
    let mut tukey_tick: f32 = 0.1;
    let mut tukey_gap: f32 = 0.;
    let (mut xadd_ps, mut yadd_ps) = (xadd, yadd);
    let skip_to_60 = if_plot == 0 && if_type == 0;
    if !skip_to_60 {
        let mut def_scale: f32 = 1.;
        if do_plot {
            let mut width_inch = 0f32;
            let (mut c2, mut c3) = (0f32, 0f32);
            ps_setup(1, &mut width_inch, &mut c2, &mut c3, 0);
            def_scale = 0.74 * width_inch / 7.5;
            iabs_plot -= 5;
        }
        num_xticks = 10;
        num_yticks = 10;
        x_grid_ofs = 0.;
        y_grid_ofs = 0.;
        xhigh = xlow + 10. * dx;
        yhigh = ylow + 10. * dy;
        xrange = 4.7 * def_scale;
        yrange = 4.7 * def_scale;
        sym_width = 0.1;
        err_len = 0.8 * sym_width;
        tick_size = 0.05;
        isym_thick = 1;
        ithick_grid = 1;
        if_box = 0;
        if do_plot {
            //
            // Get new data range, tick definition and thicknesses if negative entry
            if if_plot < 0 {
                // `101 format(' min and max values of x:',2f10.3,',  y:',2f10.3)`
                let _ = write!(
                    out,
                    " min and max values of x:{}{},  y:{}{}\n",
                    format_f(xmin as f64, 10, 3),
                    format_f(xmax as f64, 10, 3),
                    format_f(ymin as f64, 10, 3),
                    format_f(ymax as f64, 10, 3)
                );
                if if_log_x != 0 {
                    let x_lin_min = 10f32.powf(xmin);
                    let x_lin_max = 10f32.powf(xmax);
                    let _ = write!(
                        out,
                        " linear limits of x:{}{}\n",
                        format_f(x_lin_min as f64, 10, 3),
                        format_f(x_lin_max as f64, 10, 3)
                    );
                }
                if if_log_y != 0 {
                    let y_lin_min = 10f32.powf(ymin);
                    let y_lin_max = 10f32.powf(ymax);
                    let _ = write!(
                        out,
                        " linear limits of y:{}{}\n",
                        format_f(y_lin_min as f64, 10, 3),
                        format_f(y_lin_max as f64, 10, 3)
                    );
                }
                let _ = out.write_all(b" lower and upper limits of x, of y, # ticks x and y: ");
                read_list_stdin(&mut [
                    ListItem::Real(&mut xlow),
                    ListItem::Real(&mut xhigh),
                    ListItem::Real(&mut ylow),
                    ListItem::Real(&mut yhigh),
                    ListItem::Integer(&mut num_xticks),
                    ListItem::Integer(&mut num_yticks),
                ]);
                let _ =
                    out.write_all(b" tick and symbol size, grid and symbol thickness, 1 for box: ");
                read_list_stdin(&mut [
                    ListItem::Real(&mut tick_size),
                    ListItem::Real(&mut sym_width),
                    ListItem::Integer(&mut ithick_grid),
                    ListItem::Integer(&mut isym_thick),
                    ListItem::Integer(&mut if_box),
                ]);
                if igen_plt_type < 0 && if_non_zero != 0 {
                    let _ = out.write_all(b" Length of ticks at ends of error bars (inches): ");
                    read_list_stdin(&mut [ListItem::Real(&mut err_len)]);
                }
            }
            //
            // For +/- 10 entry, get size/position and possible grid offsets
            let mut x_lower_left;
            let mut y_lower_left;
            if iabs_plot >= 5 {
                x_lower_left = 0.;
                y_lower_left = 0.;
                let _ = out.write_all(b" X and Y size, lower left X and Y (- to offset grids): ");
                read_list_stdin(&mut [
                    ListItem::Real(&mut xrange),
                    ListItem::Real(&mut yrange),
                    ListItem::Real(&mut x_lower_left),
                    ListItem::Real(&mut y_lower_left),
                ]);
                if x_lower_left < 0. {
                    let _ = out.write_all(b" grid offset in x: ");
                    read_list_stdin(&mut [ListItem::Real(&mut x_grid_ofs)]);
                    x_lower_left = -x_lower_left;
                }
                if y_lower_left < 0. {
                    let _ = out.write_all(b" grid offset in y: ");
                    read_list_stdin(&mut [ListItem::Real(&mut y_grid_ofs)]);
                    y_lower_left = -y_lower_left;
                }
                if isymbol[0] == -2 {
                    let mut frac_box_width: f32 = 0.;
                    let mut frac_box_height: f32 = 0.;
                    let mut frac_box_tick: f32 = 0.;
                    let _ = out.write_all(b" fraction box width, height, tick size: ");
                    read_list_stdin(&mut [
                        ListItem::Real(&mut frac_box_width),
                        ListItem::Real(&mut frac_box_height),
                        ListItem::Real(&mut frac_box_tick),
                    ]);
                    ps_frac_box_params(frac_box_width, frac_box_height, frac_box_tick);
                }
            } else {
                x_lower_left = xll_stock[(iabs_plot - 1) as usize] * def_scale;
                y_lower_left = yll_stock[(iabs_plot - 1) as usize] * def_scale;
            }
            //
            // Box parameters for tukey box plots, ask about new page
            if igen_plt_type == 2 {
                tukey_width = 0.4;
                tukey_tick = 0.1;
                tukey_gap = 0.;
                ithick_tukey = 1;
                let _ = out
                    .write_all(b" Tukey box width, tick size, tick-symbol gap, line thickness: ");
                read_list_stdin(&mut [
                    ListItem::Real(&mut tukey_width),
                    ListItem::Real(&mut tukey_tick),
                    ListItem::Real(&mut tukey_gap),
                    ListItem::Integer(&mut ithick_tukey),
                ]);
            }
            let iabs_num_xtick = num_xticks.abs();
            let iabs_num_ytick = num_yticks.abs();
            ps_sym_size(sym_width);
            ps_setup_thick(ithick_grid);
            let _ = out.write_all(b" new page (0 or 1)?: ");
            let mut if_new_page = 0i32;
            read_list_stdin(&mut [ListItem::Integer(&mut if_new_page)]);
            if if_new_page != 0 {
                psframe();
            }
            xadd_ps = x_lower_left + 0.1;
            yadd_ps = y_lower_left + 0.1;
            //
            // Draw the axes
            if if_log_x == 0 || if_plot > 0 {
                xscale = xrange / (xhigh - xlow);
                ps_grid_line(
                    xadd_ps,
                    yadd_ps - y_grid_ofs,
                    xrange,
                    0.,
                    num_xticks,
                    tick_size,
                );
                if if_box != 0 {
                    ps_grid_line(
                        xadd_ps,
                        yadd_ps + yrange + y_grid_ofs,
                        xrange,
                        0.,
                        num_xticks,
                        -tick_size,
                    );
                }
            } else {
                xlow = xlow.log10();
                xhigh = xhigh.log10();
                let _ = write!(out, "{} x ticks: ", format_i(iabs_num_xtick, 3));
                if xtick.len() < iabs_num_xtick as usize {
                    xtick.resize(iabs_num_xtick as usize, 0.);
                }
                {
                    let mut items: Vec<ListItem> = xtick[..iabs_num_xtick as usize]
                        .iter_mut()
                        .map(ListItem::Real)
                        .collect();
                    read_list_stdin(&mut items);
                }
                xscale = xrange / (xhigh - xlow);
                ps_log_grid(
                    xadd_ps,
                    yadd_ps - y_grid_ofs,
                    xscale,
                    0.,
                    &xtick,
                    num_xticks,
                    tick_size,
                );
                if if_box != 0 {
                    ps_log_grid(
                        xadd_ps,
                        yadd_ps + yrange + y_grid_ofs,
                        xscale,
                        0.,
                        &xtick,
                        num_xticks,
                        -tick_size,
                    );
                }
            }
            if if_log_y == 0 || if_plot >= 0 {
                yscale = yrange / (yhigh - ylow);
                ps_grid_line(
                    xadd_ps - x_grid_ofs,
                    yadd_ps,
                    0.,
                    yrange,
                    num_yticks,
                    tick_size,
                );
                if if_box != 0 {
                    ps_grid_line(
                        xadd_ps + xrange + x_grid_ofs,
                        yadd_ps,
                        0.,
                        yrange,
                        num_yticks,
                        -tick_size,
                    );
                }
            } else {
                ylow = ylow.log10();
                yhigh = yhigh.log10();
                let _ = write!(out, "{} y ticks: ", format_i(iabs_num_ytick, 3));
                if ytick.len() < iabs_num_ytick as usize {
                    ytick.resize(iabs_num_ytick as usize, 0.);
                }
                {
                    let mut items: Vec<ListItem> = ytick[..iabs_num_ytick as usize]
                        .iter_mut()
                        .map(ListItem::Real)
                        .collect();
                    read_list_stdin(&mut items);
                }
                yscale = yrange / (yhigh - ylow);
                ps_log_grid(
                    xadd_ps - x_grid_ofs,
                    yadd_ps,
                    0.,
                    yscale,
                    &ytick,
                    num_yticks,
                    tick_size,
                );
                if if_box != 0 {
                    ps_log_grid(
                        xadd_ps + xrange + x_grid_ofs,
                        yadd_ps,
                        0.,
                        yscale,
                        &ytick,
                        num_yticks,
                        -tick_size,
                    );
                }
            }
            units_per_inch = ps_setup_thick(isym_thick);
        }
    }
    let xadd = xadd_ps;
    let yadd = yadd_ps;
    //
    // Came here if no plots, so need to test for plotting in this section
    // Which does the line fits and tukey box plots
    // 60
    for igroup in 1..=num_groups {
        let mut sx: f32 = 0.;
        let mut sy: f32 = 0.;
        let mut sxsq: f32 = 0.;
        let mut sysq: f32 = 0.;
        let mut sxy: f32 = 0.;
        let mut nn: i32 = 0;
        if do_plot && num_colors > 0 {
            ps_color_for(lookup_color_index(1, igroup));
        }
        for i in 1..=np {
            if igroup_num[i - 1] != igroup {
                continue;
            }
            if if_type != 0 {
                let text = if igen_plt_type == 0 {
                    // `103 format(i3,2i7,2f10.3,i10,2f10.3)`
                    format!(
                        "{}{}{}{}{}{}{}{}\n",
                        format_i(igroup, 3),
                        format_i(namex[i - 1], 7),
                        format_i(irecx[i - 1], 7),
                        format_f(xx[i - 1] as f64, 10, 3),
                        format_f(colx[i - 1] as f64, 10, 3),
                        format_i(irecy[i - 1], 10),
                        format_f(yy[i - 1] as f64, 10, 3),
                        format_f(coly[i - 1] as f64, 10, 3)
                    )
                } else if igen_plt_type > 0 {
                    // `203 format(i3,3f10.3)`
                    format!(
                        "{}{}{}\n",
                        format_i(igroup, 3),
                        format_f(xx[i - 1] as f64, 10, 3),
                        format_f(yy[i - 1] as f64, 10, 3)
                    )
                } else {
                    format!(
                        "{}{}{}{}\n",
                        format_i(igroup, 3),
                        format_f(xx[i - 1] as f64, 10, 3),
                        format_f(yy[i - 1] as f64, 10, 3),
                        format_f(colx[i - 1] as f64, 10, 3)
                    )
                };
                write_unit(iun_out, text.as_bytes());
            }
            sx += xx[i - 1];
            sy += yy[i - 1];
            sxy += xx[i - 1] * yy[i - 1];
            sxsq += xx[i - 1] * xx[i - 1];
            sysq += yy[i - 1] * yy[i - 1];
            nn += 1;
            if igen_plt_type == 2 {
                colx[(nn - 1) as usize] = yy[i - 1];
            }
        }
        if nn <= 1 {
            continue;
        }
        let rnumer = nn as f32 * sxy - sx * sy;
        let denom = nn as f32 * sxsq - sx * sx;
        let radic = denom * (nn as f32 * sysq - sy * sy);
        if radic > 0. {
            let rr = rnumer / radic.sqrt();
            let aa = (sy * sxsq - sx * sxy) / denom;
            let bb = rnumer / denom;
            let mut se: f32 = 0.;
            let term = sysq - aa * sy - bb * sxy;
            if nn > 2 && term >= 0. {
                se = (term / (nn - 2) as f32).sqrt();
            }
            let sa = se * (1. / nn as f32 + (sx * sx / nn as f32) / denom).sqrt();
            let sb = se / (denom / nn as f32).sqrt();
            // `105 format('grp',i3,', n=',i4,', r=',f6.3,', a=',f10.3,', b=',f10.3,
            // ', sa=',f9.3,', sb=',f9.3)`
            let _ = write!(
                out,
                "grp{}, n={}, r={}, a={}, b={}, sa={}, sb={}\n",
                format_i(igroup, 3),
                format_i(nn, 4),
                format_f(rr as f64, 6, 3),
                format_f(aa as f64, 10, 3),
                format_f(bb as f64, 10, 3),
                format_f(sa as f64, 9, 3),
                format_f(sb as f64, 9, 3)
            );
        }
        if igen_plt_type == 2 && nn >= 2 && do_plot {
            //
            // order values in colx
            //
            let n = nn as usize;
            for i in 1..n {
                for j in i + 1..=n {
                    if colx[j - 1] < colx[i - 1] {
                        colx.swap(i - 1, j - 1);
                    }
                }
            }
            //
            // get percentiles and draw box
            //
            ps_setup_thick(ithick_tukey);
            let tukey_adjust = (ithick_tukey - 1) as f32 / units_per_inch;
            let p10 = pctile(colx, nn, 0.10);
            let p25 = pctile(colx, nn, 0.25);
            let p50 = pctile(colx, nn, 0.50);
            let p75 = pctile(colx, nn, 0.75);
            let p90 = pctile(colx, nn, 0.90);
            // `106 format(' grp',i3,', n=',i4,',  10,25,50,75,90%s:',5f9.3)`
            let _ = write!(
                out,
                " grp{}, n={},  10,25,50,75,90%s:{}{}{}{}{}\n",
                format_i(igroup, 3),
                format_i(nn, 4),
                format_f(p10 as f64, 9, 3),
                format_f(p25 as f64, 9, 3),
                format_f(p50 as f64, 9, 3),
                format_f(p75 as f64, 9, 3),
                format_f(p90 as f64, 9, 3)
            );
            let y10 = yscale * (p10 - ylow) + yadd - tukey_adjust;
            let y25 = yscale * (p25 - ylow) + yadd - tukey_adjust;
            let y50 = yscale * (p50 - ylow) + yadd - tukey_adjust;
            let y75 = yscale * (p75 - ylow) + yadd - tukey_adjust;
            let y90 = yscale * (p90 - ylow) + yadd - tukey_adjust;
            let rx = xscale * (sx / nn as f32 - xlow) + xadd;
            let xcen = rx - tukey_adjust;
            let xleft = xcen - tukey_width / 2.;
            let xright = xcen + tukey_width / 2.;
            let tick_left = xcen - tukey_tick / 2.;
            let tick_right = xcen + tukey_tick / 2.;
            ps_move_abs(xleft, y50);
            ps_vect_abs(xright, y50);
            ps_move_abs(xleft, y25);
            ps_vect_abs(xright, y25);
            ps_vect_abs(xright, y75);
            ps_vect_abs(xleft, y75);
            ps_vect_abs(xleft, y25);
            ps_move_abs(xcen, y25);
            ps_vect_abs(xcen, y10);
            ps_move_abs(xcen, y75);
            ps_vect_abs(xcen, y90);
            ps_move_abs(tick_left, y10);
            ps_vect_abs(tick_right, y10);
            ps_move_abs(tick_left, y90);
            ps_vect_abs(tick_right, y90);
            ps_setup_thick(isym_thick);
            //
            // plot points outside 10/90 plus gap
            //
            for i in 1..=n {
                let ytrunc = ylow.max(yhigh.min(colx[i - 1]));
                let ry = yscale * (ytrunc - ylow) + yadd;
                if ry < y10 - tukey_gap || ry > y90 + tukey_gap {
                    ps_symbol(rx, ry, isymbol[(igroup - 1) as usize]);
                }
            }
        }
    }
    if do_plot && num_colors > 0 {
        pssetcolor(0, 0, 0);
    }

    if !do_plot {
        return;
    }
    //
    // Now do non-tukey plotting
    if igen_plt_type != 2 {
        let connect_adjust = (if_plot_connect - 1) as f32 / units_per_inch;
        let sym_connect_gap = PLOTVARS.with_borrow(|p| p.sym_connect_gap);
        let mut igrp_last = -1;
        let mut rx_last: f32 = -1000.;
        let mut ry_last: f32 = -1000.;
        for i in 1..=np {
            let xtrunc = xlow.max(xhigh.min(xx[i - 1]));
            let ytrunc = ylow.max(yhigh.min(yy[i - 1]));
            let do_connect =
                (if_plot_connect > 0) && (i > 1) && (!by_group || igroup_num[i - 1] == igrp_last);
            if num_colors > 0 && igroup_num[i - 1] != igrp_last {
                ps_color_for(lookup_color_index(1, igroup_num[i - 1]));
            }
            igrp_last = igroup_num[i - 1];
            let rx = xscale * (xtrunc - xlow) + xadd;
            let ry = yscale * (ytrunc - ylow) + yadd;
            let rx_adj = rx - connect_adjust;
            let ry_adj = ry - connect_adjust;
            if do_connect {
                ps_setup_thick(if_plot_connect);
                let cut_dist = sym_connect_gap * sym_width;
                let mut frac_cut: f32 = 0.;
                let dist_tot = ((rx_adj - rx_last) * (rx_adj - rx_last)
                    + (ry_adj - ry_last) * (ry_adj - ry_last))
                    .sqrt();
                if isymb_temp != 0 && dist_tot > 1.0e-4 {
                    frac_cut = cut_dist / dist_tot;
                }
                if frac_cut <= 0.45 {
                    let xcut = frac_cut * (rx_adj - rx_last);
                    let ycut = frac_cut * (ry_adj - ry_last);
                    ps_move_abs(rx_last + xcut, ry_last + ycut);
                    ps_vect_abs(rx_adj - xcut, ry_adj - ycut);
                }
                ps_setup_thick(isym_thick);
            }
            rx_last = rx_adj;
            ry_last = ry_adj;
            isymb_temp = isymbol[(igroup_num[i - 1] - 1) as usize];
            if isymb_temp == -2 {
                ps_frac_box(rx, ry, colx[i - 1]);
            } else {
                if isymb_temp < 0 {
                    isymb_temp = -(i as i32);
                }
                ps_symbol(rx, ry, isymb_temp);
            }
            if igen_plt_type < 0 && if_non_zero != 0 && colx[i - 1] >= 0. {
                let y_plus = yscale * (ylow.max(yhigh.min(yy[i - 1] + colx[i - 1])) - ylow) + yadd
                    - connect_adjust;
                let y_minus = yscale * (ylow.max(yhigh.min(yy[i - 1] - colx[i - 1])) - ylow) + yadd
                    - connect_adjust;
                if if_plot_connect > 0 {
                    ps_setup_thick(if_plot_connect);
                }
                ps_move_abs(rx_adj - err_len / 2., y_minus);
                ps_vect_abs(rx_adj + err_len / 2., y_minus);
                ps_move_abs(rx_adj - err_len / 2., y_plus);
                ps_vect_abs(rx_adj + err_len / 2., y_plus);
                ps_move_abs(rx_adj, y_minus);
                ps_vect_abs(rx_adj, y_plus);
                if do_connect {
                    ps_setup_thick(isym_thick);
                }
            }
        }
        if num_colors > 0 {
            pssetcolor(0, 0, 0);
        }
    }
    //
    // Finally draw extra things
    for _il in 1..=num_lines {
        ps_dash_interactive(xscale, xlow, xadd, yscale, ylow, yadd);
    }
    if if_plot < 0 && do_plot {
        label_axis(
            xadd,
            yadd - y_grid_ofs,
            xrange,
            xscale,
            num_xticks.abs(),
            &xtick,
            if_log_x,
            0,
        );
        label_axis(
            xadd - x_grid_ofs,
            yadd,
            yrange,
            yscale,
            num_yticks.abs(),
            &ytick,
            if_log_y,
            1,
        );
        ps_misc_items(xscale, xlow, xadd, xrange, yscale, ylow, yadd, yrange);
    }
}

/// Original `gnplt` (`bsplt.f90:461`).
#[allow(clippy::too_many_arguments)]
pub fn gnplt(
    xx: &[f32],
    yy: &[f32],
    igroup_num: &[i32],
    num_points: i32,
    isymbol: &mut [i32],
    num_groups: i32,
    if_log_x: i32,
    if_log_y: i32,
) {
    let mut colx = vec![0f32; num_points.max(1) as usize];
    let coly = [0f32; 1];
    let irecx = [0i32; 1];
    let irecy = [0i32; 1];
    let namex = [0i32; 1];
    PLOTVARS.with_borrow_mut(|p| p.igen_plt_type = 1);
    bsplt(
        &namex, xx, yy, igroup_num, num_points, isymbol, num_groups, &irecx, &irecy, &mut colx,
        &coly, if_log_x, if_log_y,
    );
}

/// Original `errplt` (`bsplt.f90:474`): `colx` holds the error bar lengths.
#[allow(clippy::too_many_arguments)]
pub fn errplt(
    xx: &[f32],
    yy: &[f32],
    igroup_num: &[i32],
    num_points: i32,
    isymbol: &mut [i32],
    num_groups: i32,
    colx: &mut [f32],
    if_log_x: i32,
    if_log_y: i32,
) {
    let coly = [0f32; 1];
    let irecx = [0i32; 1];
    let irecy = [0i32; 1];
    let namex = [0i32; 1];
    PLOTVARS.with_borrow_mut(|p| p.igen_plt_type = -1);
    bsplt(
        &namex, xx, yy, igroup_num, num_points, isymbol, num_groups, &irecx, &irecy, colx, &coly,
        if_log_x, if_log_y,
    );
}

/// Original `boxplt` (`bsplt.f90:487`): `colx` is the work array the values
/// are sorted in.
#[allow(clippy::too_many_arguments)]
pub fn boxplt(
    xx: &[f32],
    yy: &[f32],
    igroup_num: &[i32],
    num_points: i32,
    isymbol: &mut [i32],
    num_groups: i32,
    colx: &mut [f32],
    if_log_x: i32,
    if_log_y: i32,
) {
    let coly = [0f32; 1];
    let irecx = [0i32; 1];
    let irecy = [0i32; 1];
    let namex = [0i32; 1];
    PLOTVARS.with_borrow_mut(|p| p.igen_plt_type = 2);
    bsplt(
        &namex, xx, yy, igroup_num, num_points, isymbol, num_groups, &irecx, &irecy, colx, &coly,
        if_log_x, if_log_y,
    );
}

/// Original `setpos` (`bsplt.f90:501`): ANSI cursor positioning.
pub fn setpos(irow: i32, jcol: i32) {
    chrout(27);
    chrout(b'[' as i32);
    let mut idig1 = (irow + 1) / 10;
    if idig1 > 0 {
        chrout(idig1 + 48);
    }
    chrout(48 + (irow + 1 - 10 * idig1));
    chrout(b';' as i32);
    let idig2 = (jcol + 1) / 100;
    idig1 = (jcol + 1 - 100 * idig2) / 10;
    if idig2 > 0 {
        chrout(idig2 + 48);
    }
    if idig1 > 0 || idig2 > 0 {
        chrout(idig1 + 48);
    }
    chrout(48 + (jcol + 1 - 10 * idig1 - 100 * idig2));
    chrout(b'H' as i32);
}

/// Original `pctile` (`bsplt.f90:521`): the `p` percentile of the sorted
/// values `x(1..n)`.
pub fn pctile(x: &[f32], n: i32, p: f32) -> f32 {
    let mut v = n as f32 * p + 0.5;
    v = 1f32.max((n as f32).min(v));
    let iv = cvttss2si(v.min(n as f32 - 1.));
    let f = v - iv as f32;
    (1. - f) * x[(iv - 1) as usize] + f * x[iv as usize]
}
