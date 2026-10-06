//! Translation of `IMOD/flib/subrs/graphics/psmisc.f`: miscellaneous items
//! for publication-quality graphs (text labels, letters in circles, symbols
//! in boxes, solid or dashed lines), positioned in user units, inches, or
//! units relative to the graph frame.

use super::plotvars::{PLOTVARS, lookup_color_index};
use super::psdashline::ps_dashed_line;
use super::psf::{pssetcolor, pswritetext};
use super::psplotpak::{ps_move_abs, ps_setup_thick, ps_vect_abs, ps_vect_inc};
use super::pssymbol::{ps_sym_size, ps_symbol};
use crate::imod::flib::subrs::compat::gfortran_rt::{
    cvttss2si, format_i, len_trim, read_line_stdin, read_list_stdin,
};
use crate::imod::flib::subrs::hvem::frefor::ListItem;
use std::cell::Cell;
use std::io::Write as _;

thread_local! {
    /// `save ntext,ncirc,nsymb,nlines` with `data .../0/`: a `/` in the
    /// entry keeps the last values.
    static NTEXT: Cell<i32> = const { Cell::new(0) };
    static NCIRC: Cell<i32> = const { Cell::new(0) };
    static NSYMB: Cell<i32> = const { Cell::new(0) };
    static NLINES: Cell<i32> = const { Cell::new(0) };
}

/// `psSetColor` to the color of entry `icl` of `icolors`, or black.
fn set_color_index(icl: i32) {
    if icl > 0 {
        let c = PLOTVARS.with_borrow(|p| p.icolors[(icl - 1) as usize]);
        pssetcolor(c[1], c[2], c[3]);
    } else {
        pssetcolor(0, 0, 0);
    }
}

/// Original `psMiscItems` (`psmisc.f:11`).
#[allow(clippy::too_many_arguments)]
pub fn ps_misc_items(
    xscal: f32,
    xlo: f32,
    xad: f32,
    xran: f32,
    yscal: f32,
    ylo: f32,
    yad: f32,
    yran: f32,
) {
    // modified for PS - Postscript
    let pwxupi: f32 = 107.;
    let mut out = std::io::stdout();
    let _ =
        out.write_all(b" Number of text strings, letters in circles, symbols in boxes, lines: ");
    let (mut ntext, mut ncirc, mut nsymb, mut nlines) =
        (NTEXT.get(), NCIRC.get(), NSYMB.get(), NLINES.get());
    read_list_stdin(&mut [
        ListItem::Integer(&mut ntext),
        ListItem::Integer(&mut ncirc),
        ListItem::Integer(&mut nsymb),
        ListItem::Integer(&mut nlines),
    ]);
    NTEXT.set(ntext);
    NCIRC.set(ncirc);
    NSYMB.set(nsymb);
    NLINES.set(nlines);
    for itext in 1..=ntext {
        let _ = write!(
            out,
            " Enter parameters for text string # {}\n",
            format_i(itext, 2)
        );
        let (mut xpos, mut ypos) = (0f32, 0f32);
        cnvrtpos(
            xscal, xlo, xad, xran, yscal, ylo, yad, yran, &mut xpos, &mut ypos,
        );
        let _ = out.write_all(
            b" 0 to center, 1 to right justify, or -1 to left justify to that position: ",
        );
        let mut just = 0i32;
        read_list_stdin(&mut [ListItem::Integer(&mut just)]);
        let _ = out.write_all(b" Character size, orientation in degrees: ");
        let mut size = 0f32;
        let mut jor = 0i32;
        read_list_stdin(&mut [ListItem::Real(&mut size), ListItem::Integer(&mut jor)]);
        let jsize = cvttss2si(pwxupi * size);
        let _ = out.write_all(b" Enter text string\n");
        let line = read_line_stdin(160);
        let icl = lookup_color_index(6, itext);
        set_color_index(icl);
        pswritetext(xpos, ypos, &line[..len_trim(&line)], jsize, jor, just);
    }
    //
    for icirc in 1..=ncirc {
        let _ = write!(
            out,
            " Enter parameters for circled letter # {}\n",
            format_i(icirc, 2)
        );
        let (mut xpos, mut ypos) = (0f32, 0f32);
        cnvrtpos(
            xscal, xlo, xad, xran, yscal, ylo, yad, yran, &mut xpos, &mut ypos,
        );
        let _ = out.write_all(b" Circle diameter and thickness: ");
        let mut circsize = 0f32;
        let mut ithick = 0i32;
        read_list_stdin(&mut [
            ListItem::Real(&mut circsize),
            ListItem::Integer(&mut ithick),
        ]);
        let _ = out.write_all(b" Character size: ");
        let mut size = 0f32;
        read_list_stdin(&mut [ListItem::Real(&mut size)]);
        let jsize = cvttss2si(pwxupi * size);
        let _ = out.write_all(b" Letter: ");
        let letter = read_line_stdin(1);
        //
        // modified for PS - Postscript
        let c2 = ps_setup_thick(ithick);
        let thkofs = 0.5 * (ithick - 1) as f32 / c2;
        pswritetext(
            xpos + thkofs,
            ypos + thkofs - 0.007,
            &letter[..1],
            jsize,
            0,
            0,
        );
        ps_move_abs(xpos + circsize / 2., ypos);
        for i in 0..=32 {
            let theta = i as f32 * 3.14159 / 16.;
            let xx = xpos + 0.5 * circsize * theta.cos();
            let yy = ypos + 0.5 * circsize * theta.sin();
            ps_vect_abs(xx, yy);
        }
    }
    //
    for isymb in 1..=nsymb {
        let _ = write!(
            out,
            " Enter parameters for boxed symbol # {}\n",
            format_i(isymb, 2)
        );
        let (mut xpos, mut ypos) = (0f32, 0f32);
        cnvrtpos(
            xscal, xlo, xad, xran, yscal, ylo, yad, yran, &mut xpos, &mut ypos,
        );
        let _ = out.write_all(
            b" Enter symbol type (0 for none), size, and thickness,\n    and box size (0 for none) and thickness: ",
        );
        let (mut itype, mut size, mut isymthk, mut boxsiz, mut iboxthk) =
            (0i32, 0f32, 0i32, 0f32, 0i32);
        read_list_stdin(&mut [
            ListItem::Integer(&mut itype),
            ListItem::Real(&mut size),
            ListItem::Integer(&mut isymthk),
            ListItem::Real(&mut boxsiz),
            ListItem::Integer(&mut iboxthk),
        ]);
        let icl = lookup_color_index(5, itype);
        set_color_index(icl);
        if itype != 0 {
            ps_sym_size(size);
            ps_setup_thick(isymthk);
            ps_symbol(xpos, ypos, itype);
        }
        if boxsiz != 0. {
            let upi = ps_setup_thick(iboxthk);
            let xyadj = 0.5 * (iboxthk - 1) as f32 / upi;
            ps_move_abs(xpos - xyadj - boxsiz / 2., ypos - xyadj - boxsiz / 2.);
            ps_vect_inc(boxsiz, 0.);
            ps_vect_inc(0., boxsiz);
            ps_vect_inc(-boxsiz, 0.);
            ps_vect_inc(0., -boxsiz);
        }
    }
    //
    pssetcolor(0, 0, 0);
    for iline in 1..=nlines {
        let icl = lookup_color_index(6, -iline);
        set_color_index(icl);
        let _ = write!(out, " Enter parameters for line # {}\n", format_i(iline, 2));
        let _ = out.write_all(
            b" 0 for user units, 1 for absolute inches, or -1 for units relative to frame: ",
        );
        let mut iscltyp = 0i32;
        read_list_stdin(&mut [ListItem::Integer(&mut iscltyp)]);
        let _ = out.write_all(
            b" slope, intercept, start & end x (in those units),      thickness (- to switch x&y),\n",
        );
        let _ = out.write_all(b" dash on and off in inches      (0,0 no dash): ");
        let (mut b, mut a, mut xstr, mut xend) = (0f32, 0f32, 0f32, 0f32);
        let mut linthk = 0i32;
        let (mut dshon, mut dshoff) = (0f32, 0f32);
        read_list_stdin(&mut [
            ListItem::Real(&mut b),
            ListItem::Real(&mut a),
            ListItem::Real(&mut xstr),
            ListItem::Real(&mut xend),
            ListItem::Integer(&mut linthk),
            ListItem::Real(&mut dshon),
            ListItem::Real(&mut dshoff),
        ]);
        if iscltyp > 0 {
            ps_dashed_line(
                1., 0., 0., 1., 0., 0., b, a, xstr, xend, linthk, dshon, dshoff,
            );
        } else if iscltyp < 0 {
            ps_dashed_line(
                xran, 0., xad, yran, 0., yad, b, a, xstr, xend, linthk, dshon, dshoff,
            );
        } else {
            ps_dashed_line(
                xscal, xlo, xad, yscal, ylo, yad, b, a, xstr, xend, linthk, dshon, dshoff,
            );
        }
        if PLOTVARS.with_borrow(|p| p.num_colors) > 0 {
            pssetcolor(0, 0, 0);
        }
    }
}

/// Original `cnvrtpos` (`psmisc.f:128`): reads a position and converts it
/// to inches.
#[allow(clippy::too_many_arguments)]
pub fn cnvrtpos(
    xscal: f32,
    xlo: f32,
    xad: f32,
    xran: f32,
    yscal: f32,
    ylo: f32,
    yad: f32,
    yran: f32,
    xpos: &mut f32,
    ypos: &mut f32,
) {
    let _ = std::io::stdout().write_all(
        b" Enter X and Y coordinates, and 0 if user units\n    or 1 if absolute inches or -1 if relative to graph frame: ",
    );
    let mut iscltyp = 0i32;
    read_list_stdin(&mut [
        ListItem::Real(xpos),
        ListItem::Real(ypos),
        ListItem::Integer(&mut iscltyp),
    ]);
    if iscltyp < 0 {
        *xpos = xad + xran * *xpos;
        *ypos = yad + yran * *ypos;
    } else if iscltyp == 0 {
        *xpos = xscal * (*xpos - xlo) + xad;
        *ypos = yscal * (*ypos - ylo) + yad;
    }
}
