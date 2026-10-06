//! Translation of `IMOD/flib/subrs/graphics/dsaxes.f90`: `scrnAxes`, which
//! finds nice limits for X and Y, draws labelled axes on the screen and
//! returns the scaling, with `axsub` and `formatForAxis`.
//!
//! The formats `formatForAxis` builds at run time (`(i23)`, `(f23.N)`,
//! `(iM)`, `(fM.N)`) are applied by [`write_axis_label`], the runtime's
//! internal `WRITE` with such a format into a `character*25`.

use super::qtplax::{plax_next_text_align, plax_sctext};
use super::scale::scale_multi_div;
use super::screenpak::{scrn_erase, scrn_grid_line, scrn_update};
use crate::imod::flib::subrs::compat::gfortran_rt::{
    adjustl, format_f, format_i, len_trim, nint_r4, nint_r8,
};
use std::io::Write as _;

/// A value written with an axis-label format.
enum LabelValue {
    Int(i32),
    Real(f32),
}

/// `write(label, fmt = frmt, err = ...) value` into a `character*25` with
/// one of the formats [`format_for_axis`] makes: `Ok` with the blank-padded
/// record, `Err` for the `ERR=` branch (a format that does not parse, or an
/// item of the wrong type for the edit descriptor).
fn write_axis_label(frmt: &str, value: LabelValue) -> Result<Vec<u8>, ()> {
    let body = frmt
        .trim_end()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or(())?;
    let text = if let Some(width) = body.strip_prefix('i') {
        let w: usize = width.parse().map_err(|_| ())?;
        match value {
            LabelValue::Int(v) => format_i(v, w),
            LabelValue::Real(_) => return Err(()),
        }
    } else if let Some(spec) = body.strip_prefix('f') {
        let (w, d) = spec.split_once('.').ok_or(())?;
        let w: usize = w.parse().map_err(|_| ())?;
        let d: usize = d.parse().map_err(|_| ())?;
        match value {
            LabelValue::Real(v) => format_f(v as f64, w, d),
            LabelValue::Int(_) => return Err(()),
        }
    } else {
        return Err(());
    };
    // An internal record of 25 characters: a longer field is an error
    if text.len() > 25 {
        return Err(());
    }
    let mut label = text.into_bytes();
    label.resize(25, b' ');
    Ok(label)
}

/// Original `scrnAxes` (`dsaxes.f90:10`): `xmin..xmax`, `ymin..ymax` are the
/// data ranges (the caller's variables, which `scaleCommon` may raise);
/// returns the scaling and the axis start and 1/10 range.
#[allow(clippy::too_many_arguments)]
pub fn scrn_axes(
    x_min: f32,
    x_max: &mut f32,
    y_min: f32,
    y_max: &mut f32,
    x_scale: &mut f32,
    x_add: &mut f32,
    y_scale: &mut f32,
    y_add: &mut f32,
    dx: &mut f32,
    xlo: &mut f32,
    dy: &mut f32,
    ylo: &mut f32,
) {
    let itext_size: i32 = 8;
    let iy_bot: i32 = 80;
    let mut num_xdiv = 0;
    let mut num_ydiv = 0;
    let mut dx_div: f32 = 0.;
    let mut dy_div: f32 = 0.;
    let mut nint_needed_x = 0;
    let mut nint_needed_y = 0;
    let mut max_len_x = 0;
    let mut max_len_y = 0;
    // `character*25 label`, kept across iterations as the source's is
    let mut label: Vec<u8> = vec![b' '; 25];
    scrn_erase(-1);
    axsub(x_min, x_max, dx, xlo, b'X', &mut num_xdiv, &mut dx_div);
    let xformat = format_for_axis(dx_div, *xlo, num_xdiv, &mut nint_needed_x, &mut max_len_x);
    axsub(y_min, y_max, dy, ylo, b'Y', &mut num_ydiv, &mut dy_div);
    let yformat = format_for_axis(dy_div, *ylo, num_ydiv, &mut nint_needed_y, &mut max_len_y);
    let _ = max_len_x;
    let left_x = 15 + 70.max(max_len_y * 18);
    let ix_delta = 920 / num_xdiv;
    let ix_range = num_xdiv * ix_delta;
    let iy_delta = 920 / num_ydiv;
    let iy_range = num_ydiv * iy_delta;
    for i in 0..=num_xdiv / 2 {
        plax_next_text_align(1);
        let value = if nint_needed_x > 0 {
            LabelValue::Int(nint_r4(*xlo + (i * 2) as f32 * dx_div))
        } else {
            LabelValue::Real(*xlo + (i * 2) as f32 * dx_div)
        };
        if let Ok(text) = write_axis_label(&xformat, value) {
            label = text;
        }
        plax_sctext(
            1,
            itext_size,
            itext_size,
            241,
            left_x + i * 2 * ix_delta,
            iy_bot - 15,
            &adjustl(&label),
        );
    }

    for i in 0..=num_ydiv / 2 {
        plax_next_text_align(2);
        let value = if nint_needed_y > 0 {
            LabelValue::Int(nint_r4(*ylo + (i * 2) as f32 * dy_div))
        } else {
            LabelValue::Real(*ylo + (i * 2) as f32 * dy_div)
        };
        if let Ok(text) = write_axis_label(&yformat, value) {
            label = text;
        }
        plax_sctext(
            1,
            itext_size,
            itext_size,
            241,
            left_x - 15,
            iy_bot + i * 2 * iy_delta,
            &label,
        );
    }
    scrn_grid_line(left_x, iy_bot, ix_delta, 0, num_xdiv);
    scrn_grid_line(left_x + ix_range, iy_bot, 0, iy_delta, num_ydiv);
    scrn_grid_line(left_x + ix_range, iy_bot + iy_range, -ix_delta, 0, num_xdiv);
    scrn_grid_line(left_x, iy_bot + iy_range, 0, -iy_delta, num_ydiv);
    scrn_update(0);
    *x_scale = ix_delta as f32 / dx_div;
    *y_scale = iy_delta as f32 / dy_div;
    *x_add = left_x as f32 - *xlo * *x_scale;
    *y_add = iy_bot as f32 - *ylo * *y_scale;
}

/// Original `axsub` (`dsaxes.f90:68`): runs the scaling routine and prints
/// the scaling.
pub fn axsub(
    x_min: f32,
    x_max: &mut f32,
    dx: &mut f32,
    xlo: &mut f32,
    name: u8,
    num_div: &mut i32,
    dx_div: &mut f32,
) {
    scale_multi_div(x_min, x_max, dx, xlo, num_div, dx_div);
    let xhi = 10. * *dx + *xlo;
    // `30 format(1x,a2,' from',f12.5,' to',f12.5,' at',f11.4,' /div.')`: `a2`
    // right-justifies the one character
    let mut out = std::io::stdout();
    let _ = write!(
        out,
        "  {} from{} to{} at{} /div.\n",
        name as char,
        format_f(*xlo as f64, 12, 5),
        format_f(xhi as f64, 12, 5),
        format_f(*dx_div as f64, 11, 4)
    );
}

/// Original `formatForAxis` (`dsaxes.f90:84`): the format for the axis
/// labels (returned, `frmt` in the source), whether it is an integer format
/// and the maximum label length.
pub fn format_for_axis(
    dx: f32,
    xlo: f32,
    num_div: i32,
    nint_needed: &mut i32,
    max_len: &mut i32,
) -> String {
    // Determine needed decimal places from the dx value
    // The 1.7e-3 was 1e-4 and is empirical to avoid 5 decimal places for numbers like
    // 400 to 400.5, or 8 for 4.0082 to 4.0102
    let xlo8 = xlo as f64;
    let dx8 = dx as f64;
    let mut ndec: i32 = 0;
    let tol = 1.7e-3_f32 as f64;
    for ipow in 0..=9 {
        ndec = ipow;
        let dxpow = dx8 * 10_i32.pow(ipow as u32) as f64;
        let xlo_pow = xlo8 * 10_i32.pow(ipow as u32) as f64;
        if (nint_r8(dxpow) as f64 - dxpow).abs() <= dxpow * tol
            && (nint_r8(xlo_pow) as f64 - xlo_pow).abs() <= dxpow * tol
        {
            break;
        }
    }

    // Detremine if integer output is OK and set up preliminary format
    let mut frmt: String;
    if ndec == 0 && xlo < 9.0e8 && xlo + num_div as f32 * dx < 9.0e8 {
        *nint_needed = 1;
        frmt = "(i23)".to_owned();
    } else {
        *nint_needed = 0;
        frmt = format!("(f23.{})", format_i(ndec, 1));
    }

    // print the actual labels to find their maximum lengths, and to see if there is a
    // zero on the ends of all of them
    *max_len = 0;
    let mut all_end_zero = *nint_needed == 0;
    let mut label: Vec<u8> = vec![b' '; 25];
    for ilab in 0..=num_div / 2 {
        if *nint_needed > 0 {
            let mut text = format_i(nint_r4(xlo + (2 * ilab) as f32 * dx), 23).into_bytes();
            text.resize(25, b' ');
            label = text;
        } else if let Ok(text) =
            write_axis_label(&frmt, LabelValue::Real(xlo + (2 * ilab) as f32 * dx))
        {
            label = text;
            let last = len_trim(&label);
            if last == 0 || label[last - 1] != b'0' {
                all_end_zero = false;
            }
        }
        *max_len = (*max_len).max(len_trim(&adjustl(&label)) as i32);
    }

    // If there is a zero, drop it and drop the length
    if all_end_zero && ndec > 0 {
        ndec -= 1;
        *max_len -= 1;
    }

    // Make the real format
    if *max_len < 10 {
        if *nint_needed > 0 {
            frmt = format!("(i{})", format_i(*max_len, 1));
        } else {
            frmt = format!("(f{}.{})", format_i(*max_len, 1), format_i(ndec, 1));
        }
    } else {
        frmt = format!("(f{}.{})", format_i(*max_len, 2), format_i(ndec, 1));
    }
    frmt
}
