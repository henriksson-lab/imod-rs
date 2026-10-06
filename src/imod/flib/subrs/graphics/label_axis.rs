//! Translation of `IMOD/flib/subrs/graphics/label_axis.f90`: labels an axis
//! of the PostScript plot with numeric tick labels and text lines.

use super::psf::pswritetext;
use crate::imod::flib::subrs::compat::gfortran_rt::{
    cvttss2si, len_trim, read_line_stdin, read_list_stdin,
};
use crate::imod::flib::subrs::hvem::frefor::ListItem;
use std::cell::Cell;
use std::io::Write as _;

thread_local! {
    /// `save numLabs, numLines` with `data numLabs/0/, numLines/0/`.
    static NUM_LABS: Cell<i32> = const { Cell::new(0) };
    static NUM_LINES: Cell<i32> = const { Cell::new(0) };
}

/// Original `label_axis` (`label_axis.f90:13`): `x_low, y_low` start the
/// axis, `range` is its length, `scale` relates user units to inches,
/// `num_ticks` is the number of divisions (linear) or ticks (log), `ticks`
/// the tick values of a log axis, `if_yaxis` 0 for X, 1 for a left and -1
/// for a right Y axis.
///
/// Fixed in translation (BUGS.md, `label_axis`): the source holds 31
/// labels of 16 characters and writes past both for more or longer ones;
/// here the label list and each label are as long as entered.
#[allow(clippy::too_many_arguments)]
pub fn label_axis(
    x_low: f32,
    y_low: f32,
    range: f32,
    scale: f32,
    num_ticks: i32,
    ticks: &[f32],
    if_log: i32,
    if_yaxis: i32,
) {
    // modified for PS - Postscript
    let ps_upi: f32 = 107.;
    let mut out = std::io::stdout();
    //
    let axis_name = if if_yaxis > 0 {
        "left"
    } else if if_yaxis < 0 {
        "right"
    } else {
        "bottom"
    };
    //
    // `character*6 axisName`: blank-padded to six
    let _ = write!(
        out,
        " # of numeric and # of text labels for {axis_name:<6} axis: "
    );
    let (mut num_labs, mut num_lines) = (NUM_LABS.get(), NUM_LINES.get());
    read_list_stdin(&mut [
        ListItem::Integer(&mut num_labs),
        ListItem::Integer(&mut num_lines),
    ]);
    NUM_LABS.set(num_labs);
    NUM_LINES.set(num_lines);
    if num_labs > 0 {
        let n = num_labs as usize;
        let mut ipos = vec![0i32; n];
        let mut interval_pos = 0i32;
        let _ = out.write_all(b" Starting tick, # ticks between labels (or 0,0 to enter list): ");
        read_list_stdin(&mut [
            ListItem::Integer(&mut ipos[0]),
            ListItem::Integer(&mut interval_pos),
        ]);
        if interval_pos == 0 {
            let _ = out.write_all(b" Enter #'s of ticks to put labels at: ");
            let mut items: Vec<ListItem> = ipos.iter_mut().map(ListItem::Integer).collect();
            read_list_stdin(&mut items);
        } else {
            for i in 1..n {
                ipos[i] = ipos[i - 1] + interval_pos;
            }
        }
        let _ = out.write_all(b" Enter labels separated by commas or spaces\n");
        let line = read_line_stdin(80);
        let trimmed = len_trim(&line) as i32;
        let char_at = |k: i32| -> u8 {
            if k >= 1 && (k as usize) <= line.len() {
                line[k as usize - 1]
            } else {
                b' '
            }
        };
        let mut has_comma = false;
        for i in 1..=trimmed {
            if char_at(i) == b',' {
                has_comma = true;
            }
        }
        let mut labels: Vec<Vec<u8>> = vec![Vec::new(); n];
        let mut lin_ind: i32 = 1;
        for i in 0..n {
            let lab_start = lin_ind;
            loop {
                let ch = char_at(lin_ind);
                lin_ind += 1;
                if lin_ind == trimmed + 1 {
                    lin_ind += 1;
                }
                if has_comma && ch != b',' && lin_ind <= trimmed {
                    continue;
                }
                if !has_comma && ch != b' ' && ch != 9 && lin_ind <= trimmed {
                    continue;
                }
                break;
            }
            let mut label = Vec::new();
            for j in lab_start..=lin_ind - 2 {
                label.push(char_at(j));
            }
            labels[i] = label;
        }
        //
        let _ = out.write_all(b" Label size and separation from axis: ");
        let (mut size, mut axis_offset) = (0f32, 0f32);
        read_list_stdin(&mut [ListItem::Real(&mut size), ListItem::Real(&mut axis_offset)]);
        let jsize = cvttss2si(size * ps_upi);
        //
        // modification for PS - Postscript
        for ind_lab in 0..n {
            let offset = if if_log == 0 {
                (ipos[ind_lab] - 1) as f32 * range / num_ticks as f32
            } else {
                scale * (ticks[(ipos[ind_lab] - 1) as usize] / ticks[0]).log10()
            };
            if if_yaxis > 0 {
                pswritetext(
                    x_low - axis_offset,
                    y_low + offset,
                    &labels[ind_lab],
                    jsize,
                    0,
                    1,
                );
            } else if if_yaxis < 0 {
                pswritetext(
                    x_low + axis_offset,
                    y_low + offset,
                    &labels[ind_lab],
                    jsize,
                    0,
                    -1,
                );
            } else {
                pswritetext(
                    x_low + offset,
                    y_low - axis_offset - size / 2.,
                    &labels[ind_lab],
                    jsize,
                    0,
                    0,
                );
            }
        }
    }
    //
    let mut cen_offset = range / 2.;
    if if_log != 0 && num_ticks > 0 {
        cen_offset = 0.5 * scale * (ticks[(num_ticks - 1) as usize] / ticks[0]).log10();
    }
    //
    for _ind_line in 1..=num_lines {
        let _ = out.write_all(b" Text size, separation from axis, center offset along axis: ");
        let (mut size, mut axis_offset, mut offset) = (0f32, 0f32, 0f32);
        read_list_stdin(&mut [
            ListItem::Real(&mut size),
            ListItem::Real(&mut axis_offset),
            ListItem::Real(&mut offset),
        ]);
        let jsize = cvttss2si(size * ps_upi);
        let _ = out.write_all(b" Enter text label\n");
        let line = read_line_stdin(80);
        let text = &line[..len_trim(&line)];
        if if_yaxis > 0 {
            pswritetext(
                x_low - axis_offset - size / 2.,
                y_low + cen_offset + offset,
                text,
                jsize,
                90,
                0,
            );
        } else if if_yaxis < 0 {
            pswritetext(
                x_low + axis_offset + size / 2 as f32,
                y_low + cen_offset + offset,
                text,
                jsize,
                90,
                0,
            );
        } else {
            pswritetext(
                x_low + cen_offset + offset,
                y_low - axis_offset - size / 2.,
                text,
                jsize,
                0,
                0,
            );
        }
    }
}
