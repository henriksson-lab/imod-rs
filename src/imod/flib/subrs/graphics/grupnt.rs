//! Translation of `IMOD/flib/subrs/graphics/grupnt.f`.

use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, format_i, ld_int, read_list_stdin};
use crate::imod::flib::subrs::hvem::frefor::ListItem;
use std::io::Write as _;

/// Original `grupnt` (`grupnt.f:1`): given `np` points `(xx, yy)`, asks how
/// many contiguous, non-overlapping groups (`ng`) to divide them into and
/// the number of points in each (`nn`), orders the points by X and finds
/// the mean X (`avgx`) and the mean and SD of Y (`avgy`, `sdy`).
///
/// Fixed in translation (BUGS.md, `grupnt`): the source's index is an
/// `integer*2 ind(10000)`, so more than 10000 points write past it and more
/// than 32767 wrap; and counts adding up to more than `np` read index
/// entries never set.  Here the index holds every point, and a group's
/// count is cut to the points that are left.
#[allow(clippy::too_many_arguments)]
pub fn grupnt(
    xx: &[f32],
    yy: &[f32],
    np: i32,
    avgx: &mut [f32],
    avgy: &mut [f32],
    sdy: &mut [f32],
    nn: &mut [i32],
    ng: &mut i32,
) {
    let npu = np.max(0) as usize;
    let mut ind: Vec<usize> = (1..=npu).collect();
    // build an index to order the points by xx
    for i in 1..npu {
        for j in i + 1..=npu {
            if xx[ind[i - 1] - 1] > xx[ind[j - 1] - 1] {
                ind.swap(i - 1, j - 1);
            }
        }
    }
    let mut out = std::io::stdout();
    let _ = write!(out, "{}  points\n", ld_int(np));
    let _ = out.write_all(b" number of groups: ");
    read_list_stdin(&mut [ListItem::Integer(ng)]);
    let npgrp = np / 1.max(*ng);
    let nextra = np - *ng * npgrp;
    for ig in 1..=*ng {
        nn[(ig - 1) as usize] = npgrp;
        if ig <= nextra {
            nn[(ig - 1) as usize] = npgrp + 1;
        }
    }
    let _ = out.write_all(b" # of points in each group (/ for equal #s): ");
    {
        let mut items: Vec<ListItem> = nn[..(*ng).max(0) as usize]
            .iter_mut()
            .map(ListItem::Integer)
            .collect();
        read_list_stdin(&mut items);
    }
    let mut ii = 1usize;
    for ig in 1..=(*ng).max(0) as usize {
        let mut sx: f32 = 0.;
        let mut sy: f32 = 0.;
        let mut sysq: f32 = 0.;
        let left = np - ii as i32 + 1;
        if nn[ig - 1] > left {
            nn[ig - 1] = left.max(0);
        }
        let n = nn[ig - 1];
        for _in in 1..=n {
            let i = ind[ii - 1];
            ii += 1;
            sx += xx[i - 1];
            sy += yy[i - 1];
            sysq += yy[i - 1] * yy[i - 1];
        }
        avgx[ig - 1] = sx / n as f32;
        avgy[ig - 1] = sy / n as f32;
        sdy[ig - 1] = 0.;
        if n > 1 {
            sdy[ig - 1] =
                ((sysq - n as f32 * avgy[ig - 1] * avgy[ig - 1]) / (n as f32 - 1.)).sqrt();
        }
        // `101 format(3f10.4,i5)`
        let _ = write!(
            out,
            "{}{}{}{}\n",
            format_f(avgx[ig - 1] as f64, 10, 4),
            format_f(avgy[ig - 1] as f64, 10, 4),
            format_f(sdy[ig - 1] as f64, 10, 4),
            format_i(nn[ig - 1], 5)
        );
    }
}
