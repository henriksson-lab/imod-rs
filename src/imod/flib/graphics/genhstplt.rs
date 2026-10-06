//! Translation of `IMOD/flib/graphics/genhstplt.f90`: a general-purpose
//! interface to the `bshst` and `bsplt` histogram and X/Y plotting
//! routines, reading columns of data from a file.
//!
//! The program is [`genhstplt`]; `realGraphicsMain`, which `plax_initialize`
//! runs on the plotting thread, is [`real_graphics_main`]; `grupnt3`,
//! `convertColsToTypes` and `convertTypesToCols` are [`grupnt3`],
//! [`convert_cols_to_types`] and [`convert_types_to_cols`].  The program's
//! `go to` structure is a loop over its statement labels ([`Label`]).
//!
//! Fixed in translation (BUGS.md, `genhstplt`): the source's fixed arrays
//! are written past for more data or entries than they hold (data rows,
//! groups, types, selections, the columns-to-types conversion, `grupnt3`'s
//! group counts); here such input stops the program with an error message
//! (data, conversion) or is cut to what the arrays hold.  `grupnt3` reads
//! index entries never set when the counts entered add up to more than the
//! points; here a count is cut to the points left.

use crate::imod::flib::subrs::compat::gfortran_rt::{
    format_f, format_i, ld_int, len_trim, nint_r4, read_line_stdin, read_list_stdin,
    read_list_stdin_err, read_runtime_error,
};
use crate::imod::flib::subrs::graphics::bshst::gnhst;
use crate::imod::flib::subrs::graphics::bsplt::{boxplt, errplt, gnplt};
use crate::imod::flib::subrs::graphics::flnam::flnam;
use crate::imod::flib::subrs::graphics::plotvars::{LIM_COLORS, LIM_KEYS, PLOTVARS};
use crate::imod::flib::subrs::graphics::psplotpak::{pltout, ps_exit};
use crate::imod::flib::subrs::graphics::qtplax::{
    plax_initialize, plax_save_png, plax_save_tiff, plax_wait_for_close,
};
use crate::imod::flib::subrs::graphics::screenpak::{
    reverse_graph_contrast, scrn_close, scrn_open,
};
use crate::imod::flib::subrs::hvem::frefor::{
    ListItem, ListReadError, frefor2, frefor3, list_read,
};
use crate::imod::flib::subrs::statsubs::statfuncs::tvalue;
use crate::imod::libcfshr::b3dutil::exit;
use std::io::{BufRead as _, Write as _};

/// `parameter (MAX_ROWS = 1000000, MAX_AVERAGES = 50000, MAX_GROUPS = 100)`
const MAX_ROWS: usize = 1000000;
const MAX_AVERAGES: usize = 50000;
const MAX_GROUPS: usize = 100;

/// The statement labels `realGraphicsMain` jumps to.
#[derive(Clone, Copy)]
enum Label {
    L5,
    L10,
    L20,
    L30,
    L31,
    L32,
    L50,
    L60,
    L70,
    L90,
    L99,
    L110,
    L120,
    L130,
    L1130,
    L1140,
    L1150,
    L1155,
    L1158,
    L1159,
    L1160,
    L1180,
    L1190,
}

/// Original program `genhstplt` (`genhstplt.f90:10`).
pub fn genhstplt() {
    plax_initialize("genhstplt", real_graphics_main);
}

/// Stops with an error for input the arrays cannot hold (see the module
/// notes).
fn too_much(what: &str) -> ! {
    let _ = std::io::stdout().flush();
    println!();
    println!("ERROR: GENHSTPLT - {what}");
    exit(1);
}

/// `read(1, '(a)') name` from the data file: one record; end of file is the
/// runtime error.
fn read_record(unit: &mut std::io::BufReader<std::fs::File>) -> Vec<u8> {
    let mut line = Vec::new();
    match unit.read_until(b'\n', &mut line) {
        Ok(0) | Err(_) => read_runtime_error(ListReadError::End),
        Ok(_) => {}
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    line.truncate(1024);
    line
}

/// Original `realGraphicsMain` (`genhstplt.f90:15`).
pub fn real_graphics_main(_com_file_arg: &[u8]) {
    let mut dmat = vec![0f32; MAX_ROWS * 10];
    let mut xx = vec![0f32; MAX_ROWS];
    let mut zz = vec![0f32; MAX_ROWS];
    let mut yy = vec![0f32; MAX_ROWS];
    let mut ix_group = vec![0i32; MAX_ROWS];
    let mut itype = vec![0i32; MAX_ROWS];
    // `itypeGroup(MAX_GROUPS, MAX_GROUPS)`: `(i, j)` at `(i-1) + 100*(j-1)`
    let mut itype_group = vec![0i32; MAX_GROUPS * MAX_GROUPS];
    let tg = |i: i32, j: i32| -> usize { (i - 1) as usize + MAX_GROUPS * (j - 1) as usize };
    let mut i_symbol = vec![0i32; MAX_GROUPS];
    let mut num_type_in_grp = vec![0i32; MAX_GROUPS];
    let mut igroup_avg = vec![0i32; MAX_AVERAGES];
    let mut num_in_avg = vec![0i32; MAX_AVERAGES];
    let mut num_combo_field: i32 = 0;
    let mut avg_x = vec![0f32; MAX_AVERAGES];
    let mut avg_y = vec![0f32; MAX_AVERAGES];
    let mut group_sd_y = vec![0f32; MAX_ROWS];
    let mut select_min = vec![0f32; MAX_GROUPS];
    let mut select_max = vec![0f32; MAX_GROUPS];
    let mut combo_field = vec![0f32; MAX_GROUPS];
    let mut icol_select = vec![0i32; MAX_GROUPS];
    let mut if_sel_exclude = vec![0i32; MAX_GROUPS];
    let mut icol_denom: i32 = 0;
    let mut icol_divide: i32 = 0;
    let mut icol_in: i32 = 0;
    let mut icol_with_num: i32 = 0;
    let mut icol_with_sd: i32 = 0;
    let mut if_log_x: i32 = 0;
    let mut if_log_y: i32 = 0;
    let mut if_one_type_per_grp: i32 = 0;
    let mut if_retain: i32 = 0;
    let mut i_group: i32 = 0;
    let mut iopt: i32 = 0;
    let mut num_lin_combo: i32 = 0;
    let mut new_col: i32 = 0;
    let mut nfields: i32 = 0;
    let mut num_avg_groups: i32 = 0;
    let mut num_avg_tot: i32;
    let mut num_data: i32 = 0;
    let mut num_groups: i32 = 0;
    let mut num_types: i32 = 0;
    let mut quotient_hi_lim: f32 = 0.;
    let mut quotient_lo_lim: f32 = 0.;
    let mut x_scale: f32 = 0.;
    let mut y_scale: f32 = 0.;
    let mut err_bar_fac: f32 = 0.;
    let mut filename: Vec<u8> = vec![b' '; 1024];
    let mut out = std::io::stdout();
    //
    let mut if_term: i32 = -1;
    let mut if_types: i32 = 0;
    let mut num_col: i32 = -1;
    let mut num_skip: i32 = 0;
    let mut if_log_xin: i32 = 0;
    let mut num_xvals: i32 = 0;
    let mut base_log: f32 = 0.;
    let mut num_select: i32 = 0;
    let mut if_reverse: i32 = 1;
    let mut if_neg_opt_shown: i32 = 0;
    let mut num_col_orig: i32 = 0;
    let if_no_term = || PLOTVARS.with_borrow(|p| p.if_no_term);

    let mut label = Label::L5;
    loop {
        match label {
            Label::L5 => {
                if if_no_term() == 0 {
                    let _ = out.write_all(
                        b" 0 for Plax screen plots, 1 for terminal only, -1 for always screen [-1]: ",
                    );
                    read_list_stdin(&mut [ListItem::Integer(&mut if_term)]);
                    if if_term < 0 {
                        PLOTVARS.with_borrow_mut(|p| p.if_no_term = 1);
                    }
                }
                if if_no_term() != 0 {
                    if_term = 0;
                }
                //
                reverse_graph_contrast(if_reverse);
                scrn_open(if_term);
                //
                let _ = out.write_all(
                    b" 1 if there are types in first column, 0 if there are no types,\n -1 to make one column be X and make separate types for other columns [0]: ",
                );
                read_list_stdin(&mut [ListItem::Integer(&mut if_types)]);
                //
                let _ = out.write_all(
                    b" Number of columns of data values, 0 if # is on line before data,\n or -1 if there is one line per row of data [-1]: ",
                );
                read_list_stdin(&mut [ListItem::Integer(&mut num_col)]);
                //
                num_skip = 0;
                let _ = out.write_all(b" Number of lines to skip at start of file [0]: ");
                if let Err(error) = read_list_stdin_err(&mut [ListItem::Integer(&mut num_skip)]) {
                    if error == ListReadError::End {
                        read_runtime_error(error);
                    }
                }
                label = Label::L10;
            }
            Label::L10 => {
                filename = flnam(1024, 1, "0");
                let name = String::from_utf8_lossy(&filename[..len_trim(&filename)]).into_owned();
                let Ok(file) = std::fs::File::open(&name) else {
                    label = Label::L10;
                    continue;
                };
                let mut unit = std::io::BufReader::new(file);
                label = Label::L20;
                let skip = num_skip;
                for _ in 1..=skip {
                    let _ = read_record(&mut unit);
                }
                if num_col < 0 {
                    let name_line = read_record(&mut unit);
                    let card = String::from_utf8_lossy(&name_line).into_owned();
                    frefor2(
                        &card,
                        &mut avg_x,
                        &mut itype,
                        &mut nfields,
                        MAX_AVERAGES as i32,
                    );
                    num_col = 0;
                    for i in 1..=nfields {
                        if itype[(i - 1) as usize] > 0 {
                            num_col = i;
                        }
                    }
                    if num_col == 0 || (if_types > 0 && num_col == 1) {
                        println!(
                            " There are not enough numeric values at start of first line of data, try again"
                        );
                        label = Label::L5;
                        continue;
                    }
                    if if_types > 0 {
                        num_col -= 1;
                    }
                    // `rewind(1)`
                    let Ok(file) = std::fs::File::open(&name) else {
                        read_runtime_error(ListReadError::Error);
                    };
                    unit = std::io::BufReader::new(file);
                    for _ in 1..=skip {
                        let _ = read_record(&mut unit);
                    }
                    let _ = write!(out, "{} columns of data\n", format_i(num_col, 5));
                }
                if num_col == 0
                    && let Err(error) = list_read(&mut unit, &mut [ListItem::Integer(&mut num_col)])
                {
                    read_runtime_error(error);
                }
                num_col_orig = 0;
                //
                num_data = 0;
                loop {
                    // 12
                    let i = num_data + 1;
                    let j_start = (i - 1) * num_col + 1;
                    let j_end = i * num_col;
                    if i as usize > MAX_ROWS || j_end.max(0) as usize > MAX_ROWS * 10 {
                        too_much("TOO MANY DATA VALUES FOR ARRAYS");
                    }
                    let result = {
                        let mut items: Vec<ListItem> = Vec::new();
                        if if_types > 0 {
                            items.push(ListItem::Integer(&mut itype[(i - 1) as usize]));
                        }
                        if j_end >= j_start {
                            for value in dmat[(j_start - 1) as usize..j_end as usize].iter_mut() {
                                items.push(ListItem::Real(value));
                            }
                        }
                        list_read(&mut unit, &mut items)
                    };
                    match result {
                        Ok(()) => {}
                        Err(ListReadError::End) => break,
                        Err(error) => read_runtime_error(error),
                    }
                    if if_types <= 0 {
                        itype[(i - 1) as usize] = 1;
                    }
                    num_data = i;
                }
                // 18
                let _ = write!(out, "{} lines of data\n", format_i(num_data, 5));
                drop(unit);
                if if_types < 0 {
                    convert_cols_to_types(
                        &mut dmat,
                        &mut num_col,
                        &mut num_data,
                        &mut itype,
                        &mut num_xvals,
                        &mut if_types,
                        &mut num_col_orig,
                    );
                }
            }
            Label::L20 => {
                if if_types != 0 {
                    let _ = out.write_all(b" # of groups (neg. if one Type/group): ");
                    read_list_stdin(&mut [ListItem::Integer(&mut num_groups)]);
                    if_one_type_per_grp = 0;
                    if num_groups < 0 {
                        num_groups = -num_groups;
                        if_one_type_per_grp = 1;
                        num_types = 1;
                    }
                    if num_groups as usize > MAX_GROUPS {
                        num_groups = MAX_GROUPS as i32;
                    }
                    for j in 1..=num_groups {
                        let _ = write!(out, " for group #{}\n", ld_int(j));
                        if if_one_type_per_grp != 0 {
                            let _ = out.write_all(b" Type # and symbol #: ");
                            let (first, sym) = (tg(1, j), (j - 1) as usize);
                            let mut t = itype_group[first];
                            let mut s = i_symbol[sym];
                            read_list_stdin(&mut [
                                ListItem::Integer(&mut t),
                                ListItem::Integer(&mut s),
                            ]);
                            itype_group[first] = t;
                            i_symbol[sym] = s;
                        } else {
                            let _ = out.write_all(b" # of Types and symbol #: ");
                            let mut s = i_symbol[(j - 1) as usize];
                            read_list_stdin(&mut [
                                ListItem::Integer(&mut num_types),
                                ListItem::Integer(&mut s),
                            ]);
                            i_symbol[(j - 1) as usize] = s;
                            num_types = num_types.min(MAX_GROUPS as i32);
                            let _ = out.write_all(b" Types: ");
                            let start = tg(1, j);
                            let mut items: Vec<ListItem> = itype_group
                                [start..start + num_types.max(0) as usize]
                                .iter_mut()
                                .map(ListItem::Integer)
                                .collect();
                            read_list_stdin(&mut items);
                        }
                        num_type_in_grp[(j - 1) as usize] = num_types;
                    }
                } else {
                    num_groups = 1;
                    itype_group[tg(1, 1)] = 1;
                    num_type_in_grp[0] = 1;
                    let _ = out.write_all(b" Symbol #: ");
                    read_list_stdin(&mut [ListItem::Integer(&mut i_symbol[0])]);
                }
                label = Label::L30;
            }
            Label::L30 => {
                if_retain = 0;
                label = Label::L31;
            }
            Label::L31 => {
                if num_col > 1 {
                    let _ = out.write_all(b" Column number: ");
                    read_list_stdin(&mut [ListItem::Integer(&mut new_col)]);
                } else {
                    new_col = 1;
                }
                if new_col <= 0 || new_col > num_col {
                    label = Label::L20;
                    continue;
                }
                num_lin_combo = 0;
                label = Label::L32;
            }
            Label::L32 => {
                if if_retain == 0 && num_xvals > 0 {
                    if_log_y = if_log_x;
                    for i in 0..num_xvals as usize {
                        yy[i] = xx[i];
                    }
                }
                //
                let _ = out.write_all(
                    b" 1 or 2 to take log or sqr root (-1 if it already is log), base to add: ",
                );
                read_list_stdin(&mut [
                    ListItem::Integer(&mut if_log_xin),
                    ListItem::Real(&mut base_log),
                ]);
                //
                if_log_x = if_log_xin;
                num_xvals = 0;
                for igr in 1..=num_groups {
                    for k in 1..=num_data {
                        let mut in_group = 0;
                        for j in 1..=num_type_in_grp[(igr - 1) as usize] {
                            if itype[(k - 1) as usize] == itype_group[tg(j, igr)] {
                                in_group = 1;
                            }
                        }
                        //
                        // check that it passes all selections too
                        //
                        for isel in 1..=num_select as usize {
                            let sel_val =
                                dmat[((k - 1) * num_col + icol_select[isel - 1] - 1) as usize];
                            if (if_sel_exclude[isel - 1] != 0)
                                != (sel_val < select_min[isel - 1]
                                    || sel_val > select_max[isel - 1])
                            {
                                in_group = 0;
                            }
                        }
                        if in_group > 0 {
                            num_xvals += 1;
                            let n = (num_xvals - 1) as usize;
                            ix_group[n] = igr;
                            if num_lin_combo > 0 {
                                xx[n] = 0.;
                                for j in 1..=num_lin_combo as usize {
                                    xx[n] += avg_x[j - 1]
                                        * dmat
                                            [((k - 1) * num_col + num_in_avg[j - 1] - 1) as usize];
                                }
                            } else {
                                xx[n] = dmat[((k - 1) * num_col + new_col - 1) as usize];
                            }
                            if if_log_x > 0 {
                                if if_log_x == 2 {
                                    xx[n] = (xx[n] + base_log).sqrt();
                                }
                                if if_log_x != 2 {
                                    xx[n] = (xx[n] + base_log).log10();
                                }
                            }
                        }
                    }
                }
                //
                if if_log_x == 2 {
                    if_log_x = 0;
                }
                if_log_x = if_log_x.abs();
                gnhst(&xx, &ix_group, num_xvals, &i_symbol, num_groups, if_log_x);
                //
                if if_neg_opt_shown == 0 {
                    if_neg_opt_shown = 1;
                    let _ = out.write_all(
                        b"Control options also available:\n  -2 to enter X axis label and keys for symbols\n  -3 to reverse display contrast\n  -4 to enter colors for groups in screen display and postscript plots\n  -5 to set indexes of text strings for each color in postscript plots \n  -6 to set gap size when connecting symbols\n  -7/-13 to save screen display to PNG file/TIFF file\n  -8 to wait until window closes then exit\n  -9/-10 to set lower and upper limits of Y/X range in screen plot\n  -11 to remove the input file, -12 to print message\n",
                    );
                }
                label = Label::L50;
            }
            Label::L50 => {
                let _ = out.write_all(
                    b" Enter 1 for new column or 14 for new column to replace Y and retain X,\n       2 for plot of this column versus previous column or 17 with connectors,\n       3 for X/Y plot of averages of groups of points or 11 with column ratios\n       4 to define new groups/symbols,   5 to open a new file,\n       6 or 7 to plot metacode file on screen or printer,   8 to exit program,\n       9 for Tukey box plots,   10 for X/Y plot with error bars using S.D.'s,\n       12 to set columns to select on, 13 for X/Y plot dividing Y by a column\n",
                );
                if if_types == 0 && num_col > 2 {
                    let _ = out.write_all(
                        b"       15 to make a separate type from each column,   16 for ordinal column,\n       18 for new column as linear combination\n",
                    );
                } else if num_col_orig > 0 {
                    let _ = out.write_all(
                        b"       15 to restore columns from types,   16 for ordinal column,\n       18 for new column as linear combination\n",
                    );
                } else {
                    let _ = out.write_all(
                        b"       16 for ordinal column,   18 for new column as linear combination\n",
                    );
                }
                let _ = out.write_all(b"       19 to scale loaded columns for one group\n");
                read_list_stdin(&mut [ListItem::Integer(&mut iopt)]);
                if iopt == -123 {
                    label = Label::L99;
                    continue;
                }
                if iopt == -8 {
                    plax_wait_for_close();
                    label = Label::L99;
                    continue;
                }
                if iopt == -11 {
                    let name =
                        String::from_utf8_lossy(&filename[..len_trim(&filename)]).into_owned();
                    if std::fs::metadata(&name).is_ok() {
                        let _ = std::fs::remove_file(&name);
                    }
                    continue;
                }
                if iopt == -12 {
                    println!(" Enter message line");
                    let name = read_line_stdin(1024);
                    let _ = out.write_all(b" ");
                    let _ = out.write_all(&name[..len_trim(&name)]);
                    let _ = out.write_all(b"\n");
                    continue;
                }
                if iopt == -2 {
                    label = Label::L1155;
                    continue;
                }
                if iopt == -3 {
                    if_reverse = 1 - if_reverse;
                    reverse_graph_contrast(if_reverse);
                    continue;
                }
                if iopt == -4 {
                    label = Label::L1158;
                    continue;
                }
                if iopt == -5 {
                    label = Label::L1159;
                    continue;
                }
                if iopt == -6 {
                    let _ = out
                        .write_all(b" Size of connecting line gap as fraction of symbol width: ");
                    let mut gap = PLOTVARS.with_borrow(|p| p.sym_connect_gap);
                    read_list_stdin(&mut [ListItem::Real(&mut gap)]);
                    PLOTVARS.with_borrow_mut(|p| p.sym_connect_gap = gap);
                    continue;
                }
                if iopt == -7 {
                    let _ = out.write_all(b" Name of file to save PNG to: ");
                    let name = read_line_stdin(1024);
                    plax_save_png(&name);
                    continue;
                }
                if iopt == -13 {
                    let _ = out.write_all(b" Name of file to save TIFF to: ");
                    let name = read_line_stdin(1024);
                    plax_save_tiff(&name);
                    continue;
                }
                if iopt == -9 {
                    let _ = out.write_all(b" Lower and upper limits of Y range: ");
                    let (mut lo, mut hi) = PLOTVARS.with_borrow(|p| (p.screen_ymin, p.screen_ymax));
                    read_list_stdin(&mut [ListItem::Real(&mut lo), ListItem::Real(&mut hi)]);
                    PLOTVARS.with_borrow_mut(|p| {
                        p.screen_ymin = lo;
                        p.screen_ymax = hi;
                    });
                    continue;
                }
                if iopt == -10 {
                    let _ = out.write_all(b" Lower and upper limits of X range: ");
                    let (mut lo, mut hi) = PLOTVARS.with_borrow(|p| (p.screen_xmin, p.screen_xmax));
                    read_list_stdin(&mut [ListItem::Real(&mut lo), ListItem::Real(&mut hi)]);
                    PLOTVARS.with_borrow_mut(|p| {
                        p.screen_xmin = lo;
                        p.screen_xmax = hi;
                    });
                    continue;
                }
                if iopt == 209 {
                    iopt = 7;
                }
                if iopt <= 0 || iopt > 19 {
                    continue;
                }
                // `go to(30, 60, 70, 20, 5, 90, 90, 99, 110, 70, 70, 130, 1130,
                // 1140, 1150, 1160, 60, 1180, 1190) iopt`
                label = [
                    Label::L30,
                    Label::L60,
                    Label::L70,
                    Label::L20,
                    Label::L5,
                    Label::L90,
                    Label::L90,
                    Label::L99,
                    Label::L110,
                    Label::L70,
                    Label::L70,
                    Label::L130,
                    Label::L1130,
                    Label::L1140,
                    Label::L1150,
                    Label::L1160,
                    Label::L60,
                    Label::L1180,
                    Label::L1190,
                ][(iopt - 1) as usize];
            }
            Label::L60 => {
                PLOTVARS.with_borrow_mut(|p| p.if_connect = i32::from(iopt == 17));
                gnplt(
                    &yy,
                    &xx,
                    &ix_group,
                    num_xvals,
                    &mut i_symbol,
                    num_groups,
                    if_log_y,
                    if_log_x,
                );
                label = Label::L50;
            }
            Label::L1140 => {
                if_retain = 1;
                label = Label::L31;
            }
            Label::L70 => {
                let _ = out.write_all(
                    b" For the error bars, enter the # of S.D.'s, or - the # of S.E.M.'s,\n   or a large # for confidence limits at that % level: ",
                );
                read_list_stdin(&mut [ListItem::Real(&mut err_bar_fac)]);
                if iopt == 10 {
                    label = Label::L120;
                    continue;
                }
                if iopt == 11 {
                    let _ = out.write_all(b" Column to divide current column by: ");
                    read_list_stdin(&mut [ListItem::Integer(&mut icol_denom)]);
                }
                num_avg_tot = 0;
                let mut igroup_start: i32 = 1;
                //
                for igr in 1..=num_groups {
                    if num_groups > 1 {
                        let _ = write!(
                            out,
                            "  Define groupings of points for group #{}\n",
                            ld_int(igr)
                        );
                    }
                    //
                    let mut num_in_group: i32 = 0;
                    for i in igroup_start..=num_xvals {
                        if ix_group[(i - 1) as usize] != igr {
                            break;
                        }
                        num_in_group += 1;
                    }
                    if iopt == 11 {
                        let mut ix = 0usize;
                        for k in 1..=num_data {
                            let mut in_group = 0;
                            for j in 1..=num_type_in_grp[(igr - 1) as usize] {
                                if itype[(k - 1) as usize] == itype_group[tg(j, igr)] {
                                    in_group = 1;
                                }
                            }
                            if in_group > 0 {
                                ix += 1;
                                zz[ix - 1] = dmat[((k - 1) * num_col + icol_denom - 1) as usize];
                            }
                        }
                    }
                    //
                    if num_in_group > 0 {
                        let gs = (igroup_start - 1) as usize;
                        let at = num_avg_tot as usize;
                        if iopt == 11 {
                            grupnt3(
                                &yy[gs..],
                                &xx[gs..],
                                &zz,
                                num_in_group,
                                &mut avg_x[at..],
                                &mut avg_y[at..],
                                &mut group_sd_y[at..],
                                &mut num_in_avg,
                                &mut num_avg_groups,
                            );
                        } else {
                            crate::imod::flib::subrs::graphics::grupnt::grupnt(
                                &yy[gs..],
                                &xx[gs..],
                                num_in_group,
                                &mut avg_x[at..],
                                &mut avg_y[at..],
                                &mut group_sd_y[at..],
                                &mut num_in_avg,
                                &mut num_avg_groups,
                            );
                        }
                        //
                        for i in 1..=num_avg_groups.max(0) as usize {
                            let n_avg = num_in_avg[i - 1];
                            let sem_val = group_sd_y[at + i - 1] / (n_avg as f32).sqrt();
                            if err_bar_fac > 30. && n_avg > 1 {
                                let t_crit = tvalue((1. + 0.01 * err_bar_fac) / 2., n_avg - 1);
                                group_sd_y[at + i - 1] = t_crit * sem_val;
                            } else if err_bar_fac >= 0. {
                                group_sd_y[at + i - 1] *= err_bar_fac;
                            } else if n_avg > 1 {
                                group_sd_y[at + i - 1] = -err_bar_fac * sem_val;
                            }
                            igroup_avg[at + i - 1] = igr;
                        }
                        //
                        num_avg_tot += num_avg_groups;
                        igroup_start += num_in_group;
                    }
                }
                errplt(
                    &avg_x,
                    &avg_y,
                    &igroup_avg,
                    num_avg_tot,
                    &mut i_symbol,
                    num_groups,
                    &mut group_sd_y,
                    if_log_y,
                    if_log_x,
                );
                label = Label::L50;
            }
            Label::L110 => {
                group_sd_y[0] = -9999.;
                println!(
                    " Enter an X value for each group, or / to use X values in previous column"
                );
                {
                    let mut items: Vec<ListItem> = group_sd_y[..num_groups.max(0) as usize]
                        .iter_mut()
                        .map(ListItem::Real)
                        .collect();
                    read_list_stdin(&mut items);
                }
                if group_sd_y[0] != -9999. {
                    for i in 0..num_xvals as usize {
                        yy[i] = group_sd_y[(ix_group[i] - 1) as usize];
                    }
                }
                boxplt(
                    &yy,
                    &xx,
                    &ix_group,
                    num_xvals,
                    &mut i_symbol,
                    num_groups,
                    &mut group_sd_y,
                    if_log_y,
                    if_log_x,
                );
                label = Label::L50;
            }
            Label::L120 => {
                if (0. ..=30.).contains(&err_bar_fac) {
                    let _ = out.write_all(b" Column number with S.D.'s: ");
                    read_list_stdin(&mut [ListItem::Integer(&mut icol_with_sd)]);
                } else {
                    let _ = out.write_all(b" Column numbers with S.D.'s and n's: ");
                    read_list_stdin(&mut [
                        ListItem::Integer(&mut icol_with_sd),
                        ListItem::Integer(&mut icol_with_num),
                    ]);
                }
                let mut ix = 0usize;
                for igr in 1..=num_groups {
                    for k in 1..=num_data {
                        let mut in_group = 0;
                        for j in 1..=num_type_in_grp[(igr - 1) as usize] {
                            if itype[(k - 1) as usize] == itype_group[tg(j, igr)] {
                                in_group = 1;
                            }
                        }
                        if in_group > 0 {
                            ix += 1;
                            let sd_val = dmat[((k - 1) * num_col + icol_with_sd - 1) as usize];
                            group_sd_y[ix - 1] = err_bar_fac * sd_val;
                            if err_bar_fac < 0. || err_bar_fac > 30. {
                                let nn_val =
                                    crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si(
                                        dmat[((k - 1) * num_col + icol_with_num - 1) as usize],
                                    );
                                let sem_val = sd_val / (nn_val as f32).sqrt();
                                if nn_val <= 1 {
                                    group_sd_y[ix - 1] = 0.;
                                } else if err_bar_fac < 0. {
                                    group_sd_y[ix - 1] = -err_bar_fac * sem_val;
                                } else {
                                    let t_crit = tvalue((1. + 0.01 * err_bar_fac) / 2., nn_val - 1);
                                    group_sd_y[ix - 1] = t_crit * sem_val;
                                }
                            }
                        }
                    }
                }
                errplt(
                    &yy,
                    &xx,
                    &ix_group,
                    num_xvals,
                    &mut i_symbol,
                    num_groups,
                    &mut group_sd_y,
                    if_log_y,
                    if_log_x,
                );
                label = Label::L50;
            }
            Label::L90 => {
                pltout(7 - iopt);
                label = Label::L50;
            }
            Label::L130 => {
                let _ = out.write_all(b" New column to select on, or 0 to clear selections: ");
                read_list_stdin(&mut [ListItem::Integer(&mut icol_in)]);
                if icol_in <= 0 {
                    num_select = 0;
                    label = Label::L50;
                    continue;
                }
                if num_select as usize >= MAX_GROUPS {
                    too_much("TOO MANY SELECTIONS FOR ARRAYS");
                }
                num_select += 1;
                let n = (num_select - 1) as usize;
                icol_select[n] = icol_in;
                let _ = out.write_all(
                    b" Minimum and maximum values for that column, and 0 to include or \n  1 to exclude values in that range: ",
                );
                read_list_stdin(&mut [
                    ListItem::Real(&mut select_min[n]),
                    ListItem::Real(&mut select_max[n]),
                    ListItem::Integer(&mut if_sel_exclude[n]),
                ]);
                label = Label::L50;
            }
            Label::L1130 => {
                let _ = out.write_all(b" Column number to divide by: ");
                read_list_stdin(&mut [ListItem::Integer(&mut icol_divide)]);
                if icol_divide < 1 || icol_divide > num_col {
                    label = Label::L50;
                    continue;
                }
                let _ = out.write_all(b" Lower and upper limits for quotient (0,0 for none): ");
                read_list_stdin(&mut [
                    ListItem::Real(&mut quotient_lo_lim),
                    ListItem::Real(&mut quotient_hi_lim),
                ]);
                let mut nx_tmp = 0usize;
                for igr in 1..=num_groups {
                    for k in 1..=num_data {
                        let mut in_group = 0;
                        for j in 1..=num_type_in_grp[(igr - 1) as usize] {
                            if itype[(k - 1) as usize] == itype_group[tg(j, igr)] {
                                in_group = 1;
                            }
                        }
                        //
                        // check that it passes all selections too
                        //
                        for isel in 1..=num_select as usize {
                            let sel_val =
                                dmat[((k - 1) * num_col + icol_select[isel - 1] - 1) as usize];
                            if (if_sel_exclude[isel - 1] != 0)
                                != (sel_val < select_min[isel - 1]
                                    || sel_val > select_max[isel - 1])
                            {
                                in_group = 0;
                            }
                        }
                        if in_group > 0 {
                            nx_tmp += 1;
                            zz[nx_tmp - 1] = 0.;
                            let div = dmat[((k - 1) * num_col + icol_divide - 1) as usize];
                            if div.abs() > 1.0e-10 * xx[nx_tmp - 1].abs() {
                                zz[nx_tmp - 1] = xx[nx_tmp - 1] / div;
                                if quotient_lo_lim != 0. || quotient_hi_lim != 0. {
                                    zz[nx_tmp - 1] =
                                        quotient_lo_lim.max(quotient_hi_lim.min(zz[nx_tmp - 1]));
                                }
                            }
                        }
                    }
                }
                gnplt(
                    &yy,
                    &zz,
                    &ix_group,
                    num_xvals,
                    &mut i_symbol,
                    num_groups,
                    if_log_y,
                    if_log_x,
                );
                label = Label::L50;
            }
            Label::L1150 => {
                if num_col_orig > 0 {
                    convert_types_to_cols(
                        &mut dmat,
                        &mut num_col,
                        &mut num_data,
                        &mut if_types,
                        &mut num_col_orig,
                    );
                    num_xvals = 0;
                    label = Label::L20;
                } else {
                    if if_types > 0 || num_col < 2 {
                        label = Label::L50;
                        continue;
                    }
                    convert_cols_to_types(
                        &mut dmat,
                        &mut num_col,
                        &mut num_data,
                        &mut itype,
                        &mut num_xvals,
                        &mut if_types,
                        &mut num_col_orig,
                    );
                    label = if if_types > 0 { Label::L20 } else { Label::L50 };
                }
            }
            Label::L1155 => {
                let _ = out.write_all(b" X axis label (Return for none): ");
                let text = read_line_stdin(80);
                let mut num_keys = PLOTVARS.with_borrow(|p| p.num_keys);
                PLOTVARS.with_borrow_mut(|p| p.xaxis_label.copy_from_slice(&text));
                let _ = out.write_all(b" Number of key strings for next graph: ");
                read_list_stdin(&mut [ListItem::Integer(&mut num_keys)]);
                num_keys = 0.max((LIM_KEYS as i32).min(num_keys));
                PLOTVARS.with_borrow_mut(|p| p.num_keys = num_keys);
                label = Label::L50;
                if num_keys == 0 {
                    continue;
                }
                let _ = write!(
                    out,
                    "Enter the{} strings one per line:\n",
                    format_i(num_keys, 2)
                );
                for i in 0..num_keys as usize {
                    let key = read_line_stdin(80);
                    PLOTVARS.with_borrow_mut(|p| p.keys[i].copy_from_slice(&key));
                }
            }
            Label::L1158 => {
                let _ = out.write_all(b" Number of groups to define color for: ");
                let mut num_colors = PLOTVARS.with_borrow(|p| p.num_colors);
                read_list_stdin(&mut [ListItem::Integer(&mut num_colors)]);
                num_colors = 0.max((LIM_COLORS as i32).min(num_colors));
                PLOTVARS.with_borrow_mut(|p| p.num_colors = num_colors);
                label = Label::L50;
                if num_colors == 0 {
                    continue;
                }
                println!(" For each group, enter group #, red, green, blue values (0-255)");
                let mut icolors = PLOTVARS.with_borrow(|p| p.icolors);
                {
                    let mut items: Vec<ListItem> = Vec::new();
                    for row in icolors[..num_colors as usize].iter_mut() {
                        for value in row[..4].iter_mut() {
                            items.push(ListItem::Integer(value));
                        }
                    }
                    read_list_stdin(&mut items);
                }
                for row in icolors[..num_colors as usize].iter_mut() {
                    row[4] = -1;
                    row[5] = -1;
                    if row[0] > 0 && row[0] <= num_groups {
                        row[4] = i_symbol[(row[0] - 1) as usize];
                    }
                }
                PLOTVARS.with_borrow_mut(|p| p.icolors = icolors);
            }
            Label::L1159 => {
                let num_colors = PLOTVARS.with_borrow(|p| p.num_colors);
                label = Label::L50;
                if num_colors == 0 {
                    continue;
                }
                let _ = out.write_all(
                    b" For each defined color, enter index of text string to apply it to,\n   or -index of line, or 0 for none:",
                );
                let mut icolors = PLOTVARS.with_borrow(|p| p.icolors);
                {
                    let mut items: Vec<ListItem> = icolors[..num_colors as usize]
                        .iter_mut()
                        .map(|row| ListItem::Integer(&mut row[5]))
                        .collect();
                    read_list_stdin(&mut items);
                }
                PLOTVARS.with_borrow_mut(|p| p.icolors = icolors);
            }
            Label::L1160 => {
                if_log_x = 0;
                let mut j = 0;
                for i in 1..=num_xvals as usize {
                    if i > 1 && ix_group[1.max(i - 1) - 1] != ix_group[i - 1] {
                        j = 0;
                    }
                    j += 1;
                    xx[i - 1] = j as f32;
                }
                label = Label::L50;
            }
            Label::L1180 => {
                println!(" Enter series of coefficient,column pairs in one line");
                let name = read_line_stdin(1024);
                let card = String::from_utf8_lossy(&name).into_owned();
                frefor3(
                    &card,
                    &mut combo_field,
                    &mut num_in_avg,
                    0,
                    &mut num_combo_field,
                    MAX_GROUPS as i32,
                );
                label = Label::L50;
                if num_combo_field % 2 > 0 {
                    println!(" You must enter an even number of values");
                    continue;
                }
                num_lin_combo = num_combo_field / 2;
                let mut bad = false;
                for i in 1..=num_lin_combo as usize {
                    avg_x[i - 1] = combo_field[2 * i - 2];
                    num_in_avg[i - 1] = nint_r4(combo_field[2 * i - 1]);
                    if num_in_avg[i - 1] < 1 || num_in_avg[i - 1] > num_col {
                        println!(" Column number out of range:{}", ld_int(num_in_avg[i - 1]));
                        bad = true;
                        break;
                    }
                }
                if bad {
                    continue;
                }
                let _ =
                    out.write_all(b" 1 to replace existing Y and retain X, 0 to roll Y into X: ");
                read_list_stdin(&mut [ListItem::Integer(&mut if_retain)]);
                label = Label::L32;
            }
            Label::L1190 => {
                let _ = out.write_all(b" Group to scale, scaling for values loaded in X and Y: ");
                read_list_stdin(&mut [
                    ListItem::Integer(&mut i_group),
                    ListItem::Real(&mut x_scale),
                    ListItem::Real(&mut y_scale),
                ]);
                for i in 0..num_xvals as usize {
                    if ix_group[i] == i_group {
                        xx[i] *= y_scale;
                        yy[i] *= x_scale;
                    }
                }
                label = Label::L50;
            }
            Label::L99 => {
                scrn_close();
                ps_exit();
            }
        }
    }
}

/// Original `grupnt3` (`genhstplt.f90:575`): given `numToGroup` points
/// `(xx, yy)` with denominators `zz`, asks how many contiguous groups to
/// divide them into and the number of points in each (`numInAvg`), orders
/// the points by X and finds the mean X (`avgX`) and the mean and SD of Y,
/// both divided by the mean denominator (`avgY`, `groupSdY`).
#[allow(clippy::too_many_arguments)]
pub fn grupnt3(
    xx: &[f32],
    yy: &[f32],
    zz: &[f32],
    num_to_group: i32,
    avg_x: &mut [f32],
    avg_y: &mut [f32],
    group_sd_y: &mut [f32],
    num_in_avg: &mut [i32],
    num_avg_grps: &mut i32,
) {
    let n = num_to_group.max(0) as usize;
    let mut ind: Vec<usize> = vec![0; n];
    let mut zsum: f32 = 0.;
    for i in 1..=n {
        ind[i - 1] = i;
        zsum += zz[i - 1];
    }
    // build an index to order the points by xx
    for i in 1..n {
        for j in i + 1..=n {
            if xx[ind[i - 1] - 1] > xx[ind[j - 1] - 1] {
                ind.swap(i - 1, j - 1);
            }
        }
    }
    let mut out = std::io::stdout();
    let _ = write!(out, "{}  points\n", ld_int(num_to_group));
    let _ = out.write_all(b" number of groups: ");
    read_list_stdin(&mut [ListItem::Integer(num_avg_grps)]);
    let thresh_delta = zsum / 1.max(*num_avg_grps) as f32;
    let mut thresh = thresh_delta;
    let mut ncum: i32 = 0;
    let mut igrp: i32 = 1;
    let mut zcum: f32 = 0.;
    for ii in 1..=n {
        let i = ind[ii - 1];
        if zcum + zz[i - 1] >= thresh {
            thresh += thresh_delta;
            if let Some(slot) = num_in_avg.get_mut((igrp - 1) as usize) {
                *slot = ii as i32 - ncum;
            }
            ncum = ii as i32;
            igrp += 1;
        }
        zcum += zz[i - 1];
    }
    if igrp == *num_avg_grps
        && let Some(slot) = num_in_avg.get_mut((igrp - 1) as usize)
    {
        *slot = num_to_group - ncum;
    }
    let _ = out.write_all(b" # of points in each group (/ for equal denominators): ");
    {
        let count = (*num_avg_grps).clamp(0, num_in_avg.len() as i32) as usize;
        let mut items: Vec<ListItem> = num_in_avg[..count]
            .iter_mut()
            .map(ListItem::Integer)
            .collect();
        read_list_stdin(&mut items);
    }
    let mut ii = 1usize;
    for igrp in 1..=(*num_avg_grps).max(0) as usize {
        let mut xsum: f32 = 0.;
        let mut ysum: f32 = 0.;
        let mut zsum: f32 = 0.;
        let mut sum_ysq: f32 = 0.;
        let left = num_to_group - ii as i32 + 1;
        if num_in_avg[igrp - 1] > left {
            num_in_avg[igrp - 1] = left.max(0);
        }
        let ntmp = num_in_avg[igrp - 1];
        for _in in 1..=ntmp {
            let i = ind[ii - 1];
            ii += 1;
            xsum += xx[i - 1];
            ysum += yy[i - 1];
            zsum += zz[i - 1];
            sum_ysq += yy[i - 1] * yy[i - 1];
        }
        avg_x[igrp - 1] = xsum / ntmp as f32;
        let avg_yin_grp = ysum / ntmp as f32;
        let avg_z = zsum / ntmp as f32;
        avg_y[igrp - 1] = avg_yin_grp / avg_z;
        let mut sdy_grp: f32 = 0.;
        if ntmp > 1 {
            sdy_grp =
                ((sum_ysq - ntmp as f32 * (avg_yin_grp * avg_yin_grp)) / (ntmp as f32 - 1.)).sqrt();
        }
        group_sd_y[igrp - 1] = sdy_grp / avg_z;
        // `101 format(6f10.4,i5)`
        let _ = write!(
            out,
            "{}{}{}{}{}{}{}\n",
            format_f(avg_x[igrp - 1] as f64, 10, 4),
            format_f(avg_yin_grp as f64, 10, 4),
            format_f(sdy_grp as f64, 10, 4),
            format_f(avg_z as f64, 10, 4),
            format_f(avg_y[igrp - 1] as f64, 10, 4),
            format_f(group_sd_y[igrp - 1] as f64, 10, 4),
            format_i(num_in_avg[igrp - 1], 5)
        );
    }
}

/// Original `convertColsToTypes` (`genhstplt.f90:646`): makes one column X
/// and the others separate types, in place in `dmat`.
pub fn convert_cols_to_types(
    dmat: &mut [f32],
    num_col: &mut i32,
    num_data: &mut i32,
    itype: &mut [i32],
    num_xvals: &mut i32,
    if_types: &mut i32,
    num_col_orig: &mut i32,
) {
    let mut out = std::io::stdout();
    let _ = out.write_all(b" Enter column number to become X (first column), or 0 to abort: ");
    let mut ix_col = 0i32;
    read_list_stdin(&mut [ListItem::Integer(&mut ix_col)]);
    *if_types = 0;
    if ix_col <= 0 || ix_col > *num_col {
        return;
    }
    let (nc, nd) = (*num_col as usize, *num_data as usize);
    if 2 * nd * nc > dmat.len() || nd * nc > itype.len() {
        too_much("TOO MANY DATA VALUES TO CONVERT COLUMNS TO TYPES");
    }
    // `dmatIn(numCol, numData)`, `dmatOut(2, numData, numCol)`,
    // `itype(numData, numCol)`
    let tmp_mat: Vec<f32> = dmat[..nc * nd].to_vec();
    for j in 1..=nc {
        for k in 1..=nd {
            dmat[2 * (k - 1) + 2 * nd * (j - 1)] = tmp_mat[(ix_col as usize - 1) + nc * (k - 1)];
            dmat[1 + 2 * (k - 1) + 2 * nd * (j - 1)] = tmp_mat[(j - 1) + nc * (k - 1)];
            itype[(k - 1) + nd * (j - 1)] = j as i32;
        }
    }
    *num_data *= *num_col;
    *num_col_orig = *num_col;
    *num_col = 2;
    *num_xvals = 0;
    *if_types = 1;
    let _ = write!(
        out,
        "Data are now in two columns with types from 1 to{}\n",
        format_i(*num_col_orig, 4)
    );
}

/// Original `convertTypesToCols` (`genhstplt.f90:673`): restores the
/// original columns from the types `convertColsToTypes` made.
pub fn convert_types_to_cols(
    dmat: &mut [f32],
    num_col: &mut i32,
    num_data: &mut i32,
    if_types: &mut i32,
    num_col_orig: &mut i32,
) {
    let nco = *num_col_orig as usize;
    let nd = *num_data as usize / nco;
    // `dmatIn(2, numData / numColOrig, numColOrig)`, `dmatOut(numColOrig,
    // numData / numColOrig)`
    let tmp_mat: Vec<f32> = dmat[..2 * nd * nco].to_vec();
    *num_data /= *num_col_orig;
    *num_col = *num_col_orig;
    for j in 1..=nco {
        for k in 1..=nd {
            dmat[(j - 1) + nco * (k - 1)] = tmp_mat[1 + 2 * (k - 1) + 2 * nd * (j - 1)];
        }
    }
    *num_col_orig = 0;
    *if_types = 0;
    let _ = write!(
        std::io::stdout(),
        "Data have been restored to the original{} columns\n",
        format_i(*num_col, 4)
    );
}
