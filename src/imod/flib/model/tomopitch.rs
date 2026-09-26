//! Translation of `IMOD/flib/model/tomopitch.f90`.
//!
//! The main program maps to [`tomopitch`] and the three external subroutines
//! to [`add_line_pair`], [`analyze_spots`] and [`findshift`].  The
//! `fortmodel` module arrays are the [`FortModel`] that `readw_or_imod`
//! fills, passed by reference to the units that `use fortmodel`; the
//! assumed-size array arguments `xcen(ipBase)` etc. are passed as the slices
//! starting at that element.
//!
//! Degree trigonometry follows the reference build: `sind`/`cosd` are the
//! libgfortran `_gfortran_sind_r4`/`_gfortran_cosd_r4` calls
//! ([`gfortran_sind_r4`]), and `atand` is inlined by gfortran as
//! `atanf(x) * (180/pi)` with the folded constant `0x42652ee0` that
//! `tomopitch.o`'s `.rodata.cst4` holds.  Formatted output uses gfortran
//! `Fw.d`/`Iw` editing; list-directed `print *` writes an integer as a blank
//! and `I11` and a `real*4` in the `G16.9`-like list form with its blank
//! separator.

use crate::imod::flib::subrs::compat::gfortran_rt::{
    gfortran_cosd_r4, gfortran_sind_r4, maxss, minss,
};
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::objtocont::objtocont;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::model::fortmodel::FortModel;
use crate::imod::flib::subrs::model::readw_or_imod::readw_or_imod;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libcfshr::parse_params::{pip_get_float, pip_number_of_entries};
use crate::imod::libcfshr::simplestat::{ls_fit, ls_fit2};
use crate::imod::libimod::imodel_fwrap::{getimodhead, getimodmaxes, getimodscales, getimodtimes};
use std::io::{BufRead, Write};

/// `parameter (LIMTOT = 500)` (`tomopitch.f90:20`).
const LIMTOT: i32 = 500;
/// `parameter (numOptions = 9)` (`tomopitch.f90:50`).
const NUM_OPTIONS: i32 = 9;
/// Fallback PIP table `options(1)` (`tomopitch.f90:52-55`).
const OPTIONS: &str = "model:ModelFile:FNM:@extra:ExtraThickness:F:@spacing:SpacingInY:F:@\
scale:ScaleFactor:F:@angle:AngleOffsetOld:F:@zshift:ZShiftOld:F:@\
xtilt:XAxisTiltOld:F:@param:ParameterFile:PF:@help:usage:B:";

/// `(180/pi)` as gfortran folds it for the inline `atand` (`atanf(x) * c`):
/// the `.rodata.cst4` constant `0x42652ee0` in `tomopitch.o`.
const ATAND_FACTOR: f32 = f32::from_bits(0x42652ee0);

/// gfortran `Fw.d` output editing: overflow is `w` asterisks, a leading zero
/// is dropped when that is what makes the value fit, `d = 0` keeps the
/// decimal point.
fn fmt_f(value: f32, w: usize, d: usize) -> String {
    if value.is_nan() {
        return format!("{:>w$}", "NaN");
    }
    if value.is_infinite() {
        let mut text = if value < 0. { "-Infinity" } else { "Infinity" };
        if text.len() > w {
            text = if value < 0. { "-Inf" } else { "Inf" };
        }
        if text.len() > w {
            return "*".repeat(w);
        }
        return format!("{text:>w$}");
    }
    let mut text = format!("{value:.d$}");
    if d == 0 {
        text.push('.');
    }
    if text.len() > w {
        if let Some(rest) = text.strip_prefix("0.") {
            text = format!(".{rest}");
        } else if let Some(rest) = text.strip_prefix("-0.") {
            text = format!("-.{rest}");
        }
    }
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// gfortran `Iw` output editing.
fn fmt_i(value: i32, w: usize) -> String {
    let text = format!("{value}");
    if text.len() > w {
        return "*".repeat(w);
    }
    format!("{text:>w$}")
}

/// Original program `tomopitch` (`tomopitch.f90:17`).
///
/// TOMOPITCH analyzes simple models of the boundaries of the section in
/// slices from a tomogram and recommends how much to change tilt angles to
/// make the section flat, how much to shift the tilt axis in Z to produce
/// centered slices, and how thick to make the slices.  It can also recommend
/// how much X-axis tilt is needed to make the section flat in the orthogonal
/// direction as well.  It can also be used with a model drawn on a whole
/// tomogram, possibly binned down.
pub fn tomopitch() {
    //
    let mut in_file = String::new();
    //
    let mut use_times: bool;
    let mut xcen = [0.0_f32; LIMTOT as usize];
    let mut ycen = [0.0_f32; LIMTOT as usize];
    let mut thick_mid = [0.0_f32; LIMTOT as usize];
    let mut y_samp = [0.0_f32; LIMTOT as usize];
    let mut if_use = [0_i32; LIMTOT as usize];
    let mut itimes: Vec<i32> = Vec::new();
    let mut iobj_vert = [0_i32; 2];
    let mut iobj_horiz = [0_i32; LIMTOT as usize];
    let mut ind_horiz = [0_i32; LIMTOT as usize];
    let mut y_vert = [0.0_f32; 2];
    let mut z_span_vert = [0.0_f32; 2];
    let mut zmean = [0.0_f32; LIMTOT as usize];
    let mut ymean = [0.0_f32; LIMTOT as usize];
    // `character*120 message`
    let mut message: String;
    //
    let num_patch: i32;
    let mut num_files = 0_i32;
    let mut ifile: i32;
    let mut ip_base: i32;
    let mut ierr: i32;
    let mut if_flip = 0_i32;
    let mut iobj: i32;
    let mut j: i32;
    let mut indy: i32;
    let mut indz: i32;
    let (mut maxx, mut maxy, mut maxz) = (0_i32, 0_i32, 0_i32);
    let mut min_time: i32;
    let mut max_time: i32;
    let mut num_objects: i32;
    // Set on every path that reaches `addLinePair`.
    let mut iobj1 = 0_i32;
    let mut iobj2 = 0_i32;
    let mut border: f32;
    let mut delta_y: f32;
    let mut y_sample: f32;
    let (mut xy_scale, mut z_scale) = (0.0_f32, 0.0_f32);
    let (mut x_offset, mut y_offset, mut z_offset) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut x_im_scale, mut y_im_scale, mut z_im_scale) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut scale_fac: f32;
    let mut yval: f32;
    let mut z_val: f32;
    let mut z_span: f32;
    let mut diff_min: f32;
    let mut xend: f32;
    let mut y_end: f32;
    let mut z_end: f32;
    let mut y_line: f32;
    let mut zline: f32;
    let mut max_ysample: f32;
    let mut alpha_add: f32;
    let mut theta_add = 0.0_f32;
    let mut shift_add = 0.0_f32;
    // Read only when entered (`ifNo* == 0`).
    let mut alpha_old = 0.0_f32;
    let mut theta_old = 0.0_f32;
    let mut shift_old = 0.0_f32;
    let mut min_ysample: f32;
    let mut if_no_alpha: i32;
    let mut if_no_theta: i32;
    let mut if_no_shift: i32;
    let mut num_line_pairs: i32;
    let mut num_vertical: i32;
    let mut num_horiz: i32;
    let mut ip1: i32;
    let mut ip2: i32;
    let mut imod_obj = 0_i32;
    let mut imod_cont = 0_i32;
    let mut no_xaxis_tilt: bool;
    //
    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    // `use fortmodel`
    let mut fm = FortModel::default();

    // `read(5,*)` with no `END=`/`ERR=`: the gfortran runtime reports a
    // failed read and stops with status 2.
    let read_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        exit(2);
    };
    // A list-directed `real*4` item (its blank separator, or the record's
    // leading blank, included): `F` form with nine significant digits and
    // four trailing blanks for magnitudes in [0.1, 1e9), `E` form with a
    // two-digit exponent otherwise; `NaN`/`Infinity` right-justified in 17.
    let ld_real = |value: f32| -> String {
        if value.is_nan() {
            return format!("{:>17}", "NaN");
        }
        if value.is_infinite() {
            return format!("{:>17}", if value < 0. { "-Infinity" } else { "Infinity" });
        }
        if value == 0. {
            return format!("{:>13}    ", format!("{value:.8}"));
        }
        let scientific = format!("{:.8e}", value.abs());
        let (mantissa, power) = scientific.split_once('e').unwrap();
        let k = power.parse::<i32>().unwrap() + 1;
        if (0..=9).contains(&k) {
            let mut text = format!("{:.*}", (9 - k) as usize, value);
            if k == 9 {
                text.push('.');
            }
            format!("{text:>13}    ")
        } else {
            let e = k - 1;
            format!(
                "{:>17}",
                format!(
                    "{}{}E{}{:02}",
                    if value < 0. { "-" } else { "" },
                    mantissa,
                    if e < 0 { '-' } else { '+' },
                    e.abs()
                )
            )
        }
    };
    // `p_coord(k, i)`, `object(k)`, `ibase_obj(i)`, `npt_in_obj(i)`
    let pc = |fm: &FortModel, k: i32, i: i32| fm.p_coord[(i - 1) as usize][(k - 1) as usize];
    let object = |fm: &FortModel, k: i32| fm.object[(k - 1) as usize];
    let ibase_obj = |fm: &FortModel, i: i32| fm.ibase_obj[(i - 1) as usize];
    let npt_in_obj = |fm: &FortModel, i: i32| fm.npt_in_obj[(i - 1) as usize];
    // `character*120` assignment
    let to_message = |text: String| -> String {
        let mut end = text.len().min(120);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text[..end].to_owned()
    };

    num_patch = 2;
    scale_fac = 1.;
    delta_y = 0.;
    border = 2.;
    if_no_alpha = 1;
    if_no_theta = 1;
    if_no_shift = 1;
    fm.fm_mod_size_type = 2;
    no_xaxis_tilt = false;
    min_ysample = 1.0e30;
    max_ysample = -1.0e30;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "tomopitch",
        "ERROR: TOMOPITCH - ",
        true,
        1,
        0,
        0,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;
    if pip_input {
        ierr = pip_number_of_entries(b"ModelFile", &mut num_files);
        if num_files == 0 {
            exit_error("No model file entered");
        }
        ierr = pip_get_float(b"SpacingInY", &mut delta_y);
        ierr = pip_get_float(b"ExtraThickness", &mut border);
        ierr = pip_get_float(b"ScaleFactor", &mut scale_fac);
        if_no_alpha = pip_get_float(b"XAxisTiltOld", &mut alpha_old);
        if_no_theta = pip_get_float(b"AngleOffsetOld", &mut theta_old);
        if_no_shift = pip_get_float(b"ZShiftOld", &mut shift_old);
        ierr = pip_get_logical("NoXAxisTilt", &mut no_xaxis_tilt);
    } else {
        print!(" Additional thickness to add outside model lines: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Real(&mut border)],
        ) {
            read_abort(err);
        }
        //
        print!(" For analysis of X-axis tilt, enter distance between sample tomograms: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Real(&mut delta_y)],
        ) {
            read_abort(err);
        }
        //
        print!(" Number of model files: ");
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(
            &mut std::io::stdin().lock(),
            &mut [ListItem::Integer(&mut num_files)],
        ) {
            read_abort(err);
        }
    }
    if delta_y > 0. && num_files > 0 {
        println!("\n MODEL FILES OR TIMES MUST OCCUR IN ORDER OF INCREASING SAMPLE COORDINATE\n");
    }
    if delta_y < 0. && num_files > 0 {
        println!("\n MODEL FILES OR TIMES MUST OCCUR IN ORDER OF DECREASING SAMPLE COORDINATE\n");
    }

    ip_base = 1;
    y_sample = -delta_y * (num_files as f32 - 1.) / 2.;
    num_vertical = 0;
    num_horiz = 0;
    num_line_pairs = 0;
    indy = 2;
    indz = 3;

    ifile = 1;
    use_times = false;

    // Loop on the files; this turns into a loop on times if times are discovered
    while ifile <= num_files {
        // First time or if no times and not drawing in whole tomo, read a file
        if !use_times && num_vertical == 0 && num_horiz == 0 {
            if pip_input {
                // `PipGetString('ModelFile', inFile)`: the `pip_fwrap.c:206`
                // wrapper into the `character*320 inFile` (`tomopitch.f90:23`),
                // which fails and exits for a longer entry.
                let mut value = [b' '; 320];
                if crate::imod::libcfshr::pip_fwrap::pipgetstring_(b"ModelFile", &mut value) == 0 {
                    in_file = crate::imod::libcfshr::b3dutil::fortran_string(&value);
                }
            } else {
                print!(" Name of model file #{}: ", fmt_i(ifile, 2));
                let _ = std::io::stdout().flush();
                let mut line = String::new();
                if matches!(std::io::stdin().lock().read_line(&mut line), Ok(0) | Err(_)) {
                    eprintln!("Fortran runtime error: End of file");
                    exit(2);
                }
                in_file = line.trim_end_matches(['\r', '\n']).to_owned();
            }
            if !readw_or_imod(in_file.trim_end_matches(' '), &mut fm) {
                use_times = std::path::Path::new(in_file.trim_end_matches(' ')).exists();
                if !use_times {
                    println!(
                        "\nERROR: TOMOPITCH - Model file {} does not exist - did you save it \
                         from 3dmod?",
                        in_file.trim_end_matches(' ')
                    );
                } else {
                    println!(
                        "\nERROR: TOMOPITCH - Reading model file {}",
                        in_file.trim_end_matches(' ')
                    );
                }
                exit(1);
            }
            if itimes.is_empty() {
                itimes = vec![0; fm.max_obj_num as usize];
                memory_error(0, "array for object times");
            }

            ierr = getimodhead(
                &mut xy_scale,
                &mut z_scale,
                &mut x_offset,
                &mut y_offset,
                &mut z_offset,
                &mut if_flip,
            );
            ierr = getimodmaxes(&mut maxx, &mut maxy, &mut maxz);
            ierr = getimodscales(&mut x_im_scale, &mut y_im_scale, &mut z_im_scale);
            //
            // convert to centered index coordinates - maxes are correct
            for i in 1..=fm.n_point {
                let p = &mut fm.p_coord[(i - 1) as usize];
                p[0] = (p[0] - x_offset) / x_im_scale - (maxx / 2) as f32;
                p[1] = (p[1] - y_offset) / y_im_scale - (maxy / 2) as f32;
                p[2] = (p[2] - z_offset) / z_im_scale - (maxz / 2) as f32;
            }
            //
            // if there is one file, see if there are multiple times
            if num_files == 1 {
                ierr = getimodtimes(&mut itimes);
                min_time = 100000;
                max_time = 0;
                for iobj in 1..=fm.max_mod_obj {
                    if npt_in_obj(&fm, iobj) > 0 {
                        min_time = min_time.min(itimes[(iobj - 1) as usize]);
                        max_time = max_time.max(itimes[(iobj - 1) as usize]);
                    }
                }
                if max_time > 0 {
                    use_times = true;
                    num_files = max_time;
                    //
                    // DNM 11/11/03: adjust ySample here, not below
                    y_sample = -delta_y * (num_files as f32 - 1.) / 2.;
                    if min_time == 0 {
                        println!(
                            "\nERROR: TOMOPITCH - The model file has multiple times but has \
                             some contours with no time index. Either contours with time or \
                             ones without should be eliminated"
                        );
                        exit(1);
                    }
                } else {
                    //
                    // now check for whether there are multiple lines in one model
                    // 2/3/05: DO NOT swap y and Z if y is the long dimension,
                    // data are already flipped back so Z is between-slice dimension
                    //
                    for iobj in 1..=fm.max_mod_obj {
                        objtocont(iobj, &fm.obj_color, &mut imod_obj, &mut imod_cont);
                        if npt_in_obj(&fm, iobj) > 2 {
                            println!(
                                "\nERROR: TOMOPITCH - Contour{} in object{} has more than 2 \
                                 points",
                                fmt_i(imod_cont, 5),
                                fmt_i(imod_obj, 3)
                            );
                            exit(1);
                        }
                        if npt_in_obj(&fm, iobj) > 1 {
                            ip1 = object(&fm, ibase_obj(&fm, iobj) + 1);
                            ip2 = object(&fm, ibase_obj(&fm, iobj) + npt_in_obj(&fm, iobj));
                            yval = 0.5 * (pc(&fm, indy, ip1) + pc(&fm, indy, ip2));
                            z_val = 0.5 * (pc(&fm, indz, ip1) + pc(&fm, indz, ip2));
                            z_span = (pc(&fm, indz, ip1) - pc(&fm, indz, ip2)).abs();
                            if (pc(&fm, 1, ip1) - pc(&fm, 1, ip2)).abs() < 0.2 * z_span {
                                num_vertical = num_vertical + 1;
                                if num_vertical <= 2 {
                                    iobj_vert[(num_vertical - 1) as usize] = iobj;
                                    y_vert[(num_vertical - 1) as usize] = yval;
                                    z_span_vert[(num_vertical - 1) as usize] = z_span;
                                }
                            } else if z_span < 0.2 * (pc(&fm, 1, ip1) - pc(&fm, 1, ip2)).abs() {
                                num_horiz = num_horiz + 1;
                                if num_horiz > LIMTOT {
                                    exit_error("Too many lines for arrays");
                                }
                                let h = (num_horiz - 1) as usize;
                                iobj_horiz[h] = iobj;
                                zmean[h] = z_val;
                                ymean[h] = yval;
                                ind_horiz[h] = num_horiz;
                            } else {
                                println!(
                                    "\nERROR: TOMOPITCH - Contour{} in object{} is too diagonal \
                                     to analyze",
                                    fmt_i(imod_cont, 5),
                                    fmt_i(imod_obj, 3)
                                );
                                exit(1);
                            }
                        }
                    }
                    //
                    // Now require 2 horiz & 2 vert, or even number of horiz lines
                    //
                    if num_vertical > 0 && num_vertical != 2 && num_horiz != 2 {
                        exit_error(
                            "To use crossed lines, you must have 2 horizontal and 2 vertical \
                             lines",
                        );
                    }
                    if num_horiz % 2 != 0 {
                        exit_error("You must have an even number of lines in a single model file");
                    }
                    let zm = |k: i32| zmean[(k - 1) as usize];
                    if num_vertical > 0 {
                        //
                        // for crossed lines, set up for 3 "files" and handle below
                        // adjust horizontal index to  match top & bottom
                        //
                        num_files = 3;
                        if (y_vert[0] > y_vert[1] && ymean[0] < ymean[1])
                            || (y_vert[0] < y_vert[1] && ymean[0] > ymean[1])
                        {
                            ind_horiz[0] = 2;
                            ind_horiz[1] = 1;
                        }
                        // `min(zSpanVert(1), zSpanVert(2))` (`tomopitch.f90:240`):
                        // `minss zSpanVert(1), zSpanVert(2)` in the reference object.
                        if (zmean[0] - zmean[1]).abs() > 0.3 * minss(z_span_vert[0], z_span_vert[1])
                        {
                            exit_error(
                                "Distance between horizontal lines must be less than 0.3 times \
                                 length of vertical ones",
                            );
                        }
                    } else if num_horiz > 2 {
                        //
                        // for multiple horizontal lines, sort them by Z and make sure
                        // they make sense
                        //
                        num_files = num_horiz / 2;
                        for i in 1..=num_horiz - 1 {
                            for j in i + 1..=num_horiz {
                                let (iu, ju) = ((i - 1) as usize, (j - 1) as usize);
                                if zm(ind_horiz[iu]) > zm(ind_horiz[ju]) {
                                    iobj = ind_horiz[iu];
                                    ind_horiz[iu] = ind_horiz[ju];
                                    ind_horiz[ju] = iobj;
                                }
                            }
                        }

                        for i in 1..=num_files {
                            j = 2 * i;
                            let ih = |k: i32| ind_horiz[(k - 1) as usize];
                            diff_min = 1.0e10;
                            if i > 1 {
                                // `tomopitch.f90:262-263`: the reference object folds
                                // `diffMin = 1.e10` and emits `minss diff, 1.e10`.
                                diff_min = minss(zm(ih(j - 1)) - zm(ih(j - 2)), diff_min);
                            }
                            if i < num_files {
                                // `tomopitch.f90:264-265`: `minss diffMin, diff`.
                                diff_min = minss(diff_min, zm(ih(j + 1)) - zm(ih(j)));
                            }
                            if zm(ih(j)) - zm(ih(j - 1)) > 0.3 * diff_min {
                                exit_error(
                                    "The spacing between two lines of a pair must be less than \
                                     0.3 times the spacing between pairs",
                                );
                            }
                        }
                    } else {
                        //
                        // otherwise, need to restore indexes
                        //
                        indy = 2;
                        indz = 3;
                    }
                }
            }
        }
        if num_files * num_patch > LIMTOT {
            exit_error("Too many total patches for arrays");
        }
        //
        // now get the first 2 contours in model or time or next pair
        // of horizontal lines, or make up lines in crossed case
        //
        if num_vertical > 0 && ifile > 1 {
            //
            // shift the horizontal lines to the endpoints of the vertical ones
            //
            y_sample = 0.;
            for i in 1..=2 {
                //
                // get appropriate endpoint of line
                //
                let iv = iobj_vert[(i - 1) as usize];
                ip1 = object(&fm, ibase_obj(&fm, iv) + 1);
                ip2 = object(&fm, ibase_obj(&fm, iv) + npt_in_obj(&fm, iv));
                if (ifile == 2 && pc(&fm, indz, ip1) > pc(&fm, indz, ip2))
                    || (ifile == 3 && pc(&fm, indz, ip1) < pc(&fm, indz, ip2))
                {
                    ip1 = ip2;
                }
                xend = pc(&fm, 1, ip1);
                y_end = pc(&fm, indy, ip1);
                z_end = pc(&fm, indz, ip1);
                //
                // solve for y and z coordinate of existing horizontal line at
                // that X value, and shift Y and Z of endpoints to move that
                // point to the desired endpoint
                //
                iobj = iobj_horiz[(ind_horiz[(i - 1) as usize] - 1) as usize];
                ip1 = object(&fm, ibase_obj(&fm, iobj) + 1);
                ip2 = object(&fm, ibase_obj(&fm, iobj) + npt_in_obj(&fm, iobj));
                y_line = pc(&fm, indy, ip1)
                    + (xend - pc(&fm, 1, ip1)) * (pc(&fm, indy, ip2) - pc(&fm, indy, ip1))
                        / (pc(&fm, 1, ip2) - pc(&fm, 1, ip1));
                zline = pc(&fm, indz, ip1)
                    + (xend - pc(&fm, 1, ip1)) * (pc(&fm, indz, ip2) - pc(&fm, indz, ip1))
                        / (pc(&fm, 1, ip2) - pc(&fm, 1, ip1));
                let (ky, kz) = ((indy - 1) as usize, (indz - 1) as usize);
                let (u1, u2) = ((ip1 - 1) as usize, (ip2 - 1) as usize);
                fm.p_coord[u1][ky] = fm.p_coord[u1][ky] + y_end - y_line;
                fm.p_coord[u2][ky] = fm.p_coord[u2][ky] + y_end - y_line;
                fm.p_coord[u1][kz] = fm.p_coord[u1][kz] + z_end - zline;
                fm.p_coord[u2][kz] = fm.p_coord[u2][kz] + z_end - zline;
                y_sample = y_sample + 0.5 * z_end;
            }
            iobj1 = iobj_horiz[0];
            iobj2 = iobj_horiz[1];
            num_objects = 2;
            message = format!(
                "lines moved to Y ={}",
                fmt_f(y_sample + maxz as f32 / 2., 7, 0)
            );
        } else if num_horiz > 2 || num_vertical > 0 {
            //
            // get pair of horizontal lines
            //
            num_objects = 2;
            ip1 = ind_horiz[(2 * ifile - 2) as usize];
            ip2 = ind_horiz[(2 * ifile - 1) as usize];
            iobj1 = iobj_horiz[(ip1 - 1) as usize];
            iobj2 = iobj_horiz[(ip2 - 1) as usize];
            y_sample = (zmean[(ip1 - 1) as usize] + zmean[(ip2 - 1) as usize]) / 2.;
            // `tomopitch.f90:339-340`: `minss ySample, minYsample` and
            // `maxss maxYsample, ySample` in the reference object.
            min_ysample = minss(y_sample, min_ysample);
            max_ysample = maxss(max_ysample, y_sample);
            message = format!(
                "line pair at Y ={}",
                fmt_f(y_sample + maxz as f32 / 2., 7, 0)
            );
        } else if use_times {
            //
            // get lines at next time
            //
            num_objects = 0;
            for iobj in 1..=fm.max_mod_obj {
                if npt_in_obj(&fm, iobj) > 0
                    && itimes[(iobj - 1) as usize] == ifile
                    && num_objects <= 2
                {
                    num_objects = num_objects + 1;
                    if num_objects == 1 {
                        iobj1 = iobj;
                    }
                    if num_objects == 2 {
                        iobj2 = iobj;
                    }
                }
            }
            if num_objects > 2 {
                println!(
                    "\nERROR: TOMOPITCH - There are more than two contours at time index{}",
                    fmt_i(ifile, 3)
                );
                exit(1);
            }
            message = format!("time index{}", fmt_i(ifile, 3));
        } else {
            //
            // or just get first two object
            //
            num_objects = 2.min(fm.max_mod_obj);
            iobj1 = 1;
            iobj2 = 2;
            message = in_file.clone();
        }
        message = to_message(message);

        if num_objects > 0 || !use_times {
            if num_objects < 2 {
                println!(
                    "\nERROR: TOMOPITCH - Model or time{} does not have enough contours",
                    fmt_i(ifile, 3)
                );
                exit(1);
            }
            if npt_in_obj(&fm, iobj1) < 2 || npt_in_obj(&fm, iobj2) < 2 {
                println!(
                    "\nERROR: TOMOPITCH - There are fewer than two points in one of the first \
                     two contours of model or time{}",
                    fmt_i(ifile, 3)
                );
                exit(1);
            }

            add_line_pair(
                &message,
                iobj1,
                iobj2,
                scale_fac * y_sample,
                border,
                scale_fac,
                indy,
                maxx,
                &mut xcen,
                &mut ycen,
                &mut thick_mid,
                &mut if_use,
                &mut y_samp,
                num_patch,
                &mut ip_base,
                &fm,
            );
            num_line_pairs = num_line_pairs + 1;
        }
        y_sample = y_sample + delta_y;
        ifile = ifile + 1;
    }
    //
    // DNM 6/25/02: need to make the ySamp values symmetric around zero
    // i.e., assume that the samples are symmetric in data set
    // 11/11/03: remove adjustment, which must have broken the 3-model case
    //
    message = "all files".to_owned();
    if use_times {
        message = "all time indexes".to_owned();
    }
    println!(
        "{}{}{:>12}",
        ld_real(min_ysample),
        ld_real(max_ysample),
        num_horiz
    );
    if !use_times
        && num_vertical == 0
        && num_horiz > 2
        // `min(10., abs(deltaY / 10.))` (`tomopitch.f90:402`): `minss |.|, 10.`.
        && max_ysample - min_ysample < minss((delta_y / 10.).abs(), 10.0_f32)
    {
        exit_error(
            "Sample lines are not separated enough in Y - did contour times get turned off?",
        );
    }

    // BRT is looking for 'all line pairs'
    if num_horiz > 2 || num_vertical > 0 {
        message = "all line pairs".to_owned();
        delta_y = 1.;
    }
    alpha_add = 0.;
    if num_files == 1 || (use_times && num_line_pairs < 2) || no_xaxis_tilt {
        delta_y = 0.;
    }
    if no_xaxis_tilt {
        if_no_alpha = 1;
    }
    analyze_spots(
        message.trim_end_matches(' '),
        &xcen,
        &ycen,
        &thick_mid,
        &if_use,
        ip_base - 1,
        &y_samp,
        delta_y,
        &mut alpha_add,
        &mut theta_add,
        &mut shift_add,
    );
    if if_no_alpha == 0 || if_no_theta == 0 || if_no_shift == 0 {
        println!();
    }

    // BRT is looking for each of these tags up to the dash
    // BRT is looking for numbers after "Original" in these outputs
    if if_no_alpha == 0 {
        println!(
            " X axis tilt -  Original:{}   Added:{}   Total:{}",
            fmt_f(alpha_old, 8, 2),
            fmt_f(alpha_add, 8, 2),
            fmt_f(alpha_old + alpha_add, 8, 2)
        );
    }
    if if_no_theta == 0 {
        println!(
            " Angle offset - Original:{}   Added:{}   Total:{}",
            fmt_f(theta_old, 8, 2),
            fmt_f(theta_add, 8, 2),
            fmt_f(theta_old + theta_add, 8, 2)
        );
    }
    if if_no_shift == 0 {
        println!(
            " Z shift -      Original:{}   Added:{}   Total:{}",
            fmt_f(shift_old, 8, 1),
            fmt_f(shift_add, 8, 1),
            fmt_f(shift_old + shift_add, 8, 1)
        );
    }
    exit(0);
}

/// Original `subroutine addLinePair(message, iobj1, iobj2, ySample, border,
/// scaleFac, indy, maxx, xcen, ycen, thickMid, ifUse, ySamp, numPatch,
/// ipBase)` (`tomopitch.f90:423`).
///
/// The arrays are the whole host arrays; the source's `xcen(ipBase)` etc.
/// are the elements from `ipBase` on.
pub fn add_line_pair(
    message: &str,
    iobj1: i32,
    iobj2: i32,
    y_sample: f32,
    border: f32,
    scale_fac: f32,
    indy: i32,
    maxx: i32,
    xcen: &mut [f32],
    ycen: &mut [f32],
    thick_mid: &mut [f32],
    if_use: &mut [i32],
    y_samp: &mut [f32],
    num_patch: i32,
    ip_base: &mut i32,
    fm: &FortModel,
) {
    let mut ibot_top = [0_i32; 2];
    let mut slope = [0.0_f32; 2];
    let mut bintcp = [0.0_f32; 2];
    let mut x_left = [0.0_f32; 2];
    let mut x_right = [0.0_f32; 2];
    let mut ip1: i32;
    let mut ip2: i32;
    let mut ibt: i32;
    let mut line: i32;
    let (mut x1, mut y1, mut x2, mut y2): (f32, f32, f32, f32);
    let xlo: f32;
    let xhi: f32;
    let (yll, ylr, yul, yur): (f32, f32, f32, f32);
    let (mut alpha_add, mut theta_add, mut shift_add) = (0.0_f32, 0.0_f32, 0.0_f32);
    let pc = |k: i32, i: i32| fm.p_coord[(i - 1) as usize][(k - 1) as usize];
    let object = |k: i32| fm.object[(k - 1) as usize];
    let ibase_obj = |i: i32| fm.ibase_obj[(i - 1) as usize];
    let npt_in_obj = |i: i32| fm.npt_in_obj[(i - 1) as usize];

    ibot_top[0] = iobj1;
    ibot_top[1] = iobj2;
    if pc(indy, object(ibase_obj(iobj1) + 1))
        + pc(indy, object(ibase_obj(iobj1) + npt_in_obj(iobj1)))
        > pc(indy, object(ibase_obj(iobj2) + 1))
            + pc(indy, object(ibase_obj(iobj2) + npt_in_obj(iobj2)))
    {
        ibot_top[0] = iobj2;
        ibot_top[1] = iobj1;
    }

    for line in 1..=2 {
        let lu = (line - 1) as usize;
        ibt = ibot_top[lu];
        ip1 = object(ibase_obj(ibt) + 1);
        ip2 = object(ibase_obj(ibt) + npt_in_obj(ibt));
        x1 = scale_fac * pc(1, ip1);
        y1 = scale_fac * pc(indy, ip1);
        x2 = scale_fac * pc(1, ip2);
        y2 = scale_fac * pc(indy, ip2);
        if (x2 - x1).abs() < 0.1 * (y1 - y2).abs() {
            println!(
                "\nERROR: TOMOPITCH - Line from{}{} to{}{} is too steep to use",
                fmt_f(x1, 7, 0),
                fmt_f(y1, 7, 0),
                fmt_f(x2, 7, 0),
                fmt_f(y2, 7, 0)
            );
            exit(1);
        }
        slope[lu] = (y2 - y1) / (x2 - x1);
        bintcp[lu] = y2 - slope[lu] * x2;
        // `min(x1, x2)` / `max(x1, x2)` (`tomopitch.f90:471-472`).  The
        // reference object unrolls the two-pass loop and allocates registers
        // differently in each pass: `minss x1, x2` both times, but
        // `maxss x2, x1` for line 1 and `maxss x1, x2` for line 2.  (That is
        // the version for a non-unit `p_coord` stride, which is what the model
        // arrays have; the unit-stride clone is never taken.)
        x_left[lu] = minss(x1, x2);
        x_right[lu] = if lu == 0 {
            maxss(x2, x1)
        } else {
            maxss(x1, x2)
        };
    }

    // `tomopitch.f90:475-476`: `minss xLeft(1), xLeft(2)` then `minss ., -0.45 *
    // scaleFac * maxx`, and the same shape with `maxss` for `xhi`.
    xlo = minss(minss(x_left[0], x_left[1]), -0.45 * scale_fac * maxx as f32);
    xhi = maxss(
        maxss(x_right[0], x_right[1]),
        0.45 * scale_fac * maxx as f32,
    );
    if x_right[0] - x_left[0] < 0.1 * (xhi - xlo) || x_right[1] - x_left[1] < 0.1 * (xhi - xlo) {
        line = 2;
        if x_right[0] - x_left[0] < 0.1 * (xhi - xlo) {
            line = 1;
        }
        let lu = (line - 1) as usize;
        println!(
            "\nERROR: TOMOPITCH - The line going from{} to{} is too short to use",
            fmt_f(x_left[lu], 7, 0),
            fmt_f(x_right[lu], 7, 0)
        );
        exit(1);
    }

    yll = xlo * slope[0] + bintcp[0];
    ylr = xhi * slope[0] + bintcp[0];
    yul = xlo * slope[1] + bintcp[1];
    yur = xhi * slope[1] + bintcp[1];
    let b = (*ip_base - 1) as usize;
    xcen[b] = xlo;
    ycen[b] = (yll + yul) / 2.;
    thick_mid[b] = 2. * border + yul - yll;
    y_samp[b] = y_sample;
    xcen[b + 1] = xhi;
    ycen[b + 1] = (ylr + yur) / 2.;
    thick_mid[b + 1] = 2. * border + yur - ylr;
    y_samp[b + 1] = y_sample;
    if_use[b] = 1;
    if_use[b + 1] = 1;
    //
    analyze_spots(
        message.trim_end_matches(' '),
        &xcen[b..],
        &ycen[b..],
        &thick_mid[b..],
        &if_use[b..],
        num_patch,
        &y_samp[b..],
        0.,
        &mut alpha_add,
        &mut theta_add,
        &mut shift_add,
    );
    *ip_base = *ip_base + num_patch;
}

/// Original `subroutine analyzeSpots(fileLabel, xcen, ycen, thickMid, ifUse,
/// numSpots, ySamp, doXtilt, alpha, thetaAdd, shiftAdd)`
/// (`tomopitch.f90:508`).
pub fn analyze_spots(
    file_label: &str,
    xcen: &[f32],
    ycen: &[f32],
    thick_mid: &[f32],
    if_use: &[i32],
    num_spots: i32,
    y_samp: &[f32],
    do_xtilt: f32,
    alpha: &mut f32,
    theta_add: &mut f32,
    shift_add: &mut f32,
) {
    let mut xx = [0.0_f32; 1000];
    let mut yy = [0.0_f32; 1000];
    let mut zz = [0.0_f32; 1000];
    //
    let mut nd: i32;
    let angle: f32;
    let (mut slope, mut bintcp, mut ro) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut a, mut b, mut c) = (0.0_f32, 0.0_f32, 0.0_f32);
    let theta: f32;
    let cos_theta: f32;
    let sin_theta: f32;
    let cos_aplha: f32;
    let sin_alpha: f32;
    let mut zp: f32;
    //
    println!("\n\n Analysis of positions from {file_label}:");
    if num_spots > 1 && thick_mid[0] < 0. || thick_mid[1] < 0. {
        exit_error("Lines cross when extrapolated to the full range in X");
    }

    // BRT is expecting a line from this call before the angle offset line
    findshift("unrotated", ycen, thick_mid, if_use, num_spots, shift_add);
    nd = 0;
    for i in 0..num_spots as usize {
        if if_use[i] != 0 {
            nd = nd + 1;
            xx[(nd - 1) as usize] = xcen[i];
            yy[(nd - 1) as usize] = ycen[i];
        }
    }
    ls_fit(&xx, &yy, nd, &mut slope, &mut bintcp, &mut ro);
    angle = slope.atan() * ATAND_FACTOR;
    for i in 0..num_spots as usize {
        yy[i] = xcen[i] * gfortran_sind_r4(-angle) + ycen[i] * gfortran_cosd_r4(-angle);
    }

    // BRT is expecting "add" before and "to" after the angle in this line
    println!(
        " slope ={}: to make level, add{} to total angle offset",
        fmt_f(slope, 8, 4),
        fmt_f(angle, 6, 1)
    );
    findshift(" rotated ", &yy, thick_mid, if_use, num_spots, shift_add);
    *theta_add = angle;
    //
    if do_xtilt == 0. || num_spots <= 2 {
        return;
    }
    //
    nd = 0;
    for i in 0..num_spots as usize {
        if if_use[i] != 0 {
            nd = nd + 1;
            let n = (nd - 1) as usize;
            xx[n] = xcen[i];
            zz[n] = ycen[i];
            yy[n] = y_samp[i];
        }
    }
    ls_fit2(&xx, &yy, &zz, nd, &mut a, &mut b, Some(&mut c));
    theta = -(a.atan() * ATAND_FACTOR);
    cos_theta = gfortran_cosd_r4(theta);
    sin_theta = gfortran_sind_r4(theta);
    *alpha = (b / (cos_theta - a * sin_theta)).atan() * ATAND_FACTOR;
    cos_aplha = gfortran_cosd_r4(*alpha);
    sin_alpha = gfortran_sind_r4(*alpha);
    for i in 0..num_spots as usize {
        zp = xcen[i] * sin_theta + ycen[i] * cos_theta;
        yy[i] = -y_samp[i] * sin_alpha + zp * cos_aplha;
    }
    println!(
        "\n Pitch between samples can be corrected with an added X-axis tilt of{}\n In this \
         case, to make level, add{} to total angle offset",
        fmt_f(*alpha, 7, 2),
        fmt_f(-theta, 6, 1)
    );
    *theta_add = -theta;

    // BRT is expecting "x-tilted  lines"
    findshift("x-tilted ", &yy, thick_mid, if_use, num_spots, shift_add);
}

/// Original `subroutine findshift(rotLabel, ycen, thick, ifUse, numSpots,
/// shift)` (`tomopitch.f90:588`).
pub fn findshift(
    rot_label: &str,
    ycen: &[f32],
    thick: &[f32],
    if_use: &[i32],
    num_spots: i32,
    shift: &mut f32,
) {
    let mut bot: f32;
    let mut top: f32;
    let real_thick: f32;
    let mut ithick: i32;
    //
    bot = 1.0e10;
    top = -1.0e10;
    for i in 0..num_spots as usize {
        if if_use[i] != 0 {
            // `tomopitch.f90:607-608`: `minss bot, .` / `maxss top, .`.
            bot = minss(bot, ycen[i] - thick[i] / 2.);
            top = maxss(top, ycen[i] + thick[i] / 2.);
        }
    }
    real_thick = top - bot;
    *shift = -0.5 * (top + bot);
    ithick = 2 * ((real_thick / 2. + 0.99) as i32);
    ithick = nice_frame(ithick, 2, 19);

    // BRT is expecting "shift of" before and ";" after the Z shift, and thickness at end
    // And also " lines" after the "x-tilted "
    println!(
        " {rot_label} lines imply added Z shift of {}; thickness of {}, set to{}",
        fmt_f(*shift, 7, 1),
        fmt_f(real_thick, 6, 1),
        fmt_i(ithick, 5)
    );
}
