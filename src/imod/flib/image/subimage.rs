//! Translation of `IMOD/flib/image/subimage.f90`.
//!
//! SUBIMAGE subtracts one image from another: sections of file B from
//! sections of file A, writing the difference images (A-B) to file C, or only
//! reporting statistics of the differences.  Written 8-feb-1989 by Sam
//! Mitchell; multiple sections and statistics-only by DNM.
//!
//! The main program maps to [`subimage`].  `iclden`, `iclavgsd` and
//! `sums_to_avgsd8` are the `simplestat.c` Fortran wrappers.  The NaN operand
//! order of the Fortran `MIN`/`MAX` sites is taken as accumulator first
//! (`minss`/`maxss` destination); it matters only for NaN pixels.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxsd, maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::rdlist::{parselist, rdlist};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen, irdsecl};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string, set_float_output_for_entered_mode};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_two_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libcfshr::simplestat::{
    array_min_max_mean_fortran, array_min_max_mean_sd_fortran, sums_to_avg_sd_dbl,
};
use crate::imod::libiimod::mrcfiles::MRC_LABEL_SIZE;
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_lines};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_mode, iiu_alt_sample, iiu_alt_size, iiu_ret_cell, iiu_trans_header,
    iiu_write_header,
};
use std::io::Write;

/// `parameter (MAXARR = 2100)` (`subimage.f90:12`).
const MAXARR: i32 = 2100;
/// `parameter (MAXLIST = 20000)`.
const MAXLIST: i32 = 20000;
/// `parameter (numOptions = 17)` (`subimage.f90:40`).
const SUBIMAGE_NUM_OPTIONS: i32 = 17;
/// Fallback PIP table, the `options(1)` string (`subimage.f90:42-49`).
const SUBIMAGE_OPTIONS: &str = "afile:AFileSubtractFrom:FN:@bfile:BFileSubtractOff:FN:@\
output:OutputFile:FN:@text:DifferenceTextFile:FN:@mode:ModeOfOutput:I:@\
asections:ASectionList:LI:@bsections:BSectionList:LI:@zero:ZeroMeanOutput:B:@\
lower:LowerThreshold:F:@upper:UpperThreshold:F:@\
xstats:StatisticsXminAndMax:IP:@ystats:StatisticsYminAndMax:IP:@\
frac:FractionalValues:B:@minmax:ErrorMinMaxLimit:F:@sdlimit:ErrorSDLimit:F:@\
param:ParameterFile:PF:@help:usage:B:";

/// Original program `subimage` (`subimage.f90:10`).
pub fn subimage() {
    // `common / bigarr / array, brray`
    let mut array = vec![0.0_f32; (MAXARR * MAXARR) as usize];
    let mut brray = vec![0.0_f32; (MAXARR * MAXARR) as usize];
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut nxyz2 = [0_i32; 3];
    // `data nxyzst /0, 0, 0/`
    let nxyzst = [0_i32; 3];
    let mut list_asec = vec![0_i32; MAXLIST as usize];
    let mut list_bsec = vec![0_i32; MAXLIST as usize];
    let mut afile = String::new();
    let mut bfile = String::new();
    let mut cfile = String::new();
    let mut text_file = [b' '; 320];
    let mut list_string = [b' '; 10000];
    let mut dat = [b' '; 9];
    let mut tim = [b' '; 8];
    let mut limarr: i32;
    let mut mode = 0_i32;
    let mut num_asec: i32;
    let mut num_bsec: i32;
    let mut ierr: i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut dsum: f32;
    let (mut tmin, mut tmax): (f32, f32);
    let (mut real_min, mut real_max): (f32, f32);
    let (mut cmin, mut cmax, mut tmean, mut sd) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let mut low_thresh = 0.0_f32;
    let mut high_thresh = 0.0_f32;
    let mut sd_limit: f32;
    let mut diff_limit: f32;
    let range: f32;
    let mut tot_pixels: f64;
    let (mut tsum, mut tsum_sq) = (0.0_f64, 0.0_f64);
    let (mut sd_sum, mut sd_sum_sq): (f64, f64);
    let (mut sum, mut sum_sq): (f64, f64);
    let mut diff_mean: f64;
    let mut real_sum: f64;
    let mut mode_out: i32;
    let mut if_zero_mean = 0_i32;
    let if_low_thresh: i32;
    let if_high_thresh: i32;
    let mut max_unit_out: i32;
    let mut if_range_frac = 0_i32;
    let (mut min_xstat, mut min_ystat, mut max_xstat, mut max_ystat): (i32, i32, i32, i32);
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    let mut unit7: Option<std::fs::File> = None;
    // `write(i, ...)` for i = 6 .. maxUnitOut: stdout, then the text file.
    let write_units = |text: &str, max_unit_out: i32, unit7: &mut Option<std::fs::File>| {
        print!("{text}");
        if max_unit_out >= 7 {
            if let Some(file) = unit7.as_mut() {
                let _ = file.write_all(text.as_bytes());
            }
        }
    };

    limarr = MAXARR * MAXARR;
    cfile.clear();
    bfile.clear();
    if_zero_mean = 0;
    sd_limit = 0.;
    diff_limit = 0.;
    max_unit_out = 6;
    if_range_frac = 0;
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[SUBIMAGE_OPTIONS],
        SUBIMAGE_NUM_OPTIONS,
        "subimage",
        "ERROR: SUBIMAGE - ",
        true,
        1,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pip_input = num_opt_arg + num_non_opt_arg > 0;

    if !pip_input {
        println!(" This program will subtract sections of");
        println!(" file B from sections of file A.  The ");
        println!(" resulting file (C) contains the difference");
        println!(" images of (A-B).");
    }

    if pip_get_in_out_file("AFileSubtractFrom", 1, "Name of file A", &mut afile, 320) != 0 {
        exit_error("No input file A specified");
    }

    ialprt(false);
    imopen(1, &afile, "ro");
    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    mode_out = mode;

    num_asec = nxyz[2];
    for i in 1..=num_asec.min(MAXLIST) {
        list_asec[(i - 1) as usize] = i - 1;
        list_bsec[(i - 1) as usize] = i - 1;
    }
    if pip_input {
        if pipgetstring_(b"ASectionList", &mut list_string) == 0 {
            let _ = parselist(&fortran_string(&list_string), &mut list_asec, &mut num_asec);
        }
        // ierr = PipGetInteger('TestLimit', limarr)
        limarr = limarr.min(MAXARR * MAXARR);
        ierr = pip_get_integer(b"ModeOfOutput", &mut mode_out);
        if ierr == 0 {
            // `call setFloatOutputForEnteredMode(modeOut)`: the function's
            // returned mode is discarded by the CALL.
            let _ = set_float_output_for_entered_mode(mode_out);
        }
        let _ = pip_get_boolean(b"ZeroMeanOutput", &mut if_zero_mean);
        if_low_thresh = 1 - pip_get_float(b"LowerThreshold", &mut low_thresh);
        if_high_thresh = 1 - pip_get_float(b"UpperThreshold", &mut high_thresh);
        let _ = pip_get_boolean(b"FractionalValues", &mut if_range_frac);
    } else {
        if_low_thresh = 0;
        if_high_thresh = 0;
        println!(" Enter list of section numbers from file A (ranges OK, / for all)");
        let _ = std::io::stdout().flush();
        let mut stdin = std::io::stdin().lock();
        let _ = rdlist(&mut stdin, &mut list_asec, &mut num_asec);
    }
    if num_asec > MAXLIST {
        exit_error("Too many sections for arrays");
    }

    if pip_get_in_out_file(
        "BFileSubtractOff",
        2,
        "Name of file B, or return for A~",
        &mut bfile,
        320,
    ) != 0
    {
        bfile = format!("{}~", afile.trim_end_matches(' '));
    }
    num_bsec = num_asec;

    if pip_input {
        if pipgetstring_(b"BSectionList", &mut list_string) == 0 {
            let _ = parselist(&fortran_string(&list_string), &mut list_bsec, &mut num_bsec);
        }
    } else {
        println!(
            " Enter list of{:>5} corresponding sections from file B (/ for 0-{:>5})",
            num_asec,
            num_asec - 1
        );
        let _ = std::io::stdout().flush();
        let mut stdin = std::io::stdin().lock();
        let _ = rdlist(&mut stdin, &mut list_bsec, &mut num_bsec);
    }
    if num_asec != num_bsec {
        exit_error("Number of sections does not match");
    }
    let _ = pip_get_in_out_file(
        "OutputFile",
        3,
        "Name of file C, or return for statistics only",
        &mut cfile,
        320,
    );

    min_xstat = 0;
    min_ystat = 0;
    max_xstat = nxyz[0] - 1;
    max_ystat = nxyz[1] - 1;
    if pip_input {
        let _ = pip_get_two_integers(b"StatisticsXminAndMax", &mut min_xstat, &mut max_xstat);
        let _ = pip_get_two_integers(b"StatisticsYminAndMax", &mut min_ystat, &mut max_ystat);
        if min_xstat < 0
            || max_xstat >= nxyz[0]
            || min_xstat > max_xstat
            || min_ystat < 0
            || max_ystat >= nxyz[1]
            || min_ystat > max_ystat
        {
            exit_error("Coordinates for getting statistics are out of range");
        }
        let _ = pip_get_float(b"ErrorMinMaxLimit", &mut diff_limit);
        let _ = pip_get_float(b"ErrorSDLimit", &mut sd_limit);
        if pipgetstring_(b"DifferenceTextFile", &mut text_file) == 0 {
            unit7 = Some(dopen(7, &fortran_string(&text_file), "new", "f"));
            max_unit_out = 7;
        }
    }

    ialprt(true);
    imopen(2, &bfile, "ro");

    unsafe {
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    let range_a = dmax - dmean;
    unsafe {
        irdhdr(
            2,
            nxyz2.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
    }
    range = (range_a + dmax - dmean) / 2.;
    let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);

    if nxyz2[0] != nx || nxyz[1] != ny {
        exit_error("Image sizes do not match");
    }
    pip_done();

    for i in 0..num_asec as usize {
        let asec = list_asec[i];
        let bsec = list_bsec[i];
        if asec < 0 || asec >= nz || bsec < 0 || bsec >= nxyz2[2] {
            exit_error("Illegal section number");
        }
    }

    if nx > limarr {
        exit_error("Images too large for arrays");
    }

    let have_output = !cfile.trim_end_matches(' ').is_empty();
    if have_output {
        imopen(3, &cfile, "new");
        iiu_trans_header(3, 1);
        iiu_alt_mode(3, mode_out);
    }
    dsum = 0.;
    sd_sum = 0.;
    sd_sum_sq = 0.;
    real_sum = 0.;
    real_min = 1.0e37;
    real_max = -real_min;
    write_units(
        " Section      Min            Max            Mean           S.D.\n",
        max_unit_out,
        &mut unit7,
    );
    let max_lines = limarr / nx;
    let num_chunks = (ny + max_lines - 1) / max_lines;
    let read_fail = || -> ! {
        // 100 call exitError('Reading file')
        exit_error("Reading file")
    };
    for isec in 1..=num_asec {
        let asec = list_asec[(isec - 1) as usize];
        let bsec = list_bsec[(isec - 1) as usize];

        //
        // Get mean difference if needed
        diff_mean = 0.;
        if if_zero_mean != 0 {
            unsafe {
                iiu_set_position(1, asec, 0);
                iiu_set_position(2, bsec, 0);
            }
            sum = 0.;
            for i_chunk in 1..=num_chunks {
                let num_lines = max_lines.min(ny - (i_chunk - 1) * max_lines);
                if unsafe { irdsecl(1, &mut array, num_lines) }.is_err()
                    || unsafe { irdsecl(2, &mut brray, num_lines) }.is_err()
                {
                    read_fail();
                }
                array_min_max_mean_fortran(
                    &array, &nx, &num_lines, &1, &nx, &1, &num_lines, &mut cmin, &mut cmax,
                    &mut tmean,
                );
                let mut bmean = 0.0_f32;
                array_min_max_mean_fortran(
                    &brray, &nx, &num_lines, &1, &nx, &1, &num_lines, &mut cmin, &mut cmax,
                    &mut bmean,
                );
                sum += ((tmean - bmean) * num_lines as f32) as f64;
            }
            diff_mean = sum / ny as f64;
        }

        unsafe {
            iiu_set_position(1, asec, 0);
            iiu_set_position(2, bsec, 0);
        }

        sum = 0.;
        sum_sq = 0.;
        tmin = 1.0e37;
        tmax = -tmin;
        for i_chunk in 1..=num_chunks {
            let line_start = (i_chunk - 1) * max_lines;
            let num_lines = max_lines.min(ny - line_start);
            if unsafe { irdsecl(1, &mut array, num_lines) }.is_err()
                || unsafe { irdsecl(2, &mut brray, num_lines) }.is_err()
            {
                read_fail();
            }

            // -----------------------------------------
            // --- Subtract section B from section A and apply thresholds ---

            let count = (nx * num_lines) as usize;
            for i in 0..count {
                array[i] = ((array[i] - brray[i]) as f64 - diff_mean) as f32;
            }
            if if_low_thresh != 0 {
                for value in array[..count].iter_mut() {
                    *value = maxss(*value, low_thresh);
                }
            }
            if if_high_thresh != 0 {
                for value in array[..count].iter_mut() {
                    *value = minss(*value, high_thresh);
                }
            }

            // ---------------------------------
            // --- Write out the difference  ---
            if have_output {
                unsafe {
                    iiu_write_lines(3, array.as_mut_ptr().cast(), num_lines);
                }
            }
            array_min_max_mean_sd_fortran(
                &array,
                &nx,
                &num_lines,
                &1,
                &nx,
                &1,
                &num_lines,
                &mut cmin,
                &mut cmax,
                &mut tsum,
                &mut tsum_sq,
                &mut tmean,
                &mut sd,
            );
            real_sum += tsum;
            real_min = minss(real_min, cmin);
            real_max = maxss(real_max, cmax);
            if line_start <= max_ystat && line_start + num_lines - 1 >= min_ystat {
                //
                // Add to stats if within Y range, replace temporary stats if it is subarea
                if min_xstat > 0 || max_xstat < nx - 1 || min_ystat > 0 || max_ystat < ny - 1 {
                    let iy_start = (min_ystat - line_start).max(0) + 1;
                    let iy_end = (max_ystat - line_start).min(num_lines - 1) + 1;
                    array_min_max_mean_sd_fortran(
                        &array,
                        &nx,
                        &num_lines,
                        &(min_xstat + 1),
                        &(max_xstat + 1),
                        &iy_start,
                        &iy_end,
                        &mut cmin,
                        &mut cmax,
                        &mut tsum,
                        &mut tsum_sq,
                        &mut tmean,
                        &mut sd,
                    );
                }
                sum += tsum;
                sum_sq += tsum_sq;
                tmin = minss(tmin, cmin);
                tmax = maxss(tmax, cmax);
            }
        }
        sums_to_avg_sd_dbl(
            sum,
            sum_sq,
            max_xstat + 1 - min_xstat,
            max_ystat + 1 - min_ystat,
            &mut tmean,
            &mut sd,
        );
        // `4000 format(i5,4f15.4)`
        write_units(
            &format!(
                "{:>5}{}{}{}{}\n",
                asec,
                format_f(tmin as f64, 15, 4),
                format_f(tmax as f64, 15, 4),
                format_f(tmean as f64, 15, 4),
                format_f(sd as f64, 15, 4)
            ),
            max_unit_out,
            &mut unit7,
        );
        sd_sum += sum;
        sd_sum_sq += sum_sq;
        if isec == 1 {
            dmin = tmin;
            dmax = tmax;
        } else {
            dmin = minss(dmin, tmin);
            dmax = maxss(dmax, tmax);
        }
        dsum += tmean;
    }

    let mut cell = iiu_ret_cell(1);
    unsafe {
        iiu_close(1);
        iiu_close(2);
    }
    dmean = dsum / num_asec as f32;
    //
    tot_pixels = num_asec as f32 as f64;
    tot_pixels *= ((max_xstat + 1 - min_xstat) * (max_ystat + 1 - min_ystat)) as f64;
    sd = maxsd(0., (sd_sum_sq - sd_sum * sd_sum / tot_pixels) / (tot_pixels - 1.)).sqrt() as f32;
    if num_asec > 1 {
        // `5000 format(' all ',4f15.4)`, `6000 format(' range frac',f9.6, 3f15.6)`
        let mut text = format!(
            " all {}{}{}{}\n",
            format_f(dmin as f64, 15, 4),
            format_f(dmax as f64, 15, 4),
            format_f(dmean as f64, 15, 4),
            format_f(sd as f64, 15, 4)
        );
        if if_range_frac > 0 && range > 0. {
            text.push_str(&format!(
                " range frac{}{}{}{}\n",
                format_f((dmin / range).abs() as f64, 9, 6),
                format_f((dmax / range) as f64, 15, 6),
                format_f((dmean / range).abs() as f64, 15, 6),
                format_f((sd / range) as f64, 15, 6)
            ));
        }
        write_units(&text, max_unit_out, &mut unit7);
    }

    ierr = 0;
    if diff_limit > 0. && maxss(dmin.abs(), dmax.abs()) > diff_limit {
        println!(
            "THE MAXIMUM DIFFERENCE EXCEEDS {}",
            format_f(diff_limit as f64, 14, 4)
        );
        ierr = 1;
    }
    if sd_limit > 0. && sd > sd_limit {
        println!(
            "THE STANDARD DEVIATION OF THE DIFFERENCE EXCEEDS {}",
            format_f(sd_limit as f64, 14, 4)
        );
        ierr = 1;
    }
    drop(unit7);
    if !have_output {
        let _ = std::io::stdout().flush();
        exit(ierr);
    }
    tot_pixels = num_asec as f32 as f64;
    tot_pixels *= (nx * ny) as f64;
    let real_mean = (real_sum / tot_pixels) as f32;
    nxyz[2] = num_asec;
    mxyz[2] = num_asec;
    // Fixed in translation (BUGS.md, `subimage`): `nz` is EQUIVALENCEd to
    // `nxyz(3)`, which the line before has just set to `numAsec`, so the
    // source divides by `numAsec` and never rescales the Z cell; a subset of
    // sections gets a Z pixel size of (input nz / numAsec) times the input's.
    // Defined: scale by the input's section count, keeping the pixel size.
    cell[2] = (cell[2] * num_asec as f32) / nz as f32;
    iiu_alt_sample(3, &mxyz);
    iiu_alt_cell(3, &cell);
    iiu_alt_size(3, &nxyz, &nxyzst);

    b3d_date(&mut dat);
    time(&mut tim);
    // `3000 format('SUBIMAGE: Subtract section B from section A.', t57, a9, 2x, a8)`
    let mut title = [b' '; MRC_LABEL_SIZE];
    let head = b"SUBIMAGE: Subtract section B from section A.";
    title[..head.len()].copy_from_slice(head);
    title[56..65].copy_from_slice(&dat);
    title[67..75].copy_from_slice(&tim);

    iiu_write_header(3, &title, 1, real_min, real_max, real_mean);

    unsafe {
        iiu_close(3);
    }
    let _ = std::io::stdout().flush();
    exit(ierr);
}
