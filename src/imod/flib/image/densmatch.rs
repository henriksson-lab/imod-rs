//! Translation of `IMOD/flib/image/densmatch.f90`.
//!
//! DENSMATCH scales the density values in one volume so that its mean and
//! standard deviation match that of another volume (or an entered target).
//! The Fortran main program maps to [`densmatch`]; the source has no other
//! program units.

use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdlin, irdsecl};
use crate::imod::libcfshr::b3dutil::{exit, set_float_output_for_entered_mode};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_integer, pip_get_three_integers, pip_get_two_floats, pip_get_two_integers,
    pip_number_of_entries,
};
use crate::imod::libcfshr::simplestat::{array_min_max_mean_fortran, sums_to_avg_sd_dbl};
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position, iiu_write_lines};
use crate::imod::libiimod::unit_header::{iiu_alt_mode, iiu_trans_header, iiuwriteheaderstr};
use chrono::{Local, Timelike};

/// `parameter (numOptions = 12)` (`densmatch.f90:59`).
const DENSMATCH_NUM_OPTIONS: i32 = 12;

/// Fallback PIP table, the `options(1)` string (`densmatch.f90:61-65`).
const DENSMATCH_OPTIONS: &str = "reference:ReferenceFile:FN:@scaled:ScaledFile:FN:@output:OutputFile:FN:@\
target:TargetMeanAndSD:FP:@mode:ModeToOutput:I:@report:ReportOnly:B:@\
xminmax:XMinAndMax:IP:@yminmax:YMinAndMax:IP:@zminmax:ZMinAndMax:IP:@\
all:UseAllPixels:B:@offset:OffsetRefToScaledXYZ:IT:@help:usage:B:";

/// Original program `densmatch` (`densmatch.f90:17`).
///
/// The `equivalence`d scalar/array pairs (`nx`/`nxyz(1)`, `ixStart`/
/// `ixyzStart(1)`, ...) are the arrays alone, read by index where the source
/// names the scalar.  Fortran unit 6 is written with `println!`, the stream
/// `imopen`/`irdhdr` print their banners on, so the two stay in order.
pub fn densmatch() {
    unsafe {
        // `Gw.d` editing, as `header.rs` carries it (FORMAT 102's `2g14.6`).
        let g_edit = |value: f32, w: usize, d: i32| -> String {
            if value.is_nan() {
                return format!("{:>w$}", "NaN");
            }
            if value.is_infinite() {
                let text = match (value < 0.0, w) {
                    (false, 8..) => "Infinity",
                    (false, _) => "Inf",
                    (true, 9..) => "-Infinity",
                    (true, _) => "-Inf",
                };
                return format!("{text:>w$}");
            }
            let magnitude = value.abs();
            let mut digits = String::new();
            let mut exponent = 1_i32;
            if magnitude != 0.0 {
                let scientific = format!("{:.*e}", (d - 1) as usize, magnitude);
                let (mantissa, power) = scientific.split_once('e').unwrap();
                digits = mantissa.replace('.', "");
                exponent = power.parse::<i32>().unwrap() + 1;
            }
            if (0..=d).contains(&exponent) {
                let mut text = format!("{:.*}", (d - exponent) as usize, value);
                if exponent == d {
                    text.push('.');
                }
                format!("{:>1$}    ", text, w - 4)
            } else {
                format!(
                    "{:>1$}",
                    format!(
                        "{}0.{}E{}{:02}",
                        if value < 0.0 { "-" } else { "" },
                        digits,
                        if exponent < 0 { '-' } else { '+' },
                        exponent.abs()
                    ),
                    w
                )
            }
        };
        // `Fw.d`: a value too wide for the field is written as `w` asterisks.
        let f_edit = |value: f32, w: usize, d: usize| -> String {
            let text = format!("{value:>w$.d$}");
            if text.len() > w { "*".repeat(w) } else { text }
        };

        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut if_min_max = [0_i32; 3];
        let mut ixyz_start = [0_i32; 3];
        let mut nxyz_use = [0_i32; 3];
        let mut num_samp_xyz = [0_i32; 3];
        let mut dxyz_sample = [0.0_f32; 3];
        let mut array: Vec<f32> = Vec::new();
        let mut in_file = String::new();
        let mut out_file;
        let mut average = [0.0_f32; 2];
        let mut stan_dev = [0.0_f32; 2];
        let mut report = false;
        let mut all_pixels = false;
        let mut iunit_out;
        let mut ierr: i32;
        let mut mode = 0_i32;
        let mut new_mode = 0_i32;
        // `ifMode` and `newMode` are assigned only inside `if (pipInput)`
        // (`densmatch.f90:133`), and the interactive path then tests
        // `ifMode > 0` (`:225`) on an uninitialised stack slot: native writes
        // a byte (mode `setFloatOutputForEnteredMode(garbage)`) or a float
        // output at random between runs.  Zero is the no-mode-entered value.
        let mut if_mode = 0_i32;
        let mut i_shift = [0_i32; 3];
        let mut i_start = [0_i32; 3];
        let mut i_end = [0_i32; 3];
        let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
        let mut sem = 0.0_f32;
        let mut idim = 0_i32;
        let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

        out_file = String::from(" ");
        let max_dim = 100000000_i32;
        //
        // This number of samples improves SD accuracy to 0.2%, down from 1-2% with 100000
        // without increasing data access time that much
        let max_samples = 1000000_i32;
        let mut non_opt_scaled_num = 2_i32;
        let mut iun_start = 1_i32;
        for i in 0..3 {
            if_min_max[i] = 0;
            i_shift[i] = 0;
            i_start[i] = 0;
        }
        //
        // Pip startup: set error, parse options, check help, set flag if used
        //
        pip_read_or_parse_options(
            &[DENSMATCH_OPTIONS],
            DENSMATCH_NUM_OPTIONS,
            "densmatch",
            "ERROR: DENSMATCH - ",
            true,
            1,
            2,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let pip_input = num_opt_arg + num_non_opt_arg > 0;
        //
        // Determine if target mean and SD
        if pip_input {
            let (mut avg1, mut sd1) = (average[0], stan_dev[0]);
            if pip_get_two_floats(b"TargetMeanAndSD", &mut avg1, &mut sd1) == 0 {
                average[0] = avg1;
                stan_dev[0] = sd1;
                iun_start = 2;
                non_opt_scaled_num = 1;
                ierr = 0;
                pip_number_of_entries(b"ReferenceFile", &mut ierr);
                if ierr > 0 {
                    exit_error("You cannot enter both -target and -reference");
                }
            }
        }
        //
        // If not, get input file
        if iun_start == 1 {
            if pip_get_in_out_file("ReferenceFile", 1, "Name of reference volume", &mut in_file)
                != 0
            {
                exit_error("Either a reference file or a target mean/SD must be entered");
            }
            imopen(1, &in_file, "ro");
        }
        //
        // Get output file(s)
        if pip_get_in_out_file(
            "ScaledFile",
            non_opt_scaled_num,
            "Name of volume to be scaled",
            &mut in_file,
        ) != 0
        {
            exit_error("No file was specified to be scaled");
        }
        imopen(2, &in_file, "old");
        //
        let _ = pip_get_in_out_file(
            "OutputFile",
            non_opt_scaled_num + 1,
            "Name of output file, or Return to rewrite file to be scaled",
            &mut out_file,
        );
        // `outFile` is `character*320`: an empty entry is the blank record.
        let out_blank = out_file.trim_end_matches(' ').is_empty();
        //
        iunit_out = 2;
        if !out_blank {
            imopen(3, &out_file, "new");
            iunit_out = 3;
        }
        if pip_input {
            let _ = pip_get_logical("ReportOnly", &mut report);
            if_min_max[0] = 1 - pip_get_two_integers(b"XMinAndMax", &mut i_start[0], &mut i_end[0]);
            if_min_max[1] = 1 - pip_get_two_integers(b"YMinAndMax", &mut i_start[1], &mut i_end[1]);
            if_min_max[2] = 1 - pip_get_two_integers(b"ZMinAndMax", &mut i_start[2], &mut i_end[2]);
            let _ = pip_get_logical("UseAllPixels", &mut all_pixels);
            if all_pixels {
                if if_min_max[0] + if_min_max[1] + if_min_max[2] > 0 {
                    exit_error("You cannot enter -all with an option specifying a min and max");
                }
                if_min_max = [1; 3];
                i_start = [0; 3];
                i_end = [0; 3];
            }
            let [shift0, shift1, shift2] = &mut i_shift;
            if pip_get_three_integers(b"OffsetRefToScaledXYZ", shift0, shift1, shift2) == 0 {
                if if_min_max[0] + if_min_max[1] + if_min_max[2] == 0 {
                    exit_error("You must enter min and max X, Y, or Z if you enter offsets");
                }
                if iun_start == 2 {
                    exit_error("You cannot enter both -target and -offset");
                }
            }
            if_mode = 1 - pip_get_integer(b"ModeToOutput", &mut new_mode);
            if if_mode > 0 && out_blank {
                exit_error("You cannot enter a new mode unless outputting to a new file");
            }
        }
        pip_done();
        //
        // sample each volume to find mean and SD
        //
        for iunit in iun_start..=2 {
            irdhdr(
                iunit,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &raw mut mode,
                &raw mut dmin,
                &raw mut dmax,
                &raw mut dmean,
            );
            for i in 0..3 {
                nxyz_use[i] = 1.max(nxyz[i] / 2);
                ixyz_start[i] = nxyz[i] / 4;
                if if_min_max[i] != 0 {
                    ixyz_start[i] = i_start[i];
                    if i_end[i] == 0 {
                        i_end[i] = nxyz[i] - 1;
                    }
                    nxyz_use[i] = i_end[i] + 1 - i_start[i];
                    if iunit == 2 {
                        ixyz_start[i] += i_shift[i];
                    }
                    // `character errstr * 9`
                    let mut errstr = "         ";
                    if ixyz_start[i] < 0 || ixyz_start[i] >= nxyz[i] {
                        errstr = "STARTING ";
                    }
                    if ixyz_start[i] + nxyz_use[i] <= 0 || ixyz_start[i] + nxyz_use[i] > nxyz[i] {
                        errstr = "ENDING   ";
                    }
                    if errstr != "         " {
                        // `write(*,'(/,a,a,a,a,i2)')`; `char(ichar('W') + i)` is
                        // X, Y or Z for the 1-based `i`.
                        println!(
                            "\nERROR: DENSMATCH - {}{} coordinate out of range in volume #{:>2}",
                            errstr,
                            (b'W' + i as u8 + 1) as char,
                            iunit
                        );
                        exit(1);
                    }
                }
            }
            let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
            let mut idel_sample = ((nxyz_use[0] as f32 * nxyz_use[1] as f32 * nxyz_use[2] as f32
                / max_samples as f32)
                .powf(0.3333)
                + 1.0) as i32;
            if all_pixels {
                idel_sample = 1;
            }
            if all_pixels && nx as f32 * ny as f32 * nz as f32 > 2.1e9 {
                exit_error("The -all option cannot be used for volumes of more than 2 gigapixels");
            }
            //
            if iunit == iun_start {
                idim = nx.max(max_dim.min(nx.wrapping_mul(ny)));
                // `allocate(array(idim), stat=ierr)`; the contents are
                // undefined until read, so zero fill stands in for them.
                ierr = 0;
                if array.try_reserve_exact(idim.max(0) as usize).is_err() {
                    ierr = 1;
                } else {
                    array.resize(idim.max(0) as usize, 0.0);
                }
                memory_error(ierr, "array for image data");
            }
            //
            // Make sure there are at least 10 samples in each direction
            for i in 0..3 {
                num_samp_xyz[i] = ((nxyz_use[i] - 1) / idel_sample + 1).max(nxyz_use[i].min(10));
                if num_samp_xyz[i] == 1 {
                    dxyz_sample[i] = 1.0;
                    if if_min_max[i] != 0 {
                        ixyz_start[i] = (i_end[i] + 1 + i_start[i]) / 2;
                    }
                } else {
                    dxyz_sample[i] = (nxyz_use[i] as f32 - 1.0) / (num_samp_xyz[i] as f32 - 1.0);
                }
            }
            //
            let mut ndat = 0_i32;
            let mut dsum8 = 0.0_f64;
            let mut dsum_sq8 = 0.0_f64;
            for jz in 1..=num_samp_xyz[2] {
                let iz = (ixyz_start[2] as f32 + (jz - 1) as f32 * dxyz_sample[2]) as i32;
                let mut tsum8 = 0.0_f64;
                let mut tsum_sq8 = 0.0_f64;
                for jy in 1..=num_samp_xyz[1] {
                    let iy = (ixyz_start[1] as f32 + (jy - 1) as f32 * dxyz_sample[1]) as i32;
                    iiu_set_position(iunit, iz, iy);
                    if irdlin(iunit, &mut array).is_err() {
                        exit_error("Reading file");
                    }
                    for jx in 1..=num_samp_xyz[0] {
                        let ix =
                            ((1 + ixyz_start[0]) as f32 + (jx - 1) as f32 * dxyz_sample[0]) as i32;
                        ndat += 1;
                        let value = array[(ix - 1) as usize];
                        tsum8 += f64::from(value);
                        tsum_sq8 += f64::from(value * value);
                    }
                }
                dsum8 += tsum8;
                dsum_sq8 += tsum_sq8;
            }
            // `sums_to_avgsd8` (`simplestat.c:134`) takes six arguments; the
            // seventh, `sem`, is passed by the source and never set.
            let _ = &mut sem;
            let slot = (iunit - 1) as usize;
            sums_to_avg_sd_dbl(
                dsum8,
                dsum_sq8,
                ndat,
                1,
                &mut average[slot],
                &mut stan_dev[slot],
            );
            // FORMAT 103: `(' Volume',i2,': mean =',f12.4,',  SD =',f12.4)`
            println!(
                " Volume{:>2}: mean ={},  SD ={}",
                iunit + 1 - iun_start,
                f_edit(average[slot], 12, 4),
                f_edit(stan_dev[slot], 12, 4)
            );
        }
        //
        // scale second volume to match first
        //
        let scale_fac = stan_dev[0] / stan_dev[1];
        let add_fac = average[0] - average[1] * scale_fac;
        //
        if report {
            // FORMAT 102: `('Scale factors to multiply by then add:', 2g14.6)`
            println!(
                "Scale factors to multiply by then add:{}{}",
                g_edit(scale_fac, 14, 6),
                g_edit(add_fac, 14, 6)
            );
            exit(0);
        }
        //
        if iunit_out == 3 {
            iiu_trans_header(3, 2);
        }
        if iunit_out == 3 && if_mode > 0 {
            mode = set_float_output_for_entered_mode(new_mode);
            iiu_alt_mode(3, mode);
        }
        let (nx, ny, nz) = (nxyz[0], nxyz[1], nxyz[2]);
        let mut tsum = 0.0_f32;
        let mut tmin = 1.0e30_f32;
        let mut tmax = -1.0e30_f32;
        let max_lines = idim / nx;
        let num_chunks = (ny + max_lines - 1) / max_lines;
        for iz in 1..=nz {
            let mut iy_line = 0_i32;
            for i_chunk in 1..=num_chunks {
                let num_lines = max_lines.min(ny - (i_chunk - 1) * max_lines);

                iiu_set_position(2, iz - 1, iy_line);
                if irdsecl(2, &mut array, num_lines).is_err() {
                    exit_error("Reading file");
                }
                let count = (nx * num_lines) as usize;
                if mode != 0 {
                    for value in &mut array[..count] {
                        *value = scale_fac * *value + add_fac;
                    }
                } else {
                    for value in &mut array[..count] {
                        *value = 255.0_f32.min(0.0_f32.max(scale_fac * *value + add_fac));
                    }
                }
                array_min_max_mean_fortran(
                    &array, &nx, &num_lines, &1, &nx, &1, &num_lines, &mut dmin, &mut dmax,
                    &mut dmean,
                );
                tmin = tmin.min(dmin);
                tmax = tmax.max(dmax);
                tsum += dmean * num_lines as f32;
                iiu_set_position(iunit_out, iz - 1, iy_line);
                iiu_write_lines(iunit_out, array.as_mut_ptr().cast(), num_lines);
                iy_line += num_lines;
            }
        }
        let tmean = tsum / (nz * ny) as f32;
        let mut dat = [b' '; 9];
        b3d_date(&mut dat);
        // `call time(tim)`: `hh:mm:ss`.
        let local = Local::now();
        let tim = format!(
            "{:02}:{:02}:{:02}",
            local.hour(),
            local.minute(),
            local.second()
        );
        //
        // FORMAT 3000: `('DENSMATCH: Scaled volume to match another',t57,a9,2x, a8)`
        // into `character*80 titlech`.
        let mut titlech = [b' '; 80];
        let head = b"DENSMATCH: Scaled volume to match another";
        titlech[..head.len()].copy_from_slice(head);
        titlech[56..65].copy_from_slice(&dat);
        titlech[67..75].copy_from_slice(tim.as_bytes());
        iiuwriteheaderstr(&iunit_out, &titlech, &1, &tmin, &tmax, &tmean);
        iiu_close(iunit_out);
        exit(0);
    }
}
