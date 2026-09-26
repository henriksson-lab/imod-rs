//! Translation of `IMOD/flib/image/assemblevol.f90`.
//!
//! Assemblevol will assemble a single MRC file from separate files that form
//! an array of subvolumes in X, Y, and Z.  Its primary use is for
//! reassembling a tomogram after it has been chopped into pieces, using the
//! coordinates output by Tomopieces.
//!
//! The main program maps to [`assemblevol`]; the subroutines `insert_array`,
//! `getPipFileNumbers`, `getCheckLowHighLimits` and `setUndefinedLimits` map
//! to [`insert_array`], [`get_pip_file_numbers`],
//! [`get_check_low_high_limits`] and [`set_undefined_limits`].  Their
//! 1-based Fortran arrays are 0-based slices.

use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_integer, pip_get_two_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::pip_fwrap::{pipgetnonoptionarg_, pipgetstring_};
use crate::imod::libcfshr::simplestat::array_min_max_mean_fortran;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_fileio::{
    iiu_alt_print, iiu_close, iiu_set_position, iiu_write_section,
};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_tilt, iiu_alt_tilt_orig, iiu_create_header,
    iiu_ret_delta, iiu_ret_origin, iiu_ret_tilt, iiu_ret_tilt_orig, iiu_trans_labels,
    iiu_write_header,
};
use std::io::Write;

/// `parameter (numOptions = 10)` (`assemblevol.f90:52`).
const NUM_OPTIONS: i32 = 10;
/// Fallback PIP table, the `options(1)` string (`assemblevol.f90:54-59`).
const OPTIONS: &str = "input:InputFile:FNM:@output:OutputFile:FN:@\
xextract:StartEndToExtractInX:IPM:@yextract:StartEndToExtractInY:IPM:@\
zextract:StartEndToExtractInZ:IPM:@nxfiles:NumberOfFilesInX:I:@\
nyfiles:NumberOfFilesInY:I:@nzfiles:NumberOfFilesInZ:I:@\
param:ParameterFile:PF:@help:usage:B:";

/// `read(5, '(a)')` into a `character*320`: end of file is the gfortran
/// runtime error, status 2.
fn read_line_320() -> String {
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if matches!(std::io::stdin().read_line(&mut line), Ok(0) | Err(_)) {
        eprintln!("Fortran runtime error: End of file");
        exit(2);
    }
    let line = line.trim_end_matches(['\r', '\n']);
    line.chars()
        .take(320)
        .collect::<String>()
        .trim_end_matches(' ')
        .to_string()
}

/// `read(5, *)` of integers with no `END=`/`ERR=`.
fn read_ints(items: &mut [ListItem]) {
    let _ = std::io::stdout().flush();
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    if list_read(&mut lock, items).is_err() {
        eprintln!("Fortran runtime error: End of file");
        exit(2);
    }
}

/// Original program `assemblevol` (`assemblevol.f90:17`).
pub fn assemblevol() {
    unsafe {
        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut nxyz2 = [0_i32; 3];
        let mut mxyz2 = [0_i32; 3];
        let mut cell2 = [0f32; 6];
        let mut delta = [0f32; 3];
        let mut tilt = [0f32; 3];
        let mut original_tilt = [0f32; 3];
        let (mut x_origin, mut y_origin, mut z_origin) = (0f32, 0f32, 0f32);
        let (mut dmin2, mut dmax2, mut dmean2) = (0f32, 0f32, 0f32);
        let (mut tmin, mut tmax, mut temp_min) = (0f32, 0f32, 0f32);
        let mut mode = 0_i32;
        let mut mode_first = 0_i32;
        let (mut num_xfiles, mut num_yfiles, mut num_zfiles) = (0_i32, 0_i32, 0_i32);
        let (mut num_xrange_in, mut num_yrange_in, mut num_zrange_in) = (0_i32, 0_i32, 0_i32);
        let mut num_opt_files = 0_i32;
        let mut out_file: String;
        //
        let max_files_to_open = 256;
        //
        // Pip startup: set error, parse options, do help output
        //
        let (mut num_opt_arg, mut num_non_opt_arg) = (0, 0);
        pip_read_or_parse_options(
            &[OPTIONS],
            NUM_OPTIONS,
            "assemblevol",
            "ERROR: ASSEMBLEVOL - ",
            true,
            2,
            2,
            1,
            &mut num_opt_arg,
            &mut num_non_opt_arg,
        );
        let pip_input = num_opt_arg + num_non_opt_arg > 0;

        if pip_input {
            //
            // Get output file from either place, adjust non-option count down if it is that
            let mut record = [b' '; 320];
            if pipgetstring_(b"OutputFile", &mut record) != 0 {
                if num_non_opt_arg == 0 {
                    exit_error(
                        "Output filename must be entered either with option or as last non-option argument",
                    );
                }
                let _ = pipgetnonoptionarg_(num_non_opt_arg, &mut record);
                num_non_opt_arg -= 1;
            }
            out_file = fortran_string(&record);
            //
            // Input files are option entries and remaining non-opt entries
            let _ = pip_number_of_entries(b"InputFile", &mut num_opt_files);
            let num_files = num_opt_files + num_non_opt_arg;
            get_pip_file_numbers(&mut num_xfiles, &mut num_xrange_in, "X");
            get_pip_file_numbers(&mut num_yfiles, &mut num_yrange_in, "Y");
            get_pip_file_numbers(&mut num_zfiles, &mut num_zrange_in, "Z");
            if num_xfiles * num_yfiles * num_zfiles != num_files {
                exit_error(
                    "The number of input files entered does not equal the product of the number in X, Y and Z implied by the other entries",
                );
            }
        } else {
            print!(" Output file for assembled volume: ");
            out_file = read_line_320();
            //
            print!(" Numbers of input files in X, Y, and Z: ");
            read_ints(&mut [
                ListItem::Integer(&mut num_xfiles),
                ListItem::Integer(&mut num_yfiles),
                ListItem::Integer(&mut num_zfiles),
            ]);
            if num_xfiles < 0 || num_yfiles < 0 || num_zfiles < 0 {
                exit_error("Number of files must be positive");
            }
        }

        let lim_range = num_xfiles.max(num_yfiles).max(num_zfiles);
        let num_files = num_xfiles * num_yfiles * num_zfiles;
        let lr = lim_range.max(0) as usize;
        let mut ix_low = vec![0_i32; lr];
        let mut ix_high = vec![0_i32; lr];
        let mut iy_low = vec![0_i32; lr];
        let mut iy_high = vec![0_i32; lr];
        let mut iz_low = vec![0_i32; lr];
        let mut iz_high = vec![0_i32; lr];
        let mut files = vec![String::new(); num_files.max(0) as usize];
        let mut x_defined = vec![false; lr];
        let mut y_defined = vec![false; lr];
        let mut z_defined = vec![false; lr];
        memory_error(0, "arrays for ranges or filenames");

        let open_layer = num_xfiles * num_yfiles <= max_files_to_open;
        //
        if !pip_input {
            println!(
                " Enter the starting and ending index coordinates for the pixels to extract from the files at successive positions in each dimension (0,0 for full extent or -1,-1 for first pixel only)"
            );
        }
        let mut nx_out = 0;
        let mut ny_out = 0;
        let mut nz_out = 0;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut max_z = 0;

        // Get the lower and upper limits for each position on each axis
        get_check_low_high_limits(
            num_xfiles,
            num_xrange_in,
            pip_input,
            &mut nx_out,
            &mut ix_low,
            &mut ix_high,
            &mut x_defined,
            &mut max_x,
            "X",
        );
        get_check_low_high_limits(
            num_yfiles,
            num_yrange_in,
            pip_input,
            &mut ny_out,
            &mut iy_low,
            &mut iy_high,
            &mut y_defined,
            &mut max_y,
            "Y",
        );
        get_check_low_high_limits(
            num_zfiles,
            num_zrange_in,
            pip_input,
            &mut nz_out,
            &mut iz_low,
            &mut iz_high,
            &mut z_defined,
            &mut max_z,
            "Z",
        );
        //
        if !pip_input {
            println!(" Enter the input file names at successive positions in X, then Y, then Z");
        }
        let mut ifile = 1_i32;
        iiu_alt_print(0);
        for iz in 1..=num_zfiles {
            for iy in 1..=num_yfiles {
                for ix in 1..=num_xfiles {
                    let fu = (ifile - 1) as usize;
                    if pip_input {
                        let mut record = [b' '; 320];
                        if ifile <= num_opt_files {
                            let _ = pipgetstring_(b"InputFile", &mut record);
                        } else {
                            let _ = pipgetnonoptionarg_(ifile - num_opt_files, &mut record);
                        }
                        files[fu] = fortran_string(&record);
                    } else {
                        print!(" Name of file at{:>4}{:>4}{:>4}: ", ix, iy, iz);
                        files[fu] = read_line_320();
                    }
                    imopen(2, &files[fu], "ro");
                    irdhdr(
                        2,
                        nxyz.as_mut_ptr(),
                        mxyz.as_mut_ptr(),
                        &mut mode,
                        &mut dmin2,
                        &mut dmax2,
                        &mut dmean2,
                    );
                    if ix == 1 && iy == 1 && iz == 1 {
                        delta = iiu_ret_delta(2);
                        [x_origin, y_origin, z_origin] = iiu_ret_origin(2);
                        tilt = iiu_ret_tilt(2);
                        original_tilt = iiu_ret_tilt_orig(2);
                    }
                    if ifile == 1 {
                        mode_first = mode;
                    }
                    if mode != mode_first {
                        exit_error("Mode mismatch for this file");
                    }
                    //
                    // Collect coordinates if they are not defined yet
                    set_undefined_limits(
                        ix,
                        nxyz[0],
                        &mut nx_out,
                        &mut ix_low,
                        &mut ix_high,
                        &mut x_defined,
                        &mut max_x,
                    );
                    set_undefined_limits(
                        iy,
                        nxyz[1],
                        &mut ny_out,
                        &mut iy_low,
                        &mut iy_high,
                        &mut y_defined,
                        &mut max_y,
                    );
                    set_undefined_limits(
                        iz,
                        nxyz[2],
                        &mut nz_out,
                        &mut iz_low,
                        &mut iz_high,
                        &mut z_defined,
                        &mut max_z,
                    );
                    if ix_high[(ix - 1) as usize] >= nxyz[0]
                        || iy_high[(iy - 1) as usize] >= nxyz[1]
                        || iz_high[(iz - 1) as usize] >= nxyz[2]
                    {
                        exit_error("Upper coordinate too high for this file");
                    }
                    iiu_close(2);
                    ifile += 1;
                }
            }
        }
        pip_done();
        //
        let idim_out = i64::from(nx_out) * i64::from(ny_out) + 10;
        let idim_in = i64::from(max_x) * i64::from(max_y) + 10;
        let mut array: Vec<f32> = Vec::new();
        let mut brray: Vec<f32> = Vec::new();
        let mut ierr = 0;
        if idim_in < 0
            || idim_out < 0
            || array.try_reserve_exact(idim_in as usize).is_err()
            || brray.try_reserve_exact(idim_out as usize).is_err()
        {
            ierr = 1;
        }
        memory_error(ierr, "arrays for image data");
        array.resize(idim_in as usize, 0.0);
        brray.resize(idim_out as usize, 0.0);
        //
        imopen(1, &out_file, "NEW");
        nxyz2[0] = nx_out;
        nxyz2[1] = ny_out;
        nxyz2[2] = nz_out;
        mxyz2[0] = nx_out;
        mxyz2[1] = ny_out;
        mxyz2[2] = nz_out;
        cell2[0] = nx_out as f32 * delta[0];
        cell2[1] = ny_out as f32 * delta[1];
        cell2[2] = nz_out as f32 * delta[2];
        cell2[3] = 90.0;
        cell2[4] = 90.0;
        cell2[5] = 90.0;
        x_origin -= ix_low[0] as f32 * delta[0];
        y_origin -= iy_low[0] as f32 * delta[1];
        z_origin -= iz_low[0] as f32 * delta[2];
        //
        let mut cur_time = [b' '; 8];
        time(&mut cur_time);
        let mut cur_date = [b' '; 9];
        b3d_date(&mut cur_date);
        // format('ASSEMBLEVOL: Reassemble a volume from pieces',t57,a9,2x,a8)
        let mut title = [b' '; MRC_LABEL_SIZE];
        let head = b"ASSEMBLEVOL: Reassemble a volume from pieces";
        title[..head.len()].copy_from_slice(head);
        title[56..65].copy_from_slice(&cur_date);
        title[67..75].copy_from_slice(&cur_time);
        let mut labels = [[b' '; MRC_LABEL_SIZE]; MRC_NLABELS];
        labels[0] = title;
        iiu_create_header(1, &nxyz2, &mxyz2, mode, &labels, 0);
        iiu_alt_cell(1, &cell2);
        iiu_alt_origin(1, &[x_origin, y_origin, z_origin]);
        iiu_alt_tilt(1, &tilt);
        iiu_alt_tilt_orig(1, &original_tilt);
        let mut dmin = 1.0e30_f32;
        let mut dmax = -1.0e30_f32;
        let mut tmean = 0.0_f32;
        //
        ifile = 1;
        for ind_zfile in 1..=num_zfiles {
            //
            // open the files on this layer if possible
            //
            if open_layer {
                for i in 1..=num_yfiles * num_xfiles {
                    imopen(i + 1, &files[(ifile - 1) as usize], "ro");
                    irdhdr(
                        i + 1,
                        nxyz.as_mut_ptr(),
                        mxyz.as_mut_ptr(),
                        &mut mode,
                        &mut dmin2,
                        &mut dmax2,
                        &mut dmean2,
                    );
                    //
                    // DNM 7/31/02: transfer labels from first file
                    //
                    if ifile == 1 {
                        iiu_trans_labels(1, i + 1);
                    }
                    ifile += 1;
                }
            }
            //
            // loop on the sections to be composed
            //
            let mut layer_file = ifile;
            let izu = (ind_zfile - 1) as usize;
            for iz in iz_low[izu]..=iz_high[izu] {
                let mut iunit = 2;
                let mut iy_offset = 0;
                //
                // loop on the files in X and Y
                //
                layer_file = ifile;
                for ind_yfile in 1..=num_yfiles {
                    let iyu = (ind_yfile - 1) as usize;
                    let mut ix_offset = 0;
                    let ny_box = iy_high[iyu] + 1 - iy_low[iyu];
                    for ind_xfile in 1..=num_xfiles {
                        let ixu = (ind_xfile - 1) as usize;
                        //
                        // open files one at a time if necessary
                        //
                        if !open_layer {
                            imopen(2, &files[(layer_file - 1) as usize], "ro");
                            irdhdr(
                                2,
                                nxyz.as_mut_ptr(),
                                mxyz.as_mut_ptr(),
                                &mut mode,
                                &mut dmin2,
                                &mut dmax2,
                                &mut dmean2,
                            );
                            if ind_xfile == 1
                                && ind_yfile == 1
                                && ind_zfile == 1
                                && iz == iz_low[izu]
                            {
                                iiu_trans_labels(1, 2);
                            }
                            layer_file += 1;
                        }
                        //
                        // read the section, and insert into big array
                        //
                        iiu_set_position(iunit, iz, 0);
                        let nx_box = ix_high[ixu] + 1 - ix_low[ixu];
                        if irdpas(
                            iunit,
                            &mut array,
                            nx_box,
                            ny_box,
                            ix_low[ixu],
                            ix_high[ixu],
                            iy_low[iyu],
                            iy_high[iyu],
                        )
                        .is_err()
                        {
                            exit_error("Reading file");
                        }
                        insert_array(
                            &array, nx_box, ny_box, &mut brray, nx_out, ny_out, ix_offset,
                            iy_offset,
                        );
                        ix_offset += nx_box;
                        if open_layer {
                            iunit += 1;
                        } else {
                            iiu_close(2);
                        }
                    }
                    iy_offset += ny_box;
                }
                //
                // section done, get density and write it
                //
                array_min_max_mean_fortran(
                    &brray,
                    &nx_out,
                    &ny_out,
                    &1,
                    &nx_out,
                    &1,
                    &ny_out,
                    &mut tmin,
                    &mut tmax,
                    &mut temp_min,
                );
                // `assemblevol.f90:276-277`: `minss dmin, tmin` / `maxss dmax,
                // tmax` in the reference object.
                dmin = minss(dmin, tmin);
                dmax = maxss(dmax, tmax);
                tmean += temp_min;
                iiu_write_section(1, brray.as_mut_ptr().cast());
            }
            //
            // close layer files if opened; otherwise set file number for next
            // layer
            //
            if open_layer {
                for i in 1..=num_yfiles * num_xfiles {
                    iiu_close(i + 1);
                }
            } else {
                ifile = layer_file;
            }
        }
        let dmean = tmean / nz_out as f32;
        iiu_write_header(1, &title, 1, dmin, dmax, dmean);
        iiu_close(1);
        // write(*,'(/,i6,a)')
        println!(
            "\n{:>6} files reassembled",
            num_xfiles * num_yfiles * num_zfiles
        );
        exit(0);
    }
}

/// Original `insert_array` (`assemblevol.f90:307`).
///
/// Copy an array with an offset into a larger array.
pub fn insert_array(
    array: &[f32],
    nx_box: i32,
    ny_box: i32,
    brray: &mut [f32],
    nx_out: i32,
    _ny_out: i32,
    ix_offset: i32,
    iy_offset: i32,
) {
    for iy in 1..=ny_box {
        for ix in 1..=nx_box {
            brray[((ix + ix_offset - 1) as i64 + (iy + iy_offset - 1) as i64 * nx_out as i64)
                as usize] = array[((ix - 1) + (iy - 1) * nx_box) as usize];
        }
    }
}

/// Original `getPipFileNumbers` (`assemblevol.f90:323`).
///
/// Get the number of files for an axis one way or another.
pub fn get_pip_file_numbers(num_xfiles: &mut i32, num_xrange_in: &mut i32, axis: &str) {
    let _ = pip_number_of_entries(
        format!("StartEndToExtractIn{axis}").as_bytes(),
        num_xrange_in,
    );
    if pip_get_integer(format!("NumberOfFilesIn{axis}").as_bytes(), num_xfiles) == 0 {
        if *num_xrange_in > 0 {
            exit_error(&format!(
                "You cannot enter both StartEndToExtractIn{axis} and NumberOfFilesIn{axis}"
            ));
        }
    } else {
        *num_xfiles = 1.max(*num_xrange_in);
    }
}

/// Original `getCheckLowHighLimits` (`assemblevol.f90:341`).
///
/// Get the low and high limits for each position on an axis.
pub fn get_check_low_high_limits(
    num_xfiles: i32,
    num_xrange_in: i32,
    pip_input: bool,
    nx_out: &mut i32,
    ix_low: &mut [i32],
    ix_high: &mut [i32],
    x_defined: &mut [bool],
    max_x: &mut i32,
    axis: &str,
) {
    for ind in 1..=num_xfiles {
        let i = (ind - 1) as usize;
        if pip_input {
            if num_xrange_in > 0 {
                let _ = pip_get_two_integers(
                    format!("StartEndToExtractIn{axis}").as_bytes(),
                    &mut ix_low[i],
                    &mut ix_high[i],
                );
            }
        } else {
            // write(*,'(1x,a,a,i3,a,a,a,$)')
            print!(
                " {} coordinates for files at position #{:>3} in {}: ",
                axis, ind, axis
            );
            let (lo, hi) = (&mut ix_low[i..], &mut ix_high[i..]);
            read_ints(&mut [ListItem::Integer(&mut lo[0]), ListItem::Integer(&mut hi[0])]);
        }
        //
        // Special case of -1, -1: mark axis as defined and set to 0,0
        // Otherwise axis is defined if either is nonzero
        if ix_low[i] == -1 && ix_high[i] == -1 {
            x_defined[i] = true;
            ix_low[i] = 0;
            ix_high[i] = 0;
        } else {
            x_defined[i] = ix_low[i] > 0 || ix_high[i] > 0;
        }
        if ix_low[i] < 0 || ix_high[i] < ix_low[i] {
            exit_error(&format!(
                "Illegal {axis} coordinate less than zero or out of order"
            ));
        }
        if x_defined[i] {
            *nx_out += ix_high[i] + 1 - ix_low[i];
            *max_x = (*max_x).max(ix_high[i] + 1 - ix_low[i]);
        }
    }
}

/// Original `setUndefinedLimits` (`assemblevol.f90:373`).
///
/// If the limits are still undefined for a position, set them to full range.
pub fn set_undefined_limits(
    ix: i32,
    nx: i32,
    nx_out: &mut i32,
    ix_low: &mut [i32],
    ix_high: &mut [i32],
    x_defined: &mut [bool],
    max_x: &mut i32,
) {
    let i = (ix - 1) as usize;
    if x_defined[i] {
        return;
    }
    ix_low[i] = 0;
    ix_high[i] = nx - 1;
    *nx_out += nx;
    *max_x = (*max_x).max(nx);
    x_defined[i] = true;
}
