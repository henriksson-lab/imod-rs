//! Translation of `IMOD/flib/image/warpvol.f90`.
//!
//! WARPVOL will transform a volume using a series of general linear
//! transformations.  Its main use is to "warp" one tomogram from a two-axis
//! tilt series so that it matches the other tomogram.  For each position in
//! the volume, it interpolates between adjacent transformations to find the
//! transformation appropriate for that position.  Any initial alignment
//! transformation (from SOLVEMATCH) must be already contained in the
//! transformations entered into this program; this combining of
//! transformations is accomplished by FINDWARP.  It can work with either a 2-D
//! matrix of transformations (varying in X and Z) or with a general 3-D
//! matrix, as output by FINDWARP.  The program uses the same algorithm as
//! ROTATEVOL for rotating large volumes.
//!
//! Program units: the main program [`warpvol`], and the external subroutines
//! [`test_cube_faces`], [`cube_test_limits`], [`interp_inv`],
//! [`fill_in_transforms`] and [`shift_transforms`].  The `rotmatwarp` module
//! is the [`RotMatWarp`] the program owns.
//!
//! Arrays keep the source's column-major shapes: `aLoc(3,3,numLocY,numLocX,
//! numLocZ)` element `(i,j,ix,iy,iz)` is
//! `a_loc[(i-1) + 3*(j-1) + 9*((ix-1) + numLocY*((iy-1) + numLocX*(iz-1)))]`,
//! and `dxyzLoc(3,...)` likewise with 3 in place of 9.
//!
//! OpenMP: the source parallelises the interpolation over output Y
//! (`!$OMP PARALLEL DO` over `iy`).  Each iteration reads only shared input
//! and stores disjoint elements of `brray`, with no reduction, so the output
//! does not depend on the thread count; the translation runs it on a
//! rayon pool of `numOMPthreads(8)` threads (the calling thread alone when
//! that is 1, e.g. under `OMP_NUM_THREADS=1`).

use crate::imod::flib::image::rotmatwarp::RotMatWarp;
use crate::imod::flib::image::rotmatwarpsubs::{
    cube_indexes, recompose_cubes, set_memory_limit_and_hdf_chunks, setup_cubes_scratch,
    write_one_cube,
};
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::{format_f, maxss, minss};
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::warpfile3d::{read_warp_file_header, read_warp_transforms};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdpas};
use crate::imod::libcfshr::b3dutil::{cputime, exit, fortran_string, num_omp_threads, wall_time};
use crate::imod::libcfshr::linearxforms::inv_matrix;
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_boolean, pip_get_float, pip_get_integer, pip_get_three_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_fileio::{iiu_close, iiu_set_position};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_tilt, iiu_create_header, iiu_ret_delta, iiu_ret_tilt, iiu_trans_labels,
    iiu_write_header,
};
use std::io::{BufRead, Write};

/// `parameter (numOptions = 15)` (`warpvol.f90:68`).
const WARPVOL_NUM_OPTIONS: i32 = 15;
/// Fallback PIP table, the `options(1)` string (`warpvol.f90:70-76`).
const WARPVOL_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@xforms:TransformFile:FN:@\
scale:ScaleTransforms:F:@tempdir:TemporaryDirectory:CH:@\
size:OutputSizeXYZ:IT:@same:SameSizeAsInput:B:@order:InterpolationOrder:I:@\
chunk:ChunkSizesForHDF:IT:@memory:MemoryLimit:I:@verbose:VerboseOutput:I:@\
patch:PatchOutputFile:FN:@filled:FilledInOutputFile:FN:@\
param:ParameterFile:PF:@help:usage:B:";

/// `(180/pi)` as gfortran folds it for the inline `atan2d`
/// (`atan2f(y, x) * c`): the `.rodata` constant `0x42652ee0` in the
/// reference `warpvol`.
const ATAN2D_FACTOR: f32 = f32::from_bits(0x42652ee0);

/// Original program `warpvol` (`warpvol.f90:20`).
pub fn warpvol() {
    let mut rmw = RotMatWarp::default();
    // The OpenMP team for the interpolation loop, made on first use.
    let mut pool: Option<rayon::ThreadPool> = None;
    let mut cell = [0.0_f32; 6];
    let mut mxyz_in = [0_i32; 3];
    let mut a_fwd = [0.0_f32; 9];
    let mut tilt_old = [0.0_f32; 3];
    let mut iz_sec = [0_i32; 6];
    let mut in_min = [0_i32; 3];
    let mut in_max = [0_i32; 3];
    let mut icube = [0_i32; 3];
    let mut xyz_cen = [0.0_f32; 3];
    //
    let mut filein = String::new();
    let mut file_out = String::new();
    let mut file_inv = [b' '; 320];
    let mut temp_dir = [b' '; 320];
    let mut temp_ext = [b' '; 320];
    let mut patch_file = [b' '; 320];
    let mut fill_file = [b' '; 320];
    //
    // DNM 3/8/01: initialize the time in case time(tim) doesn't work
    //
    let mut dat = [b' '; 9];
    let mut tim = *b"00:00:00";
    let (mut num_input, mut num_loc_x, mut num_loc_y, mut num_loc_z) = (0_i32, 0, 0, 0);
    let (mut x_loc_start, mut y_loc_start, mut z_loc_start) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut x_loc_max, mut y_loc_max, mut z_loc_max) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut dx_loc, mut dy_loc, mut dz_loc) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut dmin, mut dmax, mut tsum): (f32, f32, f32);
    let (mut dmin_in, mut dmax_in) = (0.0_f32, 0.0_f32);
    let mut interp_order: i32;
    let mut xf_scale: f32;
    let mut base_int: f32;
    let (mut ix_cube, mut iy_cube) = (0_i32, 0_i32);
    let mut iz_end = 0_i32;
    let mut wall_cum: f64;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    // `read(5, '(a)') var` into a `character*320`, no END=.
    let read_stdin_line = |field: &mut [u8; 320]| {
        let mut line = Vec::new();
        match std::io::stdin().lock().read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => {
                let _ = std::io::stdout().flush();
                eprintln!("Fortran runtime error: End of file");
                exit(2);
            }
            Ok(_) => {}
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        line.resize(320, b' ');
        field.copy_from_slice(&line[..320]);
    };
    //
    // set defaults here
    //
    interp_order = 2;
    xf_scale = 1.;
    base_int = 0.5;
    temp_ext[..10].copy_from_slice(b"wrp      1");
    time(&mut tim);
    b3d_date(&mut dat);
    wall_cum = 0.;
    //
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[WARPVOL_OPTIONS],
        WARPVOL_NUM_OPTIONS,
        "warpvol",
        "ERROR: WARPVOL - ",
        true,
        3,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pipinput = num_opt_arg + num_non_opt_arg > 0;

    if pip_get_in_out_file("InputFile", 1, "Name of input file", &mut filein, 320) != 0 {
        exit_error("No input file specified");
    }
    imopen(5, &filein, "RO");
    unsafe {
        irdhdr(
            5,
            rmw.nxyz_in.as_mut_ptr(),
            mxyz_in.as_mut_ptr(),
            &raw mut rmw.mode,
            &raw mut dmin_in,
            &raw mut dmax_in,
            &raw mut rmw.dmean_in,
        );
    }
    rmw.nxyz_out[0] = rmw.nxyz_in[2];
    rmw.nxyz_out[1] = rmw.nxyz_in[1];
    rmw.nxyz_out[2] = rmw.nxyz_in[0];
    //
    if pipinput {
        let _ = pipgetstring_(b"PatchOutputFile", &mut patch_file);
        let _ = pipgetstring_(b"FilledInOutputFile", &mut fill_file);
    }
    let blank = |field: &[u8]| field.iter().all(|&c| c == b' ');

    if pip_get_in_out_file("OutputFile", 2, "Name of output file", &mut file_out, 320) != 0
        && blank(&patch_file)
        && blank(&fill_file)
    {
        exit_error("No output file specified");
    }
    //
    set_memory_limit_and_hdf_chunks(&mut rmw, pipinput);
    if pipinput {
        let _ = pipgetstring_(b"TemporaryDirectory", &mut temp_dir);
        let _ = pip_get_integer(b"InterpolationOrder", &mut interp_order);
        let _ = pip_get_integer(b"VerboseOutput", &mut rmw.i_verbose);
        let _ = pip_get_float(b"ScaleTransforms", &mut xf_scale);
        let mut iz = 0;
        let _ = pip_get_boolean(b"SameSizeAsInput", &mut iz);
        if iz != 0 {
            rmw.nxyz_out[0] = rmw.nxyz_in[0];
            rmw.nxyz_out[2] = rmw.nxyz_in[2];
        }
        {
            let [nx, ny, nz] = &mut rmw.nxyz_out;
            let _ = pip_get_three_integers(b"OutputSizeXYZ", nx, ny, nz);
        }
        if pipgetstring_(b"TransformFile", &mut file_inv) != 0 {
            exit_error("No file with inverse transforms specified");
        }
    } else {
        print!(
            " Enter path name of directory for temporary files, \n or Return to use current directory: "
        );
        let _ = std::io::stdout().flush();
        read_stdin_line(&mut temp_dir);
        //
        print!(
            " X, Y, and Z dimensions of the output file (/ for{:5}{:5}{:5}): ",
            rmw.nxyz_out[0], rmw.nxyz_out[1], rmw.nxyz_out[2]
        );
        let _ = std::io::stdout().flush();
        {
            let [nx, ny, nz] = &mut rmw.nxyz_out;
            let mut stdin = std::io::stdin().lock();
            if let Err(err) = list_read(
                &mut stdin,
                &mut [
                    ListItem::Integer(nx),
                    ListItem::Integer(ny),
                    ListItem::Integer(nz),
                ],
            ) {
                let _ = std::io::stdout().flush();
                match err {
                    ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                    ListReadError::Error => {
                        eprintln!("Fortran runtime error: Bad integer for item 1 in list input")
                    }
                }
                exit(2);
            }
        }
        //
        // get matrix for inverse transforms
        //
        println!(" Enter name of file with inverse transformations");
        read_stdin_line(&mut file_inv);
    }
    pip_done();
    if interp_order < 2 {
        base_int = 0.;
    }
    //
    let unit1 = read_warp_file_header(
        &mut file_inv,
        &mut num_input,
        &mut num_loc_x,
        &mut num_loc_y,
        &mut num_loc_z,
        &mut x_loc_start,
        &mut y_loc_start,
        &mut z_loc_start,
        &mut dx_loc,
        &mut dy_loc,
        &mut dz_loc,
        xf_scale,
    );

    let mut num_loc_tot = num_loc_y * num_loc_z * num_loc_x;
    if num_loc_tot == 0 {
        exit_error("The warp file specifies 0 positions on one axis");
    }
    let mut aloc_temp = vec![0.0_f32; 9 * num_loc_tot as usize];
    let mut dloc_temp = vec![0.0_f32; 3 * num_loc_tot as usize];
    let mut solve_temp = vec![false; num_loc_tot as usize];

    read_warp_transforms(
        unit1,
        num_input,
        num_loc_tot,
        num_loc_x,
        num_loc_y,
        num_loc_z,
        &mut x_loc_start,
        &mut y_loc_start,
        &mut z_loc_start,
        &mut dx_loc,
        &mut dy_loc,
        &mut dz_loc,
        xf_scale,
        &mut x_loc_max,
        &mut y_loc_max,
        &mut z_loc_max,
        &mut aloc_temp,
        &mut dloc_temp,
        &mut solve_temp,
    );
    //
    // determine additional positions needed to cover plane of volume
    let (nx_out, ny_out, nz_out) = (rmw.nxyz_out[0], rmw.nxyz_out[1], rmw.nxyz_out[2]);
    let mut nx_low_add = 0;
    let mut nx_high_add = 0;
    let mut ny_low_add = 0;
    let mut ny_high_add = 0;
    let mut nz_low_add = 0;
    let mut nz_high_add = 0;
    if num_loc_y > 1 {
        nx_low_add = 0.max(((x_loc_start + nx_out as f32 / 2.) / dx_loc).round() as i32);
        nx_high_add = 0.max(
            ((nx_out as f32 / 2. - (x_loc_start + (num_loc_y - 1) as f32 * dx_loc)) / dx_loc)
                .round() as i32,
        );
    }
    if num_loc_x > num_loc_z {
        ny_low_add = 0.max(((y_loc_start + ny_out as f32 / 2.) / dy_loc).round() as i32);
        ny_high_add = 0.max(
            ((ny_out as f32 / 2. - (y_loc_start + (num_loc_x - 1) as f32 * dy_loc)) / dy_loc)
                .round() as i32,
        );
    } else if num_loc_z > 1 {
        nz_low_add = 0.max(((z_loc_start + nz_out as f32 / 2.) / dz_loc).round() as i32);
        nz_high_add = 0.max(
            ((nz_out as f32 / 2. - (z_loc_start + (num_loc_z - 1) as f32 * dz_loc)) / dz_loc)
                .round() as i32,
        );
    }
    //
    // If any are being added, shift the transforms up in the array
    // and adjust the parameters
    let new_loc_x = num_loc_y + nx_low_add + nx_high_add;
    let new_loc_y = num_loc_x + ny_low_add + ny_high_add;
    let new_loc_z = num_loc_z + nz_low_add + nz_high_add;
    num_loc_tot = new_loc_x * new_loc_y * new_loc_z;
    let mut a_loc = vec![0.0_f32; 9 * num_loc_tot as usize];
    let mut dxyz_loc = vec![0.0_f32; 3 * num_loc_tot as usize];
    let mut solved = vec![false; num_loc_tot as usize];

    shift_transforms(
        &aloc_temp,
        &dloc_temp,
        &solve_temp,
        num_loc_y,
        num_loc_x,
        num_loc_z,
        &mut a_loc,
        &mut dxyz_loc,
        &mut solved,
        new_loc_x,
        new_loc_y,
        new_loc_z,
        nx_low_add,
        ny_low_add,
        nz_low_add,
    );
    x_loc_start -= nx_low_add as f32 * dx_loc;
    y_loc_start -= ny_low_add as f32 * dy_loc;
    z_loc_start -= nz_low_add as f32 * dz_loc;
    num_loc_y = new_loc_x;
    num_loc_x = new_loc_y;
    num_loc_z = new_loc_z;
    drop((aloc_temp, dloc_temp, solve_temp));
    //
    // Fill in missing transforms by extrapolation and weighted averaging
    fill_in_transforms(
        &mut a_loc,
        &mut dxyz_loc,
        &solved,
        num_loc_y,
        num_loc_x,
        num_loc_z,
        dx_loc,
        dy_loc,
        dz_loc,
    );
    let aloc = |i: usize, j: usize, l: usize| a_loc[(i - 1) + 3 * (j - 1) + 9 * (l - 1)];
    let dloc = |i: usize, l: usize| dxyz_loc[(i - 1) + 3 * (l - 1)];
    if !blank(&fill_file) {
        //
        // Output file of filled in transforms
        let mut unit1 = dopen(1, &fortran_string(&fill_file), "new", "f");
        // `104 format(i5,2i6,3f11.2,3f10.4)`
        let mut text = format!(
            "{:5}{:6}{:6}{}{}{}{}{}{}\n",
            num_loc_y,
            num_loc_x,
            num_loc_z,
            format_f(x_loc_start as f64, 11, 2),
            format_f(y_loc_start as f64, 11, 2),
            format_f(z_loc_start as f64, 11, 2),
            format_f(dx_loc as f64, 10, 4),
            format_f(dy_loc as f64, 10, 4),
            format_f(dz_loc as f64, 10, 4)
        );
        for iz in 1..=num_loc_z {
            for iy in 1..=num_loc_x {
                for ix in 1..=num_loc_y {
                    let l = (ix + (iy - 1) * num_loc_y + (iz - 1) * num_loc_y * num_loc_x) as usize;
                    // `103 format(3f9.1)`
                    text.push_str(&format_f(
                        (x_loc_start + (ix - 1) as f32 * dx_loc) as f64,
                        9,
                        1,
                    ));
                    text.push_str(&format_f(
                        (y_loc_start + (iy - 1) as f32 * dy_loc) as f64,
                        9,
                        1,
                    ));
                    text.push_str(&format_f(
                        (z_loc_start + (iz - 1) as f32 * dz_loc) as f64,
                        9,
                        1,
                    ));
                    text.push('\n');
                    // `102 format(3f11.6,f11.3)`
                    for i in 1..=3 {
                        for j in 1..=3 {
                            text.push_str(&format_f(aloc(i, j, l) as f64, 11, 6));
                        }
                        text.push_str(&format_f(dloc(i, l) as f64, 11, 3));
                        text.push('\n');
                    }
                }
            }
        }
        let _ = unit1.write_all(text.as_bytes());
    }
    if !blank(&patch_file) {
        //
        // Output patch file of inverse vectors for positions in output volume
        let mut unit1 = dopen(1, &fortran_string(&patch_file), "new", "f");
        let mut text = format!("{:8} positions\n", num_loc_y * num_loc_x * num_loc_z);
        for iz in 1..=num_loc_z {
            for iy in 1..=num_loc_x {
                for ix in 1..=num_loc_y {
                    let l = (ix + (iy - 1) * num_loc_y + (iz - 1) * num_loc_y * num_loc_x) as usize;
                    let xp = x_loc_start + (ix - 1) as f32 * dx_loc;
                    let yp = y_loc_start + (iy - 1) as f32 * dy_loc;
                    let zp = z_loc_start + (iz - 1) as f32 * dz_loc;
                    let dx =
                        aloc(1, 1, l) * xp + aloc(1, 2, l) * yp + aloc(1, 3, l) * zp + dloc(1, l)
                            - xp;
                    let dy =
                        aloc(2, 1, l) * xp + aloc(2, 2, l) * yp + aloc(2, 3, l) * zp + dloc(2, l)
                            - yp;
                    let dz =
                        aloc(3, 1, l) * xp + aloc(3, 2, l) * yp + aloc(3, 3, l) * zp + dloc(3, l)
                            - zp;
                    let ind_x = (xp + nx_out as f32 / 2.).round() as i32;
                    let ind_y = (yp + ny_out as f32 / 2.).round() as i32;
                    let ind_z = (zp + nz_out as f32 / 2.).round() as i32;
                    // `105 format(3i8,3f12.2)`
                    text.push_str(&format!(
                        "{:8}{:8}{:8}{}{}{}\n",
                        ind_x,
                        ind_y,
                        ind_z,
                        format_f(dx as f64, 12, 2),
                        format_f(dy as f64, 12, 2),
                        format_f(dz as f64, 12, 2)
                    ));
                }
            }
        }
        let _ = unit1.write_all(text.as_bytes());
    }
    //
    // Exit if patch or warp output files
    if !blank(&patch_file) || !blank(&fill_file) {
        exit(0);
    }
    //
    // Get mean inverse transform
    for ja in 1..=num_loc_tot as usize {
        let mut matrix = [0.0_f32; 9];
        matrix.copy_from_slice(&a_loc[9 * (ja - 1)..9 * ja]);
        inv_matrix(&matrix, &mut rmw.a_inv);
        for k in 0..9 {
            a_fwd[k] += rmw.a_inv[k] / num_loc_tot as f32;
        }
    }
    //
    // DNM 7/26/02: transfer pixel spacing to same axes
    //
    let delta = iiu_ret_delta(5);
    for i in 0..3 {
        cell[i] = rmw.nxyz_out[i] as f32 * delta[i];
        cell[i + 3] = 90.;
        rmw.cxyz_out[i] = rmw.nxyz_out[i] as f32 / 2.;
    }
    //
    // Unless one layer in Y, allow 5% of memory for transforms
    // `warpvol.f90:278` truncates the product, so `-memory 1` becomes 0 and
    // native sets up a negative number of cubes, writes a bare header and
    // exits 0 (BUGS.md).  Defined behaviour: a positive entered limit never
    // drops below 1 MB.
    if num_loc_x > 1 {
        let entered = rmw.memory_lim;
        rmw.memory_lim = (0.95_f32 * rmw.memory_lim as f32) as i32;
        if entered > 0 && rmw.memory_lim < 1 {
            rmw.memory_lim = 1;
        }
    }
    //
    // Get provisional setup of cubes then find actual limits of input
    // cubes with this setup - loop until no new extra pixels needed
    //
    let temp_dir_str = fortran_string(&temp_dir);
    let mut num_extra = 0;
    let mut new_extra = 1;
    while new_extra > 0 {
        setup_cubes_scratch(
            &mut rmw,
            &a_fwd,
            &a_loc,
            num_loc_tot,
            num_extra,
            " ",
            &temp_dir_str,
            &mut temp_ext,
            &tim,
            true,
        );
        //
        new_extra = 0;
        for iz_cube in 1..=rmw.n_cubes[2] {
            for ix_cube in 1..=rmw.n_cubes[0] {
                for iy_cube in 1..=rmw.n_cubes[1] {
                    test_cube_faces(
                        &rmw,
                        &a_loc,
                        &dxyz_loc,
                        x_loc_start,
                        dx_loc,
                        y_loc_start,
                        dy_loc,
                        z_loc_start,
                        dz_loc,
                        num_loc_y,
                        num_loc_x,
                        num_loc_z,
                        ix_cube,
                        iy_cube,
                        iz_cube,
                        &mut new_extra,
                        &mut in_min,
                        &mut in_max,
                    );
                }
            }
        }
        num_extra += new_extra;
    }
    //
    println!(
        "{:>12}  extra pixels needed in cubes for final setup",
        num_extra
    );
    //
    // Get setup again for real this time and open output file after
    // all potential errors are past
    //
    imopen(6, &file_out, "NEW");
    setup_cubes_scratch(
        &mut rmw,
        &a_fwd,
        &a_loc,
        num_loc_tot,
        num_extra,
        &filein,
        &temp_dir_str,
        &mut temp_ext,
        &tim,
        true,
    );
    //
    // Set up axes and strides and allocate the array for transforms
    let mut iy = rmw.idim_out[1];
    if num_loc_x == 1 && (num_loc_z > 1 || rmw.iout_xaxis == 3) {
        iy = 1;
    }
    let (inner_axis, iouter_axis, inner_stride, iouter_stride);
    if rmw.iout_xaxis == 3 {
        inner_axis = 3_usize;
        iouter_axis = 1_usize;
        inner_stride = rmw.idim_out[0] * rmw.idim_out[1];
        iouter_stride = 1;
    } else {
        inner_axis = 1;
        iouter_axis = 3;
        inner_stride = 1;
        iouter_stride = rmw.idim_out[0] * rmw.idim_out[1];
    }
    let ix = rmw.idim_out[inner_axis - 1];
    // `offsetIn(3, ix, iy)`, `aPix(3, 3, ix, iy)`.
    let dim_ix = ix as usize;
    let n_pix = dim_ix * iy as usize;
    let mut offset_in: Vec<f32> = Vec::new();
    let mut a_pix: Vec<f32> = Vec::new();
    if offset_in.try_reserve_exact(3 * n_pix).is_err()
        || a_pix.try_reserve_exact(9 * n_pix).is_err()
    {
        exit_error("Failed to allocate arrays for precomputed transforms");
    }
    offset_in.resize(3 * n_pix, 0.);
    a_pix.resize(9 * n_pix, 0.);
    let pix = |ix: i32, iy: i32| (ix - 1) as usize + dim_ix * (iy - 1) as usize;
    //
    //
    let mut labels = [[0_u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    labels[0] = rmw.title;
    iiu_create_header(6, &rmw.nxyz_out, &rmw.nxyz_out, rmw.mode, &labels, 0);
    iiu_alt_cell(6, &cell);
    iiu_trans_labels(6, 5);
    // `302 format('WARPVOL: 3-D warping of tomogram:',t57,a9,2x,a8)`
    let mut titlech = [b' '; 80];
    let head = b"WARPVOL: 3-D warping of tomogram:";
    titlech[..head.len()].copy_from_slice(head);
    titlech[56..65].copy_from_slice(&dat);
    titlech[67..75].copy_from_slice(&tim);
    rmw.title = titlech;
    dmin = 1.0e20;
    dmax = -dmin;
    tsum = 0.;
    let mut num_done = 0;
    tilt_old = iiu_ret_tilt(5);
    iiu_alt_tilt(6, &tilt_old);
    let (cx_out, cy_out, cz_out) = (rmw.cxyz_out[0], rmw.cxyz_out[1], rmw.cxyz_out[2]);
    //
    // loop on layers of cubes in Z, do all I/O to complete layer
    //
    iz_sec[5] = 0;
    for iz_cube in 1..=rmw.n_cubes[2] {
        icube[2] = iz_cube;
        //
        // initialize files and counters
        //
        for i in 1..=4 {
            if rmw.need_scratch[i - 1] {
                unsafe { iiu_set_position(i as i32, 0, 0) };
            }
            iz_sec[i - 1] = 0;
        }
        //
        // loop on the cubes in the layer
        //
        for ind_outer in 1..=rmw.lim_outer {
            for ind_inner in 1..=rmw.lim_inner {
                let cpu_start = cputime();
                let wall_start = wall_time();
                cube_indexes(
                    rmw.iout_xaxis,
                    ind_inner,
                    ind_outer,
                    &mut ix_cube,
                    &mut iy_cube,
                );
                icube[0] = ix_cube;
                icube[1] = iy_cube;
                let mut new_extra_arg = -1;
                test_cube_faces(
                    &rmw,
                    &a_loc,
                    &dxyz_loc,
                    x_loc_start,
                    dx_loc,
                    y_loc_start,
                    dy_loc,
                    z_loc_start,
                    dz_loc,
                    num_loc_y,
                    num_loc_x,
                    num_loc_z,
                    ix_cube,
                    iy_cube,
                    iz_cube,
                    &mut new_extra_arg,
                    &mut in_min,
                    &mut in_max,
                );
                let mut if_empty = 0;
                for i in 0..3 {
                    if in_min[i] > in_max[i] {
                        if_empty = 1;
                    }
                }
                //
                // load the input cube
                let plane = rmw.input_dim[0] as usize * rmw.input_dim[1] as usize;
                if if_empty == 0 {
                    if rmw.i_verbose > 1 {
                        println!(
                            "{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}",
                            ix_cube,
                            iy_cube,
                            iz_cube,
                            in_min[2],
                            in_max[2],
                            in_max[2] + 1 - in_min[2]
                        );
                    }
                    for iz in in_min[2]..=in_max[2] {
                        unsafe { iiu_set_position(5, iz, 0) };
                        if rmw.i_verbose > 1 {
                            println!(
                                "{:5}{:5}{:5}{:5}{:5}{:5}{:5}{:5}{:5}",
                                ix_cube,
                                iy_cube,
                                iz_cube,
                                iz,
                                iz + 1 - in_min[2],
                                in_min[0],
                                in_max[0],
                                in_min[1],
                                in_max[1]
                            );
                            let _ = std::io::stdout().flush();
                        }
                        let offset = (iz - in_min[2]) as usize * plane;
                        if unsafe {
                            irdpas(
                                5,
                                &mut rmw.array[offset..],
                                rmw.input_dim[0],
                                rmw.input_dim[1],
                                in_min[0],
                                in_max[0],
                                in_min[1],
                                in_max[1],
                            )
                        }
                        .is_err()
                        {
                            exit_error("Reading image file");
                        }
                    }
                }
                let cube_x = rmw.nxyz_cube[(ix_cube - 1) as usize][0];
                let cube_y = rmw.nxyz_cube[(iy_cube - 1) as usize][1];
                let start_y = rmw.ixyz_cube[(iy_cube - 1) as usize][1];
                //
                // prepare offsets and limits
                let x_offs_out = (rmw.ixyz_cube[(icube[inner_axis - 1] - 1) as usize]
                    [inner_axis - 1]
                    - 1) as f32
                    - rmw.cxyz_out[inner_axis - 1];
                let ix_limit = in_max[0] + 1 - in_min[0];
                let iy_limit = in_max[1] + 1 - in_min[1];
                let iz_limit = in_max[2] + 1 - in_min[2];
                //
                // If one layer of cubes in Z, precompute all transforms because it
                // is invariant in Z
                if if_empty == 0 && num_loc_z == 1 && rmw.iout_xaxis != 3 {
                    if rmw.i_verbose > 0 {
                        println!(" Precomputing for layer");
                    }
                    let z_cen = rmw.ixyz_cube[(iz_cube - 1) as usize][2] as f32 + cz_out;
                    for iy in 1..=cube_y {
                        let y_cen = (start_y + iy - 1) as f32 - cy_out;
                        for ix in 1..=cube_x {
                            let x_cen = ix as f32 + x_offs_out;
                            let p = pix(ix, iy);
                            let (ap, op) = (
                                <&mut [f32; 9]>::try_from(&mut a_pix[9 * p..9 * p + 9]).unwrap(),
                                <&mut [f32; 3]>::try_from(&mut offset_in[3 * p..3 * p + 3])
                                    .unwrap(),
                            );
                            interp_inv(
                                &a_loc,
                                &dxyz_loc,
                                x_loc_start,
                                dx_loc,
                                y_loc_start,
                                dy_loc,
                                z_loc_start,
                                dz_loc,
                                num_loc_y,
                                num_loc_x,
                                num_loc_z,
                                x_cen,
                                y_cen,
                                z_cen,
                                ap,
                                op,
                            );
                            for i in 0..3 {
                                op[i] = op[i] + 1. + rmw.nxyz_in[i] as f32 / 2. - in_min[i] as f32;
                            }
                        }
                    }
                }
                //
                // loop over Z segments of the output cube (multiple slices in case
                // where the X axis is being output as the Z axis)
                let mut iz_start = 1;
                let mut num_zleft = rmw.nxyz_cube[(iz_cube - 1) as usize][2];
                while num_zleft > 0 {
                    let num_zto_do = num_zleft.min(rmw.max_zout);
                    iz_end = iz_start + num_zto_do - 1;
                    if if_empty == 0 {
                        // Set up index limits for inner and outer loops
                        let (inner_start, inner_end, iouter_start, iouter_end) =
                            if rmw.iout_xaxis == 3 {
                                (iz_start, iz_end, 1, cube_x)
                            } else {
                                (1, cube_x, iz_start, iz_end)
                            };
                        let ind_base =
                            -rmw.idim_out[0] - iz_start * rmw.idim_out[0] * rmw.idim_out[1];
                        //
                        // Loop on the outer axis in the Z segment
                        for iouter in iouter_start..=iouter_end {
                            xyz_cen[iouter_axis - 1] = (rmw.ixyz_cube
                                [(icube[iouter_axis - 1] - 1) as usize][iouter_axis - 1]
                                + iouter
                                - 1) as f32
                                - rmw.cxyz_out[iouter_axis - 1];
                            let outer_cen = xyz_cen[iouter_axis - 1];
                            //
                            // get matrices for each inner position either for the one
                            // position in Y or for every one
                            if !(num_loc_z == 1 && rmw.iout_xaxis != 3) {
                                let mut iyxf = cube_y;
                                if num_loc_x == 1 {
                                    iyxf = 1;
                                }
                                // Parallelism native lacks (owner-approved, 2026-09-26,
                                // under the standing rule for parallelism added once
                                // single-thread parity is reached; precedent:
                                // `sliceMedianFilter`'s general path).  The source runs
                                // this loop serially (`warpvol.f90:455-466`).  Row `iy`
                                // writes only `aPix(:, :, ix, iy)` / `offsetIn(:, ix, iy)`,
                                // which is row `iy - 1` of `dim_ix` pixels, and reads
                                // nothing any row writes; each pixel's `interpInv` call
                                // and offset update are exactly the serial ones.
                                // `xyzCen` is per row: its Y and inner components are set
                                // before every call and the outer one (distinct axis) is
                                // the value set above, so each row sees the same centres;
                                // nothing reads the inner/Y components after this loop
                                // (both are set again before their next use).  Rows can
                                // therefore run on any number of threads with the same
                                // result; serial and parallel run the same closure.
                                assert!(
                                    dim_ix > 0 && iyxf as usize * dim_ix <= a_pix.len() / 9,
                                    "warpvol: precompute rows outside aPix"
                                );
                                let xyz_cen0 = xyz_cen;
                                let (a_loc, dxyz_loc, nxyz_in) = (&a_loc, &dxyz_loc, rmw.nxyz_in);
                                let pre_row =
                                    |(iy0, (a_row, o_row)): (usize, (&mut [f32], &mut [f32]))| {
                                        let iy = iy0 as i32 + 1;
                                        let mut xyz_cen = xyz_cen0;
                                        xyz_cen[1] = (start_y + iy - 1) as f32 - cy_out;
                                        for ix in inner_start..=inner_end {
                                            xyz_cen[inner_axis - 1] = ix as f32 + x_offs_out;
                                            // `pix(ix, iy)` within row `iy`.
                                            let p = (ix - 1) as usize;
                                            let (ap, op) = (
                                                <&mut [f32; 9]>::try_from(
                                                    &mut a_row[9 * p..9 * p + 9],
                                                )
                                                .unwrap(),
                                                <&mut [f32; 3]>::try_from(
                                                    &mut o_row[3 * p..3 * p + 3],
                                                )
                                                .unwrap(),
                                            );
                                            interp_inv(
                                                a_loc,
                                                dxyz_loc,
                                                x_loc_start,
                                                dx_loc,
                                                y_loc_start,
                                                dy_loc,
                                                z_loc_start,
                                                dz_loc,
                                                num_loc_y,
                                                num_loc_x,
                                                num_loc_z,
                                                xyz_cen[0],
                                                xyz_cen[1],
                                                xyz_cen[2],
                                                ap,
                                                op,
                                            );
                                            for i in 0..3 {
                                                op[i] = op[i] + 1. + nxyz_in[i] as f32 / 2.
                                                    - in_min[i] as f32;
                                            }
                                        }
                                    };
                                let num_threads = num_omp_threads(8);
                                if num_threads > 1 && iyxf > 1 {
                                    let pool = pool.get_or_insert_with(|| {
                                        rayon::ThreadPoolBuilder::new()
                                            .num_threads(num_threads as usize)
                                            .build()
                                            .unwrap()
                                    });
                                    pool.install(|| {
                                        use rayon::prelude::*;
                                        a_pix
                                            .par_chunks_mut(9 * dim_ix)
                                            .zip(offset_in.par_chunks_mut(3 * dim_ix))
                                            .take(iyxf as usize)
                                            .enumerate()
                                            .for_each(&pre_row)
                                    });
                                } else {
                                    a_pix
                                        .chunks_mut(9 * dim_ix)
                                        .zip(offset_in.chunks_mut(3 * dim_ix))
                                        .take(iyxf as usize)
                                        .enumerate()
                                        .for_each(&pre_row);
                                }
                            }
                            let thread_wall = wall_time();
                            let num_threads = num_omp_threads(8);
                            //
                            // parallelize loop on Y and inner axes
                            //
                            let dim1 = rmw.input_dim[0] as usize;
                            let dmean_in = rmw.dmean_in;
                            let idim_out1 = rmw.idim_out[0];
                            let array = &rmw.array;
                            // `!$OMP PARALLEL DO` over `iy`: iteration `iy` stores only
                            // `brray(indBase + ix*innerStride + iy*idimOut(1) +
                            // iouter*iouterStride)` for its own `iy`, and with
                            // `ix < idimOut(inner)`, `iy < idimOut(2)` these are distinct
                            // elements for distinct `iy`; every other value it uses is
                            // read-only.  So the rows can run on any number of threads
                            // and each element is the same store of the same value.
                            struct Out(*mut f32, usize);
                            unsafe impl Sync for Out {}
                            unsafe impl Send for Out {}
                            let out = Out(rmw.brray.as_mut_ptr(), rmw.brray.len());
                            // Every read below is at `(x, y, z)` with `1 <= x <= ixLimit`,
                            // `1 <= y <= iyLimit`, `1 <= z <= izLimit` (the loads are guarded by
                            // the range test and the `+1`/`-1` neighbours are clamped into it), so
                            // one check of the three limits against the allocated shape bounds
                            // every index; after it the reads need no per-element check.  The
                            // source reads out of its array if the limits ever exceeded the
                            // shape; here that stops the program instead.
                            assert!(
                                ix_limit as usize <= dim1
                                    && (iy_limit as usize) * dim1 <= plane
                                    && (iz_limit.max(1) as usize) * plane <= array.len(),
                                "input region larger than the allocated array"
                            );
                            let arr = move |x: i32, y: i32, z: i32| -> f32 {
                                // SAFETY: bounded by the assertion above (see its comment).
                                unsafe {
                                    *array.get_unchecked(
                                        (x - 1) as usize
                                            + dim1 * (y - 1) as usize
                                            + plane * (z - 1) as usize,
                                    )
                                }
                            };
                            let row = |iy: i32| {
                                // Capture `out` whole, not its raw-pointer field.
                                let out = &out;
                                // Copy the loop invariants into locals: read through
                                // the closure's captured references they would be
                                // reloaded after every store through `out`, which
                                // may alias them as far as the compiler knows.
                                let (arr, ix_limit, iy_limit, iz_limit, dmean_in) =
                                    (arr, ix_limit, iy_limit, iz_limit, dmean_in);
                                let (ind_base, inner_stride, idim_out1, iouter_stride) =
                                    (ind_base, inner_stride, idim_out1, iouter_stride);
                                let (inner_start, inner_end, x_offs_out, base_int) =
                                    (inner_start, inner_end, x_offs_out, base_int);
                                let (inner_axis, iouter_axis, outer_cen, interp_order) =
                                    (inner_axis, iouter_axis, outer_cen, interp_order);
                                let (a_pix, offset_in): (&[f32], &[f32]) = (&a_pix, &offset_in);
                                let y_cen = (start_y + iy - 1) as f32 - cy_out;
                                let mut iyxf = iy;
                                if num_loc_x == 1 {
                                    iyxf = 1;
                                }
                                // One check per row bounds every `aPix`/`offsetIn`
                                // read and every store below: `p` and the output
                                // index are both increasing in `ix` (`innerStride`
                                // is positive), so checking the two ends of the row
                                // covers every pixel between them.
                                if inner_start <= inner_end {
                                    let ind = |ix: i32| {
                                        ind_base as i64
                                            + ix as i64 * inner_stride as i64
                                            + iy as i64 * idim_out1 as i64
                                            + iouter as i64 * iouter_stride as i64
                                            - 1
                                    };
                                    assert!(
                                        inner_start >= 1
                                            && iyxf >= 1
                                            && inner_stride > 0
                                            && 9 * pix(inner_end, iyxf) + 9 <= a_pix.len()
                                            && 3 * pix(inner_end, iyxf) + 3 <= offset_in.len()
                                            && ind(inner_start) >= 0
                                            && ind(inner_end) < out.1 as i64,
                                        "warpvol: row outside its arrays"
                                    );
                                }
                                // Fortran `int()` is a bare `cvttss2si` (NaN or
                                // out of range gives `i32::MIN`); Rust's `as`
                                // saturates.  Either value fails the `>= 1` /
                                // `<= limit` test below, so the outcome is the
                                // same; the bare instruction just skips the clamp.
                                #[inline(always)]
                                fn cvt(x: f32) -> i32 {
                                    #[cfg(target_arch = "x86_64")]
                                    {
                                        // SAFETY: SSE is part of the x86-64 baseline.
                                        unsafe {
                                            core::arch::x86_64::_mm_cvttss_si32(
                                                core::arch::x86_64::_mm_set_ss(x),
                                            )
                                        }
                                    }
                                    #[cfg(not(target_arch = "x86_64"))]
                                    {
                                        crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si(x)
                                    }
                                }
                                for ix in inner_start..=inner_end {
                                    let x_cen = ix as f32 + x_offs_out;
                                    let p = pix(ix, iyxf);
                                    // SAFETY: `p <= pix(innerEnd, iyxf)`, checked
                                    // against both arrays above.
                                    let ap =
                                        unsafe { &*(a_pix.as_ptr().add(9 * p) as *const [f32; 9]) };
                                    let op = unsafe {
                                        &*(offset_in.as_ptr().add(3 * p) as *const [f32; 3])
                                    };
                                    // aPix(i, j, ix, iyxf) is ap[(i-1) + 3*(j-1)]
                                    let ia = 3 * (inner_axis - 1);
                                    let io = 3 * (iouter_axis - 1);
                                    //
                                    // get indices in array of input data
                                    let xp =
                                        ap[ia] * x_cen + ap[3] * y_cen + ap[io] * outer_cen + op[0];
                                    let yp = ap[1 + ia] * x_cen
                                        + ap[4] * y_cen
                                        + ap[1 + io] * outer_cen
                                        + op[1];
                                    let zp = ap[2 + ia] * x_cen
                                        + ap[5] * y_cen
                                        + ap[2 + io] * outer_cen
                                        + op[2];
                                    let mut bval = dmean_in;
                                    //
                                    // do generalized evaluation of whether pixel is doable
                                    //
                                    let ixp = cvt(xp + base_int);
                                    let iyp = cvt(yp + base_int);
                                    let izp = cvt(zp + base_int);
                                    if ixp >= 1
                                        && ixp <= ix_limit
                                        && iyp >= 1
                                        && iyp <= iy_limit
                                        && izp >= 1
                                        && izp <= iz_limit
                                    {
                                        let dx = xp - ixp as f32;
                                        let dy = yp - iyp as f32;
                                        let dz = zp - izp as f32;
                                        let ixp_p1 = ix_limit.min(ixp + 1);
                                        let iyp_p1 = iy_limit.min(iyp + 1);
                                        let izp_p1 = iz_limit.min(izp + 1);
                                        //
                                        if interp_order >= 2 {
                                            let ixp_m1 = 1.max(ixp - 1);
                                            let iyp_m1 = 1.max(iyp - 1);
                                            let izp_m1 = 1.max(izp - 1);
                                            //
                                            // Set up terms for quadratic interpolation
                                            // No longer omit higher-order terms, and no longer limit value to
                                            // min/max of values used for interp
                                            let dx_sq = dx * dx;
                                            let dy_sq = dy * dy;
                                            let dz_sq = dz * dz;
                                            let fx = 1. - dx_sq;
                                            let fx_m1 = 0.5 * (dx_sq - dx);
                                            let fx_p1 = fx_m1 + dx;
                                            let fy = 1. - dy_sq;
                                            let fy_m1 = 0.5 * (dy_sq - dy);
                                            let fy_p1 = fy_m1 + dy;
                                            let fz = 1. - dz_sq;
                                            let fz_m1 = 0.5 * (dz_sq - dz);
                                            let fz_p1 = fz_m1 + dz;

                                            bval = fz_m1
                                                * (fy_m1
                                                    * (fx_m1 * arr(ixp_m1, iyp_m1, izp_m1)
                                                        + fx * arr(ixp, iyp_m1, izp_m1)
                                                        + fx_p1 * arr(ixp_p1, iyp_m1, izp_m1))
                                                    + fy * (fx_m1 * arr(ixp_m1, iyp, izp_m1)
                                                        + fx * arr(ixp, iyp, izp_m1)
                                                        + fx_p1 * arr(ixp_p1, iyp, izp_m1))
                                                    + fy_p1
                                                        * (fx_m1 * arr(ixp_m1, iyp_p1, izp_m1)
                                                            + fx * arr(ixp, iyp_p1, izp_m1)
                                                            + fx_p1 * arr(ixp_p1, iyp_p1, izp_m1)))
                                                + fz * (fy_m1
                                                    * (fx_m1 * arr(ixp_m1, iyp_m1, izp)
                                                        + fx * arr(ixp, iyp_m1, izp)
                                                        + fx_p1 * arr(ixp_p1, iyp_m1, izp))
                                                    + fy * (fx_m1 * arr(ixp_m1, iyp, izp)
                                                        + fx * arr(ixp, iyp, izp)
                                                        + fx_p1 * arr(ixp_p1, iyp, izp))
                                                    + fy_p1
                                                        * (fx_m1 * arr(ixp_m1, iyp_p1, izp)
                                                            + fx * arr(ixp, iyp_p1, izp)
                                                            + fx_p1 * arr(ixp_p1, iyp_p1, izp)))
                                                + fz_p1
                                                    * (fy_m1
                                                        * (fx_m1 * arr(ixp_m1, iyp_m1, izp_p1)
                                                            + fx * arr(ixp, iyp_m1, izp_p1)
                                                            + fx_p1 * arr(ixp_p1, iyp_m1, izp_p1))
                                                        + fy * (fx_m1 * arr(ixp_m1, iyp, izp_p1)
                                                            + fx * arr(ixp, iyp, izp_p1)
                                                            + fx_p1 * arr(ixp_p1, iyp, izp_p1))
                                                        + fy_p1
                                                            * (fx_m1
                                                                * arr(ixp_m1, iyp_p1, izp_p1)
                                                                + fx * arr(ixp, iyp_p1, izp_p1)
                                                                + fx_p1
                                                                    * arr(ixp_p1, iyp_p1, izp_p1)));
                                        } else {
                                            //
                                            // Set up terms for linear interpolation
                                            //
                                            let d11 = (1. - dx) * (1. - dy);
                                            let d12 = (1. - dx) * dy;
                                            let d21 = dx * (1. - dy);
                                            let d22 = dx * dy;
                                            bval = (1. - dz)
                                                * (d11 * arr(ixp, iyp, izp)
                                                    + d12 * arr(ixp, iyp_p1, izp)
                                                    + d21 * arr(ixp_p1, iyp, izp)
                                                    + d22 * arr(ixp_p1, iyp_p1, izp))
                                                + dz * (d11 * arr(ixp, iyp, izp_p1)
                                                    + d12 * arr(ixp, iyp_p1, izp_p1)
                                                    + d21 * arr(ixp_p1, iyp, izp_p1)
                                                    + d22 * arr(ixp_p1, iyp_p1, izp_p1));
                                        }
                                    }
                                    let index = (ind_base
                                        + ix * inner_stride
                                        + iy * idim_out1
                                        + iouter * iouter_stride
                                        - 1)
                                        as usize;
                                    // SAFETY: in bounds by the row assertion; distinct
                                    // `iy` store to distinct elements (see `out`).
                                    unsafe { *out.0.add(index) = bval };
                                }
                            };
                            if num_threads > 1 {
                                let pool = pool.get_or_insert_with(|| {
                                    rayon::ThreadPoolBuilder::new()
                                        .num_threads(num_threads as usize)
                                        .build()
                                        .unwrap()
                                });
                                pool.install(|| {
                                    use rayon::prelude::*;
                                    (1..=cube_y).into_par_iter().for_each(&row)
                                });
                            } else {
                                for iy in 1..=cube_y {
                                    row(iy);
                                }
                            }
                            wall_cum += wall_time() - thread_wall;
                        }
                    } else {
                        let count = (rmw.idim_out[0] * rmw.idim_out[1] * num_zto_do) as usize;
                        rmw.brray[..count].fill(rmw.dmean_in);
                    }

                    write_one_cube(
                        &mut rmw,
                        ix_cube,
                        iy_cube,
                        num_zto_do,
                        iz_start,
                        &mut iz_sec,
                        &mut dmin,
                        &mut dmax,
                        &mut tsum,
                    );
                    //
                    // End of do while: update start and number left to do
                    iz_start = iz_end + 1;
                    num_zleft -= num_zto_do;
                }
                num_done += 1;
                if rmw.i_verbose > 0 {
                    println!(
                        "Cube CPU time:{}   Wall time:{}",
                        format_f(cputime() - cpu_start, 10, 4),
                        format_f(wall_time() - wall_start, 10, 4)
                    );
                }
                println!(
                    "Finished{:6} of{:6}",
                    num_done,
                    rmw.n_cubes[0] * rmw.n_cubes[1] * rmw.n_cubes[2]
                );
                let _ = std::io::stdout().flush();
            }
        }
        //
        // whole layer of cubes in z is done.  now reread and compose one
        // row of the output section at a time in array
        //
        recompose_cubes(&mut rmw, iz_cube, &mut dmin, &mut dmax, &mut tsum);
        if rmw.chunked_hdf {
            iz_sec[5] += iz_end;
        }
    }
    //
    if rmw.i_verbose > 0 {
        println!("Thread wall time {}", format_f(wall_cum, 10, 4));
    }
    let dmean = tsum / nz_out as f32;
    iiu_write_header(6, &rmw.title, 1, dmin, dmax, dmean);
    for i in 1..=4 {
        if rmw.need_scratch[i - 1] {
            unsafe { iiu_close(i as i32) };
        }
    }
    unsafe {
        iiu_close(5);
        iiu_close(6);
    }
    exit(0);
}

/// Original `testCubeFaces` (`warpvol.f90:625`).
///
/// TESTCUBEFACES tests the faces of an output cube at its corners and at all
/// positions corresponding to transforms.
pub fn test_cube_faces(
    rmw: &RotMatWarp,
    a_loc: &[f32],
    dxyz_loc: &[f32],
    x_loc_start: f32,
    dx_loc: f32,
    y_loc_start: f32,
    dy_loc: f32,
    z_loc_start: f32,
    dz_loc: f32,
    num_loc_y: i32,
    num_loc_x: i32,
    num_loc_z: i32,
    ix_cube: i32,
    iy_cube: i32,
    iz_cube: i32,
    new_extra_arg: &mut i32,
    in_min: &mut [i32; 3],
    in_max: &mut [i32; 3],
) {
    let (mut x_cube_low, mut x_cube_high) = (0.0_f32, 0.0_f32);
    let (mut y_cube_low, mut y_cube_high) = (0.0_f32, 0.0_f32);
    let (mut z_cube_low, mut z_cube_high) = (0.0_f32, 0.0_f32);
    let (mut ix_low, mut ix_high, mut iy_low, mut iy_high, mut iz_low, mut iz_high) =
        (0_i32, 0_i32, 0_i32, 0_i32, 0_i32, 0_i32);
    let (mut dx_use, mut dy_use, mut dz_use) = (0.0_f32, 0.0_f32, 0.0_f32);
    let (mut x_loc_use, mut yloc_use, mut zloc_use) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut dxyz = [0.0_f32; 3];
    let mut am_inv = [0.0_f32; 9];
    //
    // back-transform the faces of the output cube at the corners and
    // at positions corresponding to transforms to
    // find the limiting index coordinates of the input cube
    //
    for i in 0..3 {
        in_min[i] = 100000;
        in_max[i] = -100000;
    }
    cube_test_limits(
        rmw.ixyz_cube[(iy_cube - 1) as usize][1],
        rmw.nxyz_cube[(iy_cube - 1) as usize][1],
        rmw.cxyz_out[1],
        num_loc_x,
        y_loc_start,
        dy_loc,
        &mut y_cube_low,
        &mut y_cube_high,
        &mut iy_low,
        &mut iy_high,
        &mut yloc_use,
        &mut dy_use,
    );
    cube_test_limits(
        rmw.ixyz_cube[(iz_cube - 1) as usize][2],
        rmw.nxyz_cube[(iz_cube - 1) as usize][2],
        rmw.cxyz_out[2],
        num_loc_z,
        z_loc_start,
        dz_loc,
        &mut z_cube_low,
        &mut z_cube_high,
        &mut iz_low,
        &mut iz_high,
        &mut zloc_use,
        &mut dz_use,
    );
    cube_test_limits(
        rmw.ixyz_cube[(ix_cube - 1) as usize][0],
        rmw.nxyz_cube[(ix_cube - 1) as usize][0],
        rmw.cxyz_out[0],
        num_loc_y,
        x_loc_start,
        dx_loc,
        &mut x_cube_low,
        &mut x_cube_high,
        &mut ix_low,
        &mut ix_high,
        &mut x_loc_use,
        &mut dx_use,
    );
    for jfx in ix_low..=ix_high {
        for jfy in iy_low..=iy_high {
            for jfz in iz_low..=iz_high {
                if jfx == ix_low
                    || jfx == ix_high
                    || jfy == iy_low
                    || jfy == iy_high
                    || jfz == iz_low
                    || jfz == iz_high
                {
                    // gfortran `MAX`/`MIN` of finite reals.
                    let x_cen = x_cube_low.max(x_cube_high.min(x_loc_use + jfx as f32 * dx_use));
                    let y_cen = y_cube_low.max(y_cube_high.min(yloc_use + jfy as f32 * dy_use));
                    let z_cen = z_cube_low.max(z_cube_high.min(zloc_use + jfz as f32 * dz_use));
                    interp_inv(
                        a_loc,
                        dxyz_loc,
                        x_loc_start,
                        dx_loc,
                        y_loc_start,
                        dy_loc,
                        z_loc_start,
                        dz_loc,
                        num_loc_y,
                        num_loc_x,
                        num_loc_z,
                        x_cen,
                        y_cen,
                        z_cen,
                        &mut am_inv,
                        &mut dxyz,
                    );
                    for i in 0..3 {
                        let jval = (am_inv[i] * x_cen
                            + am_inv[i + 3] * y_cen
                            + am_inv[i + 6] * z_cen
                            + dxyz[i]
                            + rmw.nxyz_in[i] as f32 / 2.)
                            .round() as i32;
                        in_min[i] = 0.max(in_min[i].min(jval - 2));
                        in_max[i] = (rmw.nxyz_in[i] - 1).min(in_max[i].max(jval + 2));
                        //
                        // See if any extra pixels are needed in input
                        //
                        if *new_extra_arg >= 0 {
                            *new_extra_arg =
                                (*new_extra_arg).max(in_max[i] + 1 - in_min[i] - rmw.input_dim[i]);
                        }
                    }
                }
            }
        }
    }
}

/// Original `cubeTestLimits` (`warpvol.f90:697`).
///
/// CUBETESTLIMITS determines limits for cube testing on one axis.
pub fn cube_test_limits(
    icube: i32,
    ncube: i32,
    cout: f32,
    nunm_loc: i32,
    start_loc: f32,
    dxyz_loc: f32,
    cube_low: &mut f32,
    cube_high: &mut f32,
    ilo: &mut i32,
    ihi: &mut i32,
    start_loc_use: &mut f32,
    dloc_use: &mut f32,
) {
    *cube_low = icube as f32 - cout;
    *cube_high = (icube + ncube) as f32 - cout;
    if nunm_loc > 1 {
        *ilo = ((*cube_low - start_loc) / dxyz_loc).floor() as i32;
        *ihi = ((*cube_high - start_loc) / dxyz_loc).ceil() as i32;
        *start_loc_use = start_loc;
        *dloc_use = dxyz_loc;
    } else {
        *start_loc_use = *cube_low;
        *dloc_use = (*cube_high - *cube_low) / 2.;
        *ilo = 0;
        *ihi = 2;
    }
}

/// Original `interpInv` (`warpvol.f90:722`).
///
/// INTERPINV takes the array of transforms and a given position xcen, ycen,
/// zcen and determines the interpolated transform minv, dxyz.
#[inline(always)]
pub fn interp_inv(
    a_loc: &[f32],
    dxyz_loc: &[f32],
    x_loc_start: f32,
    dx_loc: f32,
    y_loc_start: f32,
    dy_loc: f32,
    z_loc_start: f32,
    dz_loc: f32,
    num_loc_y: i32,
    num_loc_x: i32,
    num_loc_z: i32,
    x_cen: f32,
    y_cen: f32,
    z_cen: f32,
    a_inv: &mut [f32; 9],
    dxyz: &mut [f32; 3],
) {
    let (ix, ix1, fx1, fx, iy, iy1, fy1, fy, iz, iz1, fz1, fz);
    // Every index below lies in `1..=numLoc*` (see the clamps), so one check of
    // the three counts against the two arrays bounds all 32 corner reads; the
    // counts are also kept under 2^24 so `float(numLoc*)` is exact and the
    // clamped `x` truncates to at most `numLoc*`.
    assert!(
        (1..1 << 24).contains(&num_loc_y)
            && (1..1 << 24).contains(&num_loc_x)
            && (1..1 << 24).contains(&num_loc_z)
            && (num_loc_y as u64 * num_loc_x as u64 * num_loc_z as u64)
                .checked_mul(9)
                .is_some_and(|n| n <= a_loc.len() as u64 && n / 3 <= dxyz_loc.len() as u64),
        "interpInv: transform grid larger than its arrays"
    );
    // gfortran compiles `min(float(n), max(1., e))` to `maxss e, 1.` then
    // `minss x, float(n)` (reference `interpinv_`), and `ix = x` to a bare
    // `cvttss2si`; `x` is in `[1, n]` (a NaN `e` gives 1.), so the truncation
    // is exact integer conversion either way.
    #[inline(always)]
    fn cvt(x: f32) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE is part of the x86-64 baseline.
            unsafe { core::arch::x86_64::_mm_cvttss_si32(core::arch::x86_64::_mm_set_ss(x)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            crate::imod::flib::subrs::compat::gfortran_rt::cvttss2si(x)
        }
    }
    if num_loc_y > 1 {
        let x = minss(
            maxss(1. + (x_cen - x_loc_start) / dx_loc, 1.),
            num_loc_y as f32,
        );
        ix = cvt(x);
        ix1 = (ix + 1).min(num_loc_y);
        fx1 = x - ix as f32;
        fx = 1. - fx1;
    } else {
        ix = 1;
        ix1 = 1;
        fx1 = 0.;
        fx = 1.;
    }
    //
    if num_loc_x > 1 {
        let y = minss(
            maxss(1. + (y_cen - y_loc_start) / dy_loc, 1.),
            num_loc_x as f32,
        );
        iy = cvt(y);
        iy1 = (iy + 1).min(num_loc_x);
        fy1 = y - iy as f32;
        fy = 1. - fy1;
    } else {
        iy = 1;
        iy1 = 1;
        fy1 = 0.;
        fy = 1.;
    }
    //
    if num_loc_z > 1 {
        let z = minss(
            maxss(1. + (z_cen - z_loc_start) / dz_loc, 1.),
            num_loc_z as f32,
        );
        iz = cvt(z);
        iz1 = (iz + 1).min(num_loc_z);
        fz1 = z - iz as f32;
        fz = 1. - fz1;
    } else {
        iz = 1;
        iz1 = 1;
        fz1 = 0.;
        fz = 1.;
    }
    //
    let pos = |x: i32, y: i32, z: i32| -> usize {
        (x - 1) as usize
            + num_loc_y as usize * ((y - 1) as usize + num_loc_x as usize * (z - 1) as usize)
    };
    let (p000, p100, p001, p101) = (
        pos(ix, iy, iz),
        pos(ix1, iy, iz),
        pos(ix, iy, iz1),
        pos(ix1, iy, iz1),
    );
    let (p010, p110, p011, p111) = (
        pos(ix, iy1, iz),
        pos(ix1, iy1, iz),
        pos(ix, iy1, iz1),
        pos(ix1, iy1, iz1),
    );
    // SAFETY (all reads below): every `p` is below `numLocY*numLocX*numLocZ`
    // by the clamps, and the assertion bounds that against both arrays.
    let m = |p: usize| -> &[f32; 9] { unsafe { &*(a_loc.as_ptr().add(9 * p) as *const [f32; 9]) } };
    let v =
        |p: usize| -> &[f32; 3] { unsafe { &*(dxyz_loc.as_ptr().add(3 * p) as *const [f32; 3]) } };
    let (m000, m100, m001, m101) = (m(p000), m(p100), m(p001), m(p101));
    let (m010, m110, m011, m111) = (m(p010), m(p110), m(p011), m(p111));
    let (v000, v100, v001, v101) = (v(p000), v(p100), v(p001), v(p101));
    let (v010, v110, v011, v111) = (v(p010), v(p110), v(p011), v(p111));
    // The source's `do i / do j` computes each `aInv(i, j)` independently, so
    // the nine run in storage order `k = i + 3*(j-1)` here; each is the
    // source's expression term for term, and LLVM evaluates them in 4-wide
    // lanes (as gfortran pairs them), each lane the scalar IEEE chain.
    for k in 0..9 {
        a_inv[k] = fy
            * (fz * (fx * m000[k] + fx1 * m100[k]) + fz1 * (fx * m001[k] + fx1 * m101[k]))
            + fy1 * (fz * (fx * m010[k] + fx1 * m110[k]) + fz1 * (fx * m011[k] + fx1 * m111[k]));
    }
    for i in 0..3 {
        dxyz[i] = fy * (fz * (fx * v000[i] + fx1 * v100[i]) + fz1 * (fx * v001[i] + fx1 * v101[i]))
            + fy1 * (fz * (fx * v010[i] + fx1 * v110[i]) + fz1 * (fx * v011[i] + fx1 * v111[i]));
    }
}

/// Original `fillInTransforms` (`warpvol.f90:804`).
///
/// FILLINTRANSFORMS fills in a regular array of transforms by extrapolation
/// from the nearest transforms to each missing one, where the extrapolation
/// is a weighted mean with weights proportional to the square of distance
/// from each existing transform.
///
/// `atan2d` is inlined by gfortran as `atan2f(y, x) * 57.29578f`
/// ([`ATAN2D_FACTOR`]).  `minX`, `minY`, `minZ` are unset when no position
/// is solved (the source then reads uninitialised locals); they are 0 here.
pub fn fill_in_transforms(
    a_loc: &mut [f32],
    dxyz_loc: &mut [f32],
    solved: &[bool],
    num_loc_y: i32,
    num_loc_x: i32,
    num_loc_z: i32,
    dx_loc: f32,
    dy_loc: f32,
    dz_loc: f32,
) {
    let ix_step: [i32; 12] = [1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 0];
    let iy_step: [i32; 12] = [-1, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 1];
    let range = 2.0_f32;
    let (mut min_x, mut min_y, mut min_z) = (0_i32, 0_i32, 0_i32);
    let mut dxyz = [0.0_f32; 3];
    let pos = |x: i32, y: i32, z: i32| -> usize {
        ((x - 1) + num_loc_y * ((y - 1) + num_loc_x * (z - 1))) as usize
    };
    //
    for iz in 1..=num_loc_z {
        for iy in 1..=num_loc_x {
            for ix in 1..=num_loc_y {
                let mut dmin = 1.0e30_f32;
                let here = pos(ix, iy, iz);
                if !solved[here] {
                    //
                    // Find closest position with transform
                    for jz in 1..=num_loc_z {
                        for jy in 1..=num_loc_x {
                            for jx in 1..=num_loc_y {
                                if solved[pos(jx, jy, jz)] {
                                    let tx = (jx - ix) as f32 * dx_loc;
                                    let ty = (jy - iy) as f32 * dy_loc;
                                    let tz = (jz - iz) as f32 * dz_loc;
                                    let dist = tx * tx + ty * ty + tz * tz;
                                    if dist < dmin {
                                        dmin = dist;
                                        min_x = jx;
                                        min_y = jy;
                                        min_z = jz;
                                    }
                                }
                            }
                        }
                    }
                    //
                    // zero out the sum
                    for jx in 0..3 {
                        dxyz_loc[jx + 3 * here] = 0.;
                        for jy in 0..3 {
                            a_loc[jx + 3 * jy + 9 * here] = 0.;
                        }
                    }
                    let mut wsum = 0.0_f32;
                    //
                    // Get actual distance to look, range of indexes to search, and
                    // the criterion which is square of maximum distance
                    let mut dist = range * dmin.sqrt();
                    // gfortran `MAX(dxLoc, 1.)` of finite reals.
                    let jx_min = 1.max((ix as f32 - dist / dx_loc.max(1.) - 1.).round() as i32);
                    let jx_max =
                        num_loc_y.min((ix as f32 + dist / dx_loc.max(1.) + 1.).round() as i32);
                    let jy_min = 1.max((iy as f32 - dist / dy_loc.max(1.) - 1.).round() as i32);
                    let jy_max =
                        num_loc_x.min((iy as f32 + dist / dy_loc.max(1.) + 1.).round() as i32);
                    let jz_min = 1.max((iz as f32 - dist / dz_loc.max(1.) - 1.).round() as i32);
                    let jz_max =
                        num_loc_z.min((iz as f32 + dist / dz_loc.max(1.) + 1.).round() as i32);
                    let dist_crit = dist * dist;
                    //
                    // Get dominant index for second dimension
                    let mut indyz = 2;
                    let mut iy_stride_fac = 1;
                    let mut iz_stride_fac = 0;
                    if num_loc_z > num_loc_x {
                        indyz = 3;
                        iz_stride_fac = 1;
                        iy_stride_fac = 0;
                    }
                    //
                    // Loop in the neighborhood, find boundary points within range
                    for jz in jz_min..=jz_max {
                        for jy in jy_min..=jy_max {
                            for jx in jx_min..=jx_max {
                                let there = pos(jx, jy, jz);
                                if solved[there] {
                                    dxyz[0] = (jx - ix) as f32 * dx_loc;
                                    dxyz[1] = (jy - iy) as f32 * dy_loc;
                                    dxyz[2] = (jz - iz) as f32 * dz_loc;
                                    dist =
                                        dxyz[0] * dxyz[0] + dxyz[1] * dxyz[1] + dxyz[2] * dxyz[2];
                                    if dist <= dist_crit {
                                        //
                                        // Find dominant direction to the point
                                        let mut angle =
                                            dxyz[indyz - 1].atan2(dxyz[0]) * ATAN2D_FACTOR + 157.5;
                                        if angle < 0. {
                                            angle += 360.;
                                        }
                                        let mut ind_dom = (angle / 45. + 1.) as i32;
                                        ind_dom = 1.max(8.min(ind_dom));
                                        //
                                        // Check that this point is a boundary, i.e. does not
                                        // have a neighbor in any one of the 5 directions toward
                                        // or at right angles to the dominant direction
                                        let mut boundary = false;
                                        for is in ind_dom..=ind_dom + 4 {
                                            let neigh_x = jx + ix_step[(is - 1) as usize];
                                            let neigh_y =
                                                jy + iy_stride_fac * iy_step[(is - 1) as usize];
                                            let neigh_z =
                                                jz + iz_stride_fac * iy_step[(is - 1) as usize];
                                            if neigh_x >= 1
                                                && neigh_x <= num_loc_y
                                                && neigh_y >= 1
                                                && neigh_y <= num_loc_x
                                                && neigh_z >= 1
                                                && neigh_z <= num_loc_z
                                                && !solved[pos(neigh_x, neigh_y, neigh_z)]
                                            {
                                                boundary = true;
                                            }
                                        }
                                        //
                                        // For boundary or min point, add to weighted sum
                                        if boundary || (jx == min_x && jy == min_y && jz == min_z) {
                                            for kx in 0..3 {
                                                dxyz_loc[kx + 3 * here] +=
                                                    dxyz_loc[kx + 3 * there] / dist;
                                                for ky in 0..3 {
                                                    a_loc[kx + 3 * ky + 9 * here] +=
                                                        a_loc[kx + 3 * ky + 9 * there] / dist;
                                                }
                                            }
                                            wsum += 1. / dist;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    //
                    // divide by weight sum
                    for jx in 0..3 {
                        dxyz_loc[jx + 3 * here] /= wsum;
                        for jy in 0..3 {
                            a_loc[jx + 3 * jy + 9 * here] /= wsum;
                        }
                    }
                }
            }
        }
    }
}

/// Original `shiftTransforms` (`warpvol.f90:945`).
///
/// SHIFTTRANSFORMS shifts the transforms to fill the center of a larger
/// array.
pub fn shift_transforms(
    aloc_in: &[f32],
    dloc_in: &[f32],
    solve_temp: &[bool],
    num_loc_y: i32,
    num_loc_x: i32,
    num_loc_z: i32,
    a_loc: &mut [f32],
    dxyz_loc: &mut [f32],
    solved: &mut [bool],
    new_loc_x: i32,
    new_loc_y: i32,
    _new_loc_z: i32,
    nx_add: i32,
    ny_add: i32,
    nz_add: i32,
) {
    for iz in (1..=num_loc_z).rev() {
        let izn = iz + nz_add;
        for iy in (1..=num_loc_x).rev() {
            let iyn = iy + ny_add;
            for ix in (1..=num_loc_y).rev() {
                let old = ((ix - 1) + num_loc_y * ((iy - 1) + num_loc_x * (iz - 1))) as usize;
                if solve_temp[old] {
                    let ixn = ix + nx_add;
                    let new =
                        ((ixn - 1) + new_loc_x * ((iyn - 1) + new_loc_y * (izn - 1))) as usize;
                    solved[new] = true;
                    for jy in 0..3 {
                        dxyz_loc[jy + 3 * new] = dloc_in[jy + 3 * old];
                        for jx in 0..3 {
                            a_loc[jx + 3 * jy + 9 * new] = aloc_in[jx + 3 * jy + 9 * old];
                        }
                    }
                }
            }
        }
    }
}
