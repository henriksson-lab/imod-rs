//! Translation of `IMOD/flib/image/xfproduct.f`.
//!
//! The whole unit is the main program, [`xfproduct`].  Transforms are held as
//! `[f32; 6]` in Fortran `(2,3)` storage order; the grid arrays
//! `dxGrid(nxgDim, nygDim, 3)` are flat, slot `s` starting at
//! `(s - 1) * nxgDim * nygDim`.  Library calls go to the C entry points with
//! the Fortran wrappers' `iz - 1` and `rows = 2` inlined (`warpwrapfort.c`,
//! `linearxforms.c`).

use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::flib::subrs::xfsubs::xfwrite::xfwrite;
use crate::imod::libcfshr::linearxforms::{xf_apply, xf_copy, xf_mult};
use crate::imod::libcfshr::parse_params::{pip_done, pip_get_integer, pip_get_two_floats};
use crate::imod::libwarp::warpfiles::{
    get_grid_parameters, get_linear_transform, get_num_warp_points, get_warp_grid, get_warp_points,
    grid_size_from_spacing, new_warp_file, set_current_warp_file, set_linear_transform,
    set_warp_grid, set_warp_points, write_warp_file,
};
use crate::imod::libwarp::warputils::{
    expand_and_extrap_grid, multiply_warpings, read_check_warp_file,
};
use std::io::{BufReader, BufWriter, Write};

/// `parameter (nflimit=100000)` (`xfproduct.f:12`).
const NFLIMIT: i32 = 100000;

/// `parameter (numOptions = 6)` (`xfproduct.f:44`).
const XFPRODUCT_NUM_OPTIONS: i32 = 6;

/// Fallback PIP table `options(1)` (`xfproduct.f:46-48`).
const XFPRODUCT_OPTIONS: &str = "in1:InputFile1:FN:@in2:InputFile2:FN:@output:OutputFile:FN:@\
scale:ScaleShifts:FP:@one:OneXformToMultiply:I:@help:usage:B:";

/// Original program `xfproduct` (`xfproduct.f:1`).
pub fn xfproduct() {
    let mut f: [Vec<[f32; 6]>; 2] = [Vec::new(), Vec::new()];
    let mut ftmp = [[0.0_f32; 6]; 3];
    let mut gfile = [String::new(), String::new()];
    let mut out_file = String::new();
    let mut err_string = String::new();
    //
    let mut ierr: i32;
    let nout: i32;
    let mut i: usize;
    let mut nsingle: i32;
    let mut indcopy: i32;
    let mut inds = [0_i32; 2];
    let mut scales = [0.0_f32; 2];
    let mut x_all_str = [0.0_f32; 2];
    let mut x_all_end = [0.0_f32; 2];
    let mut y_all_str = [0.0_f32; 2];
    let mut y_all_end = [0.0_f32; 2];
    let mut warping = [false; 2];
    let mut control_pts = [false; 3];
    let mut linear = [false; 2];
    let mut linear_only = [false; 3];
    let mut need_grid: bool;
    let mut ind_warp_file = [0_i32; 3];
    // `equivalence (nfirst, numXforms(1)), (nsecond, numXforms(2))`
    let mut num_xforms = [0_i32; 2];
    let mut iflags = 0_i32;
    let mut ibin = 0_i32;
    let mut nx = [0_i32; 3];
    let mut ny = [0_i32; 3];
    let mut iz: i32;
    let mut nxg_dim: i32;
    let mut nyg_dim: i32;
    let mut n_control = 0_i32;
    let (mut nx_gr_tmp, mut ny_gr_tmp) = (0_i32, 0_i32);
    let mut max_control = [0_i32; 2];
    let mut x_all_int: f32;
    let mut y_all_int: f32;
    let (mut x_int_tmp, mut y_int_tmp, mut x_str_tmp, mut y_str_tmp) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let (mut xcen, mut ycen) = (0.0_f32, 0.0_f32);
    let mut xvec_tmp = [0.0_f32; 3];
    let mut yvec_tmp = [0.0_f32; 3];
    let mut xcon_tmp = [0.0_f32; 3];
    let mut ycon_tmp = [0.0_f32; 3];
    let mut pixel_size = [0.0_f32; 3];
    let mut x_control: Vec<f32> = Vec::new();
    let mut y_control: Vec<f32> = Vec::new();
    let mut x_vector: Vec<f32> = Vec::new();
    let mut y_vector: Vec<f32> = Vec::new();
    let mut dx_grid: Vec<f32> = Vec::new();
    let mut dy_grid: Vec<f32> = Vec::new();
    let mut if_use_2nd: i32;
    let mut nx_grids = [0_i32; 2];
    let mut ny_grids = [0_i32; 2];
    // Fixed in translation (BUGS.md, `xftoxg` / `xfproduct`): uninitialised
    // in the source when there is no PIP input (`xfproduct.f:95` assigns it
    // only inside `if (pipinput)`), and read when both inputs are warpings,
    // so native's interactive run takes the "entered scales" branch or not
    // depending on stack residue.  Interactive entry cannot enter scales, so
    // it is defined as 0: no scales entered, the first file is scaled by the
    // warpings' pixel-size ratio.
    let mut if_scales = 0_i32;
    let (mut x_new_cont, mut y_new_cont, mut x_new_cpv, mut y_new_cpv): (f32, f32, f32, f32);
    let warp_scale: f32;
    let mut x_starts = [0.0_f32; 2];
    let mut y_starts = [0.0_f32; 2];
    let mut x_intervals = [0.0_f32; 2];
    let mut y_intervals = [0.0_f32; 2];
    //
    let pipinput: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);
    //
    scales[0] = 1.;
    scales[1] = 1.;
    nsingle = -1;
    //
    pip_read_or_parse_options(
        &[XFPRODUCT_OPTIONS],
        XFPRODUCT_NUM_OPTIONS,
        "xfproduct",
        "ERROR: XFPRODUCT - ",
        true,
        3,
        2,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pipinput = num_opt_arg + num_non_opt_arg > 0;
    //
    // Get all the filenames
    if pip_get_in_out_file(
        "InputFile1",
        1,
        "File of transforms applied first",
        &mut gfile[0],
        320,
    ) != 0
    {
        exit_error("NO FIRST INPUT FILE SPECIFIED");
    }
    if pip_get_in_out_file(
        "InputFile2",
        2,
        "File of transforms applied second",
        &mut gfile[1],
        320,
    ) != 0
    {
        exit_error("NO SECOND INPUT FILE SPECIFIED");
    }
    if pip_get_in_out_file(
        "OutputFile",
        3,
        "New file for product transforms",
        &mut out_file,
        320,
    ) != 0
    {
        exit_error("NO OUTPUT FILE SPECIFIED");
    }
    //
    // Open the input files and read transforms if linear files
    for ifs in 0..2 {
        ind_warp_file[ifs] = read_check_warp_file(
            gfile[ifs].trim_end_matches(' '),
            0,
            1,
            &mut nx[ifs],
            &mut ny[ifs],
            &mut num_xforms[ifs],
            &mut ibin,
            &mut pixel_size[ifs],
            &mut iflags,
            &mut err_string,
        );
        if ind_warp_file[ifs] < -1 {
            exit_error(&err_string);
        }
        warping[ifs] = ind_warp_file[ifs] >= 0;
        control_pts[ifs] = false;
        if warping[ifs] {
            control_pts[ifs] = (iflags / 2) % 2 != 0;
            println!(
                "Old warping file opened: {}",
                gfile[ifs].trim_end_matches(' ')
            );
            f[ifs] = vec![[0.0; 6]; num_xforms[ifs].max(0) as usize];
        } else {
            //
            // read regular linear xforms
            let unit = dopen(1, &gfile[ifs], "ro", "f");
            ierr = xfrdall2(&mut BufReader::new(unit), &mut f[ifs], NFLIMIT);
            num_xforms[ifs] = f[ifs].len() as i32;
            if ierr == 2 {
                exit_error("READING TRANSFORM FILE");
            }
            if ierr == 1 {
                exit_error("TOO MANY TRANSFORMS IN FILE FOR ARRAY");
            }
            // `close(1)`: the reader is dropped with the call above.
        }
    }
    let nfirst = num_xforms[0];
    let nsecond = num_xforms[1];
    //
    if nfirst == 0 {
        exit_error("NO TRANSFORMS IN FIRST INPUT FILE");
    }
    println!(" {nfirst:>11}  transforms in first file");
    if nsecond == 0 {
        exit_error("NO TRANSFORMS IN SECOND INPUT FILE");
    }
    println!(" {nsecond:>11}  transforms in second file");
    if nfirst > NFLIMIT || nsecond > NFLIMIT {
        exit_error("TOO MANY TRANSFORMS FOR ARRAY SIZE");
    }
    if pipinput {
        pip_get_integer(b"OneXformToMultiply", &mut nsingle);
        let (mut first_scale, mut second_scale) = (scales[0], scales[1]);
        if_scales = 1 - pip_get_two_floats(b"ScaleShifts", &mut first_scale, &mut second_scale);
        scales = [first_scale, second_scale];
    }
    pip_done();
    //
    // Check that warpings are the same size except for scaling
    if warping[0] && warping[1] {
        warp_scale = pixel_size[0] / pixel_size[1];
        if (warp_scale * nx[0] as f32).round() as i32 != nx[1]
            || (warp_scale * ny[0] as f32).round() as i32 != ny[1]
        {
            exit_error("WARPINGS THAT ARE NOT OF THE SAME SIZE IMAGE AREA CANNOT BE MULTIPLIED");
        }
        if if_scales != 0 {
            if (scales[0] / warp_scale - scales[1]).abs() > 1.0e-5 {
                exit_error("YOU MUST ENTER SCALES THAT KEEP THE WARPED IMAGE AREA THE SAME SIZE");
            }
        } else {
            scales[0] = warp_scale;
        }
    }
    //
    // Determine maximum number of control points and array sizes
    // Use preliminary assessment of whether there is control point output
    control_pts[2] = (warping[0] != warping[1]) && (control_pts[0] || control_pts[1]);
    nxg_dim = 0;
    nyg_dim = 0;
    for ifs in 0..2 {
        max_control[ifs] = 0;
        linear_only[ifs] = !warping[ifs];
        if warping[ifs] {
            set_current_warp_file(ind_warp_file[ifs]);
            x_all_str[ifs] = nx[ifs] as f32;
            y_all_str[ifs] = ny[ifs] as f32;
            x_all_end[ifs] = 0.;
            y_all_end[ifs] = 0.;
            x_all_int = nx[ifs] as f32;
            y_all_int = ny[ifs] as f32;
            for iz in 1..=num_xforms[ifs] {
                if get_linear_transform(iz - 1, &mut f[ifs][(iz - 1) as usize], 2) != 0 {
                    exit_error("GETTING LINEAR TRANSFORM FROM WARP FILE");
                }
                need_grid = !control_pts[ifs];
                if control_pts[ifs] {
                    if get_num_warp_points(iz - 1, &mut n_control) != 0 {
                        exit_error("GETTING NUMBER OF CONTROL POINTS");
                    }
                    max_control[ifs] = max_control[ifs].max(n_control);
                    if n_control <= 3 {
                        if get_warp_points(
                            iz - 1,
                            &mut xcon_tmp,
                            &mut ycon_tmp,
                            &mut xvec_tmp,
                            &mut yvec_tmp,
                        ) != 0
                        {
                            exit_error("GETTING CONTROL POINTS");
                        }
                        for i in 0..n_control as usize {
                            if xvec_tmp[i].abs() > 1.0e-3 || yvec_tmp[i].abs() > 1.0e-3 {
                                exit_error(
                                    "CANNOT WORK WITH 3 OR FEWER CONTROL POINTS THAT HAVE NON-ZERO VECTORS",
                                );
                            }
                        }
                    } else if !control_pts[2] {
                        if grid_size_from_spacing(iz - 1, -1., -1., 1) != 0 {
                            exit_error("SETTING GRID SIZE FROM SPACING OF CONTROL POINTS");
                        }
                        need_grid = true;
                    }
                }
                if need_grid {
                    if get_grid_parameters(
                        iz - 1,
                        &mut nx_gr_tmp,
                        &mut ny_gr_tmp,
                        &mut x_str_tmp,
                        &mut y_str_tmp,
                        &mut x_int_tmp,
                        &mut y_int_tmp,
                    ) != 0
                    {
                        exit_error("GETTING GRID PARAMETERS");
                    }
                    nxg_dim = nxg_dim.max(nx_gr_tmp);
                    nyg_dim = nyg_dim.max(ny_gr_tmp);
                    // `xfproduct.f:157-162`: the reference object has the
                    // running value as `minss` destination and the product as
                    // `maxss` destination.
                    x_all_str[ifs] = minss(x_all_str[ifs], x_str_tmp);
                    x_all_int = minss(x_all_int, x_int_tmp);
                    x_all_end[ifs] = maxss((nx_gr_tmp - 1) as f32 * x_int_tmp, x_all_end[ifs]);
                    y_all_str[ifs] = minss(y_all_str[ifs], y_str_tmp);
                    y_all_int = minss(y_all_int, y_int_tmp);
                    y_all_end[ifs] = maxss((ny_gr_tmp - 1) as f32 * y_int_tmp, y_all_end[ifs]);
                }
            }

            if x_all_str[ifs] <= 0.
                || x_all_end[ifs] >= nx[ifs] as f32
                || y_all_str[ifs] <= 0.
                || y_all_end[ifs] >= ny[ifs] as f32
            {
                exit_error("CANNOT WORK WITH GRIDS THAT EXTEND OUTSIDE THE DEFINED IMAGE AREA");
            }
            if x_all_str[ifs] < nx[ifs] as f32 {
                // `xfproduct.f:173-179`, operand order from the reference
                // object: `maxss xAllEnd, nx - .` but `maxss ny - ., yAllEnd`,
                // and `maxss expr, 2.` for the grid counts.
                x_all_str[ifs] = minss(x_all_str[ifs], x_all_int / 2.);
                x_all_end[ifs] = maxss(x_all_end[ifs], nx[ifs] as f32 - x_all_int / 2.);
                y_all_str[ifs] = minss(y_all_str[ifs], y_all_int / 2.);
                y_all_end[ifs] = maxss(ny[ifs] as f32 - y_all_int / 2., y_all_end[ifs]);
                iz = maxss(
                    (x_all_end[ifs] - x_all_str[ifs]) / x_all_int + 1.05,
                    2.0_f32,
                ) as i32;
                nxg_dim = nxg_dim.max(iz);
                iz = maxss(
                    (y_all_end[ifs] - y_all_str[ifs]) / y_all_int + 1.05,
                    2.0_f32,
                ) as i32;
                nyg_dim = nyg_dim.max(iz);
            }
            linear_only[ifs] = control_pts[ifs] && max_control[ifs] <= 3;
        }
    }
    linear_only[2] = linear_only[0] && linear_only[1];
    //
    // set up output file and allocate arrays
    let mut out_unit: Option<BufWriter<std::fs::File>> = None;
    if !linear_only[2] {
        let mut ifs = 2_usize;
        if linear_only[1] {
            ifs = 1;
        }
        let k = ifs - 1;
        nx[2] = (nx[k] as f32 * scales[k]).round() as i32;
        ny[2] = (ny[k] as f32 * scales[k]).round() as i32;
        xcen = nx[2] as f32 / 2.;
        ycen = ny[2] as f32 / 2.;
        pixel_size[2] = pixel_size[k] / scales[k];
        control_pts[2] = control_pts[k] && (linear_only[0] || linear_only[1]);
        iflags = 1;
        if control_pts[2] {
            iflags = 3;
        }
        ind_warp_file[2] = new_warp_file(nx[2], ny[2], ibin, pixel_size[2], iflags);
        if ind_warp_file[2] < 0 {
            exit_error("OPENING A NEW WARPING FILE");
        }

        if control_pts[2] {
            iz = max_control[0].max(max_control[1]);
            x_control = vec![0.0; iz as usize];
            y_control = vec![0.0; iz as usize];
            x_vector = vec![0.0; iz as usize];
            y_vector = vec![0.0; iz as usize];
        } else {
            dx_grid = vec![0.0; (nxg_dim * nyg_dim * 3) as usize];
            dy_grid = vec![0.0; (nxg_dim * nyg_dim * 3) as usize];
        }
        memory_error(0, "ARRAYS FOR WARPING DATA");
    } else {
        out_unit = Some(BufWriter::new(dopen(3, &out_file, "new", "f")));
    }
    let slot = (nxg_dim * nyg_dim) as usize;

    //
    // Determine how to sample the transforms when multiplying
    let mut nout_value = nfirst.min(nsecond);
    if nsecond != nfirst {
        if nsecond == 1 {
            if nsingle >= 0 && nsingle < nfirst {
                println!(" Single second transform applied to first transform #{nsingle:>12}");
            } else {
                println!(" Single second transform applied to all first transforms");
            }
            nout_value = nfirst;
        } else if nfirst == 1 {
            if nsingle >= 0 && nsingle < nsecond {
                println!(" Single first transform applied to second transform #{nsingle:>12}");
            } else {
                println!(" Single first transform applied to all second transforms");
            }
            nout_value = nsecond;
        } else {
            println!(" WARNING: XFPRODUCT - Number of transforms does not match");
        }
    }
    nout = nout_value;
    //
    // Loop on the output
    for iz in 1..=nout {
        //
        // Set up indexes to take first and second from and copy index for copy
        inds[0] = iz;
        inds[1] = iz;
        indcopy = 0;
        if nsecond != nfirst {
            if nsecond == 1 {
                inds[1] = 1;
                if nsingle >= 0 && nsingle < nfirst && nsingle != iz - 1 {
                    indcopy = 1;
                }
            } else if nfirst == 1 {
                inds[0] = 1;
                if nsingle >= 0 && nsingle < nsecond && nsingle != iz - 1 {
                    indcopy = 2;
                }
            }
        }
        //
        // Get the transforms and scale them to the output scale
        for ifs in 0..2 {
            xf_copy(&f[ifs][(inds[ifs] - 1) as usize], 2, &mut ftmp[ifs], 2);
            ftmp[ifs][4] *= scales[ifs];
            ftmp[ifs][5] *= scales[ifs];
            linear[ifs] = linear_only[ifs];
            if !linear[ifs] && control_pts[ifs] {
                set_current_warp_file(ind_warp_file[ifs]);
                get_num_warp_points(inds[ifs] - 1, &mut n_control);
                linear[ifs] = n_control <= 3;
            }
        }
        if linear[0] && linear[1] {
            //
            // If both are linear, do ordinary product or copy, then write or put in file
            if indcopy > 0 {
                let source = ftmp[(indcopy - 1) as usize];
                xf_copy(&source, 2, &mut ftmp[2], 2);
            } else {
                let (first, second) = (ftmp[0], ftmp[1]);
                xf_mult(&first, &second, &mut ftmp[2], 2);
            }
            if linear_only[2] {
                if xfwrite(out_unit.as_mut().unwrap(), &ftmp[2]).is_err() {
                    exit_error("WRITING OUT NEW TRANSFORM FILE");
                }
            } else {
                set_current_warp_file(ind_warp_file[2]);
                if set_linear_transform(iz - 1, &ftmp[2], 2) != 0 {
                    exit_error("ADDING LINEAR TRANSFORM TO NEW WARPING");
                }
                if !control_pts[2] {
                    //
                    // Output a zero grid
                    nx_grids[1] = 9.min(nxg_dim);
                    ny_grids[1] = 9.min(nyg_dim);
                    // `nx(3) / nxGrids(2)` is an integer division.
                    x_intervals[1] = (nx[2] / nx_grids[1]) as f32;
                    x_starts[1] = x_intervals[1] / 2.;
                    y_intervals[1] = (ny[2] / ny_grids[1]) as f32;
                    y_starts[1] = y_intervals[1] / 2.;
                    for jy in 0..ny_grids[1] {
                        for ix in 0..nx_grids[1] {
                            let index = slot + (ix + jy * nxg_dim) as usize;
                            dx_grid[index] = 0.;
                            dy_grid[index] = 0.;
                        }
                    }
                    if set_warp_grid(
                        iz - 1,
                        nx_grids[1],
                        ny_grids[1],
                        x_starts[1],
                        y_starts[1],
                        x_intervals[1],
                        y_intervals[1],
                        &dx_grid[slot..],
                        &dy_grid[slot..],
                        nxg_dim,
                    ) != 0
                    {
                        exit_error("STORING ZERO WARPING GRID");
                    }
                }
            }
        } else {
            //
            // Warping involved somewhere: get data for first and second in turn
            for ifs in 0..2 {
                if linear[ifs] || (indcopy > 0 && indcopy != ifs as i32 + 1) {
                    continue;
                }
                set_current_warp_file(ind_warp_file[ifs]);
                if control_pts[ifs] {
                    //
                    // If there are control points and we are writing control points, get them
                    if control_pts[2] {
                        get_num_warp_points(inds[ifs] - 1, &mut n_control);
                        if get_warp_points(
                            inds[ifs] - 1,
                            &mut x_control,
                            &mut y_control,
                            &mut x_vector,
                            &mut y_vector,
                        ) != 0
                        {
                            exit_error("GETTING WARP CONTROL POINTS");
                        }
                        //
                        // Scale all components
                        for j in 0..n_control as usize {
                            x_control[j] *= scales[ifs];
                            y_control[j] *= scales[ifs];
                            x_vector[j] *= scales[ifs];
                            y_vector[j] *= scales[ifs];
                        }
                    } else {
                        //
                        // Otherwise set up to get a grid
                        if grid_size_from_spacing(inds[ifs] - 1, -1., -1., 1) != 0 {
                            exit_error("SETTING GRID SIZE FROM SPACING OF CONTROL POINTS");
                        }
                    }
                }
                if !control_pts[2] {
                    //
                    // Now get a grid and scale it
                    let base = ifs * slot;
                    if get_warp_grid(
                        inds[ifs] - 1,
                        &mut nx_grids[ifs],
                        &mut ny_grids[ifs],
                        &mut x_starts[ifs],
                        &mut y_starts[ifs],
                        &mut x_intervals[ifs],
                        &mut y_intervals[ifs],
                        &mut dx_grid[base..base + slot],
                        &mut dy_grid[base..base + slot],
                        nxg_dim,
                    ) != 0
                    {
                        exit_error("GETTING A WARP GRID");
                    }
                    if !(linear[0] || control_pts[ifs] || indcopy != 0)
                        && expand_and_extrap_grid(
                            &mut dx_grid[base..base + slot],
                            &mut dy_grid[base..base + slot],
                            nxg_dim,
                            nyg_dim,
                            &mut nx_grids[ifs],
                            &mut ny_grids[ifs],
                            &mut x_starts[ifs],
                            &mut y_starts[ifs],
                            x_intervals[ifs],
                            y_intervals[ifs],
                            x_all_str[ifs],
                            y_all_str[ifs],
                            x_all_end[ifs],
                            y_all_end[ifs],
                            0,
                            nx[ifs],
                            0,
                            ny[ifs],
                        ) != 0
                    {
                        exit_error("EXPANDING A WARP GRID");
                    }
                    for jy in 0..ny_grids[ifs] {
                        for ix in 0..nx_grids[ifs] {
                            let index = base + (ix + jy * nxg_dim) as usize;
                            dx_grid[index] *= scales[ifs];
                        }
                    }
                    for jy in 0..ny_grids[ifs] {
                        for ix in 0..nx_grids[ifs] {
                            let index = base + (ix + jy * nxg_dim) as usize;
                            dy_grid[index] *= scales[ifs];
                        }
                    }
                    x_starts[ifs] *= scales[ifs];
                    y_starts[ifs] *= scales[ifs];
                    x_intervals[ifs] *= scales[ifs];
                    y_intervals[ifs] *= scales[ifs];
                }
            }
            //
            // Now do something unless copying
            if_use_2nd = -1;
            if indcopy == 0 {
                if linear[0] {
                    //
                    // If first one is linear, multiply transforms and set up to copy the warp
                    let (first, second) = (ftmp[0], ftmp[1]);
                    xf_mult(&first, &second, &mut ftmp[1], 2);
                    indcopy = 2;
                } else if linear[1] && control_pts[2] {
                    //
                    // If second one is linear and first one is still control points, transform
                    // them
                    for j in 0..n_control.max(0) as usize {
                        (x_new_cont, y_new_cont) =
                            xf_apply(&ftmp[1], xcen, ycen, x_control[j], y_control[j], 2);
                        (x_new_cpv, y_new_cpv) = xf_apply(
                            &ftmp[1],
                            xcen,
                            ycen,
                            x_control[j] + x_vector[j],
                            y_control[j] + y_vector[j],
                            2,
                        );
                        x_control[j] = x_new_cont;
                        y_control[j] = y_new_cont;
                        x_vector[j] = x_new_cpv - x_new_cont;
                        y_vector[j] = y_new_cpv - y_new_cont;
                    }
                    let (first, second) = (ftmp[0], ftmp[1]);
                    xf_mult(&first, &second, &mut ftmp[1], 2);
                    indcopy = 2;
                } else {
                    //
                    // multiplication of two grids: make new grid the one with smaller interval
                    if_use_2nd = 0;
                    if linear[1] {
                        nx_grids[1] = 0;
                        ny_grids[1] = 0;
                    } else if x_intervals[1] * y_intervals[1] < x_intervals[0] * y_intervals[0] {
                        if_use_2nd = 1;
                    }
                    let (grids_12, grid_3) = dx_grid.split_at_mut(2 * slot);
                    let (dy_grids_12, dy_grid_3) = dy_grid.split_at_mut(2 * slot);
                    let (first, second) = (ftmp[0], ftmp[1]);
                    if multiply_warpings(
                        &grids_12[..slot],
                        &dy_grids_12[..slot],
                        nxg_dim,
                        nx_grids[0],
                        ny_grids[0],
                        x_starts[0],
                        y_starts[0],
                        x_intervals[0],
                        y_intervals[0],
                        &first,
                        xcen,
                        ycen,
                        &grids_12[slot..],
                        &dy_grids_12[slot..],
                        nxg_dim,
                        nx_grids[1],
                        ny_grids[1],
                        x_starts[1],
                        y_starts[1],
                        x_intervals[1],
                        y_intervals[1],
                        &second,
                        grid_3,
                        dy_grid_3,
                        &mut ftmp[2],
                        if_use_2nd,
                        2,
                    ) != 0
                    {
                        exit_error("MULTIPLYING WARPINGS");
                    }
                    indcopy = 3;
                }
            }
            //
            // Put out the warping now indicated by indcopy
            set_current_warp_file(ind_warp_file[2]);
            if set_linear_transform(iz - 1, &ftmp[(indcopy - 1) as usize], 2) != 0 {
                exit_error("ADDING LINEAR TRANSFORM TO NEW WARPING");
            }
            if control_pts[2] {
                if set_warp_points(
                    iz - 1,
                    n_control,
                    &x_control,
                    &y_control,
                    &x_vector,
                    &y_vector,
                ) != 0
                {
                    exit_error("STORING NEW WARP CONTROL POINTS");
                }
            } else {
                i = (if_use_2nd + 1) as usize;
                if i == 0 {
                    i = indcopy as usize;
                }
                let base = (indcopy - 1) as usize * slot;
                if set_warp_grid(
                    iz - 1,
                    nx_grids[i - 1],
                    ny_grids[i - 1],
                    x_starts[i - 1],
                    y_starts[i - 1],
                    x_intervals[i - 1],
                    y_intervals[i - 1],
                    &dx_grid[base..],
                    &dy_grid[base..],
                    nxg_dim,
                ) != 0
                {
                    exit_error("STORING NEW WARPING GRID");
                }
            }
        }
    }
    if linear_only[2] {
        // `close(3)`
        if let Some(mut unit) = out_unit.take() {
            if unit.flush().is_err() {
                exit_error("WRITING OUT NEW TRANSFORM FILE");
            }
        }
        println!(" {nout:>11}  new transforms written");
    } else {
        if write_warp_file(out_file.trim_end_matches(' '), 0) != 0 {
            exit_error("WRITING NEW WARPING FILE");
        }
        println!(
            "{nout:>5} new transforms written to warping file: {}",
            out_file.trim_end_matches(' ')
        );
    }
    let _ = std::io::stdout().flush();
    crate::imod::libcfshr::b3dutil::exit(0);
}
