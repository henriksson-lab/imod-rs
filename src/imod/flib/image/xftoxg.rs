//! Translation of `IMOD/flib/image/xftoxg.f90`.
//!
//! The main program maps to [`xftoxg`]; its three contained procedures map to
//! [`cumulative_warp`], [`fit_warp_component`] and [`robust_fit_to_nat`], and
//! the two external program units to [`group_rotations`] and [`angle_diff`].
//! A contained procedure reaches its host's variables by host association;
//! here the host variables it reads or writes are passed explicitly, and the
//! ones it merely uses as scratch (`i`, `j`, `kl`, `ipow`, `x`, `y`,
//! `slopeTmp`, `bint`) are its own locals, because no host code reads them
//! after the call.  The stale `nxGrTmp`..`yIntTmp` that `cumulativeWarp`
//! reuses for a section without a grid, and the `numFit` that the host prints
//! after `robustFitToNat`, are passed by reference so their host lifetime is
//! kept.
//!
//! Transforms are held as `[f32; 6]` in Fortran `(2,3)` storage order, so a
//! Fortran element `(i,j)` is index `(i-1) + 2*(j-1)`.  Library calls go to
//! the C entry points with the Fortran wrappers' `iz - 1` and `rows = 2`
//! inlined (`warpwrapfort.c`, `linearxforms.c`, `amat_to_rotmagstr.c`).

use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, memory_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::statsubs::polyfit::polyfit;
use crate::imod::flib::subrs::xfsubs::xflincom::xflincom;
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::flib::subrs::xfsubs::xfwrite::xfwrite;
use crate::imod::libcfshr::amat_to_rotmagstr::{amat_to_rotmag, rotmag_to_amat};
use crate::imod::libcfshr::linearxforms::{xf_copy, xf_invert, xf_mult, xf_unit};
use crate::imod::libcfshr::parse_params::{pip_get_float, pip_get_integer, pip_get_three_floats};
use crate::imod::libcfshr::regression::robust_regress;
use crate::imod::libwarp::warpfiles::{
    clear_warp_file, get_grid_parameters, get_linear_transform, get_num_warp_points, get_warp_grid,
    grid_size_from_spacing, new_warp_file, separate_linear_transform, set_current_warp_file,
    set_grid_size_to_make, set_linear_transform, set_warp_grid, write_warp_file,
};
use crate::imod::libwarp::warputils::{
    expand_and_extrap_grid, invert_warp_grid, multiply_warpings, read_check_warp_file,
};
use std::io::{BufReader, BufWriter, Write};

/// `parameter (LIMSEC = 100000)` (`xftoxg.f90:13`).
const LIMSEC: i32 = 100000;

/// `parameter (numOptions = 8)` (`xftoxg.f90:44`).
const XFTOXG_NUM_OPTIONS: i32 = 8;

/// Fallback PIP table `options(1)` (`xftoxg.f90:46-49`).
const XFTOXG_OPTIONS: &str = "input:InputFile:FN:@goutput:GOutputFile:FN:@nfit:NumberToFit:I:@\
ref:ReferenceSection:I:@order:OrderOfPolynomialFit:I:@\
mixed:HybridFits:I:@range:RangeOfAnglesInAverage:F:@help:usage:B:";

/// Original program `xftoxg` (`xftoxg.f90:11`).
pub fn xftoxg() {
    let mut f: Vec<[f32; 6]> = Vec::new();
    let mut nat: Vec<[f32; 6]>;
    let mut g_avg = [0.0_f32; 6];
    let mut gcen = [0.0_f32; 6];
    let mut nat_avg = [0.0_f32; 6];
    let mut gcen_inv = [0.0_f32; 6];
    let mut prod = [0.0_f32; 6];
    let mut slope = [[0.0_f32; 6]; 10];
    let mut intcp = [0.0_f32; 6];
    let mut nat_prod = [0.0_f32; 6];
    let mut x = vec![0.0_f32; LIMSEC as usize];
    let mut y = vec![0.0_f32; LIMSEC as usize];
    let mut slope_tmp = [0.0_f32; 10];
    let mut igroup = vec![0_i32; LIMSEC as usize];
    let mut n_control = vec![0_i32; LIMSEC as usize];
    let mut in_file = String::new();
    let mut out_file = String::new();
    let mut err_string = String::new();
    let mut dx_grid: Vec<f32> = Vec::new();
    let mut dy_grid: Vec<f32> = Vec::new();
    let mut dx_cum: Vec<f32> = Vec::new();
    let mut dy_cum: Vec<f32> = Vec::new();
    let mut dx_prod: Vec<f32> = Vec::new();
    let mut dy_prod: Vec<f32> = Vec::new();
    let mut g_warp: Vec<[f32; 6]> = Vec::new();

    let mut nhybrid: i32;
    let mut if_shift: i32;
    let mut iorder: i32;
    let mut nlist = 0_i32;
    let mut kl_low = 0_i32;
    let mut kl_high = 0_i32;
    let (mut nx, mut ny) = (0_i32, 0_i32);
    let mut num_fit = 0_i32;
    let mut ierr: i32;
    let mut num_groups = 0_i32;
    let mut num_in_first = 0_i32;
    let mut iorder_use = 0_i32;
    let mut ibin = 0_i32;
    let mut ind_fit_center = 0_i32;
    let mut iref_sec: i32;
    let ind_warp_input: i32;
    let mut ind_warp_output = 0_i32;
    let mut nx_grid = 0_i32;
    let mut ny_grid = 0_i32;
    let (mut nx_gr_tmp, mut ny_gr_tmp) = (0_i32, 0_i32);
    let mut iflags = 0_i32;
    let (mut x_start, mut y_start) = (0.0_f32, 0.0_f32);
    let (mut x_interval, mut y_interval) = (0.0_f32, 0.0_f32);
    let (mut x_str_tmp, mut y_str_tmp, mut x_int_tmp, mut y_int_tmp) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let warping: bool;
    let mut control = false;
    let mut robust_linear: bool;
    let mut delta_angle: f32;
    let mut ang_diff: f32;
    let mut bint = 0.0_f32;
    let mut angle_range: f32;
    let mut pixel_size = 0.0_f32;
    let (mut xcen, mut ycen) = (0.0_f32, 0.0_f32);
    let (mut x_end, mut y_end) = (0.0_f32, 0.0_f32);
    let num_cols: i32;
    let mut num_rows = 0_i32;
    let mut max_rob_iter: i32;
    let mut r_mat: Vec<f32> = Vec::new();
    let mut r_sds: Vec<f32> = Vec::new();
    let mut r_work: Vec<f32> = Vec::new();
    let mut r_means: Vec<f32> = Vec::new();
    let mut b_solve: Vec<f32> = Vec::new();
    let mut c_solve: Vec<f32> = Vec::new();
    let mut scale_kfactor: f32;
    let mut rob_change_max: f32;
    let mut rob_oscill_max: f32;
    let mut frac_zero_wgt: f32;

    let pip_input: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    //
    // defaults
    //
    if_shift = 7;
    nhybrid = 0;
    iorder = 1;
    iref_sec = 0;
    angle_range = 999.;
    robust_linear = false;
    max_rob_iter = 200;
    scale_kfactor = 1.;
    rob_change_max = 0.02;
    rob_oscill_max = 0.04;
    frac_zero_wgt = 0.2;
    //
    // initialize
    //
    pip_read_or_parse_options(
        &[XFTOXG_OPTIONS],
        XFTOXG_NUM_OPTIONS,
        "xftoxg",
        "ERROR: XFTOXG - ",
        true,
        1,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    pip_input = num_opt_arg + num_non_opt_arg > 0;

    //
    // get input parameters
    //
    if pip_input {
        if pip_get_integer(b"ReferenceSection", &mut iref_sec) == 0 {
            if iref_sec < 1 {
                exit_error("Reference section number must be positive");
            }
            if_shift = 0;
        }
        pip_get_integer(b"NumberToFit", &mut if_shift);
        if if_shift < 0 {
            exit_error("A negative value for nfit is not allowed");
        }
        if iref_sec > 0 && if_shift > 0 {
            exit_error("A reference section can only be used with global alignment");
        }

        pip_get_integer(b"OrderOfPolynomialFit", &mut iorder);
        pip_get_integer(b"HybridFits", &mut nhybrid);

        if nhybrid != 0 {
            if if_shift == 0 {
                exit_error("You cannot use hybrid and global alignment together");
            }
            if nhybrid < 0 {
                exit_error("A negative value for hybrid alignment is not allowed");
            }
            nhybrid = 1.max(4.min(nhybrid));
        }

        pip_get_float(b"RangeOfAnglesInAverage", &mut angle_range);
        pip_get_logical("RobustFit", &mut robust_linear);
        pip_get_float(b"KFactorScaling", &mut scale_kfactor);
        pip_get_integer(b"MaximumIterations", &mut max_rob_iter);
        pip_get_three_floats(
            b"IterationParams",
            &mut frac_zero_wgt,
            &mut rob_change_max,
            &mut rob_oscill_max,
        );
    } else {
        // `read(*,*)` / `read(5,*)` with no `END=`/`ERR=`: the gfortran
        // runtime reports a failed read and stops with status 2.
        let read_abort = |err: ListReadError| -> ! {
            let _ = std::io::stdout().flush();
            match err {
                ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
            }
            crate::imod::libcfshr::b3dutil::exit(2);
        };
        let stdin = std::io::stdin();
        let mut stdin = stdin.lock();
        println!(" Enter 0 to align all sections to a single average central alignment;");
        println!("    or 1 to align to an average alignment that shifts based");
        println!("          on a polynomial fit to the whole stack;");
        println!("    or N to align each section to an average alignment based");
        // `write(*,'(1x,a,/,a,$)')`
        print!(
            "          on a polynomial fit to the nearest N sections,\n   or -1 or -N for a hybrid of central and shifting alignments: "
        );
        let _ = std::io::stdout().flush();
        if let Err(err) = list_read(&mut stdin, &mut [ListItem::Integer(&mut if_shift)]) {
            read_abort(err);
        }
        //
        if if_shift < 0 {
            print!(
                " Enter # of parameters to do central alignment on (1 for rotation only,\n2 for translation only, 3 for both, 4 for trans/rot/mag): "
            );
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(&mut stdin, &mut [ListItem::Integer(&mut nhybrid)]) {
                read_abort(err);
            }
            if_shift = if_shift.wrapping_abs();
            nhybrid = 1.max(4.min(nhybrid));
        }
        //
        if if_shift > 0 {
            print!(" Order of polynomial to fit (1 for linear): ");
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(&mut stdin, &mut [ListItem::Integer(&mut iorder)]) {
                read_abort(err);
            }
        }
    }
    if if_shift > 1 {
        iorder = (if_shift - 1).min(iorder);
    }
    iorder = 1.max(10.min(iorder));
    //
    if pip_get_in_out_file(
        "InputFile",
        1,
        "Input file of f transforms",
        &mut in_file,
        320,
    ) != 0
    {
        exit_error("No input file specified");
    }
    if pip_get_in_out_file(
        "GOutputFile",
        2,
        "Output file of g transforms",
        &mut out_file,
        320,
    ) != 0
    {
        // `i = len_trim(inFile)`; `inFile(i - 1:i) .ne. 'xf'`
        let trimmed = in_file.trim_end_matches(' ');
        let i = trimmed.len();
        if i < 2 || &trimmed.as_bytes()[i - 2..i] != b"xf" {
            exit_error("No output file specified and input filename does not end in xf");
        }
        out_file = format!("{}g", &trimmed[..i - 1]);
    }
    //
    // Determine if there is warping
    ind_warp_input = read_check_warp_file(
        in_file.trim_end_matches(' '),
        0,
        1,
        &mut nx,
        &mut ny,
        &mut nlist,
        &mut ibin,
        &mut pixel_size,
        &mut iflags,
        &mut err_string,
    );
    if ind_warp_input < -1 {
        exit_error(&err_string);
    }
    warping = ind_warp_input >= 0;
    let mut out_unit: Option<BufWriter<std::fs::File>> = None;
    if !warping {
        //
        // Regular: open files, read the whole list of f's
        //
        let in_unit = dopen(1, &in_file, "old", "f");
        out_unit = Some(BufWriter::new(dopen(2, &out_file, "new", "f")));
        ierr = xfrdall2(&mut BufReader::new(in_unit), &mut f, LIMSEC);
        nlist = f.len() as i32;
        if ierr == 2 {
            exit_error("Reading transform file");
        }
        if ierr == 1 {
            exit_error("Too many transforms in file for arrays");
        }
    }
    if nlist > LIMSEC {
        exit_error("Too many transforms for arrays");
    }
    if iref_sec > nlist {
        exit_error("Reference section number too large for number of transforms");
    }
    // `f`, `g` and `nat` are `(2,3,LIMSEC)` arrays in the source; they are
    // sized to the list here, with at least one entry for the `xfUnit(g(1,1,1))`
    // that runs even on an empty list.
    let list_size = nlist.max(1) as usize;
    f.resize(list_size, [0.0; 6]);
    let mut g: Vec<[f32; 6]> = vec![[0.0; 6]; list_size];
    nat = vec![[0.0; 6]; list_size];
    //
    if if_shift > 0 && nlist < 3 {
        println!(
            "\nWARNING: XFTOXG - Computing a global alignment since there are fewer than 3 transforms"
        );
        if_shift = 0;
    }
    //
    // Set up for linear robust fits
    if if_shift > 0 {
        num_cols = iorder + 4;
        num_rows = nlist;
        if if_shift > 1 {
            num_rows = if_shift;
        }
        r_mat = vec![0.0; (num_rows * (num_cols + 2)) as usize];
        b_solve = vec![0.0; (iorder * 4) as usize];
        c_solve = vec![0.0; 4];
        r_means = vec![0.0; num_cols as usize];
        r_sds = vec![0.0; num_cols as usize];
        r_work = vec![0.0; (num_cols * num_cols + 2 * num_rows) as usize];
        memory_error(0, "arrays for robust regression");
    }
    if warping {
        //
        // warping
        println!("Old warping file opened: {}", in_file.trim_end_matches(' '));
        control = (iflags / 2) % 2 != 0;
        xcen = nx as f32 / 2.;
        ycen = ny as f32 / 2.;
        //
        // Get the linear transforms and determine smallest spacing needed
        x_start = nx as f32;
        y_start = ny as f32;
        x_end = 0.;
        y_end = 0.;
        x_interval = nx as f32;
        y_interval = ny as f32;
        nx_grid = 0;
        ny_grid = 0;
        for kl in 1..=nlist {
            let k = (kl - 1) as usize;
            n_control[k] = 4;
            if control && get_num_warp_points(kl - 1, &mut n_control[k]) != 0 {
                exit_error("Getting number of control points");
            }
            // IS THIS NEEDED?
            if n_control[k] > 2 && separate_linear_transform(kl - 1) != 0 {
                exit_error("Separating out the linear transform from the warping");
            }
            if get_linear_transform(kl - 1, &mut f[k], 2) != 0 {
                exit_error("Getting linear transform from warp file");
            }
            if n_control[k] >= 4 {
                if control && grid_size_from_spacing(kl - 1, -1., -1., 1) != 0 {
                    exit_error("Setting grid size from spacing of control points");
                }
                if get_grid_parameters(
                    kl - 1,
                    &mut nx_gr_tmp,
                    &mut ny_gr_tmp,
                    &mut x_str_tmp,
                    &mut y_str_tmp,
                    &mut x_int_tmp,
                    &mut y_int_tmp,
                ) != 0
                {
                    exit_error("Getting grid parameters");
                }
                // `xftoxg.f90:219-224`.  The reference object does the four
                // `min`s as one `minps` with the running values as
                // destination, but recomputes `xInterval` for line 221 as a
                // scalar `minss xIntTmp, xInterval` (the other operand order);
                // both `max`es are `maxss product, running`.
                let x_int_scalar = minss(x_int_tmp, x_interval);
                x_start = minss(x_start, x_str_tmp);
                x_interval = minss(x_interval, x_int_tmp);
                x_end = maxss((nx_gr_tmp - 1) as f32 * x_int_scalar, x_end);
                y_start = minss(y_start, y_str_tmp);
                y_interval = minss(y_interval, y_int_tmp);
                y_end = maxss((ny_gr_tmp - 1) as f32 * y_interval, y_end);
            }
        }
        if x_start <= 0. || x_end >= nx as f32 || y_start <= 0. || y_end >= ny as f32 {
            exit_error("Cannot work with grids that extend outside the defined image area");
        }
        // Fixed in translation (BUGS.md, `xftoxg` / `xfproduct`): when no
        // section of a control-point file has 4 or more points, no section
        // contributes a grid above, `xStart`/`xEnd` stay at their "no grid"
        // starting values, and the common grid computed below has zero
        // extent and a zero interval.  Native then carries NaN through the
        // cumulative warps into the rotation angles, and `groupRotations`
        // loops forever.  There is no warping grid to align, so this is an
        // error here.
        if control && !n_control[..nlist as usize].iter().any(|&n| n >= 4) {
            exit_error("No section has enough control points (4) to define a warping grid");
        }
        //
        // Figure out the grid size and interval that fits in the range, but make it
        // fill the range
        //
        // `xftoxg.f90:234-241`.  In the reference object `xStart`/`yStart`
        // are `minss start, interval / 2.`; the `max`es are computed twice:
        // a scalar `maxss n - interval / 2., end` feeds the grid size and
        // interval, while the value stored back into `xEnd`/`yEnd` (read
        // again later) is a `maxps end, n - interval / 2.`.  The grid counts
        // are `maxss expr, 2.`.
        x_start = minss(x_start, x_interval / 2.);
        let x_end_scalar = maxss(nx as f32 - x_interval / 2., x_end);
        y_start = minss(y_start, y_interval / 2.);
        let y_end_scalar = maxss(ny as f32 - y_interval / 2., y_end);
        x_end = maxss(x_end, nx as f32 - x_interval / 2.);
        y_end = maxss(y_end, ny as f32 - y_interval / 2.);
        nx_grid = maxss((x_end_scalar - x_start) / x_interval + 1.05, 2.0_f32) as i32;
        x_interval = (x_end_scalar - x_start) / (nx_grid - 1) as f32;
        ny_grid = maxss((y_end_scalar - y_start) / y_interval + 1.05, 2.0_f32) as i32;
        y_interval = (y_end_scalar - y_start) / (ny_grid - 1) as f32;
        //
        if control {
            //
            // Then set all the parameters for control point grids
            for kl in 1..=nlist {
                set_grid_size_to_make(
                    kl - 1,
                    nx_grid,
                    ny_grid,
                    x_start,
                    y_start,
                    x_interval,
                    y_interval,
                );
            }
        }
        //
        // Allocate arrays
        let nxy = (nx_grid * ny_grid) as usize;
        dx_grid = vec![0.0; nxy];
        dy_grid = vec![0.0; nxy];
        dx_cum = vec![0.0; nxy * nlist as usize];
        dy_cum = vec![0.0; nxy * nlist as usize];
        dx_prod = vec![0.0; nxy];
        dy_prod = vec![0.0; nxy];
        g_warp = vec![[0.0; 6]; nlist as usize];
        memory_error(0, "arrays for warping grids");
        //
        // Get the cumulative transforms over the whole list, put cumulative linear in g
        cumulative_warp(
            1,
            nlist,
            &mut g,
            &f,
            &n_control,
            control,
            nx,
            ny,
            xcen,
            ycen,
            nx_grid,
            ny_grid,
            x_start,
            y_start,
            x_interval,
            y_interval,
            x_end,
            y_end,
            &mut dx_grid,
            &mut dy_grid,
            &mut dx_cum,
            &mut dy_cum,
            &mut nx_gr_tmp,
            &mut ny_gr_tmp,
            &mut x_str_tmp,
            &mut y_str_tmp,
            &mut x_int_tmp,
            &mut y_int_tmp,
        );
        //
        // Find the mean grid and take its inverse, leave in dxGrid, dyGrid
        dx_prod.fill(0.);
        dy_prod.fill(0.);
        for kl in 1..=nlist {
            let k = (kl - 1) as usize;
            xf_copy(&g[k], 2, &mut g_warp[k], 2);
            for j in 1..=ny_grid {
                for i in 1..=nx_grid {
                    let ij = ((i - 1) + (j - 1) * nx_grid) as usize;
                    dx_prod[ij] += dx_cum[ij + k * nxy] / nlist as f32;
                    dy_prod[ij] += dy_cum[ij + k * nxy] / nlist as f32;
                }
            }
        }

        invert_warp_grid(
            &dx_prod,
            &dy_prod,
            nx_grid,
            nx_grid,
            ny_grid,
            x_start,
            y_start,
            x_interval,
            y_interval,
            &g[0],
            xcen,
            ycen,
            &mut dx_grid,
            &mut dy_grid,
            &mut prod,
            2,
        );

        // Or take the inverse at the reference section
        if iref_sec > 0 {
            let r = (iref_sec - 1) as usize;
            invert_warp_grid(
                &dx_cum[r * nxy..],
                &dy_cum[r * nxy..],
                nx_grid,
                nx_grid,
                ny_grid,
                x_start,
                y_start,
                x_interval,
                y_interval,
                &g[r],
                xcen,
                ycen,
                &mut dx_grid,
                &mut dy_grid,
                &mut prod,
                2,
            );
        }

        //
        // Start a new warp file
        if if_shift < 2 {
            clear_warp_file(ind_warp_input);
        }
        iflags = 1;
        ind_warp_output = new_warp_file(nx, ny, ibin, pixel_size, iflags);
        if ind_warp_output < 0 {
            exit_error("Opening a new warping file");
        }
    }

    if !warping || if_shift > 1 {
        //
        // Regular transforms: compute g's to align all sections to the first
        // Do this for warping with local fits instead of trusting the g computed above
        xf_unit(&mut g[0], 1., 2);
        for i in 2..=nlist {
            let (before, after) = g.split_at_mut((i - 1) as usize);
            xf_mult(
                &f[(i - 1) as usize],
                &before[(i - 2) as usize],
                &mut after[0],
                2,
            );
        }
    }
    //
    // Usual treatment of regular transforms now that we have cumulative g in each case
    // Convert to "natural" transforms
    //
    delta_angle = 0.;
    for kl in 1..=nlist {
        let k = (kl - 1) as usize;
        // `amat_to_rotmag(g(1,1,kl), nat(1,1,kl), nat(1,2,kl), nat(2,1,kl),
        // nat(2,2,kl))`: the wrapper passes `amat[0], amat[2], amat[1],
        // amat[3]` (`amat_to_rotmagstr.c:254`).
        let (theta, ydtheta, smag, ydmag) = amat_to_rotmag(g[k][0], g[k][2], g[k][1], g[k][3]);
        nat[k][0] = theta;
        nat[k][2] = ydtheta;
        nat[k][1] = smag;
        nat[k][3] = ydmag;
        nat[k][4] = g[k][4];
        nat[k][5] = g[k][5];
        //
        // if rotation angle differs greatly from last one, adjust the
        // DELTANG value appropriately so that angles change in continuous
        // increments around + or - 180 degrees
        //
        if kl > 1 {
            ang_diff = nat[k - 1][0] - (nat[k][0] + delta_angle);
            if ang_diff.abs() > 180. {
                delta_angle += (360.0_f32).copysign(ang_diff);
            }
        }
        nat[k][0] += delta_angle;
    }
    //
    // average natural g's, convert back to xform, take inverse of average;
    // ginv is used if no linear fits, natav is used in hybrid cases
    // First get group with restricted rotation range
    //
    group_rotations(
        &nat,
        1,
        nlist,
        angle_range,
        &mut igroup,
        &mut num_groups,
        &mut num_in_first,
    );
    xf_unit(&mut nat_avg, 0., 2);
    for kl in 1..=nlist {
        let k = (kl - 1) as usize;
        if igroup[k] == 1 {
            let previous = nat_avg;
            xflincom(
                &previous,
                1.,
                &nat[k],
                1. / num_in_first as f32,
                &mut nat_avg,
            );
        }
    }
    // `rotmag_to_amat(natAvg(1,1), natAvg(1,2), natAvg(2,1), natAvg(2,2), gAvg)`
    g_avg[..4].copy_from_slice(&rotmag_to_amat(
        nat_avg[0], nat_avg[2], nat_avg[1], nat_avg[3],
    ));
    g_avg[4] = nat_avg[4];
    g_avg[5] = nat_avg[5];
    xf_invert(&g_avg, &mut gcen_inv, 2);
    //
    // If doing reference section, just invert its transform
    //
    if iref_sec > 0 {
        xf_invert(&g[(iref_sec - 1) as usize], &mut gcen_inv, 2);
    }
    //
    // DNM 1/24/04: fixed bug in hybrid 3 or 4:
    // do not convert natav to be the inverse!
    //
    // loop over each section
    //
    for ilist in 1..=nlist {
        if if_shift > 0 {
            // if doing line fits, set up section limits for this fit
            if if_shift == 1 {
                kl_low = 1; //ifshift = 1: take all sections
                kl_high = nlist;
            } else {
                kl_low = 1.max(ilist - if_shift / 2); //>1: take N sections centered
                kl_high = nlist.min(kl_low + if_shift - 1); //on this one, or offset
                kl_low = 1.max(kl_low.min(kl_high + 1 - if_shift));
            }
            if if_shift > 1 || (if_shift == 1 && ilist == 1) {
                //
                // do line fit to each component of the natural parameters
                // first time only if doing all sections, or each time for N
                //
                group_rotations(
                    &nat,
                    kl_low,
                    kl_high,
                    angle_range,
                    &mut igroup,
                    &mut num_groups,
                    &mut num_in_first,
                );
                if num_in_first < 2 {
                    exit_error("Only 1 point in fit; increase NumberToFit or RangeOfAngles");
                }
                iorder_use = 1.max(iorder.min(num_in_first - 2));
                ind_fit_center = (kl_high + kl_low) / 2;
                ierr = 1;
                if robust_linear && num_in_first > 3 {
                    // Do the robust fitting if called for and there are enough points
                    if max_rob_iter < 0 {
                        println!(" Robust fitting for rot/mag of section{ilist:>12}");
                    }
                    robust_fit_to_nat(
                        1,
                        2,
                        &mut ierr,
                        kl_low,
                        kl_high,
                        &igroup,
                        iorder_use,
                        ind_fit_center,
                        &nat,
                        &mut r_mat,
                        num_rows,
                        frac_zero_wgt,
                        &mut b_solve,
                        iorder,
                        &mut c_solve,
                        &mut r_means,
                        &mut r_sds,
                        &mut r_work,
                        scale_kfactor,
                        max_rob_iter,
                        rob_change_max,
                        rob_oscill_max,
                        &mut slope,
                        &mut intcp,
                        &mut num_fit,
                    );
                    // `101 format('Robust fitting to ',a,' failed for section',i5,
                    // ', falling back to regular fit')`
                    if ierr == 0 {
                        if max_rob_iter < 0 {
                            println!(" Robust fitting for shifts of section{ilist:>12}");
                        }
                        robust_fit_to_nat(
                            3,
                            3,
                            &mut ierr,
                            kl_low,
                            kl_high,
                            &igroup,
                            iorder_use,
                            ind_fit_center,
                            &nat,
                            &mut r_mat,
                            num_rows,
                            frac_zero_wgt,
                            &mut b_solve,
                            iorder,
                            &mut c_solve,
                            &mut r_means,
                            &mut r_sds,
                            &mut r_work,
                            scale_kfactor,
                            max_rob_iter,
                            rob_change_max,
                            rob_oscill_max,
                            &mut slope,
                            &mut intcp,
                            &mut num_fit,
                        );
                        if ierr != 0 {
                            println!(
                                "Robust fitting to shifts failed for section{ilist:>5}, falling back to regular fit"
                            );
                        }
                    } else {
                        println!(
                            "Robust fitting to rot/mag failed for section{ilist:>5}, falling back to regular fit"
                        );
                    }
                }

                // Otherwise, or if robust failed, do regular linear fit
                if ierr != 0 {
                    for i in 1..=2 {
                        for j in 1..=3 {
                            let ij = ((i - 1) + (j - 1) * 2) as usize;
                            num_fit = 0;
                            for kl in kl_low..=kl_high {
                                if igroup[(kl - 1) as usize] == 1 {
                                    num_fit += 1;
                                    x[(num_fit - 1) as usize] = (kl - ind_fit_center) as f32;
                                    y[(num_fit - 1) as usize] = nat[(kl - 1) as usize][ij];
                                }
                            }
                            //
                            polyfit(&x, &y, num_fit, iorder_use, &mut slope_tmp, &mut bint);
                            for ipow in 1..=iorder_use {
                                slope[(ipow - 1) as usize][ij] = slope_tmp[(ipow - 1) as usize];
                            }
                            intcp[ij] = bint;
                        }
                    }
                }
            }
            if if_shift == 1 && ilist == 1 {
                println!(" constants and coefficients of fit to{num_fit:>12}  natural parameters");
                let mut stdout = std::io::stdout();
                if xfwrite(&mut stdout, &intcp).is_err() {
                    exit_error("Writing file");
                }
                for ipow in 1..=iorder_use {
                    if xfwrite(&mut stdout, &slope[(ipow - 1) as usize]).is_err() {
                        exit_error("Writing file");
                    }
                }
            }
            //
            // calculate the g transform at this position along the linear fit
            // and take its inverse
            //
            xf_copy(&intcp, 2, &mut nat_prod, 2);
            for ipow in 1..=iorder_use {
                let previous = nat_prod;
                xflincom(
                    &previous,
                    1.,
                    &slope[(ipow - 1) as usize],
                    ((ilist - ind_fit_center) as f32).powi(ipow),
                    &mut nat_prod,
                );
            }
            //
            // for hybrid method, restore global average for translations of nhybrid > 1
            // restore rotation if nhybrid is 1, 3 or 4; restore overall mag
            // if nhybrid is 4
            //
            if nhybrid > 0 {
                if nhybrid > 1 {
                    nat_prod[4] = nat_avg[4];
                    nat_prod[5] = nat_avg[5];
                }
                if nhybrid == 1 || nhybrid > 2 {
                    nat_prod[0] = nat_avg[0];
                }
                if nhybrid > 3 {
                    nat_prod[1] = nat_avg[1];
                }
            }
            //
            gcen[..4].copy_from_slice(&rotmag_to_amat(
                nat_prod[0],
                nat_prod[2],
                nat_prod[1],
                nat_prod[3],
            ));
            gcen[4] = nat_prod[4];
            gcen[5] = nat_prod[5];
            xf_invert(&gcen, &mut gcen_inv, 2);
        }
        //
        // multiply this g by the inverse of the grand average or the locally
        // fitted average: generates a xform to the common central place or
        // to the locally fitted center.  This stuff about the linearly fitted
        // center position is pretty ad hoc and hokey, but it seems to give
        // good results in the final images.
        //
        if warping {
            //
            // If there are warpings, multiply cumulative warp by inverse warp based on
            // this transform and the inverse average warp in the global case
            let nxy = (nx_grid * ny_grid) as usize;

            if if_shift > 0 {
                // For local fits, get a cumulative warp over the fit range
                if if_shift > 1 {
                    if set_current_warp_file(ind_warp_input) != 0 {
                        exit_error("Switching back to input warp file");
                    }
                    cumulative_warp(
                        kl_low,
                        kl_high,
                        &mut g_warp,
                        &f,
                        &n_control,
                        control,
                        nx,
                        ny,
                        xcen,
                        ycen,
                        nx_grid,
                        ny_grid,
                        x_start,
                        y_start,
                        x_interval,
                        y_interval,
                        x_end,
                        y_end,
                        &mut dx_grid,
                        &mut dy_grid,
                        &mut dx_cum,
                        &mut dy_cum,
                        &mut nx_gr_tmp,
                        &mut ny_gr_tmp,
                        &mut x_str_tmp,
                        &mut y_str_tmp,
                        &mut x_int_tmp,
                        &mut y_int_tmp,
                    );
                }

                // For all fits, now fit to the cumulative warps, put the fitted values in dxProd,
                // and invert that, assuming the current linear forward transform there
                fit_warp_component(
                    &dx_cum,
                    &mut dx_prod,
                    nx_grid,
                    ny_grid,
                    kl_low,
                    kl_high,
                    ind_fit_center,
                    iorder,
                    ilist,
                );
                fit_warp_component(
                    &dy_cum,
                    &mut dy_prod,
                    nx_grid,
                    ny_grid,
                    kl_low,
                    kl_high,
                    ind_fit_center,
                    iorder,
                    ilist,
                );
                invert_warp_grid(
                    &dx_prod,
                    &dy_prod,
                    nx_grid,
                    nx_grid,
                    ny_grid,
                    x_start,
                    y_start,
                    x_interval,
                    y_interval,
                    &gcen,
                    xcen,
                    ycen,
                    &mut dx_grid,
                    &mut dy_grid,
                    &mut gcen_inv,
                    2,
                );
            }
            if iref_sec > 0 {
                //
                // For a reference section, set to unit transform and warping (did verify that
                // product warp was <= 0.001)
                xf_unit(&mut prod, 1., 2);
                dx_prod[..nxy].fill(0.);
                dy_prod[..nxy].fill(0.);
            } else {
                //
                // Otherwise multiply the local cumulative warp plus the true global G times
                // the fitted local warp plus the true fitted center.  Not perfect but very close.
                // The alternative would be to do the fit with the local cumulative gWarp's
                let k = (ilist - 1) as usize;
                if multiply_warpings(
                    &dx_cum[k * nxy..],
                    &dy_cum[k * nxy..],
                    nx_grid,
                    nx_grid,
                    ny_grid,
                    x_start,
                    y_start,
                    x_interval,
                    y_interval,
                    &g[k],
                    xcen,
                    ycen,
                    &dx_grid,
                    &dy_grid,
                    nx_grid,
                    nx_grid,
                    ny_grid,
                    x_start,
                    y_start,
                    x_interval,
                    y_interval,
                    &gcen_inv,
                    &mut dx_prod,
                    &mut dy_prod,
                    &mut prod,
                    0,
                    2,
                ) != 0
                {
                    exit_error("Multiplying two warpings together for final warping");
                }
            }
            if set_current_warp_file(ind_warp_output) != 0 {
                exit_error("Switching to output warp file");
            }
            if set_linear_transform(ilist - 1, &prod, 2) != 0
                || set_warp_grid(
                    ilist - 1,
                    nx_grid,
                    ny_grid,
                    x_start,
                    y_start,
                    x_interval,
                    y_interval,
                    &dx_prod,
                    &dy_prod,
                    nx_grid,
                ) != 0
            {
                exit_error("Storing final warping");
            }
        } else {
            xf_mult(&g[(ilist - 1) as usize], &gcen_inv, &mut prod, 2);
            if xfwrite(out_unit.as_mut().unwrap(), &prod).is_err() {
                exit_error("Writing file");
            }
        }
    }
    if warping {
        if write_warp_file(out_file.trim_end_matches(' '), 0) != 0 {
            exit_error("Writing new warp file");
        }
        println!(
            "New warping file written: {}",
            out_file.trim_end_matches(' ')
        );
    } else {
        // `close(2)`
        if let Some(mut unit) = out_unit.take() {
            if unit.flush().is_err() {
                exit_error("Writing file");
            }
        }
    }
    let _ = std::io::stdout().flush();
    crate::imod::libcfshr::b3dutil::exit(0);
}

/// Original contained `cumulativeWarp` (`xftoxg.f90:521`).
///
/// Form cumulative product of warping transforms to align to the first section in a range
#[allow(clippy::too_many_arguments)]
fn cumulative_warp(
    kl_start: i32,
    kl_end: i32,
    gcw: &mut [[f32; 6]],
    f: &[[f32; 6]],
    n_control: &[i32],
    control: bool,
    nx: i32,
    ny: i32,
    xcen: f32,
    ycen: f32,
    nx_grid: i32,
    ny_grid: i32,
    x_start: f32,
    y_start: f32,
    x_interval: f32,
    y_interval: f32,
    x_end: f32,
    y_end: f32,
    dx_grid: &mut [f32],
    dy_grid: &mut [f32],
    dx_cum: &mut [f32],
    dy_cum: &mut [f32],
    nx_gr_tmp: &mut i32,
    ny_gr_tmp: &mut i32,
    x_str_tmp: &mut f32,
    y_str_tmp: &mut f32,
    x_int_tmp: &mut f32,
    y_int_tmp: &mut f32,
) {
    let nxy = (nx_grid * ny_grid) as usize;
    // initialize first one to unit transform
    xf_unit(&mut gcw[(kl_start - 1) as usize], 1., 2);
    let start = (kl_start - 1) as usize * nxy;
    dx_cum[start..start + nxy].fill(0.); //ARRAY OPERATIONS
    dy_cum[start..start + nxy].fill(0.);
    for kl in kl_start + 1..=kl_end {
        let k = (kl - 1) as usize;
        if n_control[k] > 3 {
            if get_warp_grid(
                kl - 1,
                nx_gr_tmp,
                ny_gr_tmp,
                x_str_tmp,
                y_str_tmp,
                x_int_tmp,
                y_int_tmp,
                dx_grid,
                dy_grid,
                nx_grid,
            ) != 0
            {
                exit_error("Getting warp grid");
            }
            if !control
                && expand_and_extrap_grid(
                    dx_grid, dy_grid, nx_grid, ny_grid, nx_gr_tmp, ny_gr_tmp, x_str_tmp, y_str_tmp,
                    *x_int_tmp, *y_int_tmp, x_start, y_start, x_end, y_end, 0, nx, 0, ny,
                ) != 0
            {
                exit_error("Expanding a warp grid");
            }
        } else {
            *nx_gr_tmp = nx_grid;
            *ny_gr_tmp = ny_grid;
            // Fixed in translation (BUGS.md, `xftoxg` / `xfproduct`): the
            // source (`xftoxg.f90:541-545`) sets only the zero grid's size to
            // the common layout and leaves its start and interval at whatever
            // the previous section's grid had -- or unset, when no earlier
            // section had a grid.  The zero grid is meant to be in the common
            // layout, so its start and interval are the common ones here.
            *x_str_tmp = x_start;
            *y_str_tmp = y_start;
            *x_int_tmp = x_interval;
            *y_int_tmp = y_interval;
            dx_grid.fill(0.);
            dy_grid.fill(0.);
        }
        let (cum_before, cum_after) = dx_cum.split_at_mut(k * nxy);
        let (dy_before, dy_after) = dy_cum.split_at_mut(k * nxy);
        let (g_before, g_after) = gcw.split_at_mut(k);
        if multiply_warpings(
            dx_grid,
            dy_grid,
            nx_grid,
            *nx_gr_tmp,
            *ny_gr_tmp,
            *x_str_tmp,
            *y_str_tmp,
            *x_int_tmp,
            *y_int_tmp,
            &f[k],
            xcen,
            ycen,
            &cum_before[(k - 1) * nxy..],
            &dy_before[(k - 1) * nxy..],
            nx_grid,
            nx_grid,
            ny_grid,
            x_start,
            y_start,
            x_interval,
            y_interval,
            &g_before[k - 1],
            &mut cum_after[..nxy],
            &mut dy_after[..nxy],
            &mut g_after[0],
            // Fixed in translation (BUGS.md, `xftoxg` / `xfproduct`): the
            // source passes `useSecond = 0` (`xftoxg.f90:548-552`), so each
            // cumulative product is stored in *this section's* grid layout
            // (its own interval, expanded to the common range), while every
            // later reader -- the next product, the mean grid, the fits --
            // treats `dxCum(:, :, kl)` as the common `nxGrid x nyGrid`
            // layout.  When a section's grid differs from the common one,
            // native therefore reads positions it never wrote (heap residue,
            // different from run to run).  The product is meant to live in
            // the common layout of the second (cumulative) grid, so it is
            // stored there (`useSecond = 1`) whenever this section's grid
            // size differs from the common one.  When the sizes agree every
            // common position is written, and the source's own call
            // (`useSecond = 0`) is kept, so those files stay byte-identical
            // to native.
            if *nx_gr_tmp == nx_grid && *ny_gr_tmp == ny_grid {
                0
            } else {
                1
            },
            2,
        ) != 0
        {
            exit_error("Multiplying two warpings together for cumulative warping");
        }
    }
}

/// Original contained `fitWarpComponent` (`xftoxg.f90:566`).
///
/// Do the polynomial fit to the x or y component of set of cumulative warping transforms
/// and place the fitted position at "ilist" in the dxyProd array
#[allow(clippy::too_many_arguments)]
fn fit_warp_component(
    dxy_cum: &[f32],
    dxy_prod: &mut [f32],
    nx_grid: i32,
    ny_grid: i32,
    kl_low: i32,
    kl_high: i32,
    ind_fit_center: i32,
    iorder: i32,
    ilist: i32,
) {
    let nxy = (nx_grid * ny_grid) as usize;
    let mut x = vec![0.0_f32; (kl_high - kl_low + 1).max(0) as usize];
    let mut y = vec![0.0_f32; x.len()];
    let mut slope_tmp = [0.0_f32; 10];
    let mut bint = 0.0_f32;
    for i in 1..=nx_grid {
        for j in 1..=ny_grid {
            let ij = ((i - 1) + (j - 1) * nx_grid) as usize;
            let mut num_fit = 0_i32;
            for kl in kl_low..=kl_high {
                num_fit += 1;
                x[(num_fit - 1) as usize] = (kl - ind_fit_center) as f32;
                y[(num_fit - 1) as usize] = dxy_cum[ij + (kl - 1) as usize * nxy];
            }
            polyfit(&x, &y, num_fit, iorder, &mut slope_tmp, &mut bint);
            for ipow in 1..=iorder {
                // `(ilist - indFitCenter)**ipow` is an integer power.
                bint += slope_tmp[(ipow - 1) as usize]
                    * (ilist - ind_fit_center).wrapping_pow(ipow as u32) as f32;
            }
            dxy_prod[ij] = bint;
        }
    }
}

/// Original contained `robustFitToNat` (`xftoxg.f90:589`).
///
/// Do a robust fit to either the geometric parameters or the shifts over current range
#[allow(clippy::too_many_arguments)]
fn robust_fit_to_nat(
    jstart: i32,
    jend: i32,
    iret: &mut i32,
    kl_low: i32,
    kl_high: i32,
    igroup: &[i32],
    iorder_use: i32,
    ind_fit_center: i32,
    nat: &[[f32; 6]],
    r_mat: &mut [f32],
    num_rows: i32,
    frac_zero_wgt: f32,
    b_solve: &mut [f32],
    iorder: i32,
    c_solve: &mut [f32],
    r_means: &mut [f32],
    r_sds: &mut [f32],
    r_work: &mut [f32],
    scale_kfactor: f32,
    max_rob_iter: i32,
    rob_change_max: f32,
    rob_oscill_max: f32,
    slope: &mut [[f32; 6]; 10],
    intcp: &mut [f32; 6],
    num_fit: &mut i32,
) {
    // `real*4 fitScale(2, 3) /1., 100., 1., 100., 1., 1./` and
    // `fitAdd(2, 3) /0., -1., 0., 0., 0., 0./`, in storage order.
    let fit_scale: [f32; 6] = [1., 100., 1., 100., 1., 1.];
    let fit_add: [f32; 6] = [0., -1., 0., 0., 0., 0.];
    let mut ind_col: i32;
    let num_bcol: i32;
    let mut num_iter = 0_i32;
    let max_zero_wgt: i32;
    *iret = 0;
    // `rMat(row, col)` is `r_mat[(row - 1) + (col - 1) * numRows]`.
    let at = |row: i32, col: i32| -> usize { ((row - 1) + (col - 1) * num_rows) as usize };

    // Load the data matrix, scaling mag and dmag
    *num_fit = 0;
    for kl in kl_low..=kl_high {
        if igroup[(kl - 1) as usize] == 1 {
            *num_fit += 1;
            ind_col = 1;
            for ipow in 1..=iorder_use {
                r_mat[at(*num_fit, ind_col)] = ((kl - ind_fit_center) as f32).powi(ipow);
                ind_col += 1;
            }
            for i in 1..=2 {
                for j in jstart..=jend {
                    let ij = ((i - 1) + (j - 1) * 2) as usize;
                    r_mat[at(*num_fit, ind_col)] =
                        (nat[(kl - 1) as usize][ij] + fit_add[ij]) * fit_scale[ij];
                    ind_col += 1;
                }
            }
            r_mat[at(*num_fit, ind_col)] = 1.;
        }
    }

    // Do the regression
    max_zero_wgt = 1.max((frac_zero_wgt * *num_fit as f32).round() as i32);
    num_bcol = 2 * (jend + 1 - jstart);
    *iret = robust_regress(
        r_mat,
        num_rows,
        0,
        iorder_use,
        *num_fit,
        num_bcol,
        b_solve,
        iorder,
        Some(c_solve),
        r_means,
        r_sds,
        r_work,
        scale_kfactor * 4.685,
        &mut num_iter,
        max_rob_iter,
        max_zero_wgt,
        rob_change_max,
        rob_oscill_max,
    );
    if *iret != 0 {
        return;
    }

    // unpack the result
    ind_col = 1;
    for i in 1..=2 {
        for j in jstart..=jend {
            let ij = ((i - 1) + (j - 1) * 2) as usize;
            for ipow in 1..=iorder_use {
                slope[(ipow - 1) as usize][ij] =
                    b_solve[((ipow - 1) + (ind_col - 1) * iorder) as usize] / fit_scale[ij];
            }
            intcp[ij] = c_solve[(ind_col - 1) as usize] / fit_scale[ij] - fit_add[ij];
            ind_col += 1;
        }
    }
}

/// Original `groupRotations` (`xftoxg.f90:639`).
///
/// groupRotations finds groups of sections whose rotation angles are
/// all within the range given by RANGE.  The angles are assumed to be
/// the first element of the transforms in F.  Only sections from KSTART
/// to KEND are considered.  IGROUP is returned with a group number for
/// each section; numGroups with the number of groups, and numInFirst
/// with the number of sections in the first (i.e., largest) group.
pub fn group_rotations(
    f: &[[f32; 6]],
    k_start: i32,
    k_end: i32,
    range: f32,
    igroup: &mut [i32],
    num_groups: &mut i32,
    num_in_first: &mut i32,
) {
    let mut diff: f32;
    let mut angle_min: f32;
    let mut angle_max: f32;
    let mut num_free: i32;
    let mut i: i32;
    let mut num_in_group: i32;
    let mut max_in_group: i32;
    // Uninitialised in the source; it is always assigned before use because
    // every ungrouped item is within range of itself unless `range < 0` or
    // the angle is NaN, which the loop below now stops on.
    let mut i_max = k_start;
    //
    num_free = k_end + 1 - k_start;
    *num_groups = 0;
    angle_min = 1.0e10;
    angle_max = -1.0e10;
    for i in k_start..=k_end {
        let k = (i - 1) as usize;
        igroup[k] = 0;
        // `min(angleMin, f(1,1,i))` / `max(angleMax, f(1,1,i))`: the
        // reference build compiles these to `minss`/`maxss` with the running
        // value as the destination (`grouprotations_` in the reference
        // `xftoxg`), i.e. `acc < f ? acc : f`, so a NaN angle replaces the
        // running value.  With a NaN left in `angleMax - angleMin` the range
        // test fails and the grouping loop below never assigns anything; the
        // reference then loops forever, and this stops with an error.  (For more than six
        // sections the reference vectorises this loop four lanes wide, which
        // can change which NaN survives but not whether one does when the
        // last section's angle is NaN.)
        angle_min = if angle_min < f[k][0] {
            angle_min
        } else {
            f[k][0]
        };
        angle_max = if angle_max > f[k][0] {
            angle_max
        } else {
            f[k][0]
        };
    }
    //
    // If the angles all fit within the range, we are all set
    //
    if angle_max - angle_min <= range {
        for i in k_start..=k_end {
            igroup[(i - 1) as usize] = 1;
        }
        *num_groups = 1;
        *num_in_first = num_free;
        return;
    }
    //
    while num_free > 0 {
        //
        // Find biggest group of angles that fit within range
        //
        i = k_start;
        max_in_group = 0;
        while i <= k_end && max_in_group < num_free {
            if igroup[(i - 1) as usize] == 0 {
                //
                // For each ungrouped item, count up number of items that would
                // fit within a range starting at this item
                //
                num_in_group = 0;
                for j in k_start..=k_end {
                    if igroup[(j - 1) as usize] == 0 {
                        diff = angle_diff(f[(i - 1) as usize][0], f[(j - 1) as usize][0]);
                        if diff >= 0. && diff <= range {
                            num_in_group += 1;
                        }
                    }
                }
                if num_in_group > max_in_group {
                    max_in_group = num_in_group;
                    i_max = i;
                }
            }
            i += 1;
        }
        // Fixed in translation (BUGS.md, `xftoxg` / `xfproduct`): when no
        // ungrouped angle fits within the range even of itself -- a NaN
        // angle, or a negative range -- `maxInGroup` stays 0, `numFree`
        // never falls, and the source loops forever (`xftoxg.f90:661-689`).
        // That is an error here.
        if max_in_group == 0 {
            exit_error(
                "Rotation angles cannot be grouped: an angle is not a number or the range is negative",
            );
        }
        //
        // Now assign the angles within range to the group
        //
        *num_groups += 1;
        for j in k_start..=k_end {
            if igroup[(j - 1) as usize] == 0 {
                diff = angle_diff(f[(i_max - 1) as usize][0], f[(j - 1) as usize][0]);
                if diff >= 0. && diff <= range {
                    igroup[(j - 1) as usize] = *num_groups;
                }
            }
        }
        if *num_groups == 1 {
            *num_in_first = max_in_group;
        }
        num_free -= max_in_group;
    }
}

/// Original `angleDiff` (`xftoxg.f90:716`).
pub fn angle_diff(ang1: f32, ang2: f32) -> f32 {
    let mut angle_diff = ang2 - ang1;
    while angle_diff > 180. {
        angle_diff -= 360.;
    }
    while angle_diff <= -180. {
        angle_diff += 360.;
    }
    angle_diff
}
