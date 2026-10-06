//! Translation of `IMOD/flib/image/maxjoinsize.f`.
//!
//! MAXJOINSIZE computes the maximum size and offsets needed to hold
//! transformed data when joining serial sections.
//!
//! The main program maps to [`maxjoinsize`].  The warp routines are the
//! `warpwrapfort.c` Fortran wrappers, which subtract one from the section
//! index (`getLinearTransform`, `getSizeAdjustedGrid`).

use crate::imod::flib::subrs::compat::gfortran_rt::{maxss, minss};
use crate::imod::flib::subrs::hvem::frefor::{ListItem, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{exit_error, memory_error, set_exit_prefix};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen};
use crate::imod::flib::subrs::xfsubs::xfrdall::xfrdall2;
use crate::imod::libcfshr::b3dutil::{exit, program_args};
use crate::imod::libcfshr::linearxforms::xfapply;
use crate::imod::libiimod::unit_fileio::iiu_close;
use crate::imod::libiimod::unit_header::iiu_ret_delta;
use crate::imod::libwarp::warpfiles::get_linear_transform;
use crate::imod::libwarp::warputils::{
    find_inverse_point, find_max_grid_size, get_size_adjusted_grid, read_check_warp_file,
};
use std::io::{BufRead, BufReader, Write};

/// `parameter (maxfiles = 100000)` (`maxjoinsize.f:19`).
const MAXFILES: i32 = 100000;

/// Original program `maxjoinsize` (`maxjoinsize.f:16`).
pub fn maxjoinsize() {
    let mut f: Vec<[f32; 6]> = Vec::new();
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut delta_first = [0.0_f32; 3];
    let mut nx = vec![0_i32; MAXFILES as usize];
    let mut ny = vec![0_i32; MAXFILES as usize];
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let mut mode = 0_i32;
    let mut num_f = 0_i32;
    let mut num_files = 0_i32;
    let mut line_skip = 0_i32;
    let (mut nxwarp, mut nywarp, mut ibinning, mut iflags) = (0_i32, 0_i32, 0_i32, 0_i32);
    let (mut max_nxg, mut max_nyg, mut nx_grid, mut ny_grid) = (0_i32, 0_i32, 0_i32, 0_i32);
    let (mut x_grid_strt, mut y_grid_strt, mut x_grid_intrv, mut y_grid_intrv) =
        (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    let (mut dx, mut dy) = (0.0_f32, 0.0_f32);
    let mut pixel_size = 0.0_f32;
    let mut n_control: Vec<i32> = Vec::new();
    let mut dx_grid: Vec<f32> = Vec::new();
    let mut dy_grid: Vec<f32> = Vec::new();
    let mut err_string = String::new();
    let mut warp_scale = 0.0_f32;
    let args = program_args();
    let iargc = args.len() as i32 - 1;
    // `getarg` into a `character*320`
    let getarg = |index: usize| -> String {
        let arg = args[index].as_bytes();
        String::from_utf8_lossy(&arg[..arg.len().min(320)])
            .trim_end_matches(' ')
            .to_owned()
    };
    //
    let mut if_control = 0_i32;
    if iargc == 0 {
        println!(" Usage: maxjoinsize number_of_sections lines_to_skip root_name");
        println!("  Computes size and offsets needed to contain all data from joined tomograms");
        println!("  It use transforms in root_name.tomoxg and gets image filenames from");
        println!("  root_name.info, skipping the given number of lines to get to the filenames");
        let _ = std::io::stdout().flush();
        exit(0);
    }
    set_exit_prefix("ERROR: MAXJOINSIZE - ");
    if iargc != 3 {
        exit_error("THERE MUST BE THREE ARGUMENTS");
    }
    // get number of files and lines to skip
    // read(rootname, *, err=91, end = 91)
    let read_int = |text: String, value: &mut i32| {
        let mut unit = std::io::Cursor::new(text.into_bytes());
        if list_read(&mut unit, &mut [ListItem::Integer(value)]).is_err() {
            // 91
            exit_error("READING NUMBER OF FILES OR LINES TO SKIP");
        }
    };
    read_int(getarg(1), &mut num_files);
    read_int(getarg(2), &mut line_skip);
    // Get the rootname, open tomoxg and read transforms
    let rootname = getarg(3);
    let filename = format!("{rootname}.tomoxg");
    let ierr = read_check_warp_file(
        &filename,
        0,
        1,
        &mut nxwarp,
        &mut nywarp,
        &mut num_f,
        &mut ibinning,
        &mut pixel_size,
        &mut iflags,
        &mut err_string,
    );
    if ierr < -1 {
        exit_error(&err_string);
    }
    let warping = ierr >= 0;
    if warping {
        if (iflags / 2) % 2 != 0 {
            if_control = 1;
        }
    } else {
        // open(3, file=filename, form='formatted', status='old', err=92)
        let Ok(file) = std::fs::File::open(&filename) else {
            exit_error("OPENING .tomoxg FILE")
        };
        let mut unit3 = BufReader::new(file);
        let ierr = xfrdall2(&mut unit3, &mut f, MAXFILES);
        if ierr == 2 {
            exit_error("READING TRANSFORM FILE");
        }
        if ierr == 1 {
            exit_error("TOO MANY TRANSFORMS IN FILE FOR TRANSFORM ARRAY");
        }
        num_f = f.len() as i32;
    }
    let _ = if_control;
    if num_f != num_files {
        exit_error("WRONG NUMBER OF TRANSFORMS IN FILE");
    }
    if f.len() < num_files.max(0) as usize {
        f.resize(num_files.max(0) as usize, [0.0; 6]);
    }
    // open the info file and skip lines
    let filename = format!("{rootname}.info");
    let Ok(file) = std::fs::File::open(&filename) else {
        // 94
        exit_error("OPENING .info FILE")
    };
    let mut unit3 = BufReader::new(file);
    // `read(3, '(a)', err=.., end=..) filename` into a `character*320`
    let mut read_line = |unit3: &mut BufReader<std::fs::File>| -> Option<String> {
        let mut line = Vec::new();
        match unit3.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => {
                if line.last() == Some(&b'\n') {
                    line.pop();
                }
                line.truncate(320);
                Some(String::from_utf8_lossy(&line).trim_end_matches(' ').to_owned())
            }
        }
    };
    for _ in 1..=line_skip {
        if read_line(&mut unit3).is_none() {
            exit_error("OPENING .info FILE");
        }
    }
    ialprt(false);
    // open image files, get sizes and max size
    let mut maxx = 0_i32;
    let mut maxy = 0_i32;
    for i in 1..=num_files {
        let Some(filename) = read_line(&mut unit3) else {
            // 95
            exit_error("READING FILENAME FROM .info FILE")
        };
        imopen(1, &filename, "ro");
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
        if i == 1 {
            delta_first = iiu_ret_delta(1);
        }
        unsafe {
            iiu_close(1);
        }
        nx[(i - 1) as usize] = nxyz[0];
        ny[(i - 1) as usize] = nxyz[1];
        maxx = maxx.max(nx[(i - 1) as usize]);
        maxy = maxy.max(ny[(i - 1) as usize]);
    }
    drop(unit3);
    // Set up for warping
    if warping {
        warp_scale = pixel_size / delta_first[0];
        n_control = vec![0; num_files.max(0) as usize];
        memory_error(0, "ARRAY FOR NUMBER OF CONTROL POINTS");
        if find_max_grid_size(
            0.,
            maxx as f32 / warp_scale,
            0.,
            maxy as f32 / warp_scale,
            &mut n_control,
            &mut max_nxg,
            &mut max_nyg,
            &mut err_string,
        ) != 0
        {
            exit_error(&err_string);
        }
        if max_nxg * max_nyg > 0 {
            dx_grid = vec![0.0; (max_nxg * max_nyg) as usize];
            dy_grid = vec![0.0; (max_nxg * max_nyg) as usize];
            memory_error(0, "ARRAYS FOR WARPING FIELDS");
        }
    }
    // transform the 4 corners in the coordinate system of the maximum size
    let xcen = maxx as f32 / 2.;
    let ycen = maxy as f32 / 2.;
    let mut xmin = xcen;
    // Fixed in translation (BUGS.md, `maxjoinsize`): the source starts `ymin`
    // and `ymax` at `xcen`, so the Y range always includes the X center.
    let mut ymin = ycen;
    let mut xmax = xcen;
    let mut ymax = ycen;
    for i in 1..=num_files {
        let k = (i - 1) as usize;
        let xhalf = nx[k] as f32 / 2.;
        let yhalf = ny[k] as f32 / 2.;
        let mut has_warp = false;
        if warping {
            if get_linear_transform(i - 1, &mut f[k], 2) != 0 {
                exit_error("GETTING LINEAR TRANSFORM FROM WARP FILE");
            }
            has_warp = n_control[k] > 2;
            if has_warp
                && get_size_adjusted_grid(
                    i - 1,
                    maxx as f32 / warp_scale,
                    maxy as f32 / warp_scale,
                    0.,
                    0.,
                    1,
                    warp_scale,
                    1,
                    &mut nx_grid,
                    &mut ny_grid,
                    &mut x_grid_strt,
                    &mut y_grid_strt,
                    &mut x_grid_intrv,
                    &mut y_grid_intrv,
                    &mut dx_grid,
                    &mut dy_grid,
                    max_nxg,
                    max_nyg,
                    &mut err_string,
                ) != 0
            {
                exit_error(&err_string);
            }
        }
        for idirx in [-1_i32, 1] {
            for idiry in [-1_i32, 1] {
                let (mut xcorn, mut ycorn) = xfapply(
                    &f[k],
                    xcen,
                    ycen,
                    xcen + idirx as f32 * xhalf,
                    ycen + idiry as f32 * yhalf,
                );
                if has_warp {
                    // Fixed in translation (BUGS.md, `maxjoinsize`): the source
                    // passes `yGridIntrv` for both grid intervals.
                    let (x_in, y_in) = (xcorn, ycorn);
                    find_inverse_point(
                        x_in,
                        y_in,
                        &dx_grid,
                        &dy_grid,
                        max_nxg,
                        nx_grid,
                        ny_grid,
                        x_grid_strt,
                        y_grid_strt,
                        x_grid_intrv,
                        y_grid_intrv,
                        &mut xcorn,
                        &mut ycorn,
                        &mut dx,
                        &mut dy,
                    );
                }
                xmin = minss(xmin, xcorn);
                ymin = minss(ymin, ycorn);
                xmax = maxss(xmax, xcorn);
                ymax = maxss(ymax, ycorn);
            }
        }
    }
    // Compute and return numbers
    let ixofs = (0.5 * (xmax + xmin) - xcen).round() as i32;
    let iyofs = (0.5 * (ymax + ymin) - ycen).round() as i32;
    let newx = 2 * (0.5 * (xmax - xmin)).round() as i32;
    let newy = 2 * (0.5 * (ymax - ymin)).round() as i32;
    // `101 format('Maximum size required:',2i9,/,'Offset needed to center:', 2i8)`
    println!("Maximum size required:{:>9}{:>9}", newx, newy);
    println!("Offset needed to center:{:>8}{:>8}", ixofs, iyofs);
    let _ = std::io::stdout().flush();
    exit(0);
}
