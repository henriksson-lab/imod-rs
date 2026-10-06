//! Translation of `IMOD/flib/image/rotatevol.f90`.
//!
//! ROTATEVOL will rotate all or part of a three-dimension volume of data.
//! The rotations may be by any angles about the three axes.  Tilt angles and
//! origin information in the header are properly maintained so that the new
//! data stack will have a coordinate system congruent with the old one.
//!
//! The program can work on an arbitrarily large volume.  It reconstructs a
//! series of sub-regions of the output volume, referred to as cubes but
//! actually rectangles.  For each cube, it reads into memory a cube from the
//! input volume that contains all of the image area that rotates into that
//! cube of output volume.  It then uses linear or triquadratic interpolation
//! to find each pixel of the output cube, and writes the cube to a scratch
//! file.  When all of the cubes in one layer are done, it reads back data
//! from the scratch files and assembles each section in that layer.
//!
//! The Fortran main program maps to [`rotatevol`]; the shared machinery is
//! `rotmatwarpsubs.rs`, and the `rotmatwarp` module variables are the
//! [`RotMatWarp`] it owns (`nxIn`/`cxIn`/... are equivalenced to the first
//! elements of `nxyzIn`/`cxyzIn`/...).  The `real*4 (3,3)` matrices are column
//! major: `a(i,j)` is `a[(i-1) + 3*(j-1)]`.

use crate::imod::flib::image::rotmatwarp::RotMatWarp;
use crate::imod::flib::image::rotmatwarpsubs::{
    set_memory_limit_and_hdf_chunks, setup_cubes_scratch, transform_cubes,
};
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_get_logical, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::linearxforms::{icalc_angles, icalc_matrix, inv_matrix};
use crate::imod::libcfshr::parse_params::{
    pip_done, pip_get_float, pip_get_integer, pip_get_three_floats, pip_get_three_integers,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_tilt, iiu_alt_tilt_orig, iiu_alt_tilt_rot,
    iiu_create_header, iiu_ret_delta, iiu_ret_origin, iiu_ret_tilt, iiu_trans_labels,
};
use std::io::{BufRead, Write};

/// `parameter (numOptions = 15)` (`rotatevol.f90:47`).
const ROTATEVOL_NUM_OPTIONS: i32 = 15;
/// Fallback PIP table, the `options(1)` string (`rotatevol.f90:49-55`).
const ROTATEVOL_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@tempdir:TemporaryDirectory:CH:@\
size:OutputSizeXYZ:IT:@center:RotationCenterXYZ:FT:@\
angles:RotationAnglesZYX:FT:@back:BackRotate:B:@order:InterpolationOrder:I:@\
query:QuerySizeNeeded:B:@fill:FillValue:F:@chunk:ChunkSizesForHDF:IT:@\
memory:MemoryLimit:I:@verbose:VerboseOutput:I:@param:ParameterFile:PF:@\
help:usage:B:";

/// Original program `rotatevol` (`rotatevol.f90:23`).
///
/// A list-directed `read` with no `END=`/`ERR=` that fails is the gfortran
/// runtime error, status 2.
pub fn rotatevol() {
    let mut rmw = RotMatWarp::default();
    let mut cell = [0.0_f32; 6];
    let mut mxyz_in = [0_i32; 3];
    let mut max_dim = [0_i32; 3];
    let mut center_in = [0.0_f32; 3];
    let mut a_fwd = [0.0_f32; 9];
    let mut a_old = [0.0_f32; 9];
    let mut a_new = [0.0_f32; 9];
    let mut a_old_inv = [0.0_f32; 9];
    let mut angles = [0.0_f32; 3];
    let mut x_temp = [0.0_f32; 3];
    //
    let mut file_in = String::new();
    let mut file_out = String::new();
    let mut temp_dir = [b' '; 320];
    let mut temp_ext = [b' '; 320];
    //
    // DNM 3/8/01: initialize the time in case time(tim) doesn't work
    //
    let mut dat = [b' '; 9];
    let mut tim = *b"00:00:00";
    let mut interp_order: i32;
    let (mut dmin_in, mut dmax_in) = (0.0_f32, 0.0_f32);
    let mut query: bool;
    let mut back_rotate: bool;
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    // `read(5,*) ...` with no END=/ERR=: a failure is the runtime's error.
    let read_stdin_list = |items: &mut [ListItem]| {
        let mut stdin = std::io::stdin().lock();
        if let Err(err) = list_read(&mut stdin, items) {
            let _ = std::io::stdout().flush();
            match err {
                ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
            }
            exit(2);
        }
    };
    //
    // set defaults here
    //
    interp_order = 2;
    temp_dir.fill(b' ');
    query = false;
    back_rotate = false;
    //
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[ROTATEVOL_OPTIONS],
        ROTATEVOL_NUM_OPTIONS,
        "rotatevol",
        "ERROR: ROTATEVOL - ",
        true,
        3,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );
    let pip_input = num_opt_arg + num_non_opt_arg > 0;

    if pip_get_in_out_file("InputFile", 1, "Name of input file", &mut file_in, 320) != 0 {
        exit_error("No input file specified");
    }

    if pip_input {
        let _ = pip_get_logical("QuerySizeNeeded", &mut query);
        if query {
            ialprt(false);
        }
    }

    imopen(5, &file_in, "RO");
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
    for i in 0..3 {
        // `centerIn(i) = nxyzIn(i) / 2`: an integer division.
        center_in[i] = (rmw.nxyz_in[i] / 2) as f32;
        rmw.nxyz_out[i] = rmw.nxyz_in[i];
        angles[i] = 0.;
    }
    //
    if !query
        && pip_get_in_out_file("OutputFile", 2, "Name of output file", &mut file_out, 320) != 0
    {
        exit_error("No output file specified");
    }

    set_memory_limit_and_hdf_chunks(&mut rmw, pip_input);
    if pip_input {
        let _ = pipgetstring_(b"TemporaryDirectory", &mut temp_dir);
        {
            let [cx, cy, cz] = &mut center_in;
            let _ = pip_get_three_floats(b"RotationCenterXYZ", cx, cy, cz);
        }
        let _ = pip_get_integer(b"InterpolationOrder", &mut interp_order);
        let _ = pip_get_integer(b"VerboseOutput", &mut rmw.i_verbose);
        {
            let [nx, ny, nz] = &mut rmw.nxyz_out;
            let _ = pip_get_three_integers(b"OutputSizeXYZ", nx, ny, nz);
        }
        {
            let [a1, a2, a3] = &mut angles;
            let _ = pip_get_three_floats(b"RotationAnglesZYX", a3, a2, a1);
        }
        let _ = pip_get_float(b"FillValue", &mut rmw.dmean_in);
        let _ = pip_get_logical("BackRotate", &mut back_rotate);
    } else {
        //
        print!(
            " Enter path name of directory for temporary files, \n or Return to use current directory: "
        );
        let _ = std::io::stdout().flush();
        // read(5, '(a)') tempDir
        {
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
            temp_dir.copy_from_slice(&line[..320]);
        }
        //
        print!(" X, Y, and Z dimensions of the output file: ");
        let _ = std::io::stdout().flush();
        {
            let [nx, ny, nz] = &mut rmw.nxyz_out;
            read_stdin_list(&mut [
                ListItem::Integer(nx),
                ListItem::Integer(ny),
                ListItem::Integer(nz),
            ]);
        }
        //
        print!(
            " Enter X, Y, and Z index coordinates of the center of rotation\n   in the input file (/ for center of file): "
        );
        let _ = std::io::stdout().flush();
        {
            let [c1, c2, c3] = &mut center_in;
            read_stdin_list(&mut [ListItem::Real(c1), ListItem::Real(c2), ListItem::Real(c3)]);
        }
        println!(
            " Rotations are applied in the order that you enter them: rotation about the   Z axis, then rotation about the Y axis, then rotation about the X axis"
        );
        print!(" Rotations about Z, Y, and X axes (gamma, beta, alpha): ");
        let _ = std::io::stdout().flush();
        {
            let [a1, a2, a3] = &mut angles;
            read_stdin_list(&mut [ListItem::Real(a3), ListItem::Real(a2), ListItem::Real(a1)]);
        }
    }
    //
    pip_done();
    if rmw.nxyz_out[0] < 1 || rmw.nxyz_out[1] < 1 || rmw.nxyz_out[2] < 1 {
        exit_error("A positive output size must be entered on all axes");
    }
    //
    // get matrices for forward and inverse rotations
    //
    if back_rotate {
        icalc_matrix(&angles, &mut rmw.a_inv);
        let a_inv = rmw.a_inv;
        inv_matrix(&a_inv, &mut a_fwd);
        icalc_angles(&mut angles, &a_fwd);
    } else {
        icalc_matrix(&angles, &mut a_fwd);
        inv_matrix(&a_fwd, &mut rmw.a_inv);
    }
    //
    // Compute maximum dimensions required if requested
    //
    if query {
        for i in 0..3 {
            max_dim[i] = 0;
        }
        let (nx_in, ny_in, nz_in) = (rmw.nxyz_in[0], rmw.nxyz_in[1], rmw.nxyz_in[2]);
        for idir_y in [-1_i32, 1] {
            for idir_z in [-1_i32, 1] {
                for i in 0..3 {
                    let value = (a_fwd[i] * nx_in as f32
                        + idir_y as f32 * a_fwd[i + 3] * ny_in as f32
                        + idir_z as f32 * a_fwd[i + 6] * nz_in as f32)
                        .abs();
                    // nint: round half away from zero.
                    max_dim[i] = max_dim[i].max(value.round() as i32);
                }
            }
        }
        // write(*,'(3i8)')
        println!("{:>8}{:>8}{:>8}", max_dim[0], max_dim[1], max_dim[2]);
        let _ = std::io::stdout().flush();
        exit(0);
    }
    //
    imopen(6, &file_out, "NEW");
    //
    // get true centers of index coordinate systems
    //
    let delta = iiu_ret_delta(5);
    for i in 0..3 {
        rmw.cxyz_in[i] = (rmw.nxyz_in[i] - 1) as f32 / 2. + center_in[i]
            - (rmw.nxyz_in[i] / 2) as f32;
        rmw.cxyz_out[i] = (rmw.nxyz_out[i] - 1) as f32 / 2.;
        cell[i] = rmw.nxyz_out[i] as f32 * delta[i];
        cell[i + 3] = 90.;
    }
    //
    //
    let mut labels = [[0_u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    labels[0] = rmw.title;
    iiu_create_header(6, &rmw.nxyz_out, &rmw.nxyz_out, rmw.mode, &labels, 0);
    iiu_alt_cell(6, &cell);
    iiu_trans_labels(6, 5);
    time(&mut tim);
    b3d_date(&mut dat);
    temp_ext[..10].copy_from_slice(b"rot      1");
    //
    // `302 format('ROTATEVOL: 3D rotation by angles:',3f7.1,t57,a9,2x,a8)`
    let mut title_ch = [b' '; 80];
    let mut head = b"ROTATEVOL: 3D rotation by angles:".to_vec();
    for angle in angles {
        head.extend_from_slice(format_f(angle as f64, 7, 1).as_bytes());
    }
    let count = head.len().min(56);
    title_ch[..count].copy_from_slice(&head[..count]);
    title_ch[56..65].copy_from_slice(&dat);
    title_ch[67..75].copy_from_slice(&tim);
    rmw.title = title_ch;
    //
    // calculate new tilt angles and origin information from old
    //
    let tilt_old = iiu_ret_tilt(5);
    icalc_matrix(&tilt_old, &mut a_old);
    inv_matrix(&a_old, &mut a_old_inv);
    iiu_alt_tilt_orig(6, &tilt_old);
    iiu_alt_tilt(6, &tilt_old);
    iiu_alt_tilt_rot(6, &angles);
    let tilt_new = iiu_ret_tilt(6);
    icalc_matrix(&tilt_new, &mut a_new);
    let mut orig = iiu_ret_origin(5);
    //
    // Need to add 0.5 back to all these center coordinates to get
    // it right for actual 90 degree rotation
    //
    let x_cen = rmw.cxyz_in[0] + 0.5 - orig[0] / delta[0];
    let y_cen = rmw.cxyz_in[1] + 0.5 - orig[1] / delta[1];
    let z_cen = rmw.cxyz_in[2] + 0.5 - orig[2] / delta[2];
    for i in 0..3 {
        x_temp[i] = a_old_inv[i] * x_cen + a_old_inv[i + 3] * y_cen + a_old_inv[i + 6] * z_cen;
    }
    for i in 0..3 {
        orig[i] = delta[i]
            * (rmw.cxyz_out[i] + 0.5
                - (a_new[i] * x_temp[0] + a_new[i + 3] * x_temp[1] + a_new[i + 6] * x_temp[2]));
    }
    iiu_alt_origin(6, &orig);
    //
    // Set up the arrangement of input and output data
    let a_inv = rmw.a_inv;
    setup_cubes_scratch(
        &mut rmw,
        &a_fwd,
        &a_inv,
        1,
        0,
        &file_in,
        &fortran_string(&temp_dir),
        &mut temp_ext,
        &tim,
        false,
    );
    //
    // Do all the work and exit
    transform_cubes(&mut rmw, interp_order);
}
