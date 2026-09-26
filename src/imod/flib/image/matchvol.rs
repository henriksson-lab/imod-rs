//! Translation of `IMOD/flib/image/matchvol.f90`.
//!
//! MATCHVOL will transform a volume using a general linear transformation.
//! Its main use is to transform one tomogram from a two-axis tilt series so
//! that it matches the other tomogram.  To do so, it can combine an initial
//! alignment transformation and any number of successive refining
//! transformations.  The program uses the same algorithm as ROTATEVOL for
//! rotating large volumes.
//!
//! The Fortran main program maps to [`matchvol`]; the shared machinery is
//! `rotmatwarpsubs.rs`, and the `rotmatwarp` module variables are the
//! [`RotMatWarp`] it owns.

use crate::imod::flib::image::rotmatwarp::RotMatWarp;
use crate::imod::flib::image::rotmatwarpsubs::{
    set_memory_limit_and_hdf_chunks, setup_cubes_scratch, transform_cubes,
};
use crate::imod::flib::subrs::compat::datetime::time;
use crate::imod::flib::subrs::compat::gfortran_rt::format_f;
use crate::imod::flib::subrs::hvem::b3ddate::b3d_date;
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    exit_error, pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::hvem::xfcopy3d::xfcopy3d;
use crate::imod::flib::subrs::hvem::xfinv3d::xfinv3d;
use crate::imod::flib::subrs::hvem::xfmult3d::xfmult3d;
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libcfshr::parse_params::{
    pip_allow_comma_defaults, pip_done, pip_get_float_array, pip_get_integer, pip_get_three_floats,
    pip_get_three_integers, pip_number_of_entries,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libiimod::mrcfiles::{MRC_LABEL_SIZE, MRC_NLABELS};
use crate::imod::libiimod::unit_header::{
    iiu_alt_cell, iiu_alt_origin, iiu_alt_tilt, iiu_create_header, iiu_ret_delta, iiu_ret_origin,
    iiu_ret_tilt, iiu_trans_labels,
};
use std::io::{BufRead, BufReader, Write};

/// `parameter (numOptions = 14)` (`matchvol.f90:43`).
const MATCHVOL_NUM_OPTIONS: i32 = 14;
/// Fallback PIP table, the `options(1)` string (`matchvol.f90:45-50`).
const MATCHVOL_OPTIONS: &str = "input:InputFile:FN:@output:OutputFile:FN:@inverse:InverseFile:FN:@\
tempdir:TemporaryDirectory:CH:@size:OutputSizeXYZ:IT:@center:CenterXYZ:FT:@\
xffile:TransformFile:FNM:@3dxform:3DTransform:FAM:@\
order:InterpolationOrder:I:@chunk:ChunkSizesForHDF:IT:@memory:MemoryLimit:I:@\
verbose:VerboseOutput:I:@param:ParameterFile:PF:@help:usage:B:";

/// Original program `matchvol` (`matchvol.f90:14`).
///
/// `aFwd(3,3)` and the other `real*4 (3,3)` matrices are column major:
/// `aFwd(i,j)` is `a_fwd[(i-1) + 3*(j-1)]`.  The `character*320` names are
/// fixed-width byte buffers where PIP fills them, so a string too long for the
/// variable behaves as the Fortran wrapper makes it.  A list-directed `read`
/// with no `END=`/`ERR=` that fails is the gfortran runtime error, status 2.
pub fn matchvol() {
    let mut rmw = RotMatWarp::default();
    let mut cell = [0.0_f32; 12];
    let mut dxyz_in = [0.0_f32; 3];
    let mut mxyz_in = [0_i32; 3];
    let mut center_in = [0.0_f32; 3];
    let mut a_fwd = [0.0_f32; 9];
    let mut tilt_old = [0.0_f32; 3];
    let mut a_tmp1 = [0.0_f32; 9];
    let mut a_tmp2 = [0.0_f32; 9];
    let mut dxyz_temp1 = [0.0_f32; 3];
    let mut dxyz_temp2 = [0.0_f32; 3];
    let (mut dmin_in, mut dmax_in) = (0.0_f32, 0.0_f32);
    //
    let mut file_in = String::new();
    let mut file_out = [b' '; 320];
    let mut temp_dir = [b' '; 320];
    let mut temp_ext = [b' '; 320];
    let mut im_file_out = String::new();
    //
    // DNM 3/8/01: initialize the time in case time(tim) doesn't work
    //
    let mut dat = [b' '; 9];
    let mut tim = *b"00:00:00";
    let mut interp_order: i32;
    let mut num_trans: i32;
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
    // `read(5, '(a)') var` into a `character*320`.
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
    // `read(1,*) ((aFwd(i, j), j = 1, 3), dxyzIn(i), i = 1, 3)` from a file.
    let read_xform = |reader: &mut dyn BufRead, a_fwd: &mut [f32; 9], dxyz_in: &mut [f32; 3]| {
        let mut values = [0.0_f32; 12];
        for i in 0..3 {
            for j in 0..3 {
                values[4 * i + j] = a_fwd[i + 3 * j];
            }
            values[4 * i + 3] = dxyz_in[i];
        }
        let result = {
            let mut items: Vec<ListItem> = values.iter_mut().map(ListItem::Real).collect();
            let mut reader = reader;
            list_read(&mut reader, &mut items)
        };
        for i in 0..3 {
            for j in 0..3 {
                a_fwd[i + 3 * j] = values[4 * i + j];
            }
            dxyz_in[i] = values[4 * i + 3];
        }
        if let Err(err) = result {
            let _ = std::io::stdout().flush();
            match err {
                ListReadError::End => eprintln!("Fortran runtime error: End of file"),
                ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
            }
            exit(2);
        }
    };
    // `write(*,102) ((a(i, j), j = 1, 3), d(i), i = 1, 3)`,
    // `102 format(3f10.6,f10.3)`.
    let format102 = |a: &[f32; 9], d: &[f32; 3]| -> String {
        let mut text = String::new();
        for i in 0..3 {
            for j in 0..3 {
                text.push_str(&format_f(a[i + 3 * j] as f64, 10, 6));
            }
            text.push_str(&format_f(d[i] as f64, 10, 3));
            text.push('\n');
        }
        text
    };
    //
    // set defaults here
    //
    interp_order = 2;
    temp_dir.fill(b' ');
    temp_ext[..10].copy_from_slice(b"mat      1");
    time(&mut tim);
    b3d_date(&mut dat);
    //
    // Pip startup: set error, parse options, check help, set flag if used
    //
    pip_read_or_parse_options(
        &[MATCHVOL_OPTIONS],
        MATCHVOL_NUM_OPTIONS,
        "matchvol",
        "ERROR: MATCHVOL - ",
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
    //
    for i in 0..3 {
        center_in[i] = rmw.nxyz_in[i] as f32 / 2.;
        rmw.nxyz_out[i] = rmw.nxyz_in[i];
    }
    //
    if pip_get_in_out_file(
        "OutputFile",
        2,
        "Name of output file",
        &mut im_file_out,
        320,
    ) != 0
    {
        exit_error("No output file specified");
    }
    //
    set_memory_limit_and_hdf_chunks(&mut rmw, pip_input);
    if pip_input {
        let _ = pipgetstring_(b"TemporaryDirectory", &mut temp_dir);
        {
            let [cx, cy, cz] = &mut center_in;
            let _ = pip_get_three_floats(b"CenterXYZ", cx, cy, cz);
        }
        let _ = pip_get_integer(b"InterpolationOrder", &mut interp_order);
        let _ = pip_get_integer(b"VerboseOutput", &mut rmw.i_verbose);
        {
            let [nx, ny, nz] = &mut rmw.nxyz_out;
            let _ = pip_get_three_integers(b"OutputSizeXYZ", nx, ny, nz);
        }
        //
        // transforms
        //
        let (mut num_xfiles, mut num_xlines) = (0_i32, 0_i32);
        let _ = pip_number_of_entries(b"TransformFile", &mut num_xfiles);
        let _ = pip_number_of_entries(b"3DTransform", &mut num_xlines);
        num_trans = num_xfiles + num_xlines;
        if num_trans == 0 {
            exit_error("No transforms specified");
        }
        pip_allow_comma_defaults(0);
        for i_trans in 1..=num_trans {
            if i_trans <= num_xfiles {
                //
                // read the files in turn
                //
                let _ = pipgetstring_(b"TransformFile", &mut file_out);
                let mut unit1 = BufReader::new(dopen(1, &fortran_string(&file_out), "ro", "f"));
                read_xform(&mut unit1, &mut a_fwd, &mut dxyz_in);
            } else {
                //
                // Then read the in-line transforms in turn
                //
                let mut num_to_get = 12;
                let _ = pip_get_float_array(b"3DTransform", &mut cell, &mut num_to_get, 12);
                for i in 1..=3 {
                    for j in 1..=3 {
                        a_fwd[(i - 1) + 3 * (j - 1)] = cell[j + 4 * (i - 1) - 1];
                    }
                    dxyz_in[i - 1] = cell[4 * i - 1];
                }
            }
            if i_trans > 1 {
                xfcopy3d(&a_fwd, &dxyz_in, &mut a_tmp2, &mut dxyz_temp2);
                xfmult3d(
                    &a_tmp1,
                    &dxyz_temp1,
                    &a_tmp2,
                    &dxyz_temp2,
                    &mut a_fwd,
                    &mut dxyz_in,
                );
            }
            xfcopy3d(&a_fwd, &dxyz_in, &mut a_tmp1, &mut dxyz_temp1);
        }
    } else {
        //
        // interactive input: transforms can alternate between coming
        // from file or from input arbitrarily
        //
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
            read_stdin_list(&mut [
                ListItem::Integer(nx),
                ListItem::Integer(ny),
                ListItem::Integer(nz),
            ]);
        }
        //
        print!(" Number of successive transformations to apply: ");
        let _ = std::io::stdout().flush();
        num_trans = 0;
        read_stdin_list(&mut [ListItem::Integer(&mut num_trans)]);
        for i_trans in 1..=num_trans {
            println!(
                " For transformation matrix #{:12} , either enter the name of a file with the transformation, or Return to enter the transformation directly",
                i_trans
            );
            read_stdin_line(&mut file_out);
            if file_out.iter().all(|&c| c == b' ') {
                println!(" Enter transformation matrix #{:12}", i_trans);
                let mut stdin = std::io::stdin().lock();
                read_xform(&mut stdin, &mut a_fwd, &mut dxyz_in);
            } else {
                let mut unit1 = BufReader::new(dopen(1, &fortran_string(&file_out), "ro", "f"));
                read_xform(&mut unit1, &mut a_fwd, &mut dxyz_in);
            }
            if i_trans > 1 {
                xfcopy3d(&a_fwd, &dxyz_in, &mut a_tmp2, &mut dxyz_temp2);
                xfmult3d(
                    &a_tmp1,
                    &dxyz_temp1,
                    &a_tmp2,
                    &dxyz_temp2,
                    &mut a_fwd,
                    &mut dxyz_in,
                );
            }
            xfcopy3d(&a_fwd, &dxyz_in, &mut a_tmp1, &mut dxyz_temp1);
        }
    }
    //
    if rmw.nxyz_out[0] < 1 || rmw.nxyz_out[1] < 1 || rmw.nxyz_out[2] < 1 {
        exit_error("Illegal output size");
    }
    println!(" Forward matrix:");
    print!("{}", format102(&a_fwd, &dxyz_in));
    //
    // get matrix for inverse transform
    //
    xfinv3d(&a_fwd, &dxyz_in, &mut rmw.a_inv, &mut rmw.cxyz_in);
    println!(" Inverse matrix:");
    print!("{}", format102(&rmw.a_inv, &rmw.cxyz_in));

    if pip_input {
        file_out.fill(b' ');
        let _ = pipgetstring_(b"InverseFile", &mut file_out);
    } else {
        println!(" Enter name of file to place inverse transformation in, or Return for none");
        read_stdin_line(&mut file_out);
    }
    if !file_out.iter().all(|&c| c == b' ') {
        let mut unit1 = dopen(1, &fortran_string(&file_out), "new", "f");
        let _ = unit1.write_all(format102(&rmw.a_inv, &rmw.cxyz_in).as_bytes());
    }
    pip_done();
    //
    imopen(6, &im_file_out, "NEW");
    //
    // Set up the arrangement of input/output data, allocate arrays
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
    // DNM 7/26/02: transfer pixel spacing to same axes; could be weird
    // if spacings are not isotropic
    //
    let mut delta = iiu_ret_delta(5);
    let mut cell6 = [0.0_f32; 6];
    for i in 0..3 {
        cell6[i] = rmw.nxyz_out[i] as f32 * delta[i];
        cell6[i + 3] = 90.;
        // cxyzin(i) =cxyzin(i) +nxyzin(i) /2.
        rmw.cxyz_in[i] += center_in[i];
        rmw.cxyz_out[i] = rmw.nxyz_out[i] as f32 / 2.;
    }
    //
    let mut labels = [[0_u8; MRC_LABEL_SIZE]; MRC_NLABELS];
    labels[0] = rmw.title;
    iiu_create_header(6, &rmw.nxyz_out, &rmw.nxyz_out, rmw.mode, &labels, 0);
    iiu_alt_cell(6, &cell6);
    iiu_trans_labels(6, 5);
    //
    // `302 format('MATCHVOL: 3-D transformation of tomogram:',t57,a9,2x,a8)`
    let mut titlech = [b' '; 80];
    let head = b"MATCHVOL: 3-D transformation of tomogram:";
    titlech[..head.len()].copy_from_slice(head);
    titlech[56..65].copy_from_slice(&dat);
    titlech[67..75].copy_from_slice(&tim);
    rmw.title = titlech;
    tilt_old = iiu_ret_tilt(5);
    iiu_alt_tilt(6, &tilt_old);
    //
    // Preserve origin in case of volume scaling
    //
    delta = iiu_ret_origin(5);
    iiu_alt_origin(6, &delta);
    //
    // Do the work and exit
    transform_cubes(&mut rmw, interp_order);
}
