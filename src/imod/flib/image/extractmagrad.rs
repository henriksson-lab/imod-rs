//! Translation of `IMOD/flib/image/extractmagrad.f`.
//!
//! The main program maps to [`extractmagrad`] and the external subroutine to
//! [`errorexit`].  Library calls go to the Fortran wrappers of
//! `extraheader.c` (`*_fortran`), which print and exit on error as the C
//! wrappers do.  Formatted output uses gfortran `Iw`/`Fw.d` editing
//! (overflow is `w` asterisks, a leading zero is dropped when that is what
//! makes the value fit); list-directed `print *` writes an integer as a
//! blank and `I11` and puts a blank between a number and a following string.

use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, frefor, list_read};
use crate::imod::flib::subrs::hvem::parse_input_params::{
    pip_get_in_out_file, pip_read_or_parse_options,
};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::imopen;
use crate::imod::libcfshr::b3dutil::exit;
use crate::imod::libcfshr::extraheader::{
    get_extra_header_items_fortran, get_extra_header_pieces_fortran, get_extra_header_tilts_fortran,
};
use crate::imod::libcfshr::parse_params::{pip_done, pip_get_float};
use crate::imod::libiimod::unit_fileio::iiu_close;
use crate::imod::libiimod::unit_header::{
    iiu_ret_delta, iiu_ret_extended_data, iiu_ret_extended_type, iiu_ret_num_extended,
};
use std::io::{BufRead, BufReader, BufWriter, Write};

/// `parameter (maxtab = 1000)` (`extractmagrad.f:26`).
const MAXTAB: usize = 1000;
/// `parameter (numOptions = 8)` (`extractmagrad.f:54`).
const NUM_OPTIONS: i32 = 8;
/// Fallback PIP table `options(1)` (`extractmagrad.f:56-60`).
const OPTIONS: &str = "input:InputImageFile:FN:@output:OutputFile:FN:@\
gradient:GradientTable:FN:@rotation:RotationAngle:F:@\
pixel:PixelSize:F:@dgrad:DeltaGradient:F:@\
drot:DeltaRotation:F:@help:usage:B:";

/// Original program `extractmagrad` (`extractmagrad.f:1`).
///
/// Extractmaggrad will read the intensity values and tilt angles from an
/// image file header, and read a table of magnification gradients as a
/// function of intensity, and output a file with a list of tilt angles and
/// mag gradients for each section.
pub fn extractmagrad() {
    let mut nxyz = [0_i32; 3];
    let mut mxyz = [0_i32; 3];
    let maxextra: i32;
    let maxtilts: i32;
    let maxpiece: i32;
    let delta: [f32; 3];
    // `real*4 tableC2(maxtab), ...`: fixed arrays; a table longer than
    // `maxtab` lines writes past them in the source, and grows them here.
    let mut table_c2: Vec<f32> = vec![0.0; MAXTAB];
    let mut table_grad: Vec<f32> = vec![0.0; MAXTAB];
    let mut table_rot: Vec<f32> = vec![0.0; MAXTAB];
    let mut table_lin: Vec<f32> = vec![0.0; MAXTAB];
    let mut tilt: Vec<f32>;
    let mut c2val: Vec<f32>;
    let mut array: Vec<u8>;
    let mut ixpiece: Vec<i32>;
    let mut iypiece: Vec<i32>;
    let mut izpiece: Vec<i32>;
    //
    let mut filin = String::new();
    let mut filout = String::new();
    let mut grad_table = String::new();
    let mut line: String;
    //
    let mut ierr: i32;
    let mut npiece = 0_i32;
    let mut i: i32;
    let nbyte: i32;
    let iflags: i32;
    let i_version: i32;
    let mut maxz: i32;
    let mut mode = 0_i32;
    let mut ntilt = 0_i32;
    let mut nbsym: i32;
    let mut ntiltout: i32;
    let mut num_in_table = 0_i32;
    let mut j: i32;
    let mag_version: i32;
    let mut n_field = 0_i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut axis_rot = 0.0_f32;
    let mut pixel_size: f32;
    // Uninitialised in the source when no table interval brackets the
    // intensity (a NaN intensity, or a non-monotonic table): native writes
    // stack residue for the first such section and the previous section's
    // values after that.  BUGS.md, fixed in translation: they start at 0.
    let mut rot = 0.0_f32;
    let mut grad = 0.0_f32;
    let mut frac: f32;
    let mut delta_grad: f32;
    let mut delta_rot: f32;
    let mut crossover = 0.0_f32;
    let mut xnum = [0.0_f32; 20];
    let mut looking: bool;
    //
    let (mut num_opt_arg, mut num_non_opt_arg) = (0_i32, 0_i32);

    // gfortran `Fw.d` and `Iw` output editing.
    let fmt_f = |value: f32, w: usize, d: usize| -> String {
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
    };
    let fmt_i = |value: i32, w: usize| -> String {
        let text = format!("{value}");
        if text.len() > w {
            return "*".repeat(w);
        }
        format!("{text:>w$}")
    };
    // `read` with no `END=`/`ERR=` that fails: the gfortran runtime reports
    // it and stops with status 2.
    let read_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        exit(2);
    };
    // The Fortran wrappers `pipgetstring`/`pipgetfloat`: the variable is left
    // untouched unless the option is found.
    // `c2fString` there copies at most the variable's declared length
    // (extractmagrad.f:35: `character*320 gradTable`); a longer entry fails with `In PipGetString, string is too
    // long for character variable`, which exits under the exit prefix.
    let get_string = |option: &[u8], length: usize, string: &mut String| -> i32 {
        let mut record = vec![b' '; length];
        let err = crate::imod::libcfshr::pip_fwrap::pipgetstring_(option, &mut record);
        if err == 0 {
            *string = crate::imod::libcfshr::b3dutil::fortran_string(&record);
        }
        err
    };
    //
    pixel_size = 0.;
    mag_version = 1;
    delta_grad = 0.;
    delta_rot = 0.;
    //
    // Pip startup: set error, parse options, check help
    //
    pip_read_or_parse_options(
        &[OPTIONS],
        NUM_OPTIONS,
        "extractmagrad",
        "ERROR: EXTRACTMAGRAD - ",
        false,
        3,
        1,
        1,
        &mut num_opt_arg,
        &mut num_non_opt_arg,
    );

    if pip_get_in_out_file("InputImageFile", 1, " ", &mut filin, 320) != 0 {
        errorexit("NO INPUT FILE SPECIFIED");
    }
    if pip_get_in_out_file("OutputFile", 2, " ", &mut filout, 320) != 0 {
        errorexit("NO OUTPUT FILE SPECIFIED");
    }
    ierr = pip_get_in_out_file(
        "OutputFile",
        2,
        "Name of output file, or return to print out values",
        &mut filout,
        320,
    );
    //
    if get_string(b"GradientTable", 320, &mut grad_table) != 0 {
        errorexit("YOU MUST SPECIFY A GRADIENT TABLE");
    }
    if pip_get_float(b"RotationAngle", &mut axis_rot) != 0 {
        errorexit("YOU MUST ENTER A ROTATION ANGLE");
    }
    ierr = pip_get_float(b"PixelSize", &mut pixel_size);
    ierr = pip_get_float(b"DeltaGradient", &mut delta_grad);
    ierr = pip_get_float(b"DeltaRotation", &mut delta_rot);
    pip_done();

    imopen(1, &filin, "RO");
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
    let nz = nxyz[2];
    //
    // Get pixel size from header if it was not entered
    //
    pixel_size = pixel_size * 10.;
    if pixel_size == 0. {
        delta = iiu_ret_delta(1);
        if delta[0] == 1.0 {
            errorexit("YOU MUST ENTER A PIXEL SIZE SINCE THE PIXEL SPACING IN THE HEADER IS 1");
        }
        pixel_size = delta[0];
    }
    //
    nbsym = iiu_ret_num_extended(1);
    maxextra = nbsym + 1024;
    maxpiece = nz + 1024;
    maxtilts = maxpiece;
    tilt = vec![0.0; maxtilts as usize];
    c2val = vec![0.0; maxtilts as usize];
    array = vec![0; (maxextra / 4 * 4) as usize];
    ixpiece = vec![0; maxpiece as usize];
    iypiece = vec![0; maxpiece as usize];
    izpiece = vec![0; maxpiece as usize];

    // `call irtsym(1,nbsym,array)`
    {
        let mut extra: Vec<u8> = Vec::new();
        let _ = iiu_ret_extended_data(1, &mut extra);
        nbsym = iiu_ret_num_extended(1);
        let count = extra.len().min(array.len());
        array[..count].copy_from_slice(&extra[..count]);
    }
    [nbyte, iflags] = iiu_ret_extended_type(1);
    get_extra_header_pieces_fortran(
        &array,
        nbsym,
        nbyte,
        iflags,
        nz,
        &mut ixpiece,
        &mut iypiece,
        &mut izpiece,
        &mut npiece,
        maxpiece,
    );
    if npiece == 0 {
        for i in 1..=nz {
            izpiece[(i - 1) as usize] = i - 1;
        }
        maxz = nz;
    } else {
        maxz = 0;
        for i in 1..=npiece {
            maxz = maxz.max(izpiece[(i - 1) as usize]);
        }
        // BUGS.md, fixed in translation: `extractmagrad.f:117` lacks the `+ 1`
        // that `extracttilts.f90:285` has (`izpiece` is 0-based), so native
        // leaves the last Z slot unmarked.  Harmless there -- that slot is
        // always filled -- but the count of Z slots is `max + 1`.
        maxz += 1;
    }
    //
    // set up a marker value for empty slots
    //
    for i in 1..=maxz {
        tilt[(i - 1) as usize] = -999.;
    }
    //
    get_extra_header_tilts_fortran(
        &array, nbsym, nbyte, iflags, nz, &mut tilt, &mut ntilt, &izpiece,
    );
    if ntilt == 0 {
        errorexit("THERE ARE NO TILT ANGLES IN THE IMAGE FILE HEADER");
    }
    // `c2val` is passed as both value arrays; intensities fill only the first.
    get_extra_header_items_fortran(
        &array, nbsym, nbyte, iflags, nz, 5, &mut c2val, None, &mut ntilt, &izpiece,
    );
    if ntilt == 0 {
        errorexit("THERE ARE NO INTENSITIES IN THE IMAGE FILE HEADER");
    }
    //
    // Open gradient table file and look for version and crossover
    //
    let Ok(file) = std::fs::File::open(&grad_table) else {
        errorexit("OPENING MAG GRADIENT TABLE FILE");
    };
    let mut unit1 = BufReader::new(file);
    // `read(1, '(a)') line`: `line` is `character*320`
    {
        let mut record: Vec<u8> = Vec::new();
        match unit1.read_until(b'\n', &mut record) {
            Ok(0) | Err(_) => read_abort(ListReadError::End),
            Ok(_) => {}
        }
        if record.last() == Some(&b'\n') {
            record.pop();
        }
        record.truncate(320);
        line = String::from_utf8_lossy(&record).into_owned();
    }
    frefor(&line, &mut xnum, &mut n_field);
    if n_field == 1 {
        i_version = xnum[0].round() as i32;
        if let Err(err) = list_read(&mut unit1, &mut [ListItem::Real(&mut crossover)]) {
            read_abort(err);
        }
    } else {
        i_version = 1;
        drop(unit1);
        let Ok(file) = std::fs::File::open(&grad_table) else {
            errorexit("OPENING MAG GRADIENT TABLE FILE");
        };
        unit1 = BufReader::new(file);
    }
    //
    // read the gradient table
    //
    i = 1;
    loop {
        let iu = (i - 1) as usize;
        if iu >= table_c2.len() {
            table_c2.push(0.);
            table_grad.push(0.);
            table_rot.push(0.);
            table_lin.push(0.);
        }
        let (mut c2, mut gr, mut ro) = (table_c2[iu], table_grad[iu], table_rot[iu]);
        let result = list_read(
            &mut unit1,
            &mut [
                ListItem::Real(&mut c2),
                ListItem::Real(&mut gr),
                ListItem::Real(&mut ro),
            ],
        );
        (table_c2[iu], table_grad[iu], table_rot[iu]) = (c2, gr, ro);
        match result {
            Ok(()) => {}
            Err(ListReadError::End) => break,
            Err(ListReadError::Error) => errorexit("READING MAG GRADIENT TABLE FILE"),
        }
        if i_version > 1 {
            table_lin[iu] = 1. / (table_c2[iu] - crossover);
        }
        num_in_table = i;
        i = i + 1;
    }

    let mut unit2 = BufWriter::new(dopen(2, &filout, "new", "f"));
    //
    // pack the values down and get count
    //
    ntiltout = 0;
    for i in 1..=ntilt {
        let iu = (i - 1) as usize;
        if tilt[iu] != -999. {
            ntiltout = ntiltout + 1;
            tilt[(ntiltout - 1) as usize] = tilt[iu];
            c2val[(ntiltout - 1) as usize] = c2val[iu];
        }
    }

    let _ = writeln!(unit2, "{}", fmt_i(mag_version, 8));
    let _ = writeln!(
        unit2,
        "{}{}{}",
        fmt_i(ntiltout, 6),
        fmt_f(pixel_size, 8, 3),
        fmt_f(axis_rot, 9, 2)
    );

    let t = |k: i32| (k - 1) as usize;
    for i in 1..=ntiltout {
        let iu = t(i);
        if tilt[iu] != -999. {
            if c2val[iu] < table_c2[0] {
                grad = table_grad[0];
                rot = table_rot[0];
            } else if c2val[iu] > table_c2[t(num_in_table)] {
                grad = table_grad[t(num_in_table)];
                rot = table_rot[t(num_in_table)];
            } else {
                looking = true;
                j = 1;
                while looking && j < num_in_table {
                    let ju = t(j);
                    frac = (c2val[iu] - table_c2[ju]) / (table_c2[ju + 1] - table_c2[ju]);
                    if frac >= 0. && frac <= 1. {
                        if i_version > 1 {
                            frac = (1. / (c2val[iu] - crossover) - table_lin[ju])
                                / (table_lin[ju + 1] - table_lin[ju]);
                        }
                        //
                        // For version 2, do not extrapolation closer to crossover
                        // It should already have been done when making the table
                        //
                        if (i_version as f32) > 1.
                            && table_c2[ju] < crossover
                            && table_c2[ju + 1] > crossover
                        {
                            if c2val[iu] < crossover {
                                grad = table_grad[ju];
                                rot = table_rot[ju];
                            } else {
                                grad = table_grad[ju + 1];
                                rot = table_rot[ju + 1];
                            }
                        } else {
                            grad = (1. - frac) * table_grad[ju] + frac * (table_grad[ju + 1]);
                            rot = (1. - frac) * table_rot[ju] + frac * (table_rot[ju + 1]);
                        }
                        looking = false;
                    }
                    j = j + 1;
                }
            }
            let _ = writeln!(
                unit2,
                "{}{}{}",
                fmt_f(tilt[iu], 8, 2),
                fmt_f(grad + delta_grad, 8, 3),
                fmt_f(rot + delta_rot, 8, 3)
            );
        }
    }
    let _ = unit2.flush();

    println!("{:>12} {}", ntiltout, " gradients output to file");

    unsafe {
        iiu_close(1);
    }
    //
    exit(0);
}

/// Original `subroutine errorexit(message)` (`extractmagrad.f:230`).
pub fn errorexit(message: &str) -> ! {
    println!();
    println!(" ERROR: EXTRACTMAGRAD - {message}");
    exit(1);
}
