//! Translation of `IMOD/flib/image/fixboundaries.f90`.
//!
//! FIXBOUNDARIES rewrites data near chunk boundaries after direct writing in
//! parallel to an output file.  Takes two arguments, the name of the main
//! image file and the name of the boundary info file.  Uses the entries in
//! the info file to rewrite all of the data in the boundary files into the
//! main file.
//!
//! The main program maps to [`fixboundaries`].

use crate::imod::flib::subrs::hvem::getinout::getinout;
use crate::imod::flib::subrs::hvem::parse_input_params::{exit_error, set_exit_prefix};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{imopen, irdlin};
use crate::imod::libcfshr::b3dutil::{exit, fortran_string};
use crate::imod::libiimod::parallelwrite::{
    iiu_par_wrt_initialize, par_wrt_get_region, par_wrt_properties,
};
use crate::imod::libiimod::unit_fileio::{
    iiu_close, iiu_file_type, iiu_set_position, iiu_write_lines,
};

/// `parameter (IDIM = 1000000)` (`fixboundaries.f90:14`).
const IDIM: usize = 1000000;

/// Original program `fixboundaries` (`fixboundaries.f90:11`).
pub fn fixboundaries() {
    unsafe {
        let mut nxyz = [0_i32; 3];
        let mut mxyz = [0_i32; 3];
        let mut nxyz2 = [0_i32; 3];
        let mut mode = 0_i32;
        let mut iz_secs = [0_i32; 2];
        let mut line_start = [0_i32; 2];
        let (mut num_files, mut if_all_sec, mut lines_guard) = (0_i32, 0_i32, 0_i32);
        let (mut dmin, mut dmax, mut dmean) = (0f32, 0f32, 0f32);
        //
        // `character*320 mainFile, infoFile`: `getarg`/`read(5, '(a)')` into
        // a 320-character variable keeps the first 320 characters.
        let (main_arg, info_arg) = match getinout(2) {
            Ok(files) => files,
            Err(_) => exit(2),
        };
        let mut main_record = [b' '; 320];
        let mut info_record = [b' '; 320];
        for (record, text) in [
            (&mut main_record, main_arg.as_bytes()),
            (&mut info_record, info_arg.as_bytes()),
        ] {
            let count = text.len().min(320);
            record[..count].copy_from_slice(&text[..count]);
        }
        let main_file = fortran_string(&main_record);
        let mut info_file = fortran_string(&info_record);
        set_exit_prefix("ERROR: fixboundaries - ");
        imopen(1, &main_file, "old");
        irdhdr(
            1,
            nxyz.as_mut_ptr(),
            mxyz.as_mut_ptr(),
            &mut mode,
            &mut dmin,
            &mut dmax,
            &mut dmean,
        );
        let mut iz = nxyz[0];
        if iiu_file_type(1) == 5 {
            iz = -iz;
        }
        // `parWrtInitialize` from Fortran is the wrapper `parwrtinitialize`
        // (`parallelwrite.c:516`), i.e. `iiuParWrtInitialize` of the
        // blank-trimmed name (`f2cString`).
        let ierr = iiu_par_wrt_initialize(&info_file, 2, iz, nxyz[1], nxyz[2]);
        if ierr != 0 {
            exit_error("Initializing from the parallel write information file");
        }
        let ierr = par_wrt_properties(&mut if_all_sec, &mut lines_guard, &mut num_files);
        if ierr > 0 {
            exit_error("Getting properties from the parallel write information file");
        }
        if ierr < 0 {
            println!(" Fixboundaries: Nothing to do for an HDF file");
            exit(0);
        }
        //
        // `real*4 array(IDIM)`: one line of the main file is read into it.  A
        // line longer than IDIM would overrun the Fortran array; the buffer
        // is made at least a line long instead of reproducing the overrun.
        let mut array = vec![0f32; IDIM.max(nxyz[0].max(0) as usize)];
        //
        // Loop on all the files
        for ifile in 1..=num_files {
            if par_wrt_get_region(ifile, &mut info_record, &mut iz_secs, &mut line_start) != 0 {
                exit_error("Getting information for one region");
            }
            info_file = fortran_string(&info_record);
            imopen(2, &info_file, "ro");
            irdhdr(
                2,
                nxyz2.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
            if if_all_sec != 0 {
                //
                // If writing all sections, loop on all sections
                for iz in 1..=nxyz[2] {
                    for j in 1..=2 {
                        if line_start[j as usize - 1] >= 0 {
                            iiu_set_position(2, iz * 2 + j - 3, 0);
                            iiu_set_position(1, iz - 1, line_start[j as usize - 1]);
                            for _i in 1..=lines_guard {
                                if irdlin(2, &mut array).is_err() {
                                    exit_error("Reading from guard file");
                                }
                                iiu_write_lines(1, array.as_mut_ptr().cast(), 1);
                            }
                        }
                    }
                }
            } else {
                //
                // Otherwise just write the two sections
                for j in 1..=2 {
                    if iz_secs[j as usize - 1] >= 0 {
                        iiu_set_position(2, j - 1, 0);
                        iiu_set_position(1, iz_secs[j as usize - 1], line_start[j as usize - 1]);
                        for _i in 1..=lines_guard {
                            if irdlin(2, &mut array).is_err() {
                                exit_error("Reading from guard file");
                            }
                            iiu_write_lines(1, array.as_mut_ptr().cast(), 1);
                        }
                    }
                }
            }
            iiu_close(2);
        }
        iiu_close(1);
        exit(0);
    }
}
