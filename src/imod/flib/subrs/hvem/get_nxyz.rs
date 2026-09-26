//! Translation of `IMOD/flib/subrs/hvem/get_nxyz.f`.

use crate::imod::flib::subrs::hvem::frefor::{ListItem, list_read};
use crate::imod::flib::subrs::imsubs::irdhdr::irdhdr;
use crate::imod::flib::subrs::imsubs::wrap_iiunit::{ialprt, imopen};
use crate::imod::libcfshr::b3dutil::fortran_string;
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use crate::imod::libiimod::unit_fileio::iiu_close;
use std::io::{BufRead, Write};

/// Original `get_nxyz` (`get_nxyz.f:15`).
///
/// GET_NXYZ takes a line of input and attempts to read it as 3 integers.  If
/// this succeeds, the values are returned in [nxyz].  If this generates an
/// error, or the line contains letters, it then uses the line as a filename
/// and tries to open an MR file on unit [iunit].  [nxyz] is then fetched
/// from the header.
///
/// If [pipinput] is TRUE, then the string is fetched with the given
/// [option].  Set [progname] to the name of the program if the option is
/// mandatory.  If this entry is optional, pass [progname] as a blank string.
///
/// `line` is the source's `character*320`.  `read(5,'(a)') line` has no
/// `END=`, so end of input is the gfortran runtime's end-of-file error.
pub fn get_nxyz(pipinput: bool, option: &str, progname: &str, iunit: i32, nxyz: &mut [i32]) {
    let mut line = [b' '; 320];
    let mut mxyz = [0_i32; 3];
    let mut mode = 0_i32;
    let (mut dmin, mut dmax, mut dmean) = (0.0_f32, 0.0_f32, 0.0_f32);

    if !pipinput {
        let mut record: Vec<u8> = Vec::new();
        let _ = std::io::stdout().flush();
        match std::io::stdin().lock().read_until(b'\n', &mut record) {
            Ok(0) | Err(_) => {
                eprintln!("Fortran runtime error: End of file");
                crate::imod::libcfshr::b3dutil::exit(2);
            }
            Ok(_) => {}
        }
        if record.last() == Some(&b'\n') {
            record.pop();
        }
        record.resize(320, b' ');
        line.copy_from_slice(&record[..320]);
    } else if pipgetstring_(option.as_bytes(), &mut line) > 0 {
        if progname.trim_end_matches(' ').is_empty() {
            return;
        }
        println!(
            "\nERROR: {} - you must enter option {} to specify file size or file name",
            progname.trim_end_matches(' '),
            option.trim_end_matches(' ')
        );
        crate::imod::libcfshr::b3dutil::exit(1);
    }
    if line[0] == b'/' || line.iter().all(|&b| b == b' ') {
        return;
    }
    let mut goto10 = line_is_filename(&line);
    if !goto10 {
        let (mut n1, mut n2, mut n3) = (nxyz[0], nxyz[1], nxyz[2]);
        let result = list_read(
            &mut &line[..],
            &mut [
                ListItem::Integer(&mut n1),
                ListItem::Integer(&mut n2),
                ListItem::Integer(&mut n3),
            ],
        );
        // Items stored before a failure stay stored.
        nxyz[0] = n1;
        nxyz[1] = n2;
        nxyz[2] = n3;
        if result.is_ok() {
            return;
        }
        goto10 = true;
    }
    if goto10 {
        ialprt(false);
        imopen(iunit, &fortran_string(&line), "ro");
        unsafe {
            irdhdr(
                iunit,
                nxyz.as_mut_ptr(),
                mxyz.as_mut_ptr(),
                &mut mode,
                &mut dmin,
                &mut dmax,
                &mut dmean,
            );
            iiu_close(iunit);
        }
        ialprt(true);
    }
}

/// Original `line_is_filename` (`get_nxyz.f:55`).
///
/// LINE_IS_FILENAME tests for whether a line could be a valid numeric input
/// and returns TRUE if not.  String must contain spaces, tabs, period, comma,
/// plus, minus, digits.
pub fn line_is_filename(line: &[u8]) -> bool {
    for &byte in line {
        let ich = byte as i32;
        if ich != 32 && ich != 11 && (!(43..=57).contains(&ich) || ich == 47) {
            return true;
        }
    }
    false
}
