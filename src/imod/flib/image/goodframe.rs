//! Translation of `IMOD/flib/image/goodframe.f90`.
//!
//! GOODFRAME reads a number and prints out the next highest valid frame size
//! for an FFT.  By default, it requires no prime factor greater than 19
//! regardless of whether FFTW is being used, because blendmont does the same.
//! With the option -n, it will use the limit from calling niceFFTlimit
//! instead.
//!
//! The main program maps to [`goodframe`].  `niceframe` is the C routine
//! `niceFrame` (`filtxcorr.c`) through its Fortran wrapper, and
//! `niceFFTlimit` is `libfft`'s.  A list-directed `read` with no
//! `END=`/`ERR=` that fails is the gfortran runtime error, status 2.

use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::libcfshr::b3dutil::{exit, program_args};
use crate::imod::libcfshr::filtxcorr::nice_frame;
use crate::imod::libfft::nice_fft_limit;
use std::io::Write;

/// Original program `goodframe` (`goodframe.f90:8`).
pub fn goodframe() {
    let mut nices = [0_i32; 10];
    let mut nx = 0_i32;
    let mut limit: i32;
    let mut ind_first: i32;
    let args = program_args();
    // iargc()
    let iargc = args.len() as i32 - 1;
    // `character*120 arg`: getarg keeps the first 120 characters.
    let getarg = |index: i32| -> Vec<u8> {
        let arg = args[index as usize].as_bytes();
        let mut field = arg[..arg.len().min(120)].to_vec();
        field.resize(120, b' ');
        field
    };
    // A failed list-directed read is the runtime's error.
    let read_abort = |err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => {
                eprintln!("Fortran runtime error: Bad integer for item 1 in list input")
            }
        }
        exit(2);
    };
    limit = 19;
    ind_first = 1;
    if iargc == 0 {
        // read(5,*) nx
        let mut stdin = std::io::stdin().lock();
        if let Err(err) = list_read(&mut stdin, &mut [ListItem::Integer(&mut nx)]) {
            read_abort(err);
        }
        nx += nx % 2;
        let nxp = nice_frame(nx, 2, 19);
        // print *,nxp
        println!("{:>12}", nxp);
        let _ = std::io::stdout().flush();
        exit(0);
    }
    let ndo = iargc.min(10);
    let arg = getarg(1);
    // trim(arg) == '-n'
    if arg.trim_ascii_end() == b"-n" {
        limit = nice_fft_limit();
        ind_first = 2;
    }
    for i in ind_first..=ndo {
        let arg = getarg(i);
        // read(arg, *) nx
        let mut unit = std::io::Cursor::new(arg);
        if let Err(err) = list_read(&mut unit, &mut [ListItem::Integer(&mut nx)]) {
            read_abort(err);
        }
        nices[(i - 1) as usize] = nice_frame(nx, 2, limit);
    }
    // print *,(nices(i), i = indFirst, ndo)
    let mut line = String::new();
    for i in ind_first..=ndo {
        line.push_str(&format!("{:>12}", nices[(i - 1) as usize]));
    }
    // An empty list writes an empty record (no leading blank).
    println!("{line}");
    let _ = std::io::stdout().flush();
    exit(0);
}
