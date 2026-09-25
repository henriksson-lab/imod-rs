//! Translation of `IMOD/flib/subrs/hvem/get_tilt_angles.f90`.

use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use crate::imod::libcfshr::parse_params::{
    pip_get_float, pip_get_float_array, pip_number_of_entries,
};
use crate::imod::libcfshr::pip_fwrap::pipgetstring_;
use std::io::{BufRead, BufReader, Write};

/// Original `get_tilt_angles` (`get_tilt_angles.f90:18`).
///
/// GET_TILT_ANGLES will read in tilt angles by the user's method of
/// choice.
/// ^ [numViews] should be the number of views, if known, or 0 if it
/// not known, in which case the routine will return the number in this
/// variable.
/// ^ [nunit] should contain the number of a free logical unit.
/// ^ Tilt angles are returned in [tilt].
/// ^ [limTilt] should contain the dimensions of [tilt].
/// ^ [ifpip] should be set to 0 for interactive input, or nonzero for
/// input via PIP.  In the later case, the program must define the
/// options FirstTiltAngle, TiltIncrement, TiltFile, and TiltAngles.
/// If a TiltFile is entered, or if TiltAngles are entered, they
/// supercede entries of FirstTiltAngle and TiltIncrement.
///
/// Unit 5 is standard input.  A list-directed `READ` without `ERR=`/`END=`
/// that fails is a gfortran runtime error, which terminates with status 2.
/// `filename` is the source's `character*320`, filled by the Fortran
/// `pipgetstring` wrapper or an `(a)` read and handed on with its blank
/// padding trimmed, as Fortran `OPEN` does.
pub fn get_tilt_angles(
    num_views: &mut i32,
    nunit: i32,
    tilt: &mut [f32],
    lim_tilt: i32,
    ifpip: i32,
) {
    let mut filename = [b' '; 320];
    let mut n_view_in: i32 = 0;
    let mut tilt_start: f32 = 0.;
    let mut tilt_inc: f32 = 0.;
    let mut num_lines: i32 = 0;
    let stdin = std::io::stdin();
    //
    if ifpip != 0 {
        let mut ierr = pip_get_float(b"FirstTiltAngle", &mut tilt_start);
        let mut ierr2 = pip_get_float(b"TiltIncrement", &mut tilt_inc);
        let start_inc_ok = ierr == 0 && ierr2 == 0;

        ierr = pipgetstring_(b"TiltFile", &mut filename);
        ierr2 = pip_number_of_entries(b"TiltAngles", &mut num_lines);
        let _ = ierr2;

        if ierr > 0 && num_lines == 0 {
            if !start_inc_ok {
                gta_errorexit(
                    "No tilt angles specified by start and increment, file, or individual values",
                );
            }
            for i in 1..=*num_views {
                tilt[(i - 1) as usize] = tilt_start + (i - 1) as f32 * tilt_inc;
            }
            return;
        }

        if num_lines > 0 {
            if ierr == 0 {
                gta_errorexit("You cannot specify both a tilt angle file and individual entries");
            }
            let mut index: i32 = 0;
            for _i in 1..=num_lines {
                let mut nin_line: i32 = 0;
                ierr = pip_get_float_array(
                    b"TiltAngles",
                    &mut tilt[index as usize..],
                    &mut nin_line,
                    lim_tilt - index,
                );
                index += nin_line;
            }
            if *num_views == 0 {
                *num_views = index;
            }
            if index != *num_views {
                // print *
                // print *,'ERROR: GET_TILT_ANGLES -', numViews, &
                //     ' angles expected but only', index, ' entered'
                // gfortran list-directed layout: a leading blank, each
                // integer*4 in 12 columns, and a separator blank between an
                // integer and the character item after it.
                println!();
                println!(
                    " ERROR: GET_TILT_ANGLES -{:>12}  angles expected but only{:>12}  entered",
                    *num_views, index
                );
                crate::imod::libcfshr::b3dutil::exit(1);
            }
            return;
        }
    } else {
        let mut stdin = stdin.lock();
        if *num_views == 0 {
            print!(
                " Enter the # of tilt angles to specify them by starting and increment angle,\n     - the # of angles to specify each individual value,\n    or 0 to read angles from a file: "
            );
            let _ = std::io::stdout().flush();
            // read(5,*) nViewIn
            if let Err(err) = list_read(&mut stdin, &mut [ListItem::Integer(&mut n_view_in)]) {
                fortran_read_abort(err);
            }
            *num_views = n_view_in.wrapping_abs();
        } else {
            print!(
                " Enter 1 to specify starting and increment angle,\n     -1 to specify each individual value,\n    or 0 to read angles from a file: "
            );
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(&mut stdin, &mut [ListItem::Integer(&mut n_view_in)]) {
                fortran_read_abort(err);
            }
        }
        if *num_views > lim_tilt {
            gta_errorexit("Too many views for tilt angle array");
        }
        if n_view_in > 0 {
            print!(" Starting and increment angle: ");
            let _ = std::io::stdout().flush();
            if let Err(err) = list_read(
                &mut stdin,
                &mut [
                    ListItem::Real(&mut tilt_start),
                    ListItem::Real(&mut tilt_inc),
                ],
            ) {
                fortran_read_abort(err);
            }
            for i in 1..=*num_views {
                tilt[(i - 1) as usize] = tilt_start + (i - 1) as f32 * tilt_inc;
            }
            return;
        } else if n_view_in < 0 {
            println!(" Enter all tilt angles");
            // read(5,*,err = 40, end = 30) (tilt(i), i = 1, numViews)
            let mut items: Vec<ListItem> = tilt[..*num_views as usize]
                .iter_mut()
                .map(ListItem::Real)
                .collect();
            match list_read(&mut stdin, &mut items) {
                Ok(()) => return,
                // 30 call gta_errorexit('end of input reached reading tilt angles')
                Err(ListReadError::End) => {
                    gta_errorexit("end of input reached reading tilt angles")
                }
                // 40 call gta_errorexit('error reading tilt angles')
                Err(ListReadError::Error) => gta_errorexit("error reading tilt angles"),
            }
        } else {
            print!(" Name of file with tilt angles: ");
            let _ = std::io::stdout().flush();
            // read(5, '(a)') filename
            let mut line: Vec<u8> = Vec::new();
            match stdin.read_until(b'\n', &mut line) {
                Ok(0) | Err(_) => fortran_read_abort(ListReadError::End),
                Ok(_) => {}
            }
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            let count = line.len().min(filename.len());
            filename[..count].copy_from_slice(&line[..count]);
            filename[count..].fill(b' ');
        }
    }

    let end = filename
        .iter()
        .rposition(|&b| b != b' ')
        .map_or(0, |index| index + 1);
    let name = String::from_utf8_lossy(&filename[..end]).into_owned();
    read_tilt_file(num_views, nunit, &name, tilt, lim_tilt);
}

/// Original `read_tilt_file` (`get_tilt_angles.f90:114`).
///
/// READ_TILT_FILE will read in tilt angles from a file.
/// ^ [numViews] should be the number of views, if known, or 0 if it
/// not known, in which case the routine will return the number in this
/// variable.
/// ^ [nunit] should contain the number of a free logical unit.
/// ^ [filename] should contain the filename.
/// ^ Tilt angles are returned in [tilt].
/// ^ [limTilt] should contain the dimensions of [tilt].
///
/// When the view count is found from the file, the source reads each value
/// into `tilt(ind)` before testing `ind` against `limTilt`, so the value one
/// past the limit is stored past the array and the error exit follows.  Here
/// that one value is read into a temporary and stored only if it fits the
/// slice; the error exit is the same.
pub fn read_tilt_file(
    num_views: &mut i32,
    nunit: i32,
    filename: &str,
    tilt: &mut [f32],
    lim_tilt: i32,
) {
    let mut unit = BufReader::new(dopen(nunit, filename, "ro", "f"));
    let mut ind: i32;
    let failure: ListReadError;
    if *num_views == 0 {
        loop {
            // 10
            ind = *num_views + 1;
            // read(nunit,*,end = 20, err = 40) tilt(ind)
            let mut value = tilt.get(ind as usize - 1).copied().unwrap_or(0.);
            match list_read(&mut unit, &mut [ListItem::Real(&mut value)]) {
                Ok(()) => {}
                // 20 close(nunit)
                Err(ListReadError::End) => return,
                Err(ListReadError::Error) => {
                    failure = ListReadError::Error;
                    break;
                }
            }
            if let Some(slot) = tilt.get_mut(ind as usize - 1) {
                *slot = value;
            }
            *num_views = ind;
            if *num_views > lim_tilt {
                gta_errorexit("Too many views for tilt angle array");
            }
        }
    } else {
        ind = 1;
        let mut result = Ok(());
        while ind <= *num_views {
            // read(nunit,*,err = 40, end = 30) tilt(ind)
            result = list_read(
                &mut unit,
                &mut [ListItem::Real(&mut tilt[(ind - 1) as usize])],
            );
            if result.is_err() {
                break;
            }
            ind += 1;
        }
        match result {
            // 20 close(nunit)
            Ok(()) => return,
            Err(err) => failure = err,
        }
    }
    let i5 = |value: i32| -> String {
        let text = format!("{value:>5}");
        if text.len() > 5 {
            "*****".to_string()
        } else {
            text
        }
    };
    if failure == ListReadError::End {
        // 30 write(message,'(a,i5,a,i5)')'end of file reached after reading',ind - 1,  &
        //      ' tilt angles, expected', numViews
        let message = format!(
            "end of file reached after reading{} tilt angles, expected{}",
            i5(ind - 1),
            i5(*num_views)
        );
        gta_errorexit(&message);
    }
    // 40 write(message,'(a,i5)')'error reading tilt angles on line', ind
    let message = format!("error reading tilt angles on line{}", i5(ind));
    gta_errorexit(&message);
}

/// Original `gta_errorexit` (`get_tilt_angles.f90:142`).
pub fn gta_errorexit(message: &str) -> ! {
    // write(*,'(/,a,a)') 'ERROR: GET_TILT_ANGLES - ', trim(message)
    println!(
        "\nERROR: GET_TILT_ANGLES - {}",
        message.trim_end_matches(' ')
    );
    crate::imod::libcfshr::b3dutil::exit(1);
}

/// A list-directed `READ(5,*)` with no `END=`/`ERR=` label: the gfortran
/// runtime reports the failure and stops the program with status 2.
fn fortran_read_abort(err: ListReadError) -> ! {
    let _ = std::io::stdout().flush();
    match err {
        ListReadError::End => eprintln!("Fortran runtime error: End of file"),
        ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
    }
    crate::imod::libcfshr::b3dutil::exit(2);
}
