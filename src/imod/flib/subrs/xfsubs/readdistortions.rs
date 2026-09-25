//! Translation of `IMOD/flib/subrs/xfsubs/readdistortions.f`.
use crate::imod::flib::subrs::hvem::dopen::dopen;
use crate::imod::flib::subrs::hvem::frefor::{ListItem, ListReadError, list_read};
use std::io::{BufReader, Write};

/// Original `readMagGradients` (`readdistortions.f:198`).
#[allow(clippy::too_many_arguments)]
pub fn read_mag_gradients(
    mag_file: &str,
    max_vals: i32,
    pixel_size: &mut f32,
    axis_rot: &mut f32,
    tilt: &mut [f32],
    dmag_per_um: &mut [f32],
    rot_per_um: &mut [f32],
    num_vals: &mut i32,
) {
    let mut mag_version = 0_i32;
    let mut unit14 = BufReader::new(dopen(14, mag_file, "ro", "f"));
    // `read(14, *)` with no `END=`/`ERR=`: the gfortran runtime reports the
    // failed read and stops with status 2.
    let read_abort = |line: i32, err: ListReadError| -> ! {
        let _ = std::io::stdout().flush();
        eprintln!("At line {line} of file readdistortions.f (unit = 14, file = '{mag_file}')");
        match err {
            ListReadError::End => eprintln!("Fortran runtime error: End of file"),
            ListReadError::Error => eprintln!("Fortran runtime error: Bad value during read"),
        }
        crate::imod::libcfshr::b3dutil::exit(2);
    };
    if let Err(err) = list_read(&mut unit14, &mut [ListItem::Integer(&mut mag_version)]) {
        read_abort(206, err);
    }
    if mag_version == 1 {
        if let Err(err) = list_read(
            &mut unit14,
            &mut [
                ListItem::Integer(num_vals),
                ListItem::Real(pixel_size),
                ListItem::Real(axis_rot),
            ],
        ) {
            read_abort(208, err);
        }
        if *num_vals > max_vals {
            println!();
            println!(" ERROR: readMagGradients - too many values for arrays");
            crate::imod::libcfshr::b3dutil::exit(1);
        }
        for i in 1..=*num_vals as usize {
            if let Err(err) = list_read(
                &mut unit14,
                &mut [
                    ListItem::Real(&mut tilt[i - 1]),
                    ListItem::Real(&mut dmag_per_um[i - 1]),
                    ListItem::Real(&mut rot_per_um[i - 1]),
                ],
            ) {
                read_abort(215, err);
            }
        }
    } else {
        println!();
        println!(
            " ERROR: readMagGradients - version{mag_version:12}  of gradient file not recognized"
        );
        crate::imod::libcfshr::b3dutil::exit(1);
    }
    // `close(14)` is the drop of the reader.
}
