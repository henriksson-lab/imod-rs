//! Translation of `IMOD/libcfshr/gettiltangles.c` and its `cfsemshare.h` APIs.

use super::b3dutil::{CArg, b3d_open_file, c_format, c_format_bytes};
use super::parse_params::{
    exit_error, pip_get_float, pip_get_float_array, pip_get_string, pip_number_of_entries,
};
use super::readlinevalues::{
    RLFV_SEPARATE_LINES, ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};

/// Original `getTiltAngles` (`gettiltangles.c:27`).  `limTilt` is
/// `tilt.len()`; pass a shorter slice for a smaller limit.
///
/// `PipNumberOfEntries` writes `numLines` only when the option is in the
/// table, and the C does not test its return, so a program that never
/// declared `TiltAngles` would read an indeterminate `numLines`; it starts at
/// 0 here.  Likewise `filename` is only set when `PipGetString` finds the
/// option, and is empty otherwise.
pub fn get_tilt_angles(num_views: &mut i32, tilt: &mut [f32]) {
    let lim_tilt = tilt.len() as i32;
    let mut filename: Vec<u8> = Vec::new();
    let mut ierr: i32;
    let ierr2: i32;
    let mut num_lines: i32 = 0;
    let mut index: i32;
    let mut nin_line: i32;
    let mut tilt_start: f32 = 0.;
    let mut tilt_inc: f32 = 0.;
    let start_inc_ok: i32;
    //
    if *num_views > lim_tilt {
        exit_error(b"Array for tilt angles is not big enough for the number of views");
    }
    ierr = pip_get_float(b"FirstTiltAngle", &mut tilt_start);
    ierr2 = pip_get_float(b"TiltIncrement", &mut tilt_inc);
    start_inc_ok = if ierr == 0 && ierr2 == 0 { 1 } else { 0 };

    ierr = pip_get_string(b"TiltFile", &mut filename);
    let _ = pip_number_of_entries(b"TiltAngles", &mut num_lines);

    if ierr > 0 && num_lines == 0 {
        if start_inc_ok == 0 {
            exit_error(
                b"No tilt angles specified by start and increment, file, or individual values",
            );
        }
        if *num_views <= 0 {
            exit_error(
                b"Tilt angles may not be entered by starting and increment angles because the program did not specify the number of views",
            );
        }
        for i in 0..*num_views {
            tilt[i as usize] = tilt_start + i as f32 * tilt_inc;
        }
        return;
    }

    if num_lines > 0 {
        if ierr == 0 {
            exit_error(b"You cannot specify both a tilt angle file and individual entries");
        }
        index = 0;
        for _i in 1..=num_lines {
            nin_line = 0;
            let _ = pip_get_float_array(
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
            exit_error(
                c_format(
                    "%d angles expected but only %d entered with TiltAngles",
                    &[CArg::Int(*num_views as i64), CArg::Int(index as i64)],
                )
                .as_bytes(),
            );
        }
        return;
    }
    read_tilt_file(
        num_views,
        &String::from_utf8_lossy(&filename),
        tilt,
        lim_tilt,
    );
}

/// `gettiltangles.c:75`: `#define MAX_LINE 80`.
const MAX_LINE: usize = 80;

/// Original `readTiltFile` (`gettiltangles.c:90`).
pub fn read_tilt_file(num_views: &mut i32, filename: &str, tilt: &mut [f32], lim_tilt: i32) {
    let mut line = [0u8; MAX_LINE];
    let mut fp;
    let ierr: i32;
    let mut num_to_get = *num_views;
    if *num_views > lim_tilt {
        exit_error(b"Array for tilt angles is not big enough for the number of views");
    }

    fp = b3d_open_file(filename, "r");
    ierr = read_lines_for_values(
        &mut fp,
        &mut num_to_get,
        lim_tilt,
        &mut line,
        MAX_LINE as i32,
        RLFV_SEPARATE_LINES,
        "f",
        &mut [ReadValueArray::Floats(tilt)],
    );
    if ierr == -2 {
        exit_error(&c_format_bytes(
            "End of file reached after reading %d tilt angles, expected %d\n",
            &[CArg::Int(num_to_get as i64), CArg::Int(*num_views as i64)],
        ));
    }
    if ierr != 0 {
        exit_from_value_read_error(ierr, "tilt angles");
    }
    if *num_views == 0 {
        *num_views = num_to_get;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn reads_one_tilt_angle_from_each_file_line() {
        let path = std::env::temp_dir().join(format!("imod-rs-tilts-{}", std::process::id()));
        let mut file = File::create(&path).unwrap();
        writeln!(file, "-60.0\n-1.5\n45.25").unwrap();
        drop(file);
        let mut num_views = 0;
        let mut tilt = [0.; 4];
        read_tilt_file(&mut num_views, path.to_str().unwrap(), &mut tilt, 4);
        assert_eq!(num_views, 3);
        assert_eq!(&tilt[..3], &[-60., -1.5, 45.25]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_the_requested_number_of_tilt_angles() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-tilts-requested-{}", std::process::id()));
        let mut file = File::create(&path).unwrap();
        writeln!(file, "1\n2\n3").unwrap();
        drop(file);
        let mut num_views = 2;
        let mut tilt = [0.; 3];
        read_tilt_file(&mut num_views, path.to_str().unwrap(), &mut tilt, 3);
        assert_eq!(&tilt[..2], &[1., 2.]);
        std::fs::remove_file(path).unwrap();
    }
}
