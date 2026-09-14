//! Translation of `IMOD/libcfshr/gettiltangles.c` and its `cfsemshare.h` APIs.

use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::ffi::OsStrExt;

use super::parse_params::{
    exit_error, pip_get_float, pip_get_float_array, pip_get_string, pip_number_of_entries,
};
use super::readlinevalues::{
    RLFV_SEPARATE_LINES, ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};

/// Original `getTiltAngles` (`gettiltangles.c:27`).
pub fn get_tilt_angles(num_views: &mut i32, tilt: &mut [f32]) {
    if *num_views > tilt.len() as i32 {
        exit_error(b"Array for tilt angles is not big enough for the number of views");
    }
    let mut tilt_start = 0.;
    let mut tilt_inc = 0.;
    let mut ierr = pip_get_float(b"FirstTiltAngle", &mut tilt_start);
    let ierr2 = pip_get_float(b"TiltIncrement", &mut tilt_inc);
    let start_inc_ok = ierr == 0 && ierr2 == 0;
    let mut filename: Vec<u8> = Vec::new();
    ierr = pip_get_string(b"TiltFile", &mut filename);
    let mut num_lines = 0;
    pip_number_of_entries(b"TiltAngles", &mut num_lines);

    if ierr > 0 && num_lines == 0 {
        if !start_inc_ok {
            exit_error(
                b"No tilt angles specified by start and increment, file, or individual values",
            );
        }
        if *num_views <= 0 {
            exit_error(b"Tilt angles may not be entered by starting and increment angles because the program did not specify the number of views");
        }
        for (index, angle) in tilt.iter_mut().take(*num_views as usize).enumerate() {
            *angle = tilt_start + index as f32 * tilt_inc;
        }
        return;
    }
    if num_lines > 0 {
        if ierr == 0 {
            exit_error(b"You cannot specify both a tilt angle file and individual entries");
        }
        let mut index = 0;
        for _ in 1..=num_lines {
            let mut nin_line = 0;
            let start = (index as usize).min(tilt.len());
            let remaining = tilt.len() - start;
            pip_get_float_array(
                b"TiltAngles",
                &mut tilt[start..],
                &mut nin_line,
                remaining as i32,
            );
            index += nin_line;
        }
        if *num_views == 0 {
            *num_views = index;
        }
        if index != *num_views {
            let message = format!(
                "{} angles expected but only {} entered with TiltAngles",
                *num_views, index
            );
            exit_error(message.as_bytes());
        }
        return;
    }
    read_tilt_file(num_views, &filename, tilt);
}

/// Original `readTiltFile` (`gettiltangles.c:90`).
pub fn read_tilt_file(num_views: &mut i32, filename: &[u8], tilt: &mut [f32]) {
    if *num_views > tilt.len() as i32 {
        exit_error(b"Array for tilt angles is not big enough for the number of views");
    }
    let path = OsStr::from_bytes(filename);
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) => {
            let message = format!("Error opening tilt angle file: {error}");
            exit_error(message.as_bytes());
        }
    };
    let mut num_to_get = *num_views;
    let mut reader = BufReader::new(file);
    let ierr = read_lines_for_values(
        &mut reader,
        &mut num_to_get,
        tilt.len(),
        RLFV_SEPARATE_LINES,
        "f",
        &mut [ReadValueArray::Floats(tilt)],
    );
    if ierr == -2 {
        let message = format!(
            "End of file reached after reading {} tilt angles, expected {}\n",
            num_to_get, *num_views
        );
        exit_error(message.as_bytes());
    }
    if ierr != 0 {
        let message = exit_from_value_read_error(ierr, "tilt angles").unwrap_err();
        exit_error(message.as_bytes());
    }
    if *num_views == 0 {
        *num_views = num_to_get;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_one_tilt_angle_from_each_file_line() {
        let path = std::env::temp_dir().join(format!("imod-rs-tilts-{}", std::process::id()));
        let mut file = File::create(&path).unwrap();
        writeln!(file, "-60.0\n-1.5\n45.25").unwrap();
        drop(file);
        let filename = path.as_os_str().as_bytes().to_vec();
        let mut num_views = 0;
        let mut tilt = [0.; 4];
        read_tilt_file(&mut num_views, &filename, &mut tilt);
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
        let filename = path.as_os_str().as_bytes().to_vec();
        let mut num_views = 2;
        let mut tilt = [0.; 3];
        read_tilt_file(&mut num_views, &filename, &mut tilt);
        assert_eq!(&tilt[..2], &[1., 2.]);
        std::fs::remove_file(path).unwrap();
    }
}
