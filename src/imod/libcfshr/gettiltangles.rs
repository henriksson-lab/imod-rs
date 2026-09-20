//! Translation of `IMOD/libcfshr/gettiltangles.c` and its `cfsemshare.h` APIs.

use std::ffi::OsString;
use std::fs::File;
use std::io::BufReader;
use std::os::unix::ffi::OsStringExt;
use std::path::{Path, PathBuf};

use super::parse_params::{
    exit_error, pip_get_float, pip_get_float_array, pip_get_string, pip_number_of_entries,
};
use super::readlinevalues::{
    RLFV_SEPARATE_LINES, ReadValueArray, exit_from_value_read_error, read_lines_for_values,
};

/// Complete function inventory for `gettiltangles.c`.
pub const GET_TILT_ANGLES_SOURCE_FUNCTIONS: &[&str] = &["getTiltAngles", "readTiltFile"];

/// Original `readTiltFile` (`gettiltangles.c:90`).
pub fn read_tilt_file(num_views: &mut i32, filename: &Path, tilt: &mut [f32]) {
    if *num_views > tilt.len() as i32 {
        exit_error(b"Array for tilt angles is not big enough for the number of views");
    }
    let file = match File::open(filename) {
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
        let mut num_views = 0;
        let mut tilt = [0.; 4];
        read_tilt_file(&mut num_views, &path, &mut tilt);
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
        read_tilt_file(&mut num_views, &path, &mut tilt);
        assert_eq!(&tilt[..2], &[1., 2.]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn inventory_covers_the_complete_source_unit() {
        assert_eq!(GET_TILT_ANGLES_SOURCE_FUNCTIONS.len(), 2);
    }
}
