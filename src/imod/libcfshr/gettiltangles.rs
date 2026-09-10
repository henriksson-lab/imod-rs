//! Translation of `IMOD/libcfshr/gettiltangles.c` and its `cfsemshare.h` APIs.

use core::ffi::{CStr, c_char};
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
pub unsafe fn get_tilt_angles(num_views: *mut i32, tilt: *mut f32, lim_tilt: i32) {
    if unsafe { *num_views > lim_tilt } {
        unsafe {
            exit_error(c"Array for tilt angles is not big enough for the number of views".as_ptr())
        };
    }
    let mut tilt_start = 0.;
    let mut tilt_inc = 0.;
    let mut ierr = unsafe { pip_get_float(c"FirstTiltAngle".as_ptr(), &mut tilt_start) };
    let ierr2 = unsafe { pip_get_float(c"TiltIncrement".as_ptr(), &mut tilt_inc) };
    let start_inc_ok = ierr == 0 && ierr2 == 0;
    let mut filename: *mut c_char = core::ptr::null_mut();
    ierr = unsafe { pip_get_string(c"TiltFile".as_ptr(), &mut filename) };
    let mut num_lines = 0;
    unsafe { pip_number_of_entries(c"TiltAngles".as_ptr(), &mut num_lines) };

    if ierr > 0 && num_lines == 0 {
        if !start_inc_ok {
            unsafe {
                exit_error(
                    c"No tilt angles specified by start and increment, file, or individual values"
                        .as_ptr(),
                )
            };
        }
        if unsafe { *num_views <= 0 } {
            unsafe {
                exit_error(c"Tilt angles may not be entered by starting and increment angles because the program did not specify the number of views".as_ptr())
            };
        }
        for index in 0..unsafe { *num_views } as usize {
            unsafe { *tilt.add(index) = tilt_start + index as f32 * tilt_inc };
        }
        return;
    }
    if num_lines > 0 {
        if ierr == 0 {
            unsafe {
                exit_error(
                    c"You cannot specify both a tilt angle file and individual entries".as_ptr(),
                )
            };
        }
        let mut index = 0;
        for _ in 1..=num_lines {
            let mut nin_line = 0;
            unsafe {
                pip_get_float_array(
                    c"TiltAngles".as_ptr(),
                    tilt.add(index as usize),
                    &mut nin_line,
                    lim_tilt - index,
                )
            };
            index += nin_line;
        }
        if unsafe { *num_views == 0 } {
            unsafe { *num_views = index };
        }
        if index != unsafe { *num_views } {
            let message = format!(
                "{} angles expected but only {} entered with TiltAngles",
                unsafe { *num_views },
                index
            );
            let message = std::ffi::CString::new(message).unwrap();
            unsafe { exit_error(message.as_ptr()) };
        }
        return;
    }
    unsafe { read_tilt_file(num_views, filename, tilt, lim_tilt) };
}

/// Original `readTiltFile` (`gettiltangles.c:90`).
pub unsafe fn read_tilt_file(
    num_views: *mut i32,
    filename: *const c_char,
    tilt: *mut f32,
    lim_tilt: i32,
) {
    if unsafe { *num_views > lim_tilt } {
        unsafe {
            exit_error(c"Array for tilt angles is not big enough for the number of views".as_ptr())
        };
    }
    let path = unsafe { OsStr::from_bytes(CStr::from_ptr(filename).to_bytes()) };
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) => {
            let message =
                std::ffi::CString::new(format!("Error opening tilt angle file: {error}")).unwrap();
            unsafe { exit_error(message.as_ptr()) };
            unreachable!();
        }
    };
    let mut num_to_get = unsafe { *num_views };
    let output = unsafe { core::slice::from_raw_parts_mut(tilt, lim_tilt as usize) };
    let mut reader = BufReader::new(file);
    let ierr = read_lines_for_values(
        &mut reader,
        &mut num_to_get,
        lim_tilt as usize,
        RLFV_SEPARATE_LINES,
        "f",
        &mut [ReadValueArray::Floats(output)],
    );
    if ierr == -2 {
        let message = std::ffi::CString::new(format!(
            "End of file reached after reading {} tilt angles, expected {}\n",
            num_to_get,
            unsafe { *num_views }
        ))
        .unwrap();
        unsafe { exit_error(message.as_ptr()) };
    }
    if ierr != 0 {
        let message = exit_from_value_read_error(ierr, "tilt angles").unwrap_err();
        let message = std::ffi::CString::new(message).unwrap();
        unsafe { exit_error(message.as_ptr()) };
    }
    if unsafe { *num_views == 0 } {
        unsafe { *num_views = num_to_get };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Write;

    #[test]
    fn reads_one_tilt_angle_from_each_file_line() {
        let path = std::env::temp_dir().join(format!("imod-rs-tilts-{}", std::process::id()));
        let mut file = File::create(&path).unwrap();
        writeln!(file, "-60.0\n-1.5\n45.25").unwrap();
        drop(file);
        let filename = CString::new(path.as_os_str().as_bytes()).unwrap();
        let mut num_views = 0;
        let mut tilt = [0.; 4];
        unsafe {
            read_tilt_file(
                &mut num_views,
                filename.as_ptr(),
                tilt.as_mut_ptr(),
                tilt.len() as i32,
            )
        };
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
        let filename = CString::new(path.as_os_str().as_bytes()).unwrap();
        let mut num_views = 2;
        let mut tilt = [0.; 3];
        unsafe {
            read_tilt_file(
                &mut num_views,
                filename.as_ptr(),
                tilt.as_mut_ptr(),
                tilt.len() as i32,
            )
        };
        assert_eq!(&tilt[..2], &[1., 2.]);
        std::fs::remove_file(path).unwrap();
    }
}
