//! Translation of `IMOD/libcfshr/writelist.c`.
#![allow(dead_code)]

use core::ffi::c_char;

unsafe extern "C" {
    static mut stdout: *mut libc::FILE;
}

/// Original `writeList` (`writelist.c:23`).
pub unsafe fn write_list(list: *mut i32, number_of_values: i32, line_length: i32) -> i32 {
    unsafe {
        let mut line_string = list_to_string(list, number_of_values);
        let saved_string = line_string;
        if line_string.is_null() {
            return 1;
        }
        while libc::strlen(line_string) > line_length as usize {
            let mut index = line_length - 1;
            while index > 0 {
                if *line_string.add(index as usize) == b',' as c_char {
                    break;
                }
                index -= 1;
            }
            *line_string.add(index as usize) = b'\n' as c_char;
            line_string = line_string.add(index as usize + 1);
        }
        libc::printf(c"%s".as_ptr(), saved_string);
        libc::printf(c"\n".as_ptr());
        libc::fflush(stdout);
        libc::free(saved_string.cast());
        0
    }
}

/// Original Fortran wrapper `writelist` (`writelist.c:45`).
pub unsafe fn writelist(list: *mut i32, number_of_values: *mut i32, line_length: *mut i32) -> i32 {
    unsafe { write_list(list, *number_of_values, *line_length) }
}

/// Original Fortran wrapper `wrlist` (`writelist.c:52`).
pub unsafe fn wrlist(list: *mut i32, number_of_values: *mut i32) {
    unsafe {
        write_list(list, *number_of_values, 80);
    }
}

/// Original `listToString` (`writelist.c:62`).
pub unsafe fn list_to_string(list: *mut i32, number_of_values: i32) -> *mut c_char {
    unsafe {
        let mut returned_string: *mut c_char = core::ptr::null_mut();
        let mut range_start = *list;
        for index in 1..number_of_values {
            if *list.add(index as usize) != *list.add(index as usize - 1) + 1 {
                returned_string =
                    add_range_to_line(range_start, *list.add(index as usize - 1), returned_string);
                if returned_string.is_null() {
                    return core::ptr::null_mut();
                }
                range_start = *list.add(index as usize);
            }
        }
        add_range_to_line(
            range_start,
            *list.add(number_of_values as usize - 1),
            returned_string,
        )
    }
}

/// Original static `addRangeToLine` (`writelist.c:78`).
pub unsafe fn add_range_to_line(
    number_start: i32,
    number_end: i32,
    line: *mut c_char,
) -> *mut c_char {
    unsafe {
        let mut range_string = [0_i8; 32];
        let mut end_string = [0_i8; 16];
        libc::sprintf(range_string.as_mut_ptr(), c"%d".as_ptr(), number_start);
        if number_end > number_start {
            libc::strcat(range_string.as_mut_ptr(), c"-".as_ptr());
            libc::sprintf(end_string.as_mut_ptr(), c"%d".as_ptr(), number_end);
            libc::strcat(range_string.as_mut_ptr(), end_string.as_ptr());
        }
        if !line.is_null() {
            let returned_line = libc::realloc(
                line.cast(),
                libc::strlen(line) + libc::strlen(range_string.as_ptr()) + 4,
            )
            .cast::<c_char>();
            if returned_line.is_null() {
                return core::ptr::null_mut();
            }
            libc::strcat(returned_line, c",".as_ptr());
            libc::strcat(returned_line, range_string.as_ptr());
            returned_line
        } else {
            let returned_line =
                libc::malloc(libc::strlen(range_string.as_ptr()) + 4).cast::<c_char>();
            if returned_line.is_null() {
                return core::ptr::null_mut();
            }
            libc::strcpy(returned_line, range_string.as_ptr());
            returned_line
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::ffi::CStr;

    #[test]
    fn converts_only_increasing_adjacent_values_to_source_ranges() {
        unsafe {
            let mut values = [1_i32, 2, 3, 8, 10, 11, 4];
            let text = list_to_string(values.as_mut_ptr(), values.len() as i32);
            assert_eq!(CStr::from_ptr(text).to_bytes(), b"1-3,8,10-11,4");
            libc::free(text.cast());
        }
    }
}
