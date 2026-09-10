//! Translation of `IMOD/libcfshr/parselist.c`.
#![allow(dead_code)]

use core::ffi::c_char;
use core::ptr;

use crate::imod::libcfshr::b3dutil::f2c_string;

/// Original `parselist` (`parselist.c:35`).  The returned allocation has C
/// ownership and must be released with `libc::free`.
pub unsafe fn parselist(line: *const c_char, nlist: *mut i32) -> *mut i32 {
    let mut intern = [0_i8; 10];
    let mut dashlast = false;
    let mut negnum = false;
    let mut gotcomma = false;
    let mut gotnum = false;
    let mut got_space = false;
    let nchars = unsafe { libc::strlen(line) } as i32;
    let mut list: *mut i32 = ptr::null_mut();

    unsafe { *nlist = 0 };
    if unsafe { *line } == 0 {
        return ptr::null_mut();
    }
    if unsafe { *line } == b'/' as c_char {
        unsafe { *nlist = -1 };
        return ptr::null_mut();
    }
    let mut ind = 0_i32;
    let mut lastnum = 0_i32;

    loop {
        let next = unsafe { *line.add(ind as usize) } as u8;
        if next.is_ascii_digit() {
            gotnum = true;
            let mut numst = ind;
            loop {
                ind += 1;
                let digit = unsafe { *line.add(ind as usize) } as u8;
                if !digit.is_ascii_digit() {
                    break;
                }
            }

            if negnum {
                numst -= 1;
            }
            let mut i = numst;
            while i < ind && i < numst + 9 {
                intern[(i - numst) as usize] = unsafe { *line.add(i as usize) };
                i += 1;
            }
            intern[(i - numst) as usize] = 0;
            let mut number = 0_i32;
            unsafe {
                libc::sscanf(
                    intern.as_ptr(),
                    c"%d".as_ptr(),
                    core::ptr::addr_of_mut!(number),
                );
            }

            let mut loopst = number;
            let mut idir = 1_i32;
            if dashlast {
                if lastnum > number {
                    idir = -1;
                }
                loopst = lastnum + idir;
            }
            let mut value = loopst;
            while idir * value <= idir * number {
                let byte_count = (unsafe { *nlist } as usize + 1) * core::mem::size_of::<i32>();
                list = if unsafe { *nlist } != 0 {
                    unsafe { libc::realloc(list.cast(), byte_count).cast() }
                } else {
                    unsafe { libc::malloc(byte_count).cast() }
                };
                if list.is_null() {
                    unsafe {
                        *nlist = -2;
                        libc::free(list.cast());
                    }
                    return ptr::null_mut();
                }
                unsafe {
                    *list.add(*nlist as usize) = value;
                    *nlist += 1;
                }
                value += idir;
            }
            lastnum = number;
            negnum = false;
            dashlast = false;
            gotcomma = false;
            got_space = false;
            if ind >= nchars {
                break;
            }
            continue;
        }

        if next == b' ' || next == b'\t' {
            got_space = true;
        }
        if next != b',' && next != b' ' && next != b'-' && next != b'\t' {
            if got_space {
                break;
            }
            unsafe {
                *nlist = -3;
                libc::free(list.cast());
            }
            return ptr::null_mut();
        }
        if next == b',' {
            gotcomma = true;
            got_space = false;
        }
        if next == b'-' {
            got_space = false;
            if dashlast || !gotnum || gotcomma {
                negnum = true;
            } else {
                dashlast = true;
            }
        }
        ind += 1;
        if ind >= nchars {
            break;
        }
    }

    if gotcomma || negnum || dashlast {
        unsafe {
            *nlist = -3;
            libc::free(list.cast());
        }
        return ptr::null_mut();
    }
    list
}

/// Original Fortran wrapper `parselistfw` (`parselist.c:144`).
pub unsafe fn parselistfw(
    line: *const c_char,
    list: *mut i32,
    nlist: *mut i32,
    limlist: *mut i32,
    linelen: i32,
) -> i32 {
    let tempstr = unsafe { f2c_string(line, linelen) };
    if tempstr.is_null() {
        return 1;
    }
    let mut ncopy = 0_i32;
    let retlist = unsafe { parselist(tempstr, &mut ncopy) };
    unsafe { libc::free(tempstr.cast()) };
    if retlist.is_null() && ncopy == 0 {
        unsafe { *nlist = 0 };
        return 0;
    }
    if retlist.is_null() && ncopy < 0 {
        return -ncopy - 1;
    }
    unsafe {
        *nlist = ncopy;
        if *limlist > 0 && ncopy > *limlist {
            *nlist = *limlist;
        }
        if *nlist > 0 {
            ptr::copy_nonoverlapping(retlist, list, *nlist as usize);
        }
        libc::free(retlist.cast());
        if *limlist > 0 && ncopy > *limlist {
            -1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parselist, parselistfw};
    use std::ffi::CString;

    #[test]
    fn source_ranges_errors_and_fortran_copy_limit() {
        let input = CString::new("1-5,7,9,11,15-13").unwrap();
        let mut count = 0;
        let values = unsafe { parselist(input.as_ptr(), &mut count) };
        assert_eq!(count, 11);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(values, count as usize) },
            &[1, 2, 3, 4, 5, 7, 9, 11, 15, 14, 13]
        );
        unsafe { libc::free(values.cast()) };

        let slash = CString::new("/").unwrap();
        assert!(unsafe { parselist(slash.as_ptr(), &mut count) }.is_null());
        assert_eq!(count, -1);
        let bad = CString::new("1x").unwrap();
        assert!(unsafe { parselist(bad.as_ptr(), &mut count) }.is_null());
        assert_eq!(count, -3);

        let fortran = b"1-4  ";
        let mut output = [0; 2];
        let mut out_count = 0;
        let mut limit = 2;
        assert_eq!(
            unsafe {
                parselistfw(
                    fortran.as_ptr().cast(),
                    output.as_mut_ptr(),
                    &mut out_count,
                    &mut limit,
                    5,
                )
            },
            -1
        );
        assert_eq!(out_count, 2);
        assert_eq!(output, [1, 2]);
    }
}
