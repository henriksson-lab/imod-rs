//! Translation of `IMOD/libcfshr/parselist.c`.
#![allow(dead_code)]

use super::parse_params::strtol;

/// Original `parselist` (`parselist.c:35`).
///
/// The C returns a `malloc`ed `int *` and reports the count, or an error code,
/// in `*nlist`; here the allocation is the returned `Vec` and `None` is the
/// C's NULL, so `*nlist` still carries 0 (empty), -1 (a leading `/`), -2 (the
/// allocation failed, which cannot happen here) or -3 (a bad character).
///
/// `line` is the C string's bytes without its NUL: `nchars` is `strlen(line)`,
/// and the one place the loop can read at that index sees the terminator, so
/// an index at or past the end reads 0 exactly as the C does.
pub fn parselist(line: &[u8], nlist: &mut i32) -> Option<Vec<i32>> {
    let mut intern = [0_u8; 10];
    let mut dashlast = false;
    let mut negnum = false;
    let mut gotcomma = false;
    let mut gotnum = false;
    let mut got_space = false;
    let nchars = line.len() as i32;
    let mut list: Vec<i32> = Vec::new();

    *nlist = 0;
    if line.is_empty() {
        return None;
    }
    if line[0] == b'/' {
        *nlist = -1;
        return None;
    }
    let mut ind = 0_i32;
    let mut lastnum = 0_i32;

    loop {
        let next = *line.get(ind as usize).unwrap_or(&0);
        if next.is_ascii_digit() {
            gotnum = true;
            let mut numst = ind;
            loop {
                ind += 1;
                let digit = *line.get(ind as usize).unwrap_or(&0);
                if !digit.is_ascii_digit() {
                    break;
                }
            }

            if negnum {
                numst -= 1;
            }
            let mut i = numst;
            while i < ind && i < numst + 9 {
                intern[(i - numst) as usize] = line[i as usize];
                i += 1;
            }
            intern[(i - numst) as usize] = 0;
            // `sscanf(intern, "%d", &number)`: leading white space, an optional
            // sign, then digits, which is `strtol` in base 10.
            let mut scanned = 0usize;
            let number = strtol(
                &intern[..intern.iter().position(|&b| b == 0).unwrap_or(intern.len())],
                &mut scanned,
                10,
            ) as i32;

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
                list.push(value);
                *nlist += 1;
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
            *nlist = -3;
            return None;
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
        *nlist = -3;
        return None;
    }
    Some(list)
}

/// Original Fortran wrapper `parselistfw` (`parselist.c:144`).
///
/// Both sides of this bridge are Rust now — its one caller is `rdlist.rs`,
/// itself a translated Fortran unit — so the hidden length argument is the
/// slice's own length and `f2c_string`'s trailing-blank trim
/// (`b3dutil.c:1189`) is done in place of the copy it made.
pub fn parselistfw(line: &[u8], list: &mut [i32], nlist: &mut i32, limlist: &mut i32) -> i32 {
    // `f2c_string`: drop trailing blanks, then NUL-terminate.
    let mut index = line.len();
    while index > 0 && line[index - 1] == b' ' {
        index -= 1;
    }
    let tempstr = &line[..index];
    let mut ncopy = 0_i32;
    let retlist = parselist(tempstr, &mut ncopy);
    let Some(retlist) = retlist else {
        if ncopy == 0 {
            *nlist = 0;
            return 0;
        }
        return -ncopy - 1;
    };
    *nlist = ncopy;
    if *limlist > 0 && ncopy > *limlist {
        *nlist = *limlist;
    }
    if *nlist > 0 {
        list[..*nlist as usize].copy_from_slice(&retlist[..*nlist as usize]);
    }
    if *limlist > 0 && ncopy > *limlist {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{parselist, parselistfw};

    #[test]
    fn source_ranges_errors_and_fortran_copy_limit() {
        let mut count = 0;
        let values = parselist(b"1-5,7,9,11,15-13", &mut count).unwrap();
        assert_eq!(count, 11);
        assert_eq!(values, &[1, 2, 3, 4, 5, 7, 9, 11, 15, 14, 13]);

        assert!(parselist(b"/", &mut count).is_none());
        assert_eq!(count, -1);
        assert!(parselist(b"1x", &mut count).is_none());
        assert_eq!(count, -3);

        let mut output = [0; 2];
        let mut out_count = 0;
        let mut limit = 2;
        assert_eq!(
            parselistfw(b"1-4  ", &mut output, &mut out_count, &mut limit),
            -1
        );
        assert_eq!(out_count, 2);
        assert_eq!(output, [1, 2]);
    }
}
