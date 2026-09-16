//! Translation of `IMOD/libcfshr/parselist.c`.
#![allow(dead_code)]

use super::parse_params::strtol;
pub const PARSELIST_SOURCE_FUNCTIONS: &[&str] = &["parselist", "parselistfw"];

/// Parsing failures from [`parselist`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseListError {
    LeadingSlash,
    InvalidCharacter,
}

/// Original `parselist` (`parselist.c:35`).
///
/// A leading slash is the source's special no-list marker; any other malformed
/// entry is an invalid character error.  The C terminator read is represented
/// by the byte lookup's zero fallback.
pub fn parselist(line: &str) -> Result<Vec<i32>, ParseListError> {
    let line = line.as_bytes();
    let mut intern = [0_u8; 10];
    let mut dashlast = false;
    let mut negnum = false;
    let mut gotcomma = false;
    let mut gotnum = false;
    let mut got_space = false;
    let nchars = line.len() as i32;
    let mut list: Vec<i32> = Vec::new();

    if line.is_empty() {
        return Ok(list);
    }
    if line[0] == b'/' {
        return Err(ParseListError::LeadingSlash);
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
            return Err(ParseListError::InvalidCharacter);
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
        return Err(ParseListError::InvalidCharacter);
    }
    Ok(list)
}

/// C/Fortran `parselistfw`, without the source's allocated temporary Fortran
/// string. `line` is already an owned Rust string, while `list` is the caller's
/// fixed output array. A leading slash deliberately leaves both output values
/// unchanged, as the source does.
pub fn parselistfw(line: &str, list: &mut [i32], nlist: &mut i32, limlist: &mut i32) -> i32 {
    match parselist(line) {
        Err(ParseListError::LeadingSlash) => 0,
        Err(ParseListError::InvalidCharacter) => 2,
        Ok(values) => {
            let total = values.len();
            let limit = if *limlist > 0 {
                (*limlist as usize).min(list.len())
            } else {
                list.len()
            };
            let copied = total.min(limit);
            *nlist = copied as i32;
            list[..copied].copy_from_slice(&values[..copied]);
            if *limlist > 0 && total > *limlist as usize {
                -1
            } else {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PARSELIST_SOURCE_FUNCTIONS, ParseListError, parselist, parselistfw};

    #[test]
    fn source_ranges_errors_and_fortran_copy_limit() {
        let values = parselist("1-5,7,9,11,15-13").unwrap();
        assert_eq!(values, &[1, 2, 3, 4, 5, 7, 9, 11, 15, 14, 13]);

        assert_eq!(parselist("/"), Err(ParseListError::LeadingSlash));
        assert_eq!(parselist("1x"), Err(ParseListError::InvalidCharacter));
        assert_eq!(parselist("-3--1, 2-4"), Ok(vec![-3, -2, -1, 2, 3, 4]));
    }
    #[test]
    fn fortran_wrapper_preserves_source_statuses_and_bounded_copy() {
        let mut values = [0; 2];
        let (mut count, mut limit) = (0, 2);
        assert_eq!(parselistfw("1-3", &mut values, &mut count, &mut limit), -1);
        assert_eq!(count, 2);
        assert_eq!(values, [1, 2]);
        let before = count;
        assert_eq!(parselistfw("/", &mut values, &mut count, &mut limit), 0);
        assert_eq!(count, before);
        assert_eq!(parselistfw("1x", &mut values, &mut count, &mut limit), 2);
        assert_eq!(PARSELIST_SOURCE_FUNCTIONS.len(), 2);
    }
}
