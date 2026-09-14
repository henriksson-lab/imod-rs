//! Translation of `IMOD/flib/subrs/hvem/rdlist.f90`.
#![allow(dead_code)]

use crate::imod::libcfshr::parselist::parselistfw;
use std::io::BufRead;

/// Original `rdlist` (`rdlist.f90:16`).
pub fn rdlist<R: BufRead>(
    iunit: &mut R,
    list: &mut [i32],
    num_in_list: &mut i32,
) -> Result<(), i32> {
    rdlist2(iunit, list, num_in_list, &mut 0)
}

/// Original `rdlist2` (`rdlist.f90:28`).
pub fn rdlist2<R: BufRead>(
    iunit: &mut R,
    list: &mut [i32],
    num_in_list: &mut i32,
    lim_list: &mut i32,
) -> Result<(), i32> {
    let mut line = String::new();
    iunit.read_line(&mut line).map_err(|_| 4)?;
    parselist2(
        line.trim_end_matches(['\r', '\n']),
        list,
        num_in_list,
        lim_list,
    )
}

/// Original `readBigList` (`rdlist.f90:42`).
pub fn read_big_list<R: BufRead>(
    in_unit: &mut R,
    in_list: &mut [i32],
    n_list: &mut i32,
    lim_list: &mut i32,
    big_string: &mut String,
) -> Result<(), i32> {
    big_string.clear();
    in_unit.read_line(big_string).map_err(|_| 4)?;
    parselist2(
        big_string.trim_end_matches(['\r', '\n']),
        in_list,
        n_list,
        lim_list,
    )
}

/// Original `parselist` (`rdlist.f90:53`).
pub fn parselist(line: &str, list: &mut [i32], num_in_list: &mut i32) -> Result<(), i32> {
    parselist2(line, list, num_in_list, &mut 0)
}

/// Original `parselist2` (`rdlist.f90:63`).
pub fn parselist2(
    line: &str,
    list: &mut [i32],
    num_in_list: &mut i32,
    lim_list: &mut i32,
) -> Result<(), i32> {
    let mut source_limit = lim_list.unsigned_abs() as i32;
    let ierr = parselistfw(line.as_bytes(), list, num_in_list, &mut source_limit);
    if ierr == 0 {
        if *lim_list < 0 {
            *lim_list = 0;
        }
        return Ok(());
    }
    // `rdlist.f90:71-79` prints the classified message before it decides
    // what to do with it, so the message appears whichever form the caller
    // used, and a positive LIMLIST then exits.
    if ierr < 0 {
        println!("\nERROR: PARSELIST - TOO MANY LIST VALUES FOR ARRAY");
    } else if ierr == 1 {
        println!("\nERROR: PARSELIST - FAILED TO ALLOCATE MEMORY FOR LIST");
    } else {
        println!("\nERROR: PARSELIST - BAD CHARACTER IN ENTRY");
    }
    if *lim_list > 0 {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        std::process::exit(1);
    }
    *lim_list = ierr + 2;
    Err(ierr)
}

#[cfg(test)]
mod tests {
    use super::{parselist2, read_big_list};
    use std::io::Cursor;

    #[test]
    fn list_ranges_negative_numbers_slash_and_limits_match_source() {
        let mut values = [0; 12];
        let mut count = 0;
        let mut limit = 12;
        parselist2("-3--1, 2-4, 8-6", &mut values, &mut count, &mut limit).unwrap();
        assert_eq!(&values[..count as usize], &[-3, -2, -1, 2, 3, 4, 8, 7, 6]);
        parselist2("/", &mut values, &mut count, &mut limit).unwrap();
        assert_eq!(count, 9);
        let mut short = [0; 2];
        let mut negative_limit = -2;
        assert_eq!(
            parselist2("1-3", &mut short, &mut count, &mut negative_limit),
            Err(-1)
        );
        assert_eq!(negative_limit, 1);
    }

    #[test]
    fn big_list_reads_one_source_record() {
        let mut input = Cursor::new(b"3-1\nignored\n");
        let (mut values, mut count, mut limit, mut scratch) = ([0; 3], 0, 3, String::new());
        read_big_list(
            &mut input,
            &mut values,
            &mut count,
            &mut limit,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(&values, &[3, 2, 1]);
    }
}
