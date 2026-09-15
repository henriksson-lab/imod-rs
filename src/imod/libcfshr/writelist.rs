//! Translation of `IMOD/libcfshr/writelist.c`.
#![allow(dead_code)]

use std::io::Write;

/// Original `writeList` (`writelist.c:23`).
pub fn write_list(list: &[i32], number_of_values: i32, line_length: i32) -> i32 {
    let Some(line_string) = list_to_string(list, number_of_values) else {
        return 1;
    };
    // The C advances `lineStr` through the buffer it keeps in `saveStr` and
    // overwrites the separating comma with a newline in place; `start` is that
    // advancing pointer, and the bytes stay in one owned buffer.
    let mut saved_string = line_string.into_bytes();
    let mut start = 0_usize;
    while saved_string.len() - start > line_length as usize {
        let mut index = line_length - 1;
        while index > 0 {
            if saved_string[start + index as usize] == b',' {
                break;
            }
            index -= 1;
        }
        saved_string[start + index as usize] = b'\n';
        start += index as usize + 1;
    }
    let mut out = std::io::stdout();
    let _ = out.write_all(&saved_string);
    let _ = out.write_all(b"\n");
    let _ = out.flush();
    0
}

/// Original Fortran wrapper `writelist` (`writelist.c:45`).
pub fn writelist(list: &[i32], number_of_values: &i32, line_length: &i32) -> i32 {
    write_list(list, *number_of_values, *line_length)
}

/// Original Fortran wrapper `wrlist` (`writelist.c:52`).
pub fn wrlist(list: &[i32], number_of_values: &i32) {
    write_list(list, *number_of_values, 80);
}

/// Original `listToString` (`writelist.c:62`).
///
/// The C hands back a `malloc`ed string the caller must free, and NULL for a
/// memory error; the `String` owns itself and the `None` arm keeps the caller's
/// error path.
pub fn list_to_string(list: &[i32], number_of_values: i32) -> Option<String> {
    let mut returned_string: Option<String> = None;
    let mut range_start = list[0];
    for index in 1..number_of_values {
        if list[index as usize] != list[index as usize - 1] + 1 {
            returned_string =
                add_range_to_line(range_start, list[index as usize - 1], returned_string);
            returned_string.as_ref()?;
            range_start = list[index as usize];
        }
    }
    add_range_to_line(
        range_start,
        list[number_of_values as usize - 1],
        returned_string,
    )
}

/// Original static `addRangeToLine` (`writelist.c:78`).
pub fn add_range_to_line(
    number_start: i32,
    number_end: i32,
    line: Option<String>,
) -> Option<String> {
    let mut range_string = number_start.to_string();
    if number_end > number_start {
        range_string.push_str("-");
        let end_string = number_end.to_string();
        range_string.push_str(&end_string);
    }

    if let Some(mut returned_line) = line {
        returned_line.push_str(",");
        returned_line.push_str(&range_string);
        Some(returned_line)
    } else {
        Some(range_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_only_increasing_adjacent_values_to_source_ranges() {
        let values = [1_i32, 2, 3, 8, 10, 11, 4];
        let text = list_to_string(&values, values.len() as i32);
        assert_eq!(text.as_deref(), Some("1-3,8,10-11,4"));
    }
}
