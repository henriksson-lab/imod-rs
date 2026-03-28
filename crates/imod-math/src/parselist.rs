/// Parse a comma/dash-separated list of integer ranges into a vector of integers.
///
/// Accepts strings like `"1-5,7,9,11,15-20"` and returns the expanded list of integers.
/// Ranges (e.g. `1-5`) are expanded to all values in the range, inclusive.
/// Backward ranges (e.g. `5-1`) produce values in descending order.
/// Negative numbers are supported when the minus sign immediately precedes the digit
/// (e.g. `"-3--1"` gives `[-3, -2, -1]`).
///
/// # Errors
///
/// Returns `Err` with a descriptive message when:
/// - The string starts with `'/'` (convention for "use default").
/// - An invalid character appears in a non-trailing position.
/// - A dangling comma, dash, or negative sign is left at the end.
///
/// An empty string returns `Ok(vec![])`.
pub fn parse_list(line: &str) -> Result<Vec<i32>, String> {
    if line.is_empty() {
        return Ok(Vec::new());
    }

    let bytes = line.as_bytes();
    if bytes[0] == b'/' {
        return Err("list starts with '/'".into());
    }

    let nchars = bytes.len();
    let mut result: Vec<i32> = Vec::new();
    let mut dash_last = false;
    let mut neg_num = false;
    let mut got_comma = false;
    let mut got_num = false;
    let mut got_space = false;
    let mut last_num: i32 = 0;
    let mut ind: usize = 0;

    while ind < nchars {
        let next = bytes[ind];

        if next.is_ascii_digit() {
            // Found a digit -- scan to end of digit run
            got_num = true;
            let num_start = if neg_num { ind - 1 } else { ind };
            while ind < nchars && bytes[ind].is_ascii_digit() {
                ind += 1;
            }
            // Parse the number from the slice (including possible leading '-')
            let s = &line[num_start..ind];
            let number: i32 = s
                .parse()
                .map_err(|e| format!("failed to parse '{}': {}", s, e))?;

            // Build the range to push
            let (loop_start, dir): (i32, i32) = if dash_last {
                let dir = if last_num > number { -1 } else { 1 };
                (last_num + dir, dir)
            } else {
                (number, 1)
            };

            let mut i = loop_start;
            while dir * i <= dir * number {
                result.push(i);
                i += dir;
            }

            last_num = number;
            neg_num = false;
            dash_last = false;
            got_comma = false;
            got_space = false;
            continue;
        }

        if next == b' ' || next == b'\t' {
            got_space = true;
        }

        // Non-digit, non-separator, non-dash, non-comma => stop or error
        if next != b',' && next != b' ' && next != b'-' && next != b'\t' {
            if got_space {
                break;
            }
            return Err(format!(
                "invalid character '{}' at position {}",
                next as char, ind
            ));
        }

        if next == b',' {
            got_comma = true;
            got_space = false;
        }

        if next == b'-' {
            got_space = false;
            if dash_last || !got_num || got_comma {
                neg_num = true;
            } else {
                dash_last = true;
            }
        }

        ind += 1;
    }

    // Dangling comma, negative sign, or dash is an error
    if got_comma || neg_num || dash_last {
        return Err("unexpected trailing comma, dash, or negative sign".into());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_range() {
        assert_eq!(parse_list("1-5").unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn individual_values() {
        assert_eq!(parse_list("3,7,11").unwrap(), vec![3, 7, 11]);
    }

    #[test]
    fn mixed() {
        assert_eq!(
            parse_list("1-5,7,9,11,15-20").unwrap(),
            vec![1, 2, 3, 4, 5, 7, 9, 11, 15, 16, 17, 18, 19, 20]
        );
    }

    #[test]
    fn backward_range() {
        assert_eq!(parse_list("5-1").unwrap(), vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn negative_numbers() {
        assert_eq!(parse_list("-3--1").unwrap(), vec![-3, -2, -1]);
    }

    #[test]
    fn negative_individual() {
        assert_eq!(parse_list("-3,-1,1").unwrap(), vec![-3, -1, 1]);
    }

    #[test]
    fn empty_string() {
        assert_eq!(parse_list("").unwrap(), Vec::<i32>::new());
    }

    #[test]
    fn slash_returns_error() {
        assert!(parse_list("/").is_err());
    }

    #[test]
    fn trailing_comma_error() {
        assert!(parse_list("1,2,").is_err());
    }

    #[test]
    fn trailing_dash_error() {
        assert!(parse_list("1-").is_err());
    }

    #[test]
    fn spaces_as_separators() {
        assert_eq!(parse_list("1 2 3").unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn stop_on_non_separator_after_space() {
        // After a space, a letter causes parsing to stop (not error)
        assert_eq!(parse_list("1-3 abc").unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn invalid_char_without_space_is_error() {
        assert!(parse_list("1-3abc").is_err());
    }

    #[test]
    fn single_value() {
        assert_eq!(parse_list("42").unwrap(), vec![42]);
    }

    #[test]
    fn negative_range_with_spaces() {
        assert_eq!(parse_list("-3 - -1").unwrap(), vec![-3, -2, -1]);
    }
}
