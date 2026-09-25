//! Translation of `IMOD/flib/subrs/hvem/frefor.f90`.
//!
//! The unit also carries [`list_read`], the stand-in for the gfortran
//! runtime's list-directed `READ` (`libgfortran/io/list_read.c`), which
//! `frefor3`'s internal read and the other `flib/subrs` leaves that read with
//! `*` format (`get_tilt_angles.f90`, `read_piece_list.f`) need.  It is the
//! runtime's behaviour, not an IMOD routine, in the same way `dopen.rs` stands
//! in for Fortran `OPEN`.

use std::io::BufRead;

/// The two ways a gfortran list-directed `READ` can fail: the `END=` and the
/// `ERR=` branches of the statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ListReadError {
    End,
    Error,
}

/// One item of a list-directed input list, typed as the Fortran variable it
/// is read into.
pub(crate) enum ListItem<'a> {
    Integer(&'a mut i32),
    Real(&'a mut f32),
}

/// gfortran list-directed `READ(unit, *) items` (`list_read.c`).
///
/// A `READ` statement begins at a new record and discards the rest of the
/// record it ends in.  Values are separated by blanks, tabs, a comma with
/// optional blanks around it, or the end of a record, so a list may continue
/// over as many records as it needs, and blank records are skipped.  Two
/// commas in a row, or a leading comma, give a null value, which leaves the
/// item unchanged; `r*c` repeats `c` `r` times and `r*` gives `r` nulls; a `/`
/// ends the statement and leaves every remaining item unchanged.  Running out
/// of records before the list is satisfied is the `END=` condition, even when
/// some items were already stored; a value that does not decode as the item's
/// type is `ERR=`.  Integers are an optional sign and digits; reals also take a
/// decimal point, an exponent introduced by `E`, `D` or `Q` or by a bare sign,
/// and `Inf`/`Infinity`/`NaN`, converted with correct rounding as `strtof`
/// does.
pub(crate) fn list_read<R: BufRead>(
    unit: &mut R,
    items: &mut [ListItem],
) -> Result<(), ListReadError> {
    let mut record: Vec<u8> = Vec::new();
    let mut pos: usize = 0;
    let mut have_record = false;
    let mut item = 0usize;
    // A pending repeat `r*c`: how many more items take the same value.
    let mut repeat_left: u64 = 0;
    let mut repeat_value: Option<Vec<u8>> = None;
    // Whether a comma has already served as the separator before the next
    // value (so a second comma means a null value).
    let mut comma_seen = true;
    let is_blank = |c: u8| c == b' ' || c == b'\t' || c == b'\r';
    while item < items.len() {
        if repeat_left > 0 {
            if let Some(value) = &repeat_value {
                store_list_value(&mut items[item], value)?;
            }
            repeat_left -= 1;
            item += 1;
            continue;
        }
        // Find the next value, crossing records as needed.
        loop {
            if !have_record || pos >= record.len() {
                record.clear();
                pos = 0;
                match unit.read_until(b'\n', &mut record) {
                    Ok(0) => return Err(ListReadError::End),
                    Ok(_) => {}
                    Err(_) => return Err(ListReadError::Error),
                }
                if record.last() == Some(&b'\n') {
                    record.pop();
                }
                have_record = true;
                continue;
            }
            let c = record[pos];
            if is_blank(c) {
                pos += 1;
                continue;
            }
            break;
        }
        let c = record[pos];
        if c == b'/' {
            return Ok(());
        }
        if c == b',' {
            pos += 1;
            if comma_seen {
                // Null value: the item keeps what it held.
                item += 1;
            }
            comma_seen = true;
            continue;
        }
        let start = pos;
        while pos < record.len()
            && !is_blank(record[pos])
            && record[pos] != b','
            && record[pos] != b'/'
        {
            pos += 1;
        }
        let token = record[start..pos].to_vec();
        comma_seen = false;
        if let Some(star) = token.iter().position(|&b| b == b'*') {
            let count = &token[..star];
            if count.is_empty() || !count.iter().all(u8::is_ascii_digit) {
                return Err(ListReadError::Error);
            }
            let count: u64 = std::str::from_utf8(count)
                .ok()
                .and_then(|text| text.parse().ok())
                .ok_or(ListReadError::Error)?;
            if count == 0 {
                return Err(ListReadError::Error);
            }
            let value = token[star + 1..].to_vec();
            repeat_value = if value.is_empty() { None } else { Some(value) };
            repeat_left = count;
            continue;
        }
        store_list_value(&mut items[item], &token)?;
        item += 1;
    }
    Ok(())
}

/// Decodes one list-directed value into its item (`list_read.c`'s
/// `read_integer` and `read_real`).
fn store_list_value(item: &mut ListItem, token: &[u8]) -> Result<(), ListReadError> {
    let (sign, body) = match token.first() {
        Some(b'+') => ("", &token[1..]),
        Some(b'-') => ("-", &token[1..]),
        _ => ("", token),
    };
    match item {
        ListItem::Integer(value) => {
            if body.is_empty() || !body.iter().all(u8::is_ascii_digit) {
                return Err(ListReadError::Error);
            }
            let text = format!("{sign}{}", std::str::from_utf8(body).unwrap());
            **value = text.parse::<i32>().map_err(|_| ListReadError::Error)?;
            Ok(())
        }
        ListItem::Real(value) => {
            let lower = body.to_ascii_lowercase();
            if lower == b"inf" || lower == b"infinity" {
                **value = if sign == "-" {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
                return Ok(());
            }
            if lower == b"nan" || (lower.starts_with(b"nan(") && lower.last() == Some(&b')')) {
                **value = f32::NAN;
                return Ok(());
            }
            let mut i = 0;
            let mut digits = 0;
            while i < body.len() && body[i].is_ascii_digit() {
                i += 1;
                digits += 1;
            }
            if i < body.len() && body[i] == b'.' {
                i += 1;
                while i < body.len() && body[i].is_ascii_digit() {
                    i += 1;
                    digits += 1;
                }
            }
            if digits == 0 {
                return Err(ListReadError::Error);
            }
            let mantissa = &body[..i];
            let mut exponent: &[u8] = b"";
            if i < body.len() {
                let mut j = i;
                if matches!(body[j], b'e' | b'E' | b'd' | b'D' | b'q' | b'Q') {
                    j += 1;
                } else if body[j] != b'+' && body[j] != b'-' {
                    return Err(ListReadError::Error);
                }
                let exp_start = j;
                if j < body.len() && (body[j] == b'+' || body[j] == b'-') {
                    j += 1;
                }
                let exp_digits = j;
                while j < body.len() && body[j].is_ascii_digit() {
                    j += 1;
                }
                if j == exp_digits || j != body.len() {
                    return Err(ListReadError::Error);
                }
                exponent = &body[exp_start..];
            }
            let text = format!(
                "{sign}{}e{}",
                std::str::from_utf8(mantissa).unwrap(),
                if exponent.is_empty() {
                    "0"
                } else {
                    std::str::from_utf8(exponent).unwrap()
                }
            );
            **value = text.parse::<f32>().map_err(|_| ListReadError::Error)?;
            Ok(())
        }
    }
}

/// Original `frefor` (`frefor.f90:13`).
pub fn frefor(card: &str, xnum: &mut [f32], num_fields: &mut i32) {
    let mut numeric = [0i32; 1];
    frefor3(card, xnum, &mut numeric, 0, num_fields, 0);
}

/// Original `frefor2` (`frefor.f90:29`).
pub fn frefor2(
    card: &str,
    xnum: &mut [f32],
    numeric: &mut [i32],
    num_fields: &mut i32,
    lim_num: i32,
) {
    frefor3(card, xnum, numeric, 1, num_fields, lim_num);
}

/// Original `frefor3` (`frefor.f90:49`).
///
/// `card` is the Fortran `character*(*)`; its length is `len(card)`, and a
/// character past it (which the source can index one or two places beyond
/// the end of a field ending at `len_trim`) is taken as a blank, which ends
/// the scan exactly as any character there would.  Fields and `xnum` indices
/// are 1-based in the arithmetic, as in the source, and shifted by one only
/// at each array access.
pub fn frefor3(
    card: &str,
    xnum: &mut [f32],
    numeric: &mut [i32],
    if_check_numeric: i32,
    num_fields: &mut i32,
    lim_num: i32,
) {
    let card = card.as_bytes();
    let blank = b' ';
    // card(i:i), 1-based
    let at = |i: i32| -> u8 {
        if i >= 1 && (i as usize) <= card.len() {
            card[i as usize - 1]
        } else {
            b' '
        }
    };
    // index(card(istart:), c) - 1
    let index_from = |istart: i32, c: u8| -> i32 {
        let from = (istart - 1).max(0) as usize;
        if from >= card.len() {
            return -1;
        }
        match card[from..].iter().position(|&b| b == c) {
            Some(offset) => offset as i32,
            None => -1,
        }
    };
    //
    if at(1) == b'/' {
        return;
    }
    let tab = 9u8;
    let len_card = card
        .iter()
        .rposition(|&b| b != b' ')
        .map_or(0, |index| index as i32 + 1);
    *num_fields = 0;
    let mut istart: i32 = 1;
    //
    // Search for starting point - skip separators
    while at(istart) == blank || at(istart) == b',' || at(istart) == tab {
        istart += 1;
        if istart > len_card {
            return;
        }
    }
    //
    // Decode up to limNum fiields
    while istart <= len_card && (*num_fields < lim_num || lim_num <= 0) {
        *num_fields += 1;
        //
        // Find end of field
        let mut num_to_blank = index_from(istart, blank);
        let mut num_to_comma = index_from(istart, b',');
        let mut num_to_tab = index_from(istart, tab);
        let num_to_sep;
        if num_to_blank <= 0 && num_to_comma <= 0 && num_to_tab <= 0 {
            num_to_sep = len_card + 1 - istart;
        } else {
            if num_to_blank <= 0 {
                num_to_blank = len_card + 1;
            }
            if num_to_comma <= 0 {
                num_to_comma = len_card + 1;
            }
            if num_to_tab <= 0 {
                num_to_tab = len_card + 1;
            }
            num_to_sep = num_to_blank.min(num_to_comma).min(num_to_tab);
        }
        let iend = istart + num_to_sep - 1;
        let field = &card[(istart - 1) as usize..iend as usize];
        let nf = (*num_fields - 1) as usize;
        //
        // Get the number
        if if_check_numeric > 0 {
            numeric[nf] = 0;
            xnum[nf] = 0.;
            // read(card(istart:iend),*,err = 98) xnum(numFields)
            let mut unit = field;
            if list_read(&mut unit, &mut [ListItem::Real(&mut xnum[nf])]).is_ok() {
                numeric[nf] = 1;
            }
            istart = iend + 2;
        } else {
            let mut unit = field;
            if list_read(&mut unit, &mut [ListItem::Real(&mut xnum[nf])]).is_err() {
                // 99 write(*,31) numFields
                // 31 format('WARNING: Invalid input in field',i4, '; input ignored from there on')
                let width = format!("{:>4}", *num_fields);
                let width = if width.len() > 4 {
                    "****".to_string()
                } else {
                    width
                };
                println!("WARNING: Invalid input in field{width}; input ignored from there on");
                *num_fields -= 1;
                return;
            }
            istart = iend + 2;
        }
        //
        // Skip over recurring separators
        while at(istart) == blank || at(istart) == b',' || at(istart) == tab {
            istart += 1;
            if istart > len_card {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{frefor, frefor2};

    /// Expected bits captured from the reference `libhvem.so` `frefor`/`frefor2`.
    #[test]
    fn fields_match_reference_list_directed_decoding() {
        let mut xnum = [-999f32; 8];
        let mut nf = -5;
        frefor("1d3 2D-2 3q1 4+2 5-1", &mut xnum, &mut nf);
        assert_eq!(nf, 5);
        let bits: Vec<u32> = xnum[..5].iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            bits,
            [0x447A0000, 0x3CA3D70A, 0x41F00000, 0x43C80000, 0x3F000000]
        );

        let mut xnum = [-999f32; 8];
        let mut numeric = [-7i32; 8];
        frefor2("1 abc 3", &mut xnum, &mut numeric, &mut nf, 0);
        assert_eq!(nf, 3);
        assert_eq!(&numeric[..3], &[1, 0, 1]);
        assert_eq!(&xnum[..3], &[1., 0., 3.]);

        // A field `1/2` stops its own internal read at the slash.
        let mut xnum = [-999f32; 8];
        frefor("  ,, 1/2 3*2", &mut xnum, &mut nf);
        assert_eq!(nf, 2);
        assert_eq!(&xnum[..3], &[1., 2., -999.]);

        let mut xnum = [-999f32; 8];
        frefor2("4.25e2 -7 9 10", &mut xnum, &mut numeric, &mut nf, 3);
        assert_eq!(nf, 3);
    }
}
