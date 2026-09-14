//! Translation of `IMOD/flib/subrs/hvem/b3ddate.f`.

use chrono::{Datelike, Local};

/// Original Fortran `b3ddate` (`b3ddate.f:1`).
///
/// The caller supplies the Fortran character storage.  The source `I2` field
/// is blank-padded, while the two-character year comes directly from the
/// `YYYYMMDD` value returned by `DATE_AND_TIME`.
pub fn b3d_date(dat: &mut [u8]) {
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let local = Local::now();
    let month = months[local.month0() as usize];
    let text = format!("{:>2}-{}-{:02}", local.day(), month, local.year() % 100);
    dat.fill(b' ');
    let count = dat.len().min(text.len());
    dat[..count].copy_from_slice(&text.as_bytes()[..count]);
}

#[cfg(test)]
mod tests {
    use super::b3d_date;

    #[test]
    fn b3d_date_writes_the_fortran_nine_character_shape() {
        let mut date = [0_u8; 9];
        b3d_date(&mut date);
        assert_eq!(date[2], b'-');
        assert_eq!(date[6], b'-');
        assert!(date[3..6].iter().all(u8::is_ascii_alphabetic));
    }
}
