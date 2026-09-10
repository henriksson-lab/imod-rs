//! Translation of `IMOD/flib/subrs/hvem/b3ddate.f`.

/// Original Fortran `b3ddate` (`b3ddate.f:1`).
///
/// The caller supplies the Fortran character storage.  The source `I2` field
/// is blank-padded, while the two-character year comes directly from the
/// `YYYYMMDD` value returned by `DATE_AND_TIME`.
pub fn b3d_date(dat: &mut [u8]) {
    let months = [
        b"Jan", b"Feb", b"Mar", b"Apr", b"May", b"Jun", b"Jul", b"Aug", b"Sep", b"Oct", b"Nov",
        b"Dec",
    ];
    let mut now = 0_i64;
    let mut local = unsafe { core::mem::zeroed::<libc::tm>() };
    unsafe {
        libc::time(&raw mut now);
        libc::localtime_r(&raw const now, &raw mut local);
    }
    let month = months[local.tm_mon as usize];
    let text = format!(
        "{:>2}-{}-{:02}",
        local.tm_mday,
        core::str::from_utf8(month).unwrap(),
        (local.tm_year + 1900) % 100
    );
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
