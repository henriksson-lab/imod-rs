//! Translation of `IMOD/flib/subrs/hvem/b3ddate.f`.

/// Original Fortran `b3ddate` (`b3ddate.f:1`).
///
/// The caller supplies the Fortran character storage.  The source `I2` field
/// is blank-padded, while the two-character year comes directly from the
/// `YYYYMMDD` value returned by `DATE_AND_TIME`.
pub fn b3d_date(dat: &mut [u8]) {
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as libc::time_t;
    // Foreign boundary, and the only one left in this module: `DATE_AND_TIME`
    // returns *local* civil time, and converting epoch seconds to it needs the
    // C library's timezone database (`/etc/localtime`, `$TZ`).  Rust's standard
    // library has no equivalent and this crate carries no date-time dependency,
    // so `localtime_r` stays a call into libc rather than being reimplemented.
    let local = unsafe {
        let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
        if libc::localtime_r(&raw const now, local.as_mut_ptr()).is_null() {
            dat.fill(b' ');
            return;
        }
        local.assume_init()
    };
    let month = months[local.tm_mon as usize];
    let text = format!(
        "{:>2}-{}-{:02}",
        local.tm_mday,
        month,
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
