//! Translation of `IMOD/flib/subrs/compat/datetime.f`.

use chrono::{Local, Timelike};

/// Original `time` (`datetime.f:1`).
///
/// `tim` is the caller's `character*(*)` storage.  The internal
/// `write(tim,101)` with `format(a2,':',a2,':',a2)` fills the first eight
/// characters and blank-pads the rest; a variable shorter than eight
/// characters is a gfortran "End of record" runtime error, status 2.
/// `date_and_time(TIME=packtime)` gives the local time as `hhmmss.sss`.
pub fn time(tim: &mut [u8]) {
    let now = Local::now();
    let packtime = format!("{:02}{:02}{:02}", now.hour(), now.minute(), now.second());
    let text = format!(
        "{}:{}:{}",
        &packtime[0..2],
        &packtime[2..4],
        &packtime[4..6]
    );
    if tim.len() < text.len() {
        eprintln!("Fortran runtime error: End of record");
        crate::imod::libcfshr::b3dutil::exit(2);
    }
    tim[..text.len()].copy_from_slice(text.as_bytes());
    tim[text.len()..].fill(b' ');
}
