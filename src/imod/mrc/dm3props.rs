//! Translation of `IMOD/mrc/dm3props.c`.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::iilikemrc::analyze_dm3;
use crate::imod::libiimod::iimage::RawImageInfo;

/// C `main` in `dm3props.c`.
pub fn dm3props(arguments: &[String]) -> i32 {
    let mut dmformat = 0_i32;
    if let Some(argument) = arguments.get(1) {
        let bytes = argument.as_bytes();
        let mut index = 0;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        let negative = if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
            let negative = bytes[index] == b'-';
            index += 1;
            negative
        } else {
            false
        };
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            dmformat = dmformat
                .saturating_mul(10)
                .saturating_add((bytes[index] - b'0') as i32);
            index += 1;
        }
        if negative {
            dmformat = -dmformat;
        }
    }
    if dmformat != 3 && dmformat != 4 {
        println!("ERROR: dm3props - First argument must be 3 or 4 for DM format");
        return 1;
    }
    for filename in &arguments[2..] {
        let Some(mut file) = ImodFile::open(filename, "rb") else {
            println!("ERROR: dm3props - Opening {filename}");
            return 1;
        };
        let mut info = RawImageInfo::default();
        let mut dmtype = 0;
        if analyze_dm3(
            &mut file,
            filename.as_bytes(),
            dmformat,
            &mut info,
            &mut dmtype,
        ) != 0
        {
            return 1;
        }
        println!(
            "{} {} {} {} {} {:.6} {:.6}",
            info.nx, info.ny, info.nz, dmtype, info.header_size, info.pixel, info.z_pixel
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::dm3props;

    #[test]
    fn invalid_format_has_source_error_status() {
        assert_eq!(dm3props(&["dm3props".into(), "2".into()]), 1);
    }

    #[test]
    fn atoi_style_prefix_is_accepted_before_file_opening() {
        assert_eq!(
            dm3props(&["dm3props".into(), "3junk".into(), "missing.dm3".into()]),
            1
        );
    }
}
