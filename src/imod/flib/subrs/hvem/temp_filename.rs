//! Translation of `IMOD/flib/subrs/hvem/temp_filename.f`.
#![allow(dead_code)]

/// Original Fortran `temp_filename` (`temp_filename.f:8`).
///
/// Composes a filename from an optional temporary directory `tempdir`, a
/// filename in `filein` with leading directories stripped off, and an
/// extension in `tempext`.  The source's `character*(*)` arguments are
/// blank-padded, so `len_trim` and `trim` are trailing-blank trims here, and
/// the `character*320` result is truncated to 320 characters.
pub fn temp_filename(filein: &str, tempdir: &str, tempext: &str) -> String {
    //
    // find last / in filein
    //
    let bytes = filein.as_bytes();
    let in_end = bytes
        .iter()
        .rposition(|&byte| byte != b' ')
        .map_or(0, |index| index + 1);
    let mut in_str = 1_usize;
    for i in 1..=in_end {
        if bytes[i - 1] == b'/' {
            in_str = i + 1;
        }
    }
    //
    let name = &filein[in_str - 1..in_end];
    let mut result = if tempdir.trim_end_matches(' ').is_empty() {
        format!("{}.{}", name, tempext.trim_end_matches(' '))
    } else {
        format!(
            "{}/{}.{}",
            tempdir.trim_end_matches(' '),
            name,
            tempext.trim_end_matches(' ')
        )
    };
    result.truncate(320);
    result
}

#[cfg(test)]
mod tests {
    use super::temp_filename;

    #[test]
    fn temp_filename_strips_directories_and_appends_the_extension() {
        assert_eq!(
            temp_filename("a/b/out.mrc", " ", "nws123456"),
            "out.mrc.nws123456"
        );
        assert_eq!(temp_filename("out.mrc", " ", "nws      "), "out.mrc.nws");
        assert_eq!(
            temp_filename("a/out.mrc", "/tmp", "nws1"),
            "/tmp/out.mrc.nws1"
        );
    }
}
