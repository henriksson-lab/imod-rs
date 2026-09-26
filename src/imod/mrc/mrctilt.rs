//! Translation of `IMOD/mrc/mrctilt.c`.
//!
//! Retranslated statement by statement (2026-09-26): the earlier body parsed
//! the tilts with `str::parse` and invented two "Invalid ..." errors, parsed
//! before opening the files, formatted the `%g` report with `{}`, computed
//! `first * 100.0` in `float` where the source multiplies by a double
//! literal, and added a helper and an error for `mrc_head_write`.
//!
//! Note that `mrctilt` is not in `IMOD/mrc/Makefile`'s `PROGS`, so upstream
//! never builds or installs it.

use std::io::Write;

use crate::imod::clip::clip::{ScanArg, sscanf};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format_bytes};
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read, mrc_head_write};

/// C `main` in `mrctilt.c:34`, returning its process status for the launcher.
pub fn mrctilt(argv: &[String]) -> i32 {
    let argc = argv.len();
    let argv0 = argv.first().map_or("mrctilt", String::as_str).as_bytes();
    // `float first, inc;` are uninitialised in the source and keep that value
    // when `sscanf` assigns nothing; they start at zero here.
    let mut first = 0.0_f32;
    let mut inc = 0.0_f32;

    if argc != 4 {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s version 1.0 Copyright (C)1994 Boulder Laboratory for\n",
            &[CArg::Bytes(argv0)],
        ));
        let _ =
            ImodFile::Stderr.write_all(b"3-Dimensional Fine Structure, University of Colorado.\n");
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "%s: Modify a mrc header to contain tilt information.\n",
            &[CArg::Bytes(argv0)],
        ));
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Usage: %s [image file] [first tilt] [tilt increment]\n",
            &[CArg::Bytes(argv0)],
        ));
        return 3;
    }

    let Some(mut fin) = ImodFile::open(&argv[1], "rb") else {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Error opening %s.\n",
            &[CArg::Bytes(argv[1].as_bytes())],
        ));
        return 3;
    };

    // `mrctilt.c:61` tests `fin` again rather than `fout`, so a failed
    // `rb+` open is not reported and the source goes on to `mrc_head_write`
    // through a NULL stream.  That would crash; exit 3 with the message the
    // source prints (naming `argv[2]`, as it does) instead.
    let Some(mut fout) = ImodFile::open(&argv[1], "rb+") else {
        let _ = ImodFile::Stderr.write_all(&c_format_bytes(
            "Error opening %s.\n",
            &[CArg::Bytes(argv[2].as_bytes())],
        ));
        return 3;
    };

    sscanf(&argv[2], "%f", &mut [ScanArg::Flt(&mut first)]);
    sscanf(&argv[3], "%f", &mut [ScanArg::Flt(&mut inc)]);

    let mut hdata = MrcHeader::default();
    if mrc_head_read(&mut fin, &mut hdata) != 0 {
        let _ = ImodFile::Stderr.write_all(b"Can't Read Input File Header.\n");
        return 3;
    }

    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
        "First tilt = %g, Inc = %g\n",
        &[CArg::Dbl(first as f64), CArg::Dbl(inc as f64)],
    ));

    hdata.idtype = 1;
    // `first * 100.0` is a double product converted to `short`.  Out of
    // `short` range that conversion is undefined; gcc on x86-64 truncates to
    // `int` (`cvttsd2si`) and keeps the low 16 bits, which is what a tilt of
    // 400 degrees gives natively (-25536), so the same two steps are written.
    hdata.vd1 = (first as f64 * 100.0) as i32 as i16;
    hdata.vd2 = (inc as f64 * 100.0) as i32 as i16;

    mrc_head_write(&mut fout, &mut hdata);
    drop(fin);
    drop(fout);
    0
}

#[cfg(test)]
mod tests {
    use super::mrctilt;
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FILE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn updates_a_real_mrc_header_in_place() {
        let sequence = FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "imod-rs-mrctilt-{}-{sequence}.mrc",
            std::process::id()
        ));
        let mut output = ImodFile::open(&path, "wb").expect("create MRC");
        let mut header = MrcHeader::default();
        mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE);
        assert_eq!(mrc_head_write(&mut output, &mut header), 0);
        drop(output);
        assert_eq!(
            mrctilt(&[
                "mrctilt".into(),
                path.display().to_string(),
                "-2.5".into(),
                "0.75".into()
            ]),
            0
        );
        let mut input = ImodFile::open(&path, "rb").expect("reopen MRC");
        let mut updated = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut input, &mut updated), 0);
        assert_eq!((updated.idtype, updated.vd1, updated.vd2), (1, -250, 75));
        let _ = std::fs::remove_file(path);
    }
}
