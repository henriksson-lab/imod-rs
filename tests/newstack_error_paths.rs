//! `newstack` error paths: what the program leaves behind, and which routine
//! reports a failed write.
//!
//! `newstack.f90` has 119 `call exitError` sites and **no** close in front of
//! any of them.  Its only closes are on the normal path -- unit 1 at
//! `newstack.f90:464, 1427, 2761`, unit 2 at `:2754` and `:2766`, unit 3 at
//! `:2769`.  Closing unit 2 flushes the MRC header, so a translation that
//! closes on the way to `exitError` leaves a 1024-byte output file where the
//! reference leaves a 0-byte one, and a `-replace` target rewritten rather than
//! untouched.
//!
//! The ordinary section write is `iiuSetPosition(2, isecOut - 1,
//! lineOutSt(iChunk))` + `iiuWriteLines(2, array(iChunkBase),
//! numLinesOut(iChunk))` (`newstack.f90:3256-3257`).  `iiuWriteLines`
//! (`unit_fileio.c:724-727`) prints `ERROR: iiuWriteLines - writing lines to
//! unit %d.` and exits, so a failed write is reported by that routine, after
//! the backend's own diagnostic -- not by a `NEWSTACK - Writing image file`
//! message in front of it.
//!
//! Every expectation below is the byte output of the reference `newstack`
//! (built from the vendored `IMOD/` tree) run on
//! `fixtures/newstack-mixed-byte.mrc` in its own directory with stdout taken
//! through a pipe.

use std::process::Command;

const AUTODOC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc");

/// Runs the translated `newstack` in `dir` and returns (status code, stdout,
/// stderr).
fn run(dir: &std::path::Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_newstack"))
        .current_dir(dir)
        .env("AUTODOC_DIR", AUTODOC)
        .env("IMOD_NO_IMAGE_BACKUP", "1")
        .args(args)
        .output()
        .expect("newstack executable must start");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn scratch(name: &str) -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("imod-rs-nserr-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(
        root.join("fixtures/newstack-mixed-byte.mrc"),
        dir.join("in.mrc"),
    )
    .unwrap();
    dir
}

/// A mode the MRC writer does not know fails inside `iiWriteSectionFloat`, so
/// the reference prints `mrcWriteSectionAny`'s own diagnostic and then
/// `iiuWriteLines`' -- in that order -- and exits from `iiuWriteLines`
/// (`unit_fileio.c:725-726`) rather than from `newstack`.  The created output
/// file is never closed, so it stays 0 bytes.
#[test]
fn an_unwritable_mode_is_reported_by_iiu_write_lines_and_leaves_an_empty_file() {
    let dir = scratch("mode9");
    let (status, stdout, stderr) = run(&dir, &["-mode", "9", "in.mrc", "o.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "", "the source writes every diagnostic to stdout");
    assert!(
        stdout.ends_with(
            "\n NEW image file on unit   2 : o.mrc\n\
             ERROR: mrcWriteSectionAny - unknown mode.\n\
             \n\
             ERROR: iiuWriteLines - writing lines to unit 2.\n"
        ),
        "stdout tail must be the reference's, not a NEWSTACK message in front \
         of the backend's; got:\n{stdout}"
    );
    let out = dir.join("o.mrc");
    assert!(out.is_file(), "the output file is created before the write");
    assert_eq!(
        std::fs::metadata(&out).unwrap().len(),
        0,
        "newstack.f90 never closes unit 2 on an error path, so no header is \
         flushed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:1962` fires after the output file has been created on unit 2.
/// The reference leaves the file empty; a translation that closes unit 2 in
/// front of `exitError` leaves 1024 bytes of header.
#[test]
fn an_error_after_the_output_is_created_leaves_a_zero_byte_file() {
    let dir = scratch("mode4");
    let (status, stdout, stderr) = run(&dir, &["-mode", "4", "in.mrc", "o.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with(
            "\n NEW image file on unit   2 : o.mrc\n\
             \n\
             ERROR: NEWSTACK - All input files must be complex if any are\n"
        ),
        "stdout tail must be the reference's; got:\n{stdout}"
    );
    let out = dir.join("o.mrc");
    assert!(out.is_file());
    assert_eq!(std::fs::metadata(&out).unwrap().len(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `-replace` opens the output `OLD` on unit 2 during option processing
/// (`newstack.f90:1173`).  The mode check at `newstack.f90:1757` then exits
/// without closing it, so the existing file is left exactly as it was.
#[test]
fn a_replace_mismatch_leaves_the_existing_output_untouched() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = scratch("replace");
    std::fs::copy(
        root.join("fixtures/newstack-replace-target.mrc"),
        dir.join("rt.mrc"),
    )
    .unwrap();
    let before = std::fs::read(dir.join("rt.mrc")).unwrap();
    let (status, stdout, stderr) = run(&dir, &["-mode", "1", "-replace", "1", "in.mrc", "rt.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with("\nERROR: NEWSTACK - Output mode does not match existing output file\n"),
        "got:\n{stdout}"
    );
    let after = std::fs::read(dir.join("rt.mrc")).unwrap();
    assert_eq!(
        after, before,
        "the existing output must be byte-identical: the source never closes \
         unit 2 on this path, so nothing is written back"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A failure before the output file is opened creates nothing at all.
#[test]
fn a_failure_before_the_output_opens_creates_no_file() {
    let dir = scratch("missing");
    let (status, stdout, stderr) = run(&dir, &["nope.mrc", "o.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with(
            "ERROR: iiOpen - Opening file nope.mrc (No such file or directory)\n\
             \n\
             ERROR: iiuOpen - Could not open 'nope.mrc'\n"
        ),
        "got:\n{stdout}"
    );
    assert!(
        !dir.join("o.mrc").exists(),
        "the output file must not exist at all"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A read that runs past the end of the file prints `mrcsec.c:367`'s
/// diagnostic once per attempt, and only then `newstack`'s own `exitError`.
///
/// The ordering is the point.  `b3dError` writes through libc stdout, which is
/// block buffered under a pipe, while `exitError` (`parse_input_params.f90:236`)
/// writes to Fortran unit 6; gfortran and libc interleave in program order, so
/// the reference prints the two backend lines first.  Translating `exitError`
/// with `println!` puts its line in front of both, because Rust's stream
/// flushes per line while the libc buffer waits for exit -- verified against
/// the reference on this exact input.
#[test]
fn a_short_read_reports_the_backend_diagnostic_before_the_newstack_one() {
    let dir = scratch("shortread");
    let whole = std::fs::read(dir.join("in.mrc")).unwrap();
    std::fs::write(dir.join("tr.mrc"), &whole[..1200]).unwrap();
    let (status, stdout, stderr) = run(&dir, &["tr.mrc", "o.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "", "the source writes every diagnostic to stdout");
    assert!(
        stdout.ends_with(
            "\n NEW image file on unit   2 : o.mrc\n\
             ERROR: mrcReadSectionAny - reading data from file.\n\
             \n\
             ERROR: mrcReadSectionAny - reading data from file.\n\
             \n\
             \n\
             ERROR: NEWSTACK - Reading image file\n"
        ),
        "got:\n{stdout}"
    );
    assert_eq!(
        std::fs::metadata(dir.join("o.mrc")).unwrap().len(),
        0,
        "unit 2 is never closed on the error path"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
