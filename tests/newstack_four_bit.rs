//! `newstack`'s two special output-mode entries, `newstack.f90:718-731`.
//!
//! Mode 16 is refused outright, with two different messages depending on the
//! number of input files (`newstack.f90:719-724`).  Mode 101 is not a data
//! mode at all: it means "pack bytes two to a byte", so the source sets
//! `pack4bitOutput`, rewrites `newMode` to 0 and calls `set4BitOutputMode(1)`
//! (`newstack.f90:727-731`).  Leaving `newMode` at 101 indexes `optimalMax`,
//! which has 17 entries, out of bounds.
//!
//! `pack4bitOutput` is not confined to that entry: `-replace` takes it from
//! the flags of the *existing* output file (`newstack.f90:1176-1177`), and it
//! then forces a rescale (`newstack.f90:1989`) into an output range of 15
//! (`newstack.f90:1998`).  The control test below shows the same `-replace`
//! onto an ordinary mode 0 file copying the data through unscaled.
//!
//! Every expectation is the byte output of the reference `newstack` (built
//! from the vendored `IMOD/` tree) run on `fixtures/newstack-mixed-byte.mrc`
//! in its own directory with stdout taken through a pipe.  The 4-bit output
//! file was compared byte for byte with the reference's; the only difference
//! is the `dd-Mmm-yy  HH:MM:SS` stamp in the label, which is documented as not
//! achievable.

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
    let dir = std::env::temp_dir().join(format!("imod-rs-ns4bit-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(
        root.join("fixtures/newstack-mixed-byte.mrc"),
        dir.join("in.mrc"),
    )
    .unwrap();
    std::fs::copy(
        root.join("fixtures/newstack-mixed-short.mrc"),
        dir.join("in2.mrc"),
    )
    .unwrap();
    dir
}

/// `newMode = 0` and `set4BitOutputMode(1)` (`newstack.f90:729-730`).  The
/// reference writes a 1024-byte header with mode 101 in it and 11 bytes per
/// 21-pixel row, and scales the byte data into 0-15.
#[test]
fn mode_101_writes_a_packed_four_bit_file() {
    let dir = scratch("mode101");
    let (status, stdout, stderr) = run(&dir, &["-mode", "101", "in.mrc", "o.mrc"]);
    assert_eq!(status, 0, "native exits 0; stdout was:\n{stdout}");
    assert_eq!(stderr, "", "the source writes every diagnostic to stdout");
    assert!(
        stdout.ends_with(
            "\n NEW image file on unit   2 : o.mrc\n\
             \x20section   input min&max       output min&max  &  mean\n\
             \x20      0     43.00    214.00      2.53     12.59      7.48\n\
             \x20      1     45.00    223.00      2.65     13.12      7.74\n\
             \x20      2     53.00    227.00      3.12     13.35      8.00\n"
        ),
        "the output range must be the 4-bit 0-15 of newstack.f90:1998; got:\n{stdout}"
    );
    let bytes = std::fs::read(dir.join("o.mrc")).unwrap();
    assert_eq!(
        bytes.len(),
        1024 + 3 * 13 * 11,
        "21 pixels pack into 11 bytes a row"
    );
    assert_eq!(
        i32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        101,
        "the header records mode 101 even though newMode became 0"
    );
    assert!(
        bytes[1024..].iter().all(|byte| byte >> 4 <= 15),
        "every nibble must be in range"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:722-724`, the single-input message.
#[test]
fn mode_16_with_one_input_names_colornewst() {
    let dir = scratch("mode16one");
    let (status, stdout, stderr) = run(&dir, &["-mode", "16", "in.mrc", "o.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - Cannot output color data (mode 16); use \
             colornewst instead or set another output mode\n"
        ),
        "got:\n{stdout}"
    );
    assert!(
        !dir.join("o.mrc").exists(),
        "the check precedes the output open"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `newstack.f90:719-721`, the message for more than one input file.
#[test]
fn mode_16_with_several_inputs_names_clip_and_cnfiles() {
    let dir = scratch("mode16many");
    let (status, stdout, stderr) = run(&dir, &["-mode", "16", "in.mrc", "in2.mrc", "o.mrc"]);
    assert_eq!(status, 1, "native exits 1; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with(
            "\nERROR: NEWSTACK - Cannot output color data (mode 16); use clip \
             for junk stacking multiple color files, use colornewst with the \
             -cnfiles options, or set another output mode\n"
        ),
        "got:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `pack4bitOutput = btest(iiuFlags, 5) .or. btest(iiuFlags, 6)`
/// (`newstack.f90:1177`): replacing sections in an existing 4-bit file
/// rescales into 0-15 even though the input and output modes are both 0, which
/// is the `(pack4bitOutput .and. .not. packed4bitInput)` term of
/// `newstack.f90:1989`.  The control below is the same command onto an
/// ordinary mode 0 file, which the reference copies through unscaled.
#[test]
fn replacing_sections_in_a_four_bit_file_rescales_but_a_plain_file_does_not() {
    let dir = scratch("replace4");
    let (status, _, _) = run(&dir, &["-mode", "101", "in.mrc", "four.mrc"]);
    assert_eq!(status, 0);
    let (status, _, _) = run(&dir, &["in.mrc", "plain.mrc"]);
    assert_eq!(status, 0);

    let (status, stdout, stderr) = run(&dir, &["-replace", "0,1", "in.mrc", "four.mrc"]);
    assert_eq!(status, 0, "native exits 0; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with(
            "\n section   input min&max       output min&max  &  mean\n\
             \x20      0     43.00    214.00      2.53     12.59      7.48\n\
             \x20      1     45.00    223.00      2.65     13.12      7.74\n"
        ),
        "got:\n{stdout}"
    );

    let (status, stdout, stderr) = run(&dir, &["-replace", "0,1", "in.mrc", "plain.mrc"]);
    assert_eq!(status, 0, "native exits 0; stdout was:\n{stdout}");
    assert_eq!(stderr, "");
    assert!(
        stdout.ends_with(
            "\n section   input min&max       output min&max  &  mean\n\
             \x20      0     43.00    214.00     43.00    214.00    127.10\n\
             \x20      1     45.00    223.00     45.00    223.00    131.66\n"
        ),
        "an ordinary mode 0 target must not rescale; got:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
