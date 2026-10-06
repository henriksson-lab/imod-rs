//! Native-golden coverage for `edmont` (`IMOD/flib/model/edmont.f90`).
//!
//! Every row of `fixtures/edmont/cases.tsv` was run through the native reference
//! program by `fixtures/make-edmont-goldens.sh` (inputs from
//! `fixtures/make-edmont-inputs.sh`), in a fresh directory holding copies of
//! the fixture inputs, with standard output captured through a pipe.  Exit
//! status, standard output and every output file must match byte for byte,
//! apart from MRC label time stamps (`common::small_prog`).  The two upstream
//! bugs fixed in translation that a case reaches (BUGS.md, `edmont`) are
//! tested for their defined behaviour below instead.

mod common;

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "edmont",
        fixtures: "edmont",
        min_cases: 30,
        reconcile: false,
        env: &[],
        stdout_mask: common::mask_stamps,
    });
}

/// Runs `imod edmont args` in a fresh directory holding the fixture inputs
/// and returns the exit status, standard output and the named output file.
fn run(tag: &str, args: &[&str], output: &str) -> (i32, String, Vec<u8>) {
    let dir = std::env::temp_dir().join(format!("imod-rs-edmont-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/edmont");
    for name in ["m.st", "ms.st", "m.pl"] {
        std::fs::copy(fixtures.join(name), dir.join(name)).unwrap();
    }
    let result = common::imod_cmd("edmont")
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(args)
        .output()
        .unwrap();
    let bytes = std::fs::read(dir.join(output)).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    (
        result.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&result.stdout).into_owned(),
        bytes,
    )
}

fn header_ints(bytes: &[u8], offset: usize) -> [i32; 3] {
    std::array::from_fn(|i| {
        i32::from_le_bytes(
            bytes[offset + 4 * i..offset + 4 * i + 4]
                .try_into()
                .unwrap(),
        )
    })
}

fn header_floats(bytes: &[u8], offset: usize) -> [f32; 3] {
    std::array::from_fn(|i| {
        f32::from_le_bytes(
            bytes[offset + 4 * i..offset + 4 * i + 4]
                .try_into()
                .unwrap(),
        )
    })
}

/// Defined behaviour (BUGS.md, `edmont`, fixed in translation): for an input
/// whose sampling (mxyz 32,24,12) differs from its size (16,12,12), native
/// scales the output cell by an `mxyz2` it never set and writes a zero cell.
/// The output keeps the input's sampling, so the cell keeps the pixel size
/// (2), times the binning in X and Y.
#[test]
fn sampling_unlike_size_keeps_the_pixel_size() {
    let (rc, _, out) = run(
        "samp",
        &["ms.st", "o.st", "-plin", "m.pl", "-plout", "o.pl"],
        "o.st",
    );
    assert_eq!(rc, 0);
    assert_eq!(header_ints(&out, 0), [16, 12, 12]);
    assert_eq!(header_ints(&out, 28), [32, 24, 12]);
    assert_eq!(header_floats(&out, 40), [64., 48., 24.]);
    let (rc, _, out) = run(
        "sampbin",
        &[
            "ms.st", "o.st", "-plin", "m.pl", "-plout", "o.pl", "-bin", "2", "-secs", "1",
        ],
        "o.st",
    );
    assert_eq!(rc, 0);
    assert_eq!(header_ints(&out, 0), [8, 6, 6]);
    assert_eq!(header_ints(&out, 28), [32, 24, 12]);
    assert_eq!(header_floats(&out, 40), [128., 96., 24.]);
}

/// Defined behaviour (BUGS.md, `edmont`, fixed in translation): the
/// duplicate-piece error names the output file that would hold the two
/// pieces; native prints the input file number there (2 for this run).
#[test]
fn duplicate_pieces_name_the_output_file() {
    let (rc, stdout, _) = run(
        "dup",
        &[
            "-imin", "m.st", "-imin", "m.st", "-plin", "m.pl", "-plin", "m.pl", "-imout", "o.st",
            "-plout", "o.pl",
        ],
        "o.st",
    );
    assert_eq!(rc, 1);
    assert!(
        stdout.contains(
            "ERROR: EDMONT - YOU MUST RENUMBER Z; OUTPUT FILE #    1 WOULD CONTAIN TWO SECTIONS \
             WITH X,Y,Z OF       0       0       0"
        ),
        "{stdout}"
    );
}
