//! Native-golden coverage for `sirtsetup` (`IMOD/pysrc/sirtsetup`, translated
//! in `src/imod/pysrc/sirtsetup.rs`), and `findsirtdiffs`
//! (`IMOD/pysrc/findsirtdiffs`), which the finish command file runs.
//!
//! Every row of `fixtures/sirtsetup/cases.tsv` was run through the native
//! Python script by `fixtures/make-sirtsetup-goldens.sh`, with the native
//! Python `splittilt` on PATH; here `splittilt` is ours
//! (`fixtures/sirtsetup/own_commands`).  See `tests/pysetup_common` for what
//! is compared.  The cases cover internal SIRT with and without splitting,
//! leave lists, scaling and trimming, a subarea, resuming with cleanup and
//! from a given iteration, vertical-slice resumption, external SIRT (local
//! alignments, LOG and SCALE), a varying X-tilt file, a GPU entry, and the
//! main error exits.  The tests below pin the upstream bugs fixed in
//! translation (`BUGS.md`, sirtsetup).

mod common;
mod pysetup_common;

use std::path::{Path, PathBuf};
use std::process::Output;

#[test]
fn sirtsetup_matches_native_goldens() {
    let failures = pysetup_common::run_cases("sirtsetup");
    common::remove_command_links();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/sirtsetup")
}

/// Runs our sirtsetup in a fresh directory holding `inputs` (`src:dst`).
fn run_ours(name: &str, inputs: &[(&str, &str)], args: &[&str]) -> (PathBuf, Output) {
    let work = std::env::temp_dir().join(format!("imod-rs-sirtdef-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for (source, target) in inputs {
        std::fs::copy(fixture().join("inputs").join(source), work.join(target)).unwrap();
    }
    common::imod_link("splittilt");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = common::imod_cmd("sirtsetup")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .env("AUTODOC_DIR", root.join("IMOD/autodoc"))
        .env(
            "PATH",
            format!(
                "{}:{}",
                common::command_link_directory().display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        )
        .args(args)
        .output()
        .unwrap();
    (work, output)
}

const BASE: [(&str, &str); 4] = [
    ("tilt.com", "tilt.com"),
    ("ts.ali", "ts.ali"),
    ("ts.tlt", "ts.tlt"),
    ("ts.xtilt", "ts.xtilt"),
];

/// Native: `leaveList.remove(ind)` removes the *value* `ind` from the sorted
/// leave list (ValueError when it is absent) and the loop stops one pair
/// short.  Defined: duplicates are dropped, so `3,5,5,8` gives exactly the
/// command files native writes for `3,5,8` (golden `leave_scale`).
#[test]
fn duplicate_leave_iterations_are_dropped() {
    let (work, output) = run_ours(
        "leave",
        &BASE,
        &[
            "-co",
            "tilt.com",
            "-nu",
            "4",
            "-le",
            "3,5,5,8,8",
            "-sc",
            "0,100",
        ],
    );
    assert_eq!(output.status.code(), Some(0));
    let golden = fixture().join("golden");
    let native_out = common::golden::expect(&golden.join("leave_scale.out"));
    assert!(native_out.matches(&output.stdout));
    for file in common::golden::read_to_string(&golden.join("leave_scale.files")).lines() {
        let ours = std::fs::read(work.join(file)).unwrap();
        if let Some(native) = common::golden::load(&golden.join("leave_scale").join(file)) {
            assert!(native.matches(&ours), "{file} differs");
        }
    }
    let _ = std::fs::remove_dir_all(&work);
}

/// Native: an empty X-tilt file reaches an error message naming the
/// undefined `xtiltfile`, a NameError.  Defined: the message itself.
#[test]
fn empty_xtilt_file_is_an_error_message() {
    let mut inputs = BASE.to_vec();
    inputs[3] = ("empty", "ts.xtilt");
    let (work, output) = run_ours("xtempty", &inputs, &["-co", "tilt.com"]);
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "ERROR: sirtsetup - The file of X-axis tilts, ts.xtilt, is empty\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}

/// Native: LOG with local alignments and no SCALE line takes `len(None)`, a
/// TypeError.  Defined: no SCALE line means no scale change, so the external
/// SIRT files are set up with no `SCALE` edit.
#[test]
fn log_without_scale_line_sets_up_external_sirt() {
    let mut inputs = BASE.to_vec();
    inputs[0] = ("noscale.com", "tilt.com");
    let (work, output) = run_ours(
        "noscale",
        &inputs,
        &["-co", "tilt.com", "-nu", "1", "-it", "2"],
    );
    assert_eq!(output.status.code(), Some(0));
    let reproject = std::fs::read_to_string(work.join("tilt_sirt-003-sync.com")).unwrap();
    assert!(
        reproject.contains("RecFileToReproj  ts.srec00\n"),
        "{reproject}"
    );
    for entry in std::fs::read_dir(&work).unwrap() {
        let text = std::fs::read_to_string(entry.unwrap().path()).unwrap_or_default();
        assert!(!text.contains("SCALE"));
    }
    assert!(work.join("tilt_sirt-finish.com").exists());
    let _ = std::fs::remove_dir_all(&work);
}

/// Native looks for the vertical-slice file of the iteration being resumed
/// from as `.vsrN`, while every one is written `.vsrNN`, so below iteration
/// 10 it never resumes from one.  Defined: `.vsrNN`, as for iteration 12 in
/// the `vert_resume` golden.
#[test]
fn resume_below_ten_uses_vertical_slice_file() {
    let mut inputs = BASE.to_vec();
    inputs[0] = ("xtilt.com", "tilt.com");
    inputs.push(("rec.mrc", "ts.srec05"));
    inputs.push(("rec.mrc", "ts.vsr05"));
    let (work, output) = run_ours("vsr", &inputs, &["-co", "tilt.com", "-nu", "1", "-it", "2"]);
    assert_eq!(output.status.code(), Some(0));
    let first = std::fs::read_to_string(work.join("tilt_sirt-001-sync.com")).unwrap();
    assert!(first.contains("RecFileToReproj  ts.vsr05\n"), "{first}");
    assert!(first.contains("VertForSIRTInput\n"));
    let _ = std::fs::remove_dir_all(&work);
}

/// `findsirtdiffs` against the Python script's own output for a set of
/// chunked SIRT logs (two chunks of iteration 3, one of iteration 4): the
/// per-iteration line, with the chunk statistics combined.
#[test]
fn findsirtdiffs_combines_chunk_statistics() {
    let work = std::env::temp_dir().join(format!("imod-rs-findsirtdiffs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();
    for (file, line) in [
        (
            "r_sirt-001.log",
            "Iter 3, slices    0    9, diff rec mean&sd:   0.125   1.500",
        ),
        (
            "r_sirt-002.log",
            "Iter 3, slices   10   19, diff rec mean&sd:  -0.250   2.000",
        ),
        (
            "r_sirt-003.log",
            "Iter 4, slices    0   19, diff rec mean&sd:   0.010   0.750",
        ),
    ] {
        std::fs::write(work.join(file), format!("stuff\n{line}\nmore\n")).unwrap();
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = common::imod_cmd("findsirtdiffs")
        .current_dir(&work)
        .env("IMOD_DIR", root.join("IMOD"))
        .arg("r_sirt")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    // python3 IMOD/pysrc/findsirtdiffs r_sirt on the same logs
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "Iter   3, slices     0    19, diff rec mean&sd:         -0.062          1.721\n\
         Iter   4, slices     0    19, diff rec mean&sd:          0.010          0.750\n"
    );
    let _ = std::fs::remove_dir_all(&work);
}
