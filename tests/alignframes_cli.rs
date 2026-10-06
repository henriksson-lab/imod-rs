//! Native-golden coverage for `alignframes` (`IMOD/mrc/alignframes.cpp`), the
//! movie-frame aligner (with `framealign.cpp`, `frameutil.cpp` and the no-CUDA
//! `nogpuframe.cpp` underneath).
//!
//! Every row of `fixtures/alignframes/cases.tsv` was run through the native
//! reference program by `fixtures/make-alignframes-goldens.sh` (inputs from
//! `fixtures/make-alignframes-inputs.sh`), in a fresh directory holding copies
//! of the fixture inputs, with `OMP_NUM_THREADS=1`.  Exit status, standard
//! output and every output file must match byte for byte, apart from MRC label
//! time stamps, the usage banner's compile date, and label slots native fills
//! from uninitialised memory.  Three rows reach upstream bugs fixed in the
//! translation (BUGS.md, `alignframes`); their goldens are native runs on an
//! equivalent input that does not trigger the bug (`af_mdoc_fei_axis`,
//! `af_mdoc_fs`) or native's output with the defined name (`af_combine`).
//! The tests below assert the defined behaviour where no native run can.

mod common;

use std::path::{Path, PathBuf};

#[test]
fn every_case_matches_native_golden() {
    common::small_prog::run(&common::small_prog::Suite {
        program: "alignframes",
        fixtures: "alignframes",
        min_cases: 90,
        reconcile: true,
        env: &[("OMP_NUM_THREADS", "1")],
        stdout_mask: common::small_prog::mask_banner,
    });
}

/// A scratch directory holding copies of the named fixture inputs.
fn work(name: &str, files: &[&str]) -> PathBuf {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/alignframes");
    let dir =
        std::env::temp_dir().join(format!("imod-rs-alignframes-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for file in files {
        std::fs::copy(fixtures.join(file), dir.join(file)).unwrap();
    }
    dir
}

/// Runs `imod alignframes` in `dir` and returns its exit status and stdout.
fn run(dir: &Path, args: &[&str]) -> (Option<i32>, String) {
    let output = common::imod_cmd("alignframes")
        .current_dir(dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .env("OMP_NUM_THREADS", "1")
        .args(args)
        .output()
        .unwrap();
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
    )
}

/// The image data of an MRC file with no extended header.
fn data(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap()[1024..].to_vec()
}

/// Defined behaviour (BUGS.md, `alignframes`): a block group sums its frames.
/// Native adds the running sum to itself from the second frame of a group on,
/// so three identical frames sum to 4x the frame.  Nine frames that are three
/// images each repeated three times, aligned in groups of three, must give
/// exactly what the three images times three give aligned one by one.
#[test]
fn block_groups_sum_their_frames() {
    let dir = work("blocks", &["mov9.mrc", "mov3x.mrc"]);
    let (rc, stdout) = run(&dir, &["-group", "3", "-pair", "2", "mov9.mrc", "o9.mrc"]);
    assert_eq!(rc, Some(0), "{stdout}");
    assert!(stdout.contains("Using block grouping"), "{stdout}");
    let (rc, stdout) = run(&dir, &["-pair", "2", "mov3x.mrc", "o3.mrc"]);
    assert_eq!(rc, Some(0), "{stdout}");
    assert!(data(&dir.join("o9.mrc")) == data(&dir.join("o3.mrc")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Defined behaviour (BUGS.md): byte frames are summed into a buffer of
/// shorts big enough for them (native allocates half of it).
#[test]
fn block_groups_sum_byte_frames() {
    let dir = work("blocks-byte", &["mov9b.mrc", "mov3s.mrc"]);
    let (rc, stdout) = run(&dir, &["-group", "3", "-pair", "2", "mov9b.mrc", "o9.mrc"]);
    assert_eq!(rc, Some(0), "{stdout}");
    let (rc, stdout) = run(&dir, &["-pair", "2", "mov3s.mrc", "o3.mrc"]);
    assert_eq!(rc, Some(0), "{stdout}");
    assert!(data(&dir.join("o9.mrc")) == data(&dir.join("o3.mrc")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Defined behaviour (BUGS.md): combined single-frame files are named after
/// the first file of each group, so each combined image gets its own
/// transform file.  Native writes all three into `one1.xf`, each replacing
/// the last.
#[test]
fn combined_files_get_their_own_transform_files() {
    let files = [
        "one1.mrc", "one2.mrc", "one3.mrc", "one4.mrc", "one5.mrc", "one6.mrc",
    ];
    let dir = work("combine-xf", &files);
    let mut args = vec!["-break", "2", "-xfext", "xf"];
    args.extend_from_slice(&files);
    args.push("out.mrc");
    let (rc, stdout) = run(&dir, &args);
    assert_eq!(rc, Some(0), "{stdout}");
    for (num, name) in [(1, "one1"), (2, "one3"), (3, "one5")] {
        assert!(
            stdout.contains(&format!("File {num} ({name}.mrc): 2 frames")),
            "{stdout}"
        );
        let xf = std::fs::read_to_string(dir.join(format!("{name}.xf"))).unwrap();
        assert_eq!(xf.lines().count(), 2, "{xf}");
        assert!(xf.lines().all(|l| l.starts_with(" 1.00000    0.00000")));
    }
    assert!(!dir.join("one1.xf~").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Defined behaviour (BUGS.md): the transforms of a range of frame sets not
/// starting at the first are written.  Native writes them through a stream it
/// has already closed and exits 0 with no `.xf` file.
#[test]
fn transforms_of_a_range_of_sets_are_written() {
    let dir = work("sets-xf", &["frames.txt", "fts.mrc"]);
    let (rc, stdout) = run(
        &dir,
        &[
            "-saved",
            "frames.txt",
            "-xfext",
            "xf",
            "-sets",
            "2,3",
            "fts.mrc",
            "out.mrc",
        ],
    );
    assert_eq!(rc, Some(0), "{stdout}");
    let xf = std::fs::read_to_string(dir.join("fts.xf")).unwrap();

    // Sets 2 and 3 of the frame list hold 8 and 7 frames
    assert_eq!(xf.lines().count(), 15, "{xf}");
    assert!(xf.lines().all(|l| l.starts_with(" 1.00000    0.00000")));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Defined behaviour (BUGS.md): a corresponding stack's extended header is
/// copied to every output.  Native closes the stack inside the output loop and
/// fails copying it to the second output ("Copying extended header from stack
/// to output file").
#[test]
fn stack_extended_header_reaches_every_output() {
    let dir = work(
        "stack-ext",
        &["tsstackx.mrc", "ts1.mrc", "ts2.mrc", "ts3.mrc"],
    );
    let (rc, stdout) = run(
        &dir,
        &[
            "-stack",
            "tsstackx.mrc",
            "-dtotal",
            "2",
            "-unweight",
            "unw.mrc",
            "ts1.mrc",
            "ts2.mrc",
            "ts3.mrc",
            "out.mrc",
        ],
    );
    assert_eq!(rc, Some(0), "{stdout}");
    let stack = std::fs::read(dir.join("tsstackx.mrc")).unwrap();
    for out in ["out.mrc", "unw.mrc"] {
        let bytes = std::fs::read(dir.join(out)).unwrap();
        assert_eq!(&bytes[92..96], &24i32.to_le_bytes(), "{out}");
        assert_eq!(&bytes[1024..1048], &stack[1024..1048], "{out}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
