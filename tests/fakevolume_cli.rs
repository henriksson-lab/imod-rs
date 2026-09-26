//! Native-golden coverage for `fakevolume` (`IMOD/mrc/fakevolume.cpp`,
//! translated in `src/imod/mrc/fakevolume.rs`).
//!
//! Every row of `fixtures/fakevolume/cases.tsv` was run through the native
//! reference program by `fixtures/make-fakevolume-goldens.sh`, with standard
//! output captured through a pipe.  `golden/<case>.rc` is the exit status,
//! `.stdout` the standard output and `.o.mrc` the output volume when native
//! left one.  The volume is compared byte for byte with label stamps masked
//! (`common::mask_stamps`); stdout likewise (it carries the `NEW image file`
//! line and the `exitError` messages).
//!
//! Pruned 2026-09-26: 20 of 25 rows kept (dropped: modes 0 and 6 (generic output conversion; 1 and 12 kept), the large mixed scene, the oversize error (same message as e_range) and an explicit -trunc 0 cylinder); the rest stay in cases.tsv as `#full` rows (FULL=1 / IMOD_RS_FULL_CASES=1, fixtures/README.md).

mod common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

const PROGRAM: &str = "fakevolume";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/fakevolume")
}

#[test]
fn every_case_matches_native_golden() {
    let table = std::fs::read_to_string(fixture_dir().join("cases.tsv")).unwrap();
    let golden = fixture_dir().join("golden");
    let mut failures = Vec::new();
    let mut count = 0;
    for line in common::golden::case_rows(&table) {
        let (name, args) = line.split_once('\t').unwrap();
        let dir =
            std::env::temp_dir().join(format!("imod-rs-{PROGRAM}-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let output = common::imod_cmd(PROGRAM)
            .current_dir(&dir)
            .env(
                "AUTODOC_DIR",
                concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
            )
            .args(args.split_whitespace())
            .stdin(Stdio::null())
            .output()
            .unwrap();
        count += 1;
        let rc: i32 = common::golden::read_to_string(&golden.join(format!("{name}.rc")))
            .trim()
            .parse()
            .unwrap();
        if output.status.code() != Some(rc) {
            failures.push(format!(
                "{name}: exit {:?}, native {rc}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = common::golden::expect(&golden.join(format!("{name}.stdout")));
        if !stdout.matches_masked(&output.stdout, common::mask_stamps) {
            failures.push(format!(
                "{name}: stdout differs\n--- native\n{}\n--- ours\n{}",
                stdout.display(),
                String::from_utf8_lossy(&output.stdout)
            ));
        }
        let expected = common::golden::load(&golden.join(format!("{name}.o.mrc")));
        let written = std::fs::read(dir.join("o.mrc")).ok();
        match (expected, written) {
            (None, None) => {}
            (Some(g), Some(w)) if g.matches_masked(&w, common::mask_stamps) => {}
            (g, w) => failures.push(format!(
                "{name}: o.mrc differs (native {:?}, ours {:?} bytes)",
                g.map(|b| b.describe()),
                w.map(|b| b.len())
            )),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(count >= 20, "only {count} cases read");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// BUGS.md, fixed in translation: `fakevolume.cpp:199` fills the
/// `CylinderIsTruncated` flags up to the *sphere* count, so with no spheres
/// and `-trunc` entered once for two cylinders native reads the second flag
/// from the stack (two native runs gave two volumes).  The translation copies
/// the first flag to every cylinder, like the radii and densities: `-trunc 1`
/// once gives the `c2` golden, native's `-trunc 1 -trunc 1`.
#[test]
fn single_trunc_entry_applies_to_every_cylinder() {
    let dir =
        std::env::temp_dir().join(format!("imod-rs-fakevolume-{}-trunc1", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let output = common::imod_cmd("fakevolume")
        .current_dir(&dir)
        .env(
            "AUTODOC_DIR",
            concat!(env!("CARGO_MANIFEST_DIR"), "/IMOD/autodoc"),
        )
        .args(
            "-output o.mrc -size 30,30,10 -back 1 -offsets 1.5,-2,0.5 -cstart 5,5,1 \
             -cend 25,22,9 -cstart 3,20,2 -cend 20,4,8 -trunc 1 -cradii 1,3 \
             -cradii 2,2.5 -cdens 4,2 -cdens 1,6"
                .split_whitespace(),
        )
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let golden = common::golden::expect(&fixture_dir().join("golden/c2.o.mrc"));
    let written = std::fs::read(dir.join("o.mrc")).unwrap();
    assert!(golden.matches_masked(&written, common::mask_stamps));
    let _ = std::fs::remove_dir_all(&dir);
}
